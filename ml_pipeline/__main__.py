import os
import pandas as pd
import argparse
import sklearn.model_selection
import concurrent.futures
import ml_models

from copy import deepcopy
from datetime import datetime, timedelta
from time import process_time
from collections import namedtuple

from . import io
from .utils import optimize_numeric_dtype, optimize_category_dtype
from .func_expression_helper import transform_func_params

Dataset = namedtuple('Dataset', 'X y')

def load_datasets(conf, data_path):
    # Load dataframe
    df = io.load_data(conf, data_path)

    # Memory optimization for data frame
    if conf.get('memory_optimization', False):
        exclude_from_optimize = conf.get('dtype', None)
        columns_to_optimize = df.columns.drop(exclude_from_optimize, errors='ignore') if exclude_from_optimize else df.columns
        if columns_to_optimize.any():
            print('Optimizing memory usage for df')
            df[columns_to_optimize] = optimize_numeric_dtype(df[columns_to_optimize], 'float64', 'float')
            df[columns_to_optimize] = optimize_numeric_dtype(df[columns_to_optimize], 'int64', 'integer')
            df[columns_to_optimize] = optimize_category_dtype(df[columns_to_optimize], conf.get('category_columns', None))

    print('Split data to training and testing sets')
    y_col = conf['y_column']
    X_col = df.columns.drop(y_col)

    train_size = conf.get('train_size', 0.5)
    test_size = conf.get('test_size', 0.5)
    random_state = conf.get('random_state', None)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df[X_col], df[y_col], test_size=test_size, train_size=train_size, random_state=random_state)

    return Dataset(X=X_train, y=y_train), Dataset(X=X_test, y=y_test)

def load_models(conf):

    # Preload preprocessors
    print("Loading preprocessors from config")
    def load_preprocessors(p_def):
        p_name = p_def.pop('name')
        p_id = p_def.pop('id')
        p = ml_models.get_preprocessor_by_name(p_name)
        return p(p_id, **p_def)

    preprocessor_dict = dict()
    if preprocessors := conf.get('preprocessors', None):
        for p_def in preprocessors:
            p = load_preprocessors(p_def)
            preprocessor_dict[p.preprocessor_id] = p

    print("Loading models from config")
    models = list()
    for model_conf in conf['models']:
        try:
            model_cls = ml_models.get_model_by_name(model_conf.pop('name'))
            model_id = model_conf.pop('id')
            params = model_conf.pop('params', dict())
            
            # Handle preprocessors
            if preprocessors := model_conf.pop('preprocessors', None):
                p_list = list()
                for p_def in preprocessors:
                    p = deepcopy(preprocessor_dict[p_def]) if isinstance(p_def, str) else load_preprocessors(p_def)
                    p_list.append(p)
                model_conf['preprocessors'] = p_list

            # Handle hyper-parameter tuning
            if hpt := model_conf.get('hpt', None):
                
                # Transform func expression
                params = transform_func_params(params)

                # Wrap every single param in a list if hpt is supported
                params = [ x if isinstance(x, list) else [x] for x in params]

                # Add scoring to hpt
                if 'scoring' in conf and 'scoring' not in hpt:
                    hpt['scoring'] = conf['scoring']

            models.append(model_cls(model_id=model_id, params=params, **model_conf))

        except Exception as err:
            print(f'Failed to configure model:\n{type(err)}: {err}')

    return models

def train_and_predict(model, ds_train, ds_test, scores: list, export_dir: str):
    
    performance = None

    try:
        start_time = process_time()

        # Train model
        print(f"{model.model_id}: Training")
        model.train(X=ds_train.X, y=ds_train.y)

        # Test model
        print(f"{model.model_id}: Testing")
        y_pred = model.predict(X=ds_test.X)

        # Evaluate model
        print(f"{model.model_id}: Evaluating performance")
        performance = model.get_performance(y_true=ds_test.y, y_pred=y_pred, scores=scores)

        time_elapsed = timedelta(seconds=process_time() - start_time)
        performance['time_elapsed'] = time_elapsed
        
        # Export model and performance
        if export_dir:
            print(f"{model.model_id}: Exporting")
            io.export_model(model, f'{export_dir}/{model.model_id}.joblib')
            io.export_model_performance(performance, f'{export_dir}/{model.model_id}_performance.csv')

        print(f"{model.model_id}: Finished\n{performance}")

    except Exception as err:
        print(f"Failed processing model ({model.model_id}):\n{type(err)}: {err}")

    return model, performance

def main(config_path, data_path):

    # load config from json
    config = io.load_json(config_path)
    if not config:
        raise Exception(f'Failed to read config from {config_path}')

    # load datasets
    datasets = load_datasets(config['data'], data_path)
    if not datasets:
        raise Exception(f'Failed to load datasets')

    # Split datasets
    ds_train, ds_test = datasets

    # load all models from config
    models = load_models(config)
    if not models:
        raise Exception("No available model")
    
    # Prepare for export directory
    export_dir = config.get("export_dir", None)
    if export_dir:		
        try:
            # Create dir if not exists
            io.create_dir(export_dir)

            # Create a sub-folder using the current time
            export_dir = f'{export_dir}/{datetime.now().strftime("%Y-%m-%dT%H-%M")}'
            io.create_dir(export_dir)

            # Export config for future reference
            io.copyfile(config_path, f'{export_dir}/config{os.path.splitext(config_path)[1]}')

        except Exception as err:
            print(f'Failed to create directory at {export_dir}: \n{type(err)}: {err}')
            export_dir = None
    else:
        print("Export directory not specified. Nothing will be exported.")

    # create an empty dictionary to store performances
    print("Commence training")
    reports = dict()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = list()
        for model in models:
            # Setup progress bar
            total = 3
            if export_dir:
                total += 1
            futures.append(executor.submit(train_and_predict, model=model, ds_train=ds_train, ds_test=ds_test, scores=config["scores"], export_dir=export_dir))
        
        for future in concurrent.futures.as_completed(futures):
            model, performance = future.result()
            # Add performance to reports
            if performance:
                reports[model.model_id] = performance

    df_performance = pd.DataFrame.from_dict(reports, orient='index')
    if scoring := config.get('scoring'):
        try:
            df_performance = df_performance.sort_values(by=scoring)
        except Exception as err:
            print(f"Failed to sort performance by {scoring}")

    # Print Summary
    print('\n' * 3)
    print("=" * 16)
    print("Model performance summary:")
    print("=" * 16)
    print(df_performance)

    # Export summary
    if export_dir:
        export_path = f"{export_dir}/performance_summary.csv"
        try:
            df_performance.to_csv(export_path, index=True)
        except Exception as err:
            print(f"Failed to save performance csv to {export_path}: \n{type(err)}: {err}")

if __name__ == '__main__':
    # Parse from args from cli
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='path to config.json')
    parser.add_argument('data_path', type=str, help='path to data')
    
    args = parser.parse_args()
    
    main(args.config_path, args.data_path)