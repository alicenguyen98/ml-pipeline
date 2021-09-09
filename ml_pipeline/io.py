"""
Handles Loading and Exporting
"""
import os.path
import pandas as pd
import jstyleson
import shutil
import joblib

def load_json(path):
    with open(path, 'r') as f:
        return jstyleson.load(f)

def load_data(conf, data_path):
    print(f'Loading data from path {data_path}')

    # Check file type by extension
    ext = os.path.splitext(data_path)[1]

    read_func = None
    if ext == '.csv':
        read_func = pd.read_csv
    elif ext in {'.xls', '.xlsx'}:
        read_func = pd.read_excel
    else:
        raise NotImplementedError('Unsupported data file type')

    # Drop columns
    usecols = None
    if drop_columns := conf.get('drop', None):
        cols = read_func(data_path, nrows=1)
        usecols = [x for x in cols if x not in drop_columns]
        print(f"Use columns: {usecols}")

    # Check if there's specified column data type in config
    kwargs = {x: conf.get(x, None) for x in ['header', 'index_col', 'dtype', 'nrows']}

    df = read_func(data_path, usecols=usecols, **kwargs)
    return df

def create_dir(export_dir):
    if not os.path.exists(export_dir):
        print(f'Creating directory {export_dir}')
        os.makedirs(export_dir)

def export_json(data, export_path):
    with open(export_path, 'w+') as f:
        f.write(jstyleson.dumps(data))

def export_model(model, export_path):
    try:
        print(f'Exporting model to {export_path}')
        joblib.dump(model, export_path)
        print(f"Model ({model.model_id}) exported successfully at {export_path}")
    except Exception as err:
        print(f"Failed to export model ({model.model_id}) at {export_path}: \n{type(err)}: {err}")

def export_model_performance(performance, export_path):
    try:
        print(f'Exporting model to {export_path}')
        pd.Series(performance).to_csv(export_path, header=False)
        print(f"Model performance exported successfully at {export_path}")
    except Exception as err:
        print(f"Failed to export model performance at {export_path} : \n{type(err)}: {err}")

def copyfile(src, dst):
    shutil.copyfile(src, dst)