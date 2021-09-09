import ml_models
import ml_pipeline
import joblib
import jstyleson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import statsmodels.api as sm
from sklearn.metrics import r2_score

#region Plots

def plot_performance(data: pd.DataFrame, true_label, pred_label, title, sample=None):
    fig, ax = plt.subplots()

    r2 = r2_score(data[true_label], data[pred_label])

    if sample:    
        data = data.sample(n=sample if isinstance(sample, int) else None, frac=sample if isinstance(sample, float) else None)

    x = data[pred_label]
    y = data[true_label]

    plt.plot(x, y, linestyle='', marker='o', markersize=0.5, alpha=1, color='blue')

    # R2
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', label=f'R2={r2:.3f}')

    plt.xlabel(pred_label)
    plt.ylabel(true_label)
    plt.title(title)
    plt.figlegend()
        
    return plt.gcf()

def plot_feature_importances(data: pd.Series, title, top=10):

    data = data[:top].sort_values(ascending=True)

    plt.figure()
    plt.title(title)
    plt.barh(np.arange(len(data)), data.to_numpy(), color="blue", align="center")

    plt.xlabel('Importance')
    plt.ylabel('Feature names')
    plt.yticks(np.arange(len(data)), data.index.to_numpy())

    return plt.gcf()

#endregion

def get_feature_importances(model):
    estimator = model._estimator
    params_name = model._feature_names
    params_name = [ x.replace(f'Rating Area_', ' ')for x in params_name ]

    importance = getattr(estimator, 'feature_importances_', None)
    if not isinstance(importance, type(None)):
        return pd.Series(importance, index=params_name).sort_values(ascending=False)

    importance = getattr(estimator, 'coef_', None)
    if not isinstance(importance, type(None)):
        return pd.Series(importance, index=params_name).sort_values(ascending=False)

    raise Exception("No importance found")


def main(config, model_dir, data_path):

    model_titles = config['model_titles']
    scores = config['scores']
    plot_samples = config.get('plot_samples', 1.0)
    plot_dpi = config.get('plot_dpi', 100)

    # Load pipeline config
    with open(f'{model_dir}/config.jsonc', 'r') as f:
        pipeline_config = jstyleson.loads(f)

    # Check if random_state is defined. We need that to get the testing dataset of as the time of training
    if isinstance(pipeline_config['data'].get('random_state', None), type(None)):
        raise Exception("Cannot reproduce testing dataset at time of training without random state")

    # Load datasets
    ds_train, ds_test = ml_pipeline.load_datasets(pipeline_config['data'], data_path)

    # Create reports dir
    export_dir = os.path.join(model_dir, 'reports')
    if not os.path.exists(export_dir):
        os.mkdir(export_dir)

    # Load all models
    for root, _, filenames in os.walk(model_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1] != '.joblib':
                continue

            # Load model
            model = joblib.load(os.path.join(root, filename))

            title = model_titles[model.model_id]

            # Write report in markdown
            report = ""
            report += f"## Report for {model.model_id}\n\n"

            # Training Performance
            y_pred = model.predict(ds_train.X)
            plot_filename = f'{model.model_id}_plot_training_performance.png'
            df_performance = pd.DataFrame({'Actual value': ds_train.y, 'Predicted value': y_pred})

            fig = plot_performance(df_performance, 'Actual value', 'Predicted value', f'Training prediction performance ({title})', sample=plot_samples)
            fig.savefig(f'{export_dir}/{plot_filename}', dpi=plot_dpi,  bbox_inches='tight')
            
            report += f'### Training prediction performance\n'
            report += pd.Series(model.get_performance(y_true=ds_train.y, y_pred=y_pred, scores=scores)).to_markdown(headers=["Metric", "Value"], index=True) + '\n\n'
            report += f'![performance_plot_performance](./{plot_filename})\n'

            # Testing Performance
            y_pred = model.predict(ds_test.X)
            plot_filename = f'{model.model_id}_plot_testing_performance.png'
            df_performance = pd.DataFrame({'Actual value': ds_test.y, 'Predicted value': y_pred})
            fig = plot_performance(df_performance, 'Actual value', 'Predicted value', f'Testing prediction performance ({title})', sample=plot_samples)
            fig.savefig(f'{export_dir}/{plot_filename}', dpi=plot_dpi,  bbox_inches='tight')

            report += f'### Testing prediction performance\n'
            report += pd.read_csv(f'{root}/{model.model_id}_performance.csv', header=None).to_markdown(headers=["Metric", "Value"], index=False) + '\n\n'
            report += f'![performance_plot_performance](./{plot_filename})\n'
            
            # Feature importance
            feature_importances = get_feature_importances(model)

            plot_filename =  f'{model.model_id}_plot_feature_importances.png'
            fig = plot_feature_importances(feature_importances, f'Feature importances ({title})')
            fig.savefig(f'{export_dir}/{plot_filename}', dpi=plot_dpi, bbox_inches='tight')
            
            report += f"### Feature importance\n"
            report += feature_importances.to_markdown(headers=["Feature name", "Importance"]) + '\n\n'
            report += f'![performance_path_plot_feature_importances](./{plot_filename})\n'

            # Calculate p-value using statsmodels
            if isinstance(model, ml_models.LinearRegression):
                report += f"### Regression result\n"
                X = model.preprocess(ds_train.X)
                y = ds_train.y
                X2 = sm.add_constant(X)
                est = sm.OLS(y, X2)
                est2 = est.fit()
                report += str(est2.summary()) + '\n'

            # Export file
            file_path = f'{export_dir}/{model.model_id}_report.md'
            with open(file_path, 'w+') as f:
                f.write(report)
                print(f'File exported: {file_path}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='path to config.json')
    parser.add_argument('data_path', type=str, help='path to data.csv')
    parser.add_argument('model_dir', type=str, help='path to config.json')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = jstyleson.load(f)

    main(config, args.model_dir, args.data_path)