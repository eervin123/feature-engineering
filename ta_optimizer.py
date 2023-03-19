import numpy as np
import pandas as pd
import talib as ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt


class TechnicalIndicatorOptimizer:
    def __init__(self, df, indicator_function, indicator_name, param_name, param_values, forecast_periods, threshold=0.001):
        self.df = df
        self.indicator_function = indicator_function
        self.indicator_name = indicator_name
        self.param_name = param_name
        self.param_values = param_values
        self.forecast_periods = forecast_periods
        self.threshold = threshold

    def _calculate_indicator(self, param_value):
        self.df[f'{self.indicator_name}_signal'] = self.indicator_function(self.df['Close'], length=param_value)

    def fit_models(self):
        self.models = {}
        for forecast in self.forecast_periods:
            self.df = self.df.dropna(subset=[f'{self.indicator_name}_signal'])
            X = self.df[f'{self.indicator_name}_signal'].iloc[:-forecast].values.reshape(-1, 1)
            y = self.df['Close'].pct_change(forecast).dropna().iloc[:-forecast].values.reshape(-1, 1)

            # Check if lengths are equal, if not, truncate X and y to the smallest length
            min_length = min(len(X), len(y))
            X = X[:min_length]
            y = y[:min_length]

            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y.ravel())
            self.models[forecast] = model



    def evaluate_indicator_param(self, param_value):
        self._calculate_indicator(param_value)
        self.fit_models()

        mse_scores = {}
        accuracy_scores = {}
        for forecast, model in self.models.items():
            y_val = self.df['Close'].shift(-forecast).pct_change(forecast).dropna()
            X_val = self.df[f'{self.indicator_name}_signal'][-len(y_val):]
            X_val = X_val.values.reshape(-1, 1)
            y_val = y_val.values.reshape(-1, 1)

            y_pred = model.predict(X_val)

            valid_indices = ~np.isnan(y_pred) & ~np.isnan(y_val.ravel())
            y_pred = y_pred[valid_indices]
            y_val = y_val[valid_indices]

            mse = mean_squared_error(y_pred, y_val)
            mse_scores[forecast] = mse

            correct_predictions = np.abs(y_pred - y_val.ravel()) <= self.threshold
            accuracy = np.mean(correct_predictions) * 100
            accuracy_scores[forecast] = accuracy

        rmse_scores = {forecast: np.sqrt(mse) for forecast, mse in mse_scores.items()}
        return {self.param_name: param_value, 'mse_scores': mse_scores, 'rmse_scores': rmse_scores, 'accuracy_scores': accuracy_scores}

    def optimize(self):
        results = [self.evaluate_indicator_param(param_value) for param_value in tqdm(self.param_values)]
        best_result = min(results, key=lambda x: np.mean(list(x['mse_scores'].values())))
        return best_result

    def plot_analysis_dashboard(self, results, **kwargs):
        plot_title = kwargs.get('plot_title', f'{self.indicator_name} Optimization Dashboard')
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(plot_title)

        # Plot MSE scores
        mse_scores = {r[self.param_name]: r['mse_scores'] for r in results}
        mse_df = pd.DataFrame(mse_scores).T
        mse_df.plot(kind='bar', ax=axes[0])
        axes[0].set_title('MSE Scores')
        axes[0].set_xlabel(self.param_name)
        axes[0].set_ylabel('MSE')

        # Plot RMSE scores
        rmse_scores = {r[self.param_name]: r['rmse_scores'] for r in results}
        rmse_df = pd.DataFrame(rmse_scores).T
        rmse_df.plot(kind='bar', ax=axes[1])
        axes[1].set_title('RMSE Scores')
        axes[1].set_xlabel(self.param_name)
        axes[1].set_ylabel('RMSE')

        # Plot accuracy scores
        accuracy_scores = {r[self.param_name]: r['accuracy_scores'] for r in results}
        accuracy_df = pd.DataFrame(accuracy_scores).T
        accuracy_df.plot(kind='bar', ax=axes[2])
        axes[2].set_title('Accuracy Scores')
        axes[2].set_xlabel(self.param_name)
        axes[2].set_ylabel('Accuracy (%)')

        plt.tight_layout()
        plt.show()

