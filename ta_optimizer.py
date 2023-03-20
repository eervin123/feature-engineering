import numpy as np
import pandas as pd
import talib as ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging 
import pickle

class TechnicalIndicatorOptimizer:
    def __init__(self, df, indicator_function, indicator_name, param_name, param_values, forecast_periods, indicator_type='single', threshold=0.001):
        """
        Initialize the TechnicalIndicatorOptimizer class with the input parameters.

        :param df: pandas DataFrame containing the historical price data with columns ['Open', 'High', 'Low', 'Close'].
        :param indicator_function: the function to calculate the technical indicator.
        :param indicator_name: the name of the technical indicator.
        :param param_name: the name of the parameter to optimize.
        :param param_values: a list of values for the parameter to optimize.
        :param forecast_periods: a list of forecast periods to evaluate the models.
        :param indicator_type: the type of indicator, either 'single' or 'stochastic'. Default is 'single'.
        :param threshold: the threshold used to determine the accuracy of the model's predictions. Default is 0.001.
        """
        logging.info(f"Initializing TechnicalIndicatorOptimizer for {indicator_name}")
        self.df = df
        self.indicator_function = indicator_function
        self.indicator_name = indicator_name
        self.param_name = param_name
        self.param_values = param_values
        self.forecast_periods = forecast_periods
        self.threshold = threshold
        self.indicator_type = indicator_type

    def _calculate_indicator(self, param_value):
        """
        Calculate the technical indicator using the input parameter value.

        :param param_value: the parameter value for the technical indicator.
        """
        logging.info(f"Calculating {self.indicator_name} with {self.param_name}: {param_value}")
        self.df[f'{self.indicator_name}_signal'] = self.indicator_function(self.df['Close'], length=param_value)

    def _calculate_stochastic(self, k_period, d_period):
        """
        Calculate the stochastic oscillator using the input K and D period values.

        :param k_period: the K period for the stochastic oscillator.
        :param d_period: the D period for the stochastic oscillator.
        """
        logging.info(f"Calculating {self.indicator_name} with k_period: {k_period}, d_period: {d_period}")

        self.df['stoch_k'], self.df['stoch_d'] = ta.STOCH(self.df['High'], self.df['Low'], self.df['Close'], fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
        self.df['stoch_signal'] = self.df['stoch_k'] - self.df['stoch_d']



    def fit_models(self):
        """
        Fit a random forest model for each forecast period using the technical indicator values.
        """
        logging.info(f"Fitting models for {self.indicator_name}")

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
        """
        Evaluate the performance of the technical indicator using the input parameter value.

        :param param_value: the parameter value for the technical indicator.
        :return: a dictionary containing the results of the evaluation.
        """
        try:
            logging.info(f"Evaluating {self.indicator_name} with {self.param_name}: {param_value}")
            if self.indicator_type == 'stochastic':
                k_period, d_period = param_value
                self._calculate_stochastic(k_period, d_period)
            else:
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
            if self.indicator_type == 'stochastic':
                return {'params': {'k_period': k_period, 'd_period': d_period}, 'mse_scores': mse_scores, 'rmse_scores': rmse_scores, 'accuracy_scores': accuracy_scores}
            else:
                return {'params': {self.param_name: param_value}, 'mse_scores': mse_scores, 'rmse_scores': rmse_scores, 'accuracy_scores': accuracy_scores}
        except Exception as e:
            logging.error(f"Error evaluating {self.indicator_name} with {self.param_name}: {param_value} - {str(e)}")
            if self.indicator_type == 'stochastic':
                return {'params': {'k_period': k_period, 'd_period': d_period}, 'error': str(e)}
            else:
                return {'params': {self.param_name: param_value}, 'error': str(e)}

    def optimize(self):
        """
        Optimize the technical indicator by evaluating its performance using different parameter values.

        :return: a dictionary containing the best results for the technical indicator.
        """
        logging.info(f"Optimizing {self.indicator_name}")
        results = [self.evaluate_indicator_param(param_value) for param_value in tqdm(self.param_values)]

        # Find the best result based on mean of MSE scores
        best_result = min(results, key=lambda x: np.mean(list(x['mse_scores'].values())))

        return best_result

    def plot_analysis_dashboard(self, result, **kwargs):
        """
        Plot the analysis dashboard, including the MSE, RMSE, and accuracy for each forecast period.

        :param result: a dictionary containing the results to plot.
        :param kwargs: optional keyword arguments for customizing the plot, e.g., custom title.
        """
        import matplotlib.pyplot as plt
        logging.info(f"Plotting analysis dashboard for {self.indicator_name}")
        logging.info(f'Results: {result} - Kwargs: {kwargs}')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        title = kwargs.get('title', f"{self.indicator_name.upper()} Analysis Dashboard")
        fig.suptitle(title)
        if not isinstance(result, dict):
            logging.warning("Invalid result input for plot_analysis_dashboard, expected a dictionary.")
            return

        mse_scores = result.get('mse_scores', None)
        rmse_scores = result.get('rmse_scores', None)

        if mse_scores is None or rmse_scores is None:
            logging.warning("Missing 'mse_scores' or 'rmse_scores' in the result dictionary.")
            return

        accuracy_scores = result['accuracy_scores']

        if self.indicator_type == 'stochastic':
            param_str = f"{self.param_name[0]}: {result[self.param_name[0]]}, {self.param_name[1]}: {result[self.param_name[1]]}"
        else:
            param_str = f"{self.param_name}: {result[self.param_name]}"

        axes[0].bar(mse_scores.keys(), mse_scores.values())
        axes[0].set_title(f"{self.indicator_name.upper()} MSE ({param_str})")
        axes[0].set_xlabel("Forecast Period")
        axes[0].set_ylabel("MSE")

        axes[1].bar(rmse_scores.keys(), rmse_scores.values())
        axes[1].set_title(f"{self.indicator_name.upper()} RMSE ({param_str})")
        axes[1].set_xlabel("Forecast Period")
        axes[1].set_ylabel("RMSE")

        axes[2].bar(accuracy_scores.keys(), accuracy_scores.values())
        axes[2].set_title(f"{self.indicator_name.upper()} Accuracy ({param_str})")
        axes[2].set_xlabel("Forecast Period")
        axes[2].set_ylabel("Accuracy (%)")

        plt.show()

    def save_best_result(self, best_result, filename="best_result.pkl"):
        with open(filename, "wb") as file:
            pickle.dump(best_result, file)

    def load_best_result(self, filename="best_result.pkl"):
        with open(filename, "rb") as file:
            best_result_loaded = pickle.load(file)
        return best_result_loaded

