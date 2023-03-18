import pandas as pd
import pandas_ta as ta
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import joblib  # for saving models
import numpy as np
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data into a Pandas DataFrame
df = pd.read_csv('btc.csv', parse_dates=True, index_col=0)
model_store = 'models/'

# Define parameters to optimize using grid search
param_grid = {
    'length': [5, 10, 20, 50, 100],
    'forecast': [5, 30, 60, 90, 120],
}

# Generate technical analysis features using pandas_ta with default parameters
df.ta.sma(append=True)

# Define a function to optimize the parameters for the sma() function
# Utilize joblib to parallelize the optimization process
def optimize_sma_params_single(length, forecast, df):
    df[f'forecast_{forecast}'] = df['Close'].shift(-forecast).pct_change(forecast)
    df['signal'] = (df['Close'] / ta.sma(df['Close'], length=length)) - 1
    df.dropna(inplace=True)
    error = mean_squared_error(df['signal'], df[f'forecast_{forecast}'])
    return {'length': length, 'forecast': forecast, 'error': error}

def optimize_sma_params(df, param_grid):
    results = Parallel(n_jobs=-1)(delayed(optimize_sma_params_single)(length, forecast, df.copy())
                                   for length in param_grid['length']
                                   for forecast in param_grid['forecast'])
    return pd.DataFrame(results)
# Optimize the SMA parameters
logging.info('Optimizing SMA parameters...')
results = optimize_sma_params(df, param_grid)

# Find the parameters with the lowest error for each forecast period
best_params = results.sort_values(['forecast', 'error']).groupby('forecast').first().reset_index()


def fit_forecast_model(forecast: int, length: int, df: DataFrame) -> Tuple[int, RandomForestRegressor]:
    logging.info(f'Fitting model for forecast {forecast} with parameters: length={length}')
    df['signal'] = (df['Close'] / ta.sma(df['Close'], length=length)) - 1
    y = df['Close'].shift(-forecast).pct_change(forecast)
    Xy = pd.concat([df['signal'], y], axis=1).dropna()
    X_train, X_val, y_train, y_val = train_test_split(Xy.iloc[:, :-1], Xy.iloc[:, -1], test_size=0.2, shuffle=False)
    X_train = X_train.values.reshape(-1, 1)
    X_val = X_val.values.reshape(-1, 1)
    y_train = y_train.values.ravel()
    y_val = y_val.values.ravel()
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return forecast, model

# Fit a Random Forest model with the best parameters for each forecast period
logging.info('Fitting Random Forest models...')
best_forecasts_lengths: List[Tuple[int, int]] = [(forecast, best_params.loc[best_params['forecast'] == forecast, 'length'].iloc[0]) 
                           for forecast in param_grid['forecast']]

with tqdm_joblib(tqdm(desc="Fitting Models", total=len(best_forecasts_lengths))) as progress_bar:
    models = dict(Parallel(n_jobs=-1)(
        delayed(fit_forecast_model)(forecast, length, df.copy()) for forecast, length in best_forecasts_lengths
    ))


# Evaluate the model performance on the validation set
logging.info('Evaluating model performance...')
scores = {}
length_scores = {}
for forecast, model in tqdm(models.items()):
    length = best_params.loc[best_params['forecast'] == forecast, 'length']
    df['signal'] = (df['Close'] / df.ta.sma(length=length, append=True)) - 1
    y_val = df['Close'].shift(-forecast).pct_change(forecast)[-len(y_val):]
    X_val = df['signal'][-len(y_val):]
    X_val = X_val.values.reshape(-1, 1)
    y_val = y_val.values.reshape(-1, 1)
    y_pred = model.predict(X_val)
    score = mean_squared_error(y_pred, y_val)
    scores[forecast] = score

    if length not in length_scores:
        length_scores[length] = {'mse_sum': 0, 'count': 0}
    length_scores[length]['mse_sum'] += score
    length_scores[length]['count'] += 1

# Calculate the average MSE for each SMA length
for length, data in length_scores.items():
    data['avg_mse'] = data['mse_sum'] / data['count']

# Determine the SMA length with the lowest average MSE
best_sma_length = min(length_scores, key=lambda x: length_scores[x]['avg_mse'])
logging.info(f"Best SMA length: {best_sma_length}")

print('MSE Scores:', scores)

# Save the models to disk
logging.info('Saving models to disk...')
for forecast, model in models.items():
    joblib.dump(model, model_store+f'model_forecast_{forecast}.joblib')
    
# Calculate the RMSE scores for each forecast period
rmse_scores = {forecast: np.sqrt(score) for forecast, score in scores.items()}

# Calculate the RMSE scores for each SMA length
length_rmse_scores = {length: np.sqrt(data['avg_mse']) for length, data in length_scores.items()}

logging.info('RMSE Scores:')
logging.info(rmse_scores)
logging.info('RMSE Scores per SMA length:')
logging.info(length_rmse_scores)

print('RMSE Scores:', rmse_scores)


# Define the prediction threshold
threshold = 0.001  # This is the percentage variance that we would consider a correct prediction
logging.info(f'Using a prediction threshold of {threshold} to determine accuracy...')
# Calculate prediction accuracy for each forecast period
accuracy_scores = {}
df['signal'] = (df['Close'] / df.ta.sma(length=best_sma_length, append=True)) - 1
for forecast, model in models.items():
    X_val = df['signal'][-len(y_val):]
    y_val = df['Close'].shift(-forecast).pct_change(forecast)[-len(y_val):]
    X_val = X_val.values.reshape(-1, 1)
    y_val = y_val.values.reshape(-1, 1)
    y_pred = model.predict(X_val)

    # Count the predictions that fall within the threshold
    correct_predictions = np.abs(y_pred - y_val.ravel()) <= threshold
    accuracy = np.mean(correct_predictions) * 100
    accuracy_scores[forecast] = accuracy

# Print the prediction accuracy for each forecast period
for forecast, accuracy in accuracy_scores.items():
    print(f"The {forecast}-minute forecast model is correct {accuracy:.4f}% of the time at a tolerance of {threshold:.4f}%.")

# Plot accuracy scores for each forecast period
plt.figure()
plt.plot(accuracy_scores.keys(), accuracy_scores.values(), marker='o', color='orange')
plt.xlabel('Forecast Period')
plt.ylabel('Accuracy (%)')
plt.title('Model Performance (Accuracy)')
plt.show()

# Plot MSE scores for each forecast period
plt.plot(scores.keys(), scores.values(), marker='o')
plt.xlabel('Forecast Period')
plt.ylabel('Mean Squared Error')
plt.title('Model Performance')
plt.show()

# Plot RMSE scores for each forecast period
plt.figure()
plt.plot(rmse_scores.keys(), rmse_scores.values(), marker='o')
plt.xlabel('Forecast Period')
plt.ylabel('Mean Squared Error')
plt.title('Model Performance (RMSE)')
plt.show()



