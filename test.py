
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
import os
from sklearn.impute import SimpleImputer


def evaluate_sma_length(sma_length, df, models, threshold):
    df['signal'] = (df['Close'] / ta.sma(df['Close'], length=sma_length)) - 1

    mse_scores = {}
    accuracy_scores = {}
    for forecast, model in models.items():
        y_val = df['Close'].shift(-forecast).pct_change(forecast).dropna()
        X_val = df['signal'][-len(y_val):]
        X_val = X_val.values.reshape(-1, 1)
        y_val = y_val.values.reshape(-1, 1)

        # Handle missing values using SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_val = imputer.fit_transform(X_val)

        y_pred = model.predict(X_val)

        valid_indices = ~np.isnan(y_pred) & ~np.isnan(y_val.ravel())
        y_pred = y_pred[valid_indices]
        y_val = y_val[valid_indices]

        mse = mean_squared_error(y_pred, y_val)
        mse_scores[forecast] = mse

        correct_predictions = np.abs(y_pred - y_val.ravel()) <= threshold
        accuracy = np.mean(correct_predictions) * 100
        accuracy_scores[forecast] = accuracy

    rmse_scores = {forecast: np.sqrt(mse) for forecast, mse in mse_scores.items()}
    return {'length': sma_length, 'mse_scores': mse_scores, 'rmse_scores': rmse_scores, 'accuracy_scores': accuracy_scores}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data into a Pandas DataFrame
df = pd.read_csv('btc.csv', parse_dates=True, index_col=0)
model_store = 'models/'
os.makedirs(model_store, exist_ok=True) # create the models directory if it doesn't exist

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
best_params = results.sort_values('error').groupby('forecast').first().reset_index()

def fit_forecast_model(forecast: int, length: int, df: DataFrame) -> Tuple[int, RandomForestRegressor]:
    logging.info(f'Fitting model for forecast {forecast} with parameters: length={length}')
    df['signal'] = (df['Close'] / ta.sma(df['Close'], length=length)) - 1
    y = df['Close'].shift(-forecast).pct_change(forecast)
    Xy = pd.concat([df['signal'], y], axis=1).dropna()
    X_train, X_val, y_train, y_val = train_test_split(Xy.iloc[:, :-1], Xy.iloc[:, -1], test_size=0.2, shuffle=False)
    X_train = X_train.values.reshape(-1, 1)
    X_val = X_val.values.reshape(-1, 1)
    y_train = y_train.values.ravel()
    y_val = y_val.values
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
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
y_val = None
for forecast, model in tqdm(models.items()):
    length = int(best_params.loc[best_params['forecast'] == forecast, 'length'].item())
    df['signal'] = (df['Close'] / df.ta.sma(length=length)) - 1
    if y_val is None:
        y_val = df['Close'].shift(-forecast).pct_change(forecast).dropna()
    else:
        y_val = df['Close'].shift(-forecast).pct_change(forecast)[-len(y_val):]
    X_val = df['signal'][-len(y_val):]
    X_val = X_val.values.reshape(-1, 1)
    y_val = y_val.values.reshape(-1, 1)
    y_pred = model.predict(X_val)

    # Drop rows with NaN values in y_pred and y_val
    valid_indices = ~np.isnan(y_pred) & ~np.isnan(y_val.ravel())
    y_pred = y_pred[valid_indices]
    y_val = y_val[valid_indices]

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
threshold = 0.001  # This is the percentage variance that we would consider a correct prediction
sma_length_results = {}
for sma_length in tqdm(param_grid['length'], desc="Evaluating SMA lengths"):
    sma_length_results[sma_length] = evaluate_sma_length(sma_length, df.copy(), models, threshold)

for sma_length, results in sma_length_results.items():
    print(f"SMA length: {sma_length}")
    print("Accuracy Scores:", results['accuracy_scores'])
    print("MSE Scores:", results['mse_scores'])
    print("RMSE Scores:", results['rmse_scores'])
    print()

# Define the prediction threshold
threshold = 0.001  # This is the percentage variance that we would consider a correct prediction
logging.info(f'Using a prediction threshold of {threshold} to determine accuracy...')
# Calculate prediction accuracy for each forecast period
accuracy_scores = {}
df['signal'] = (df['Close'] / df.ta.sma(length=best_sma_length, append=True)) - 1
for forecast, model in models.items():
    X_val = df['signal'][-len(y_val):]
    y_val = df['Close'].shift(-forecast).pct_change(forecast)[-len(y_val.ravel()):]
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

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(8, 18))

# Plot accuracy scores for each forecast period
axes[0].plot(accuracy_scores.keys(), accuracy_scores.values(), marker='o', color='orange')
axes[0].set_xlabel('Forecast Period')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Model Performance (Accuracy)')

# Plot MSE scores for each forecast period
axes[1].plot(scores.keys(), scores.values(), marker='o')
axes[1].set_xlabel('Forecast Period')
axes[1].set_ylabel('Mean Squared Error')
axes[1].set_title('Model Performance (MSE)')

# Plot RMSE scores for each forecast period
axes[2].plot(rmse_scores.keys(), rmse_scores.values(), marker='o')
axes[2].set_xlabel('Forecast Period')
axes[2].set_ylabel('Root Mean Squared Error')
axes[2].set_title('Model Performance (RMSE)')

# Adjust layout
plt.tight_layout()

# Show the subplots
plt.show()
