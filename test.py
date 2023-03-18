import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import joblib # for saving models
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data into a Pandas DataFrame
df = pd.read_csv('btc.csv', parse_dates=True, index_col=0)

# Define parameters to optimize using grid search
param_grid = {
    'length': [5, 10, 20, 50, 100],
    'forecast': [5, 30, 60, 90, 120],
}

# Generate technical analysis features using pandas_ta with default parameters
df.ta.sma(append=True)

# Define a function to optimize the parameters for the sma() function
def optimize_sma_params(df, param_grid):
    results = []
    for length in param_grid['length']:
        for forecast in param_grid['forecast']:
            # Convert the forecast parameter to a timedelta object
            df[f'forecast_{forecast}'] = df['Close'].shift(-forecast).pct_change(forecast)
            # Generate technical analysis features using pandas_ta with current parameters
            df['signal'] = (df['Close'] / df.ta.sma(length=length, append=True)) - 1
            # Drop NaN values
            df.dropna(inplace=True)
            # Calculate the mean squared error of the signal column compared to each of the target columns
            error = mean_squared_error(df['signal'], df[f'forecast_{forecast}'])
            # Append the results to a list   
            results.append({'length': length, 'forecast': forecast, 'error': error})
    return pd.DataFrame(results)

# Optimize the SMA parameters
logging.info('Optimizing SMA parameters...')
results = optimize_sma_params(df, param_grid)

# Find the parameters with the lowest error for each forecast period
best_params = results.sort_values(['forecast', 'error']).groupby('forecast').first().reset_index()

# Fit a Random Forest model with the best parameters for each forecast period
models = {}
for forecast in param_grid['forecast']:
    length = best_params.loc[best_params['forecast'] == forecast, 'length'].iloc[0]
    df['signal'] = (df['Close'] / df.ta.sma(length=length, append=True)) - 1
    y = df['Close'].shift(-forecast).pct_change(forecast)
    Xy = pd.concat([df['signal'], y], axis=1).dropna()
    X_train, X_val, y_train, y_val = train_test_split(Xy.iloc[:, :-1], Xy.iloc[:, -1], test_size=0.2, shuffle=False)
    X_train = X_train.values.reshape(-1, 1)
    X_val = X_val.values.reshape(-1, 1)
    y_train = y_train.values.ravel()
    y_val = y_val.values.ravel()
    logging.info(f'Fitting model for forecast {forecast} with parameters: length={length}')
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    models[forecast] = model

# Evaluate the model performance on the validation set
scores = {}
for forecast, model in tqdm(models.items()):
    length = best_params.loc[best_params['forecast'] == forecast, 'length'].iloc[0]
    df['signal'] = (df['Close'] / df.ta.sma(length=length, append=True)) - 1
    X_val = df['signal'][-len(y_val):]
    y_val = df['Close'].shift(-forecast).pct_change(forecast)[-len(y_val):]
    X_val = X_val.values.reshape(-1, 1)
    y_val = y_val.values.reshape(-1, 1)
    y_pred = model.predict(X_val)
    score = mean_squared_error(y_pred, y_val)
    scores[forecast] = score

print('MSE Scores:', scores)

# Plot MSE scores for each forecast period
plt.plot(scores.keys(), scores.values(), marker='o')
plt.xlabel('Forecast Period')
plt.ylabel('Mean Squared Error')
plt.title('Model Performance')
plt.show()

# Save the models to disk
for forecast, model in models.items():
    joblib.dump(model, f'model_forecast_{forecast}.joblib')

rmse_scores = {k: np.sqrt(v) for k, v in scores.items()}
print('RMSE Scores:', rmse_scores)


# Define the prediction threshold
threshold = 0.001 # This is the percentage variance that we would consider a correct prediction

# Calculate prediction accuracy for each forecast period
accuracy_scores = {}
for forecast, model in models.items():
    length = best_params.loc[best_params['forecast'] == forecast, 'length'].iloc[0]
    df['signal'] = (df['Close'] / df.ta.sma(length=length, append=True)) - 1
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
    print(f"The {forecast}-minute forecast model is correct {accuracy:.2f}% of the time at a tolerance of {threshold:.2f}%.")

import matplotlib.pyplot as plt

# Plot RMSE scores for each forecast period
plt.figure()
plt.plot(scores.keys(), scores.values(), marker='o')
plt.xlabel('Forecast Period')
plt.ylabel('Mean Squared Error')
plt.title('Model Performance (RMSE)')
plt.show()

# Plot accuracy scores for each forecast period
plt.figure()
plt.plot(accuracy_scores.keys(), accuracy_scores.values(), marker='o', color='orange')
plt.xlabel('Forecast Period')
plt.ylabel('Accuracy (%)')
plt.title('Model Performance (Accuracy)')
plt.show()
