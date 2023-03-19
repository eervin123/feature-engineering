#%%
import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Load data into a Pandas DataFrame
df = pd.read_csv('btc.csv', parse_dates=True, index_col=0)

# Define parameters to optimize using grid search
param_grid = {
    'length': [5, 10, 20, 50, 100],
    'forecast': [5, 30, 60, 90, 120],
}

#%% Generate technical analysis features using pandas_ta with default parameters
df.ta.sma(append=True)

# Define a function to optimize the parameters for the sma() function
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Hello World of technical analysis price crosses above sma
def optimize_sma_params(df, param_grid):
    results = []
    for length in param_grid['length']:
        for forecast in param_grid['forecast']:
            # Convert the forecast parameter to a timedelta object
            df[f'forecast_{forecast}'] = df['Close'].shift(-forecast).pct_change(forecast) # create target columns for prediction
            df.dropna(inplace=True) # drop NaN values
            # Generate technical analysis features using pandas_ta with current parameters
            df['signal'] = (df['Close'] / df.ta.sma(length=length, append=True))-1 # create signal column
            df.dropna(inplace=True)
            # Calculate the mean squared error of the signal column compared to each of the target columns
            df[f'error_{forecast}'] = mean_squared_error(df['signal'], df[f'forecast_{forecast}']) # calculate error for each forecast
            print(df)
            # Append the results to a list   
            results.append(f'length: {length}, forecast: {forecast}, error: {df[f"error_{forecast}"].mean()}' )
            
            print(f'length: {length}, forecast: {forecast}, error: {df[f"error_{forecast}"].mean()}')
    return results       
# %%
results = optimize_sma_params(df, param_grid)
# %%
