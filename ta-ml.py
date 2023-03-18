#%% Import libraries
import pandas as pd
import pandas_ta as ta


#%% Load Bitcoin CSV file
def load_data():
    file_path = '~/Documents/data/hourly_BTCUSDT.csv'
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    return df

def build_all_indicators():
    df = load_data()
    df.ta.strategy('All',verbose=True)
    # print(df)
    return df

if __name__ == '__main__':
    df=build_all_indicators()
    print(df)


# %%


# %%
