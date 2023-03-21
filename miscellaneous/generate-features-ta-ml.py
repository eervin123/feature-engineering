'''
This file uses pandas_ta to build all the technical indicators and then appends them to 
the dataframe.  It then saves the dataframe to a csv file.  This file can then be used by 
an ML model to predict the price of Bitcoin.

Note: the `#%%` is a VSCode specific comment that tells the IDE to treat the code as a
cell in a Jupyter notebook.  This is useful for running the code in the IDE while keeping the *.py file
'''

#%% Import libraries
import pandas as pd
import pandas_ta as ta


#%% Load Bitcoin CSV file
def load_data():
    file_path = '../data/btc.csv'
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    return df

def build_all_indicators():
    df = load_data()
    df.ta.strategy('All',verbose=True)
    return df

if __name__ == '__main__':
    df=build_all_indicators()
    print(df.tail(10))
    df.to_csv('../data/btc_with_all_features.csv')


