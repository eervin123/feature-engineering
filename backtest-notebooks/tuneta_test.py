import logging
import multiprocessing
from tuneta.tune_ta import TuneTA
import pandas as pd
from pandas_ta import percent_return
from sklearn.model_selection import train_test_split
import yfinance as yf
import warnings

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Filter out warnings
warnings.filterwarnings('ignore')

def main():
    try:
        logging.info("Starting script.")

        # Download data set from yahoo, calculate next day return and split into train and test
        logging.info("Downloading data from Yahoo Finance.")
        X = yf.download("SPY", period="10y", interval="1d", auto_adjust=True)
        logging.info("Data download complete. Calculating next day returns.")
        
        y = percent_return(X.Close, offset=-1)
        logging.info("Splitting data into train and test sets.")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, shuffle=False)

        # Initialize with 1 core and show trial results
        logging.info("Initializing TuneTA with 6 job.")
        tt = TuneTA(n_jobs=6, verbose=True)

        # Optimize indicators with reduced trials
        logging.info("Starting optimization of indicators.")
        tt.fit(X_train, y_train,
            indicators=['tta'],
            ranges=[(4, 30)],
            trials=5,  
            early_stop=10,
        )
        logging.info("Optimization complete.")

        # Show time duration in seconds per indicator
        logging.info("Showing fit times.")
        tt.fit_times()

        # Show correlation of indicators to target
        logging.info("Generating report on correlation of indicators to target.")
        tt.report(target_corr=True, features_corr=True)

        # Select features with at most x correlation between each other
        logging.info("Pruning features based on correlation.")
        tt.prune(max_inter_correlation=.7)

        # Show correlation of indicators to target and among themselves
        logging.info("Generating final report on correlation of indicators.")
        tt.report(target_corr=True, features_corr=True)
        # Save the report to a csv
        tt.report(target_corr=True, features_corr=True)

        # Add indicators to X_train
        logging.info("Transforming training data with selected features.")
        features = tt.transform(X_train)
        X_train = pd.concat([X_train, features], axis=1)

        # Add same indicators to X_test
        logging.info("Transforming test data with selected features.")
        features = tt.transform(X_test)
        X_test = pd.concat([X_test, features], axis=1)
        
        # Create a concatenated dataframe to save to csv
        xdf = pd.concat([X_train, X_test], axis=0)
        xdf.to_csv("X.csv")
        ydf = pd.concat([y_train, y_test], axis=0)
        ydf.to_csv("y.csv")
        full_df = pd.concat([xdf, ydf], axis=1)
        full_df.to_csv("full.csv")

        logging.info("Script finished successfully.")

    except Exception as e:
        logging.error("An error occurred: %s", e)

if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)
    logging.info("Main script starting.")
    main()
    logging.info("Main script finished.")