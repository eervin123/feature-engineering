# Backtesting and Feature Generation

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Contributing](../CONTRIBUTING.md)

## About <a name = "about"></a>

Personal repo where I test and explore various strategies and methods for generating features on market data.

## Getting Started <a name = "getting_started"></a>

Simply clone the repo and install the dependencies in the requirements.txt
Many of the files use a `data` folder and a `models` folder. If they are not already in this repo, you can create them.
In order to get you started with data, use the [get_data.ipynb](miscellaneous/get-data.ipynb) file to download yourself some data. Save some bitcoin 1m data in a csv file titled `/data/btc.csv`

Also as an added convenience, if on linux of mac just create a symbolic link to your data folder wherever that may be on your system. navigat to the root folder of your project and type `ln -s <target folder location> <alias folder name>` that will create a shortcut that looks like a folder in your project root folder

### Playing with the data

You might start with the random forest files they are probably the cleanest and seem to be the most productive.

### Prerequisites

If using conda create an environment and then activate the environment. You can either conda install each of the packages or install them from the requirements.txt file. Or, you can `pip install -r requirements.txt`

[requirements.txt](requirements.txt) should have all you need.

## Usage <a name = "usage"></a>

This repo is a collection of files and a constant work in progress. I am always testing and bouncing around on new files. I will try to keep the repo organized with good docstrings in each of the files, but I'm only human, so go easy on me.

### For a good starting point check out

[The Random Forest Notebook](random-forest/random-forest-testing-momo-v1.2.ipynb)

- The file uses a random forest to build a model based on features that I generated on btc data. I tried to comment things pretty well. The model will then make predictions on the future price moves of bitcoin.
- Here is an example of the output. This is predicting 30 minutes into the future within +/-10bps.

        - For a threshold of: 0.001
        - MSE: 1.2729619990324905e-06
        - RMSE: 0.0011
        - Accuracy: 83.8493%
        - Direction Accuracy: 96.6824% 
          - This happened 486013 times out of 502690
        - R-squared: 0.9802
    ![Actual-vs-Predicted](output/Actual-v-Predicted.png)
