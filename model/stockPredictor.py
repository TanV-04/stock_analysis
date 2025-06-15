from sklearn.ensemble import RandomForestRegressor  # good for regression problems
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import yfinance as yf

# a model that predicts stock prices based on the historical data from Yahoo Finance (yfinance)


# makes sure that we have at least 8 days of data (7 for input and 1 for prediction)
def stock_predictor_model(input_data):
    if len(input_data) < 8:
        return ValueError("not enough data to use a 7-day window")

    windowSize = 7  # each prediction is based on the past 7 days of stock data
    features = []  # stores the OHLCV values of the past 7 days
    targets = []  # stores the "Close" price for the next day (prediction target)

    # extract OHLCV for each of the 7 days (7 x 5 = 35) and flatten it to a single 1D array of 35 numbers
    for i in range(windowSize, len(input_data) - 1):
        pastSevenDays = input_data.iloc[i - windowSize : i]
        features.append(
            pastSevenDays[["Open", "High", "Low", "Close", "Volume"]].values.flatten()
        )
        targets.append(input_data["Close"].iloc[i + 1])

    # convert the lists to numpy arrays
    X = np.array(features)
    y = np.array(targets)  # close price for the day after the 7-day window

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    return model


ticker = "AAPL"
data = yf.Ticker(ticker).history(
    period="2mo"
)  # fetches two months of historical stock data for Apple
model = stock_predictor_model(data)  # trains a model using that data


with open("model/stockPredictor.pkl", "wb") as f:
    pickle.dump(model, f)
