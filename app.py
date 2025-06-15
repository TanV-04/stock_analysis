import streamlit as st
import pickle
import yfinance as yf

# loading the trained model
with open("model/stockPredictor.pkl", "rb") as f:
    model = pickle.load(f)  # deserializes the model object

st.title("Stock Market Dashboard and Predictor")
ticker = st.text_input(
    "enter stock ticker (example, AAPL): ", "AAPL"
)  # input text field to enter stock ticker (default ticker is AAPL)
if ticker:
    try:
        company_name = yf.Ticker(ticker).info.get("longName")
        # if not company_name:
        #     company_name = "Unknown Company"
        st.subheader(f"Company: {company_name}")  # display company name

        # debugging
        # info = yf.Ticker(ticker).info
        # st.write(info)

        data = yf.Ticker(ticker).history(period="1mo")

        if len(data) < 8:
            st.warning("not enough data to make a prediction. try another ticker")
            st.stop()

        st.line_chart(data["Close"])

        # predicting the next day's closing price
        st.subheader("Next day stock prediction")
        X_pred = (
            data[["Open", "High", "Low", "Close", "Volume"]]
            .iloc[-8:-1]
            .values.flatten()  # flatten the OHLCV to a single array for model input
            .reshape(1, -1)
        )
        prediction = model.predict(X_pred)[0]
        st.metric(
            label="Predicted Next Close Price", value=f"${prediction:.2f}"
        )  # display the prediction using st.metric

        # lets users choose the multiple companies to compare stock performance
        st.subheader("compare trends with other companies")
        otherTickers = st.multiselect(
            "Select other companies to compare (e.g., MSFT, GOOGL, AMZN):",
            ["MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"],
        )

        # fetches historical closing prices and combines them into a dictionary and display a multiline chart for comparison
        if otherTickers:
            compareData = {} # dictionary to store stock comparison data to hold the historical closing prices for each company
            allTickers = [ticker] + otherTickers # combine the original ticker with the additional tickers selected

            # loop through all the tickers to fetch the data
            for tk in allTickers:
                try:
                    hist = yf.Ticker(tk).history(period="1mo")["Close"] # fetch one month of historical data and select only the Closing prices
                    compareData[tk] = hist # add this to the compareData with the ticker symbol as the key
                except:
                    st.warning(f"Could not load data for {tk}")

            if compareData: # if the compareData has valid entries, pass to a line chart 
                st.line_chart(compareData)

    except Exception as e:
        st.error(f"Error: {e}")  # catching the actual error
