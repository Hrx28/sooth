import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load Data (replace with your data loading mechanism)
@st.cache_data
def load_data():
    data = pd.read_csv(r'C:\Users\harsh\sooth\result_dataset.csv')

    # Ensure correct date format with day first
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    return data


# Function to get the top 10 best-selling products
def get_top_products(data):
    product_sales = data.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False)
    return product_sales.head(10).index.tolist()


# Function to train the time series model (using Exponential Smoothing)
def train_model(stock_id, data, forecast_weeks):
    stock_data = data[data['StockCode'] == stock_id].copy()
    stock_data['InvoiceDate'] = pd.to_datetime(stock_data['InvoiceDate'])
    stock_data.set_index('InvoiceDate', inplace=True)

    # Aggregate data by date to get total quantity sold per day
    daily_data = stock_data['Quantity'].resample('W').sum()

    # Split the data into train and test sets
    train_data = daily_data.iloc[:-forecast_weeks]
    test_data = daily_data.iloc[-forecast_weeks:]

    # Train the Exponential Smoothing model (non-seasonal as default)
    model = ExponentialSmoothing(train_data, trend="add")
    model_fit = model.fit()

    # Forecast the next 'forecast_weeks' weeks
    forecast = model_fit.forecast(forecast_weeks)

    return train_data, test_data, forecast, model_fit


# Streamlit App Design
def main():
    st.sidebar.title("Input Options")

    # Load data
    data = load_data()

    # Get top 10 best-selling products
    top_products = get_top_products(data)

    # Input for Stock ID and number of weeks
    stock_id = st.sidebar.selectbox("Select a Stock Code:", top_products)
    forecast_weeks = st.sidebar.slider("Number of Weeks to Forecast", min_value=1, max_value=15, value=5)

    # Display forecast button
    if st.sidebar.button("Show Forecast"):
        # Train model and forecast
        train_data, test_data, forecast, model_fit = train_model(stock_id, data, forecast_weeks)

        # Main Plot: Train vs Predicted Data
        st.header(f"Demand Overview for {stock_id}")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_data.index, train_data, label='Train Actual Demand', color='blue')
        ax.plot(test_data.index, test_data, label='Test Actual Demand', color='green')
        ax.plot(test_data.index, forecast, label='Test Predicted Demand', color='orange')
        ax.legend()
        ax.set_title(f"Actual vs Predicted Demand for {stock_id}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Demand')
        st.pyplot(fig)

        # Error Distribution Plots
        st.write("### Training Error Distribution")
        train_pred = model_fit.fittedvalues
        train_error = train_data - train_pred
        fig2, ax2 = plt.subplots()
        ax2.hist(train_error, bins=10, color='green', alpha=0.6)
        ax2.set_title("Training Error Distribution")
        st.pyplot(fig2)

        st.write("### Testing Error Distribution")
        test_error = test_data - forecast
        fig3, ax3 = plt.subplots()
        ax3.hist(test_error, bins=10, color='red', alpha=0.6)
        ax3.set_title("Testing Error Distribution")
        st.pyplot(fig3)

        # Prepare forecast data for download
        forecast_df = pd.DataFrame(forecast, columns=['Forecasted_Quantity'])
        forecast_df.index.name = 'Date'
        csv = forecast_df.to_csv(index=True)

        # Download forecast as CSV
        st.download_button(label="Download Forecast as CSV", data=csv, file_name=f'{stock_id}_forecast.csv', mime='text/csv')


# Running the app
if __name__ == "__main__":
    main()
