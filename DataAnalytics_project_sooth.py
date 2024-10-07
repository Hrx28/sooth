import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(layout="wide", page_title="Demand Forecasting")


st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .css-1d391kg {
        background-color: #ffffff;
    }
    .stSidebar {
        background-color: #e6f3ff;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
    }
    .stButton>button:hover {
        background-color: #135a8c;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    data = pd.read_csv(r'C:\Users\harsh\sooth\result_dataset.csv')
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    return data


def get_top_products(data):
    product_sales = data.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False)
    return product_sales.head(10).index.tolist()


def train_model(stock_id, data, forecast_weeks):
    stock_data = data[data['StockCode'] == stock_id].copy()
    stock_data['InvoiceDate'] = pd.to_datetime(stock_data['InvoiceDate'])
    stock_data.set_index('InvoiceDate', inplace=True)

    daily_data = stock_data['Quantity'].resample('W').sum()

    train_data = daily_data.iloc[:-forecast_weeks]
    test_data = daily_data.iloc[-forecast_weeks:]

    model = ExponentialSmoothing(train_data, trend="add")
    model_fit = model.fit()

    forecast = model_fit.forecast(forecast_weeks)

    return train_data, test_data, forecast, model_fit

def main():
    st.sidebar.title("Input Options")


    data = load_data()


    top_products = get_top_products(data)


    stock_id = st.sidebar.selectbox("Select a Stock Code:", top_products)
    forecast_weeks = st.sidebar.slider("Number of Weeks to Forecast", min_value=1, max_value=15, value=5)


    if st.sidebar.button("Show Forecast"):

        train_data, test_data, forecast, model_fit = train_model(stock_id, data, forecast_weeks)


        st.title("Demand Forecasting")
        st.header(f"Demand Overview for {stock_id}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_data.index, y=train_data, name='Train Actual Demand', line=dict(color='#1f77b4')))
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data, name='Test Actual Demand', line=dict(color='#2ca02c')))
        fig.add_trace(go.Scatter(x=test_data.index, y=forecast, name='Test Predicted Demand', line=dict(color='#ff7f0e')))
        fig.update_layout(title=f"Actual vs Predicted Demand for {stock_id}", xaxis_title='Date', yaxis_title='Demand')
        st.plotly_chart(fig, use_container_width=True)


        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Training Error Distribution")
            train_pred = model_fit.fittedvalues
            train_error = train_data - train_pred
            fig_train = px.histogram(train_error, nbins=20, color_discrete_sequence=['#2ca02c'])
            fig_train.update_layout(title="Training Error Distribution")
            st.plotly_chart(fig_train, use_container_width=True)

        with col2:
            st.subheader("Testing Error Distribution")
            test_error = test_data - forecast
            fig_test = px.histogram(test_error, nbins=20, color_discrete_sequence=['#d62728'])
            fig_test.update_layout(title="Testing Error Distribution")
            st.plotly_chart(fig_test, use_container_width=True)


        forecast_df = pd.DataFrame(forecast, columns=['Forecasted_Quantity'])
        forecast_df.index.name = 'Date'
        csv = forecast_df.to_csv(index=True)


        st.download_button(label="Download Forecast as CSV", data=csv, file_name=f'{stock_id}_forecast.csv', mime='text/csv')


if __name__ == "__main__":
    main()
