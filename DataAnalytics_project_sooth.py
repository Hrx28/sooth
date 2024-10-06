import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@st.cache_data
def load_data():
    data = pd.read_csv(r'result_dataset.csv')

  
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

       
        forecast_df = pd.DataFrame(forecast, columns=['Forecasted_Quantity'])
        forecast_df.index.name = 'Date'
        csv = forecast_df.to_csv(index=True)

       
        st.download_button(label="Download Forecast as CSV", data=csv, file_name=f'{stock_id}_forecast.csv', mime='text/csv')



if __name__ == "__main__":
    main()
