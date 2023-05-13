import streamlit as st
from datetime import date
from indicators import MACD,stochastic
import pandas as pd
import yfinance as yf
from plotly import graph_objs as go
from tkinter import *
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.express as px

@st.cache_data
def load_local(uploaded_file):
    data = pd.read_excel(uploaded_file)
    return data

@st.cache_data
def load_data(ticker,period):
    data = yf.download(tickers = ticker, period = period ,interval = "1d", ignore_tz = True, prepost = False)
    data.reset_index(inplace=True)
    return data

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Buy And Sell')

data_type = ('Load Data from local device','Upload from Yahoo Finance')
dec = st.selectbox('Select source of data',data_type)

if(dec=='Load Data from local device'):
    uploaded_file = st.file_uploader("Choose a file",type='xlsx')
    if uploaded_file is not None:
        data_load_state = st.text('Loading data...')
        data = load_local(uploaded_file)
        data_load_state.text('Loading data... done!')
else:
    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'NQ=F')
    selected_stock = st.selectbox('Select Company', stocks)
    n_years = st.slider('Years of Stock Data:', 1, 10)
    period = n_years
    period = str(period)+'y'
    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock,period)
    data_load_state.text('Loading data... done!')

st.subheader('Raw data Head')
st.write(data.head())
st.subheader('Data Size')
st.write(data.shape[0],'data points')

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

def plot_indicator(df):
    fig = go.Figure()
    fig = make_subplots(rows=2, cols=1,  shared_xaxes=True )
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"),row=1,col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"),row=1,col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['ma'], name="moving_average"),row=1,col=1)
    # fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    # st.plotly_chart(fig)
    fig.add_trace(go.Scatter(x=data['Date'][1:],y=df['macd'], name="macd"),row=2,col=1)
    fig.add_trace(go.Scatter(x=data['Date'][1:], y=df['signal'], name="signal"),row=2,col=1)
    fig.layout.update(xaxis2_rangeslider_visible=True)
    # fig.add_trace(go.Scatter(x=data['Date'][1:],y=df['sig'], name="sig"),row=2,col=1)
    st.plotly_chart(fig)
    # plt.plot(macd)
    # plt.show()

def plot_stochas(df):
    fig = go.Figure()
    fig = make_subplots(rows=2, cols=1,  shared_xaxes=True )
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"),row=1,col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"),row=1,col=1)
    # fig.add_trace(go.Scatter(x=data['Date'], y=data['ma'], name="moving_average"),row=1,col=1)
    # fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    # st.plotly_chart(fig)
    fig.add_trace(go.Scatter(x=data['Date'][1:],y=df['k'], name="%K"),row=2,col=1)
    fig.add_trace(go.Scatter(x=data['Date'][1:], y=df['d'], name="%D"),row=2,col=1)
    fig.layout.update(xaxis2_rangeslider_visible=True)
    # fig.add_trace(go.Scatter(x=data['Date'][1:],y=df['sig'], name="sig"),row=2,col=1)
    st.plotly_chart(fig)
    # plt.plot(macd)
    # plt.show()
	
plot_raw_data()

algorithms = ('EMA','RSI','MFI','MACD','STOCHAS','WILLIAMS%R','DONCHIAN','ADX','AROON','BOLLINGER','FIBONACCI','PVT','VWAP','OBV')
selected_algo = st.selectbox('Select your choice of Algorithms', algorithms)
# for algo in selected_algo:
if(selected_algo=='MACD'):
    final,df,num_trades = MACD(data)
    # st.write(macd)
    # st.write(len(df['macd'].values))
    # st.write(len(df['signal'].values))
    # st.write(len(df['ma'].values))
    # st.write(len(df['sig'].values))
    plot_indicator(df)
    # st.write('Number of Trades :',num_trades)
    st.write('Invested Amount :',100000)
    st.write('Returned Amount :',final)
    st.write('Profit :',final-100000)
    sig = df['sig'].values
    dates = df['Date'].values
    close = df['Close'].values
    x1 = []
    buy = []
    x2 = []
    sell = []
    for i in range(len(sig)):
        if(sig[i]==1):
            x1.append(dates[i])
            buy.append(close[i])
        elif(sig[i]==-1):
            x2.append(dates[i])
            sell.append(close[i])
    tab1, tab2 = st.tabs(["Buy Data", "Sell Data"])
    with tab1:
        st.write('Optimal buy dates with prices :',pd.DataFrame({'Date':x1,'Price':buy}))
    with tab2:
        st.write('Optimal sell dates with prices :',pd.DataFrame({'Date':x2,'Price':sell}))
elif(selected_algo=='STOCHAS'):
    final,df = stochastic(data)
    plot_stochas(df)
    # st.write('Number of Trades :',num_trades)
    st.write('Invested Amount :',100000)
    st.write('Returned Amount :',final)
    st.write('Profit :',final-100000)
    sig = df['sig'].values
    dates = df['Date'].values
    close = df['Close'].values
    x1 = []
    buy = []
    x2 = []
    sell = []
    for i in range(len(sig)):
        if(sig[i]==1):
            x1.append(dates[i])
            buy.append(close[i])
        elif(sig[i]==-1):
            x2.append(dates[i])
            sell.append(close[i])
    tab1, tab2 = st.tabs(["Buy Data", "Sell Data"])
    with tab1:
        st.write('Optimal buy dates with prices :',pd.DataFrame({'Date':x1,'Price':buy}))
    with tab2:
        st.write('Optimal sell dates with prices :',pd.DataFrame({'Date':x2,'Price':sell}))
