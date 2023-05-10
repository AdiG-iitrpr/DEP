import pandas_datareader as web
import matplotlib.pyplot as plt 
import numpy as np
import random
import pandas as pd

class Indicators():
  def __init__(self,df):
    self.df = df

  # Exponential Moving Average
  def calculate_ema(self,prices, days, smoothing=2):
    ema = [sum(prices[:days]) / days]
    for price in prices[days:]:
        ema.append((price * (smoothing / (1 + days))) + ema[-1] * (1 - (smoothing / (1 + days))))
    return ema

  def plot_ema(self,currentpoint,k1 = 20,k2 = 200,toPlot = True):

    df = self.df
    stockPrice= df['Close'][max(0,currentpoint - k2):currentpoint+1]

    ema1 = [0] * len(stockPrice)
    ema2 = [0] * len(stockPrice)
    ema1[0] = stockPrice[0]
    ema2[0] = stockPrice[0]
    #ema = stockValue*(alpha /(k1 + 1)  +   ema[i-1] * ((1- alpha)/(k1 + 1)))

    for i in range(1,len(stockPrice)):
      point_gap = k1 + 1

      if i <= k1-1:
        point_gap = i + 2
      alpha = 2/(k1 + 1)
      ema1[i] = stockPrice[i] * alpha + (1 - alpha) * (ema1[i-1])

      point_gap = k2 + 1
      if i <= k2-1:
        point_gap = i + 2

      alpha = 2/(k2 + 1)
      ema2[i] = stockPrice[i] * alpha + (1 - alpha) * (ema2[i-1])

    # cond1 = ema1[-2] < ema2[-2]
    # cond2 = ema1[-2] > ema2[-2]

    cond1 = True 
    cond2 = True 

    if ema1[-1] >= ema2[-1] and cond1:
      return 1
    elif ema1[-1] <= ema2[-1] and cond2:
      return -1
    else:
      return 0
    

    if toPlot == True:
      print(f'Profit booked  : {profit}')

      plt.plot(stockPrice,label = "Stock Price",color = 'blue')
      plt.plot(ema1, label="EMA1 Values",color = 'green')
      plt.plot(ema2, label="EMA2 Values",color = 'red')
      plt.xlabel("Minutes")
      plt.ylabel("Price")
      plt.legend()
      plt.show()
    else: 
      print(k1,k2,profit)
    return profit 

  # RSI Algorithm
  def rsi(self,currentprice,period,toPlot = False):
    df = self.df
    stockPriceClose = df['Close'][max(0,currentprice - period):currentprice + 1]
    stockPriceOpen = df['Open'][max(0,currentprice - period):currentprice + 1]

    rsi_values = []
    rsi_values.append(50)

    for i in range(1,len(stockPriceClose)):
      avg_gain = 0 
      avg_loss = 0 
      loss_days = 0
      gain_days = 0 

      for j in range(i,max(i - period -1, -1),-1) : 
        if stockPriceClose[j] > stockPriceOpen[j]: 
          gain_days+=1
          avg_gain += stockPriceClose[j] - stockPriceOpen[j] 
        else : 
          loss_days+=1 
          avg_loss += stockPriceOpen[j] - stockPriceClose[j] 

      if gain_days != 0:
        avg_gain /= gain_days 
      
      if loss_days != 0:
        avg_loss /= loss_days 

      if avg_loss == 0: 
        rsi_values.append(100)
      else:
        rs = avg_gain/avg_loss  
        rsi_values.append(100 - 100/(1 + rs))  

      if(rsi_values[-1] > 70):
        return -1
      elif rsi_values[-1] < 30:
        return 1
      else:
        return 0

    if toPlot == True:
      plt.title("Stock Price")
      plt.plot(stockPriceClose,label = "Stock Price",color = 'blue')
      plt.xlabel("Minutes")
      plt.ylabel("Price")
      plt.legend()
      plt.show()
      print(' ')

      plt.title("RSI values")
      plt.axhline(80,label = 'Overbought')
      plt.axhline(20,label = 'Oversold')
      plt.xlabel("Minutes")
      plt.ylabel("RSI")
      plt.plot(rsi_values, label="RSI Value",color = 'green')
      plt.legend()
      plt.show()

  # Money Flow Index
  def mfi(self,limit,period,toPlot = False):
    df = self.df
    stockPriceClose = df['Close'][:limit] 
    stockPriceHigh = df['High'][:limit]  
    stockPriceOpen =df['Open'][:limit]
    volume = df['Volume'][:limit]

    stockPriceTypical = (stockPriceClose + stockPriceHigh )/2
    mfi = []
    raw_moneyFlow_values = []
    raw_moneyFlow_values.append(50)
    mfi.append(50)
    for i in range(1,len(stockPriceClose)):
      avg_gain = 0 
      avg_loss = 0 
      loss_days = 0
      gain_days = 0 

      raw_moneyFlow_values.append(stockPriceTypical[i] * volume[i])

      for j in range(i,max(i - period -1, -1),-1) : 
        
        
        if stockPriceClose[j] > stockPriceOpen[j]: 
          gain_days+=1
          avg_gain += raw_moneyFlow_values[j] 
        else : 
          loss_days+=1 
          avg_loss += raw_moneyFlow_values[j] 

      if gain_days != 0:
        avg_gain /= gain_days 
      
      if loss_days != 0:
        avg_loss /= loss_days 

      if avg_loss == 0: 
        mfi.append(100)
      else:
        rs = avg_gain/avg_loss  
        mfi.append(100 - 100/(1 + rs))  
    

    if toPlot == True:
      plt.title("Stock Price")
      plt.plot(stockPriceClose,label = "Stock Price",color = 'blue')
      plt.xlabel("Minutes")
      plt.ylabel("Price")
      plt.legend()
      plt.show()
      print(' ')

      plt.title("MFI values")
      plt.axhline(80,label = 'Overbought')
      plt.axhline(20,label = 'Oversold')
      plt.xlabel("Minutes")
      plt.ylabel("MFI")
      plt.plot(mfi, label="MFI Value",color = 'green')
      plt.legend()
      plt.show()

    # mfi(limit = 1000,period = 200,toPlot = True)

  # Moving Averages Convergence Divergence
  def MACD(self,data):
    price = data['Close']
    exp1 = price.ewm(span = 12, adjust = False).mean()
    exp2 = price.ewm(span = 26, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2)
    signal = pd.DataFrame(macd.ewm(span = 9, adjust = False).mean())
    macd = macd.values
    signal = signal.values
    sig = []
    for i in range(len(macd)):
      if(macd[i]>signal[i] and macd[i]<0):
        sig.append(1)
      elif(macd[i]<signal[i] and macd[i]>0):
        sig.append(-1)
      else:
        sig.append(0)
    return sig

  # Stochastic Indicator
  def stochastic(self,data):
    stochastic = []
    lookback = 14
    period = 3
    high = data['High']
    low = data['Low']
    price = data['Close']
    highh = high.rolling(lookback).max() 
    lowl = low.rolling(lookback).min()
    k = 100 * ((price - lowl) / (highh - lowl))
    d = k.rolling(period).mean()
    prices = data['Close']
    sig = []
    for i in range(prices.shape[0]):
      if(k[i]<20 and d[i]<20 and k[i]>d[i]):
        sig.append(1)
      elif(k[i]>80 and d[i]>80 and d[i]>k[i]):
        sig.append(-1)
      else:
        sig.append(0)
    return sig

  # Williams % R  
  def williams(self,data):
    william_r = []
    lookback = 14
    high = data['High']
    low = data['Low']
    price = data['Close']
    highh = high.rolling(lookback).max() 
    lowl = low.rolling(lookback).min()
    william_r = -100 * ((highh - price) / (highh - lowl))
    william_r = william_r.values
    closing_values = data['Close'].values
    prices = data['Close'].values
    sig = []
    for i in range(len(william_r)):
      if(william_r[i-1]>-80 and william_r[i]<-80 ):
        sig.append(1)
      elif(william_r[i-1]<-20 and william_r[i]>-20):
        sig.append(-1)
      else:
        sig.append(0)
    return sig

  # Donchian Indicator
  def donchian(self,data):
    don = []
    lookback = 14
    high = data['High']
    low = data['Low']
    price = data['Close']
    highh = high.rolling(lookback).max()
    lowl = low.rolling(lookback).min()
    don = (highh+lowl)/2
    don = don.values
    a = 250
    b = 500
    len = [i for i in range(a,b)]
    closing_values = data['Close'].values
    closing_values = [closing_values[x] for x in range(a,b)]
    high_values = highh.values
    high_values = [high_values[x] for x in range(a,b)]
    low_values = lowl.values
    low_values = [low_values[x] for x in range(a,b)]
    don_values = [don[x] for x in range(a,b)]
    # moving_average = [ma[x] for x in range(a,b)]
    # plt.plot(closing_values,label='Stock Prices')
    h = high.values
    l = low.values
    h = [h[i] for i in range(a,b)]
    l = [l[i] for i in range(a,b)]
    return

  # Average Directional Index
  def ADX(self,data):
    lookback = 14
    high = data['High']
    low = data['Low']
    close = data['Close']
    dm_pos = data['High'].diff()
    dm_pos[dm_pos<0] = 0
    dm_neg = data['Low'].diff()
    dm_neg[dm_neg>0] = 0
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.rolling(lookback).mean()
    plus_di = 100 * (dm_pos.ewm(alpha = 1/lookback).mean() / atr)
    minus_di = abs(100 * (dm_neg.ewm(alpha = 1/lookback).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth = adx.ewm(alpha = 1/lookback).mean()
    prices = data['Close']
    sig = []
    for i in range(1,prices.shape[0]):
      if(adx[i-1]<25 and adx[i]>25 and plus_di[i]>minus_di[i]):
        sig.append(1)
      elif(adx[i-1]<25 and adx[i]>25 and minus_di[i]>plus_di[i]):
        sig.append(-1)
      else:
        sig.append(0)
    return sig

  # Aroon
  def Aroon(self,data):
    period = 25
    high = data['High'].rolling(period+1).apply(lambda x: x.argmax(), raw=True)
    low = data['Low'].rolling(period+1).apply(lambda x: x.argmin(), raw=True)
    aroon_up = pd.Series((period - high) / period * 100, name='Aroon Up')
    aroon_down = pd.Series((period - low) / period * 100, name='Aroon Down')
    a = 300
    b = 450
    prices = data['Close']
    ma = prices.ewm(span = 200, adjust = False).mean()
    ma = ma.values
    sig = []
    isbought = False
    buy = 0
    sell = 0
    for i in range(prices.shape[0]):
      if(aroon_up[i]>=70 and aroon_down[i]<=30 and prices[i]>ma[i]):
        sig.append(1)
      elif(aroon_up[i]<=30 and aroon_down[i]>=70 and prices[i]<ma[i]):
        sig.append(-1)
      else:
        sig.append(0)
    return sig

  #Bollinger Bands
  def bollinger(self,data, t_window):
    data['SMA'] = data.Close.rolling(window=t_window).mean()
    data['STDDEV'] = data.Close.rolling(window=t_window).std()
    data['UPP_BAND' ]= data.SMA + 2*data.STDDEV
    data['LOW_BAND' ]= data.SMA - 2*data.STDDEV
    # data["BUY_SIG"] = np.where(data.LOW_BAND>data.Close , True , False)
    # data["SELL_SIG"] = np.where(data.UPP_BAND<data.Close , True , False)
    data =data.dropna()
    buy = []
    sell =[]
    open_pos = False 
    data = data.reset_index()
    for i in range(len(data)):
        if(data.LOW_BAND[i]>data.Close[i]):
          if open_pos == False:
            open_pos = True 
            buy.append(i)
        elif data.UPP_BAND[i]<data.Close[i]:
          if open_pos :
            open_pos = False
            sell.append(i)
    return data,buy,sell
  
  # Fibonacci Signal 
  def fibonacci_signal(self,data):
    df = self.df
    # Calculate Fibonacci retracement levels
    low = df['Adj Close'].min()
    high = df['Adj Close'].max()
    levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    diffs = [high - low, high - low, high - low, high - low, high - low, high - low, 0]
    retracements = [high - diff * level for level, diff in zip(levels, diffs)]

    # Generate buy and sell signals
    df['buy'] = np.where(df['Adj Close'] <= retracements[1], 1, 0)
    df['sell'] = np.where(df['Adj Close'] >= retracements[-2], 1, 0)

    # Plot Fibonacci retracement levels
    # plt.figure(figsize=(12, 6))
    # plt.plot(df.index, df['Adj Close'])
    # for level in retracements:
        # plt.axhline(level, linestyle='--', color='gray')
    # plt.show()

    # Return buy and sell signals
    return df[['Adj Close', 'buy', 'sell']]

  # Price Volume Trend 
  def price_volume_trend(self,data):
    df = self.df
    df['PVT'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) * df['Volume']
    df['PVT'] = df['PVT'].cumsum()
    plt.plot(data['PVT'].values)
    plt.title('PVT Values')
    return df['PVT'] 

  # Volume Weighted Average Price
  def vwap(self,data):
    df = self.df
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TPV'] = df['TP'] * df['Volume']
    vwap = df['TPV'].sum() / df['Volume'].sum()
    vwap_value = vwap(data)
    print("VWAP is:", vwap_value)
    plt.plot(vwap.values)
    return vwap


  # On Balance Volume Indicator
  def obv(self,data):
      df = self.df
      df['OBV'] = 0
      df['OBV'] = df['Volume'].where(df['Close'] >= df['Close'].shift(1), -df['Volume'])
      df['OBV'] = df['OBV'].cumsum()
      plt.plot(data['OBV'].values)
      plt.title('OBV Trend')
      plt.show()
      return df['OBV']

  # Price Volume Trend 
  def pvt(self,data):
      prices = data['Close'].values
      volumes = data['Volume'].values
      # Initialize the PVT value and signal list with the first value
      pvt = [0]
      signals = [0]
      # Loop through the prices and volumes to calculate the PVT value and signals
      for i in range(1, len(prices)):
          pvt_value = ((prices[i] - prices[i-1]) / prices[i-1]) * volumes[i] + pvt[-1]
          pvt.append(pvt_value)

          if pvt[-1] > pvt[-2]:
              signals.append(1)  # Buy signal
          elif pvt[-1] < pvt[-2]:
              signals.append(-1)  # Sell signal
          else:
              signals.append(0)  # No signal
      return signals
