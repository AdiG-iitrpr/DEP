import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tkinter import *

def MACD(df):
    price = df['Close']
    exp1 = price.ewm(span = 12, adjust = False).mean()
    exp2 = price.ewm(span = 26, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2)
    signal = pd.DataFrame(macd.ewm(span = 9, adjust = False).mean())
    macd = macd.values
    signal = signal.values
    macd = [0]+macd
    signal = [0]+signal
    hist = [macd[i]-signal[i] for i in range(len(macd))]
    ma = price.ewm(span = 200, adjust = False).mean()
    ma = [0]+ma
    prices = df['Close'].values
    sig = []
    isbought = False
    buy = 0
    sell = 0
    df['macd'] = macd
    df['signal'] = signal
    df['ma'] = ma
    for i in range(len(prices)):
        if(macd[i]>signal[i] and isbought==True):
            sig.append(0)
        elif(macd[i]>signal[i] and macd[i]<0 and prices[i]>ma[i] and isbought == False):
        # elif(macd[i]>signal[i] and macd[i]<0 and isbought == False):
            sig.append(1)
            buy+=1
            isbought = True
        elif(macd[i]<signal[i] and macd[i]>0 and prices[i]<ma[i] and isbought == True):
        # elif(macd[i]<signal[i] and macd[i]>0 and isbought == True):
            sig.append(-1)
            sell+=1
            isbought = False
        else:
            sig.append(0)
    if(buy>sell):
        for i in range(len(sig)-1,-1,-1):
            if(sig[i]==1):
                sig[i] = 0
                buy-=1
                break
    #   print('Buy sell number :',buy)
    df['sig'] = sig
    my = 100000
    profit = 0
    bought = 0
    for i in range(len(sig)):
        if(sig[i]==1):
            bought = my//prices[i]
            my -= bought*prices[i]
            # profit-=prices[i]
        elif(sig[i]==-1):
            my += bought*prices[i]
            # profit+=prices[i]
#   print('Final Amount :',my)
    return my,df,buy

def stochastic(df):
  stochastic = []
  lookback = 14
  period = 3
  high = df['High']
  low = df['Low']
  price = df['Close']
  highh = high.rolling(lookback).max() 
  lowl = low.rolling(lookback).min()
  k = 100 * ((price - lowl) / (highh - lowl))
  d = k.rolling(period).mean()
  df['k'] = k
  df['d'] = d
  prices = df['Close']
  sig = []
  isbought = False
  buy = 0
  sell = 0
  for i in range(prices.shape[0]):
    if(k[i]<20 and d[i]<20 and k[i]>d[i] and isbought==True):
      sig.append(0)
    elif(k[i]<20 and d[i]<20 and k[i]>d[i] and isbought == False):
      sig.append(1)
      buy+=1
      isbought = True
    elif(k[i]>80 and d[i]>80 and d[i]>k[i] and isbought==True):
      sig.append(-1)
      sell+=1
      isbought = False
    elif(k[i]>80 and d[i]>80 and d[i]>k[i] and isbought==False):
      sig.append(0)
    else:
      sig.append(0)
  if(buy>sell):
    for i in range(prices.shape[0]-1,-1,-1):
      if(sig[i]==1):
        sig[i] = 0
        buy-=1
        break
#   print(buy,sell)
  df['sig'] = sig
  my = 100000
  profit = 0
  bought = 0
  for i in range(prices.shape[0]):
    if(sig[i]==1):
      bought = my//prices[i]
      my -= bought*prices[i]
      # profit-=prices[i]
    elif(sig[i]==-1):
      my += bought*prices[i]
      # profit+=prices[i]
  return my,df

# data = yf.download(tickers = 'NQ=F', period = '5y' ,interval = "1d", ignore_tz = True, prepost = False)
# final,macd,signal,ma = MACD(data)
# print(len())
# print(final)
# print(len(macd),len(signal),len(ma))
# plt.plot(macd)
# plt.plot(signal)
# plt.show()