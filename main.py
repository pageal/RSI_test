#BASED ON tese articles:
# https://www.roelpeters.be/many-ways-to-calculate-the-rsi-in-python-pandas/
# https://towardsdatascience.com/algorithmic-trading-with-rsi-using-python-f9823e550fe0
# https://medium.com/codex/algorithmic-trading-with-relative-strength-index-in-python-d969cf22dd85

#!!!ta-lib instll and build is tied to Microsoft Visual C++
#so 'talib' it will not be used

#pip install pandas-ta
import pandas_ta as pta

# pandas_ta requires numpy version 1.20 or higher
# pip show numpy
# pip uninstall numpy
# pip install numpy

import numpy as np
import yfinance as yf
import pandas_datareader as web
import datetime as dt
from matplotlib import pyplot as plt

import pandas


def rsi(df, periods=14, ema=True):
    """
    Returns a pd.Series with the relative strength index.
    """
#    close_delta = df['close'].diff()
    close_delta = df.diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    if ema is True:
        # Use exponential moving average
        ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window=periods, min_periods=periods).mean()
        ma_down = down.rolling(window=periods, min_periods=periods).mean()

    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

'''
Basic strategy for RSI trading:  when the value leaves the overbought and oversold sections, it makes the appropriate trade. 
For example, if it leaves the oversold section, a buy trade is made. If it leaves the overbought section, a sell trade is made.
'''
def strategy_test(prices, rsi, rsi_length):
    buy_signals = [np.nan]*len(prices)
    sell_signals = [np.nan]*len(prices)
    profit = 0.0
    for i in range(rsi_length+1,len(prices)):
        if rsi[i - 1] <= 30 and rsi[i] > 30:
            buy_signals[i]=prices[i]
        elif rsi[i - 1] >= 70 and rsi[i] < 70:
            sell_signals[i]=prices[i]
    return buy_signals, sell_signals
    #print(f"profit: {profit}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    company = 'AAPL'
    rsi_length = 14
    test_days = 300

    test_end = dt.datetime.now()
    test_start = test_end - dt.timedelta(days=test_days+rsi_length)

    test_data = web.DataReader(company, "yahoo", test_start, test_end)
    #or using yfinance
    #test_data = yf.download(company, test_start, test_end)

    #CALCULATE RSI for each data point (starting from rsi_length-th)
    # first rsi_length RSI values will have numpy.nan value as there as not enough information for consistent RSI calculation
    # https://stackoverflow.com/questions/17534106/what-is-the-difference-between-nan-and-none
    rsi_ema = pta.rsi(test_data['Close'], length=rsi_length)
    rsi_sma = rsi(df=test_data['Close'], periods=rsi_length, ema=False)

    buy_signals, sell_signals = strategy_test(test_data['Close'], rsi_sma, rsi_length)

    fig = plt.figure()
    plt.plot(test_data.index, [70] * len(test_data.index), color="darkgrey", label="overbought")
    plt.plot(test_data.index, [30] * len(test_data.index), color="lightgray", label="oversold")
    plt.plot(test_data.index, rsi_ema, color="wheat", label="rsi ema")
    plt.plot(test_data.index, rsi_sma, color="goldenrod", label="rsi sma")
    plt.plot(test_data["Close"], color="sienna", label=f"{company} close price")
    plt.scatter(x=test_data.index, y=buy_signals, s=25, color='green', alpha=0.7,
                label=f"Buy signal")
    plt.scatter(x=test_data.index, y=sell_signals, s=25, color='red', alpha=0.7,
                label=f"Sell signal")
    plt.legend()
    plt.show()

    print("done")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
