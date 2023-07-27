import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt



class Bot:
    def __init__(self):
        pass

    def get_position(self, open, close, today):
        if open[today] > close[today - 1] and close[today-1] - open[today - 1] > 0:
            return 0
        elif open[today] < close[today - 1] and close[today-1] - open[today - 1] < 0:
            return 1
        elif open[today] < close[today - 1] and close[today-1] - open[today - 1] < 0:
            return -1
        else:
            return 0


    def plot(self, backtest_results, long_only, ticker):
        plt.plot(backtest_results, label = 'Backtest Results')
        plt.plot(long_only, label = 'Long Only')
        plt.title(ticker)
        plt.legend(loc = 'upper left')
        plt.show()



    def get_stock_data(self, ticker):
        stock = yf.Ticker(ticker)
        data = stock.history(period = '5y', fequency = '1d')
        return data

    def get_stock_close(self, data):
        return data['Close'].values

    def get_stock_open(self, data):
        return data['Open'].values

    def backtest_trading(self, stock):

        data = self.get_stock_data(stock)
        close_price = self.get_stock_close(data)
        open_price = self.get_stock_open(data)


        total_profit = 0
        aum = []

        long_only_total_profit = 0
        long_only = []

        for i in range(1, len(close_price)):


            position = self.get_position(open_price, close_price, i)
            profit = position * ( close_price[i] - open_price[i] )





            total_profit = total_profit + profit
            aum.append(total_profit)

            long_only_profit = close_price[i] - open_price[i]
            long_only_total_profit = long_only_total_profit + long_only_profit
            long_only.append(long_only_total_profit)

        self.plot(aum, long_only, stock)


def main():

    bot = Bot()
    bot.backtest_trading('gld')




main()


