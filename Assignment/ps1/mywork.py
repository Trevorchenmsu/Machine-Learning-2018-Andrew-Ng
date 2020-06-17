import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from time import time
import scipy.optimize as spo

# def get_max_close(symbol):
#     df = pd.read_csv("{}.csv".format((symbol)))
#     # return df['Close'].max()
#     return df['Volume'].mean()

# def test_run():
# #     df = pd.read_csv('AAPL.csv')
# #     df[['Close', 'Adj Close']].plot()
# #     plt.show()
# # test_run()

"""************************** read csv file, join data, plot data **************************************"""
def symbol_to_path(symbol, base_dir="data"):
    # return csv file path given ticker symbol
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates, addSPY=True, colname='Adj Close'):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + list(
            symbols)  # handles the case where symbols is np array of 'object'

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                              parse_dates=True, usecols=['Date', colname], na_values=['nan'])
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    import matplotlib.pyplot as plt
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
#
# def plot_selected(df, columns, start_index, end_index):
#     df = df.loc[start_index:end_index, columns]
#     plot_data(df)
#
# def normalize_data(df):
#     return df/df.iloc[0]
#
# def test_run():
#     start_date = '2010-01-01'
#     end_date = '2010-12-31'
#     dates = pd.date_range(start_date, end_date)
#     symbols = ['GOOG', 'IBM', 'GLD']
#     df = get_data(symbols, dates)
#     df = normalize_data(df)
#     plot_data(df)
#     # df1 = pd.DataFrame(index=dates)
#
# test_run()

"""*********************************** Numpy usage *********************************************"""
# def get_max_index(a):
#     """Return the index of the maximum value in given 1D array"""
#     return a.argmax()
#
# def how_long(func, *args):
#     """Execute function with given arguments, and measure execution time"""
#     t0 = time()
#     result = func(*args)
#     t1 = time()
#     return result, t1 - t0
#
# def manual_mean(arr):
#     """compute mean of all elements in the given 2D array"""
#     sum = 0
#     for i in range(0, arr.shape[0]):
#         for j in range(0, arr.shape[1]):
#             sum = sum + arr[i, j]
#     return sum / arr.size
#
# def numpy_mean(arr):
#     return arr.mean()
#
# def test_run():
    # print(np.array([2, 3, 4]))
    # print(np.array([(2, 3, 4), (5, 6, 7)]))
    # print(np.empty(5))
    # print(np.empty((5, 4)))
    # print(np.ones((5, 4), dtype=np.int_))
    # print(np.random.random((5, 4)))
    # print(np.random.rand(5, 4))
    # print(np.random.normal(size=(2, 3))) # mean = 0, s.d.=1
    # print(np.random.normal(50, 10, size=(2, 3))) # mean = 50, s.d.=10

    # random integers
    # a = np.random.randint(10) # a single inter in [0, 10)
    # a = np.random.randint(0, 10) # same as above
    # a = np.random.randint(0, 10, size=5) # 5 random integers, 1D array
    # a = np.random.randint(0, 10, size=(2, 3)) # 5 random integers. 2*3 array
    # np.random.seed(693) # seed the random number generator, make it pseudo ranodm
    # a = np.random.randint(0, 10, size=(5, 4))

    # print("Array:\n",a)
    # print("Sum of array:\n", a.sum())
    # print("Sum of each column:\n", a.sum(axis=0))
    # print("Sum of each row:\n", a.sum(axis=1))
    # similar to min, max, mean
    # a = np.array([9, 6, 2, 3, 12, 14, 7, 10], dtype=np.int32)
    # print("Array:", a)
    # # find the maximum and its index
    # print("Maximum value: ", a.max())
    # print("Index of max: ", get_max_index(a))
    # t1 = time.time()
    # print("ML4T")
    # t2 = time.time()
    # print("time used: ", t2 - t1, "seconds")

    # nd1 = np.random.random((1000, 10000))
    # res_manual, t_manual = how_long(manual_mean, nd1)
    # res_numpy, t_numpy = how_long(numpy_mean, nd1)
    # print("manual: {:.6f}({:.3f} secs.) vs. numpy: {:.6f}({:.3f} secs.)".format(res_manual, t_manual, res_numpy, t_numpy))
    # assert abs(res_manual - res_numpy) <= 10e-6
    # speedup = t_manual / t_numpy
    # print("numpy mean is ", speedup, "times faster than manual for loops.")
    # a = np.random.random((5, 4))
    # print(a)
    # print(a[3, 2])

"""*********************************** Statistics Analysis *********************************************"""
# rolling mean: take a small windows, like 20 days to calculate the mean, then forward one day to compute another mean, etc.
def get_rolling_mean(data, window):
    rm = data.rolling(window).mean()
    return rm

def get_rolling_std(data, window):
    rstd = data.rolling(window).std()
    return rstd

def get_bollinger_band(rm, rstd):
    upper_band = rm + 2 * rstd
    lower_band =  rm - 2 * rstd
    return upper_band, lower_band

def compute_daily_returns(df):
    daily_return = df.copy()
    # [1:] from second row till the end
    daily_return[1:] = (df[1:] / df[:-1].values) - 1
    # daily_return = (df / df.shift(1)) - 1
    daily_return.iloc[0, :] = 0
    return daily_return

#
# def test_run():
#     dates = pd.date_range('2012-07-01', '2012-07-31')
#     symbols = ['SPY', 'XOM']
#     df = get_data(symbols, dates)
#     # print(df.iloc[0, :])
#     plot_data(df)
#     # print(df.mean())
#     # print(df.median())
#     # print(df.std())
#     # ax = df['SPY'].plot(title="SPY rolling mean", label='SPY')
#     # rm_SPY = df['SPY'].rolling(20).mean()
#     # rm_SPY.plot(label='Rolling mean', ax=ax)
#     # ax.set_xlabel("Date")
#     # ax.set_ylabel("price")
#     # ax.legend(loc='upper left')
#     # plt.show()
#     # rm_SPY = get_rolling_mean(df['SPY'], window=20)
#     # rstd_SPY = get_rolling_std(df['SPY'], window=20)
#     # upper_band, lower_band = get_bollinger_band(rm_SPY, rstd_SPY)
#     # ax = df['SPY'].plot(title="Bollinger Bands", label='SPY')
#     # rm_SPY.plot(label='Rolling mean', ax=ax)
#     # upper_band.plot(label='upper band', ax=ax)
#     # lower_band.plot(label='lower band', ax=ax)
#     # ax.set_xlabel("Date")
#     # ax.set_ylabel("price")
#     # ax.legend(loc='upper left')
#     # plt.show()
#
#     daily_returns = compute_daily_returns(df)
#     plot_data(daily_returns, title="Daily returns", ylabel="Daily returns")

"""*********************************** 01-05 Incomplete data *********************************************"""
def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    df_data.fillna(method="ffill", inplace=True)
    df_data.fillna(method="bfill", inplace=True)

# def test_run():
#     """Function called by Test Run."""
#     # Read data
#     symbol_list = ["JAVA", "FAKE1", "FAKE2"]  # list of symbols
#     start_date = "2005-12-31"
#     end_date = "2014-12-07"
#     dates = pd.date_range(start_date, end_date)  # date range as index
#     df_data = get_data(symbol_list, dates)  # get data for each symbol
#
#     # Fill missing values
#     fill_missing_values(df_data)
#
#     # Plot
#     plot_data(df_data)


"""*********************************** 01-06 histograms and scatter plots *********************************************"""
# in daily return distribution, higher mean indicates higher return while smaller std indicates higher risk
# def test_run():
    # dates = pd.date_range('2009-01-01', '2012-12-31')
    # symbols = ['SPY']
    # df = get_data(symbols, dates)
    # # plot_data(df)
    # daily_returns = compute_daily_returns(df)
    # # plot_data(daily_returns, title="Daily retunrs", ylabel="Daily returns")
    # daily_returns.hist(bins=20)
    # mean = daily_returns['SPY'].mean()
    # # print("mean=", mean)
    # std = daily_returns['SPY'].std()
    # # print("std=", std)
    # plt.axvline(mean, color='w', linestyle='dashed', linewidth=2)
    # plt.axvline(std, color='r', linestyle='dashed', linewidth=2)
    # plt.axvline(-std, color='r', linestyle='dashed', linewidth=2)
    # # plt.show()
    # print(daily_returns.kurtosis())
    #
    # dates = pd.date_range('2009-01-01', '2012-12-31')
    # symbols = ['SPY', 'XOM']
    # df = get_data(symbols, dates)
    # # plot_data(df)
    # daily_returns = compute_daily_returns(df)
    # # plot_data(daily_returns, title="Daily returns", ylabel="Daily returns")
    # # daily_returns.hist(bins=20)
    # daily_returns['SPY'].hist(bins=20, label="SPY")
    # daily_returns['XOM'].hist(bins=20, label="XOM")
    # plt.legend(loc='upper right')
    # plt.show()
    #
    # dates = pd.date_range('2009-01-01', '2012-12-31')
    # symbols = ['SPY', 'XOM', 'GLD']
    # df = get_data(symbols, dates)
    # daily_returns = compute_daily_returns(df)
    # daily_returns.plot(kind='scatter', x='SPY', y='XOM')
    # beta_XOM, alpha_XOM = np.polyfit(daily_returns['SPY'], daily_returns['XOM'], 1)
    # plt.plot(daily_returns['SPY'], beta_XOM*daily_returns['SPY']+alpha_XOM, '-', color='r')
    #
    # daily_returns.plot(kind='scatter', x='SPY', y='GLD')
    # beta_GLD, alpha_GLD = np.polyfit(daily_returns['SPY'], daily_returns['GLD'], 1)
    # plt.plot(daily_returns['SPY'], beta_GLD * daily_returns['SPY'] + alpha_GLD, '-', color='r')
    #
    # plt.show()
    #
    # print(daily_returns.corr(method='pearson'))

"""*********************************** 01-08 optimizer *********************************************"""
# def f(X):
#     """Given a scalar X, return some value (a real number)."""
#     Y = (X - 1.5) ** 2 + 0.5
#     print( "X = {}, Y = {}".format(X, Y)) # for tracing
#     return Y
#
#
# def test_run():
#     Xguess = 2.0
#     min_result = spo.minimize(f, Xguess, method='SLSQP', options={'disp': True})
#     print("Minima found at:")
#     print( "X = {}, Y = {}".format(min_result.x, min_result.fun))
#
#     # Plot function values, mark minima
#     Xplot = np.linspace(0.5, 2.5, 21)
#     Yplot = f(Xplot)
#     plt.plot(Xplot, Yplot)
#     plt.plot(min_result.x, min_result.fun, 'ro')
#     plt.title("Minima of an objective function")
#     plt.show()


"""Minimize an objective function using SciPy."""


def error(line, data):  # error function
    """Compute error between given line model and observed data.

    Parameters
    ----------
    line: tuple/list/array (C0, C1) where C0 is slope and C1 is Y-intercept
    data: 2D array where each row is a point (x, y)

    Returns error as a single real value.
    """
    # Metric: Sum of squared Y-axis differences
    err = np.sum((data[:, 1] - (line[0] * data[:, 0] + line[1])) ** 2)
    return err


def fit_line(data, error_func):
    """Fit a line to given data, using a supplied error function.

    Parameters
    ----------
    data: 2D array where each row is a point (X0, Y)
    error_func: function that computes the error between a line and observed data

    Returns line that minimizes the error function.
    """
    # Generate initial guess for line model
    l = np.float32([0, np.mean(data[:, 1])])  # slope = 0, intercept = mean(y values)

    # Plot initial guess (optional)
    x_ends = np.float32([-5, 5])
    plt.plot(x_ends, l[0] * x_ends + l[1], 'm--', linewidth=2.0, label="Initial guess")

    # Call optimizer to minimize error function
    result = spo.minimize(error_func, l, args=(data,), method='SLSQP', options={'disp': True})
    return result.x


def test_run():
    # Define original line
    l_orig = np.float32([4, 2])
    print("Original line: C0 = {}, C1 = {}".format(l_orig[0], l_orig[1]))

    Xorig = np.linspace(0, 10, 21)
    Yorig = l_orig[0] * Xorig + l_orig[1]
    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original line")

    # Generate noisy data points
    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label="Data points")

    # Try to fit a line to this data
    l_fit = fit_line(data, error)
    print("Fitted line: C0 = {}, C1 = {}".format(l_fit[0], l_fit[1]))
    plt.plot(data[:, 0], l_fit[0] * data[:, 0] + l_fit[1], 'r--', linewidth=2.0, label="Fitted Line")

    # Add a legend and show plot
    plt.legend(loc='upper right')
    plt.show()


"""Minimize an objective function using SciPy: 3D"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo


def error_poly(C, data):  # error function
    """Compute error between given polynomial and observed data.

    Parameters
    ----------
    C: numpy.poly1d object or equivalent array representing polynomial coefficients
    data: 2D array where each row is a point (x, y)

    Returns error as a single real value.
    """
    # Metric: Sum of squared Y-axis differences
    err = np.sum((data[:, 1] - np.polyval(C, data[:, 0])) ** 2)
    return err


def fit_poly(data, error_func, degree=3):
    """Fit a polynomial to given data, using a supplied error function.

    Parameters
    ----------
    data: 2D array where each row is a point (X0, Y)
    error_func: function that computes the error between a polynomial and observed data

    Returns polynomial that minimizes the error function.
    """
    # Generate initial guess for line model (all coeffs = 1)
    Cguess = np.poly1d(np.ones(degree + 1, dtype=np.float32))

    # Plot initial guess (optional)
    x = np.linspace(-5, 5, 21)
    plt.plot(x, np.polyval(Cguess, x), 'm--', linewidth=2.0, label="Initial guess")

    # Call optimizer to minimize error function
    result = spo.minimize(error_func, Cguess, args=(data,), method='SLSQP', options={'disp': True})
    return np.poly1d(result.x)  # convert optimal result into a poly1d object and return


# def test_run():


# Define original line


# Generate noisy data points


# Try to fit a line to this data


# Add a legend and show plot

if __name__ == "__main__":
    test_run()