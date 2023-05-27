import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch.unitroot import ADF
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf


def acf_plot(timeseries, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(timeseries, ax=ax)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(title)
    plt.show()
    count=len(timeseries)
    print('the number of values is:',count)
    
    
def autocovariance(series, lag):
    len_series = len(series)
    mean = np.mean(series)

    covarianc_acc=0
    for index in range(lag, len_series):
        summand = (series[index]-mean) * (series[index-lag]-mean)
        covarianc_acc+=summand

    autocovariance =  covarianc_acc / len_series
    return autocovariance


def plot_acvf(timeseries, title):
    acvf = [autocovariance(timeseries, lag) for lag in range(len(timeseries))]
    plt.vlines([i for i in range(len(acvf))],ymax=acvf,ymin=0)
    plt.title(title)
    
    
def test_arima(timeseries: pd.DataFrame):
    order_values = [(p, d, q) for p in range(4) for d in range(2) for q in range(4)]
    best_aic = float('inf')
    best_order = None

    for p in range(4):
        for d in range(2):
            for q in range(4):
                model = ARIMA(timeseries, order=(p, d, q))
                results = model.fit()
                aic = results.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
        
    return best_order, best_aic


def decompose(time_period, period):
    decomposition = sm.tsa.seasonal_decompose(time_period, model='additive', period=period)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 6))
    axes[0].plot(decomposition.trend)
    axes[0].set_title("Trend")
    axes[1].plot(decomposition.seasonal)
    axes[1].set_title("Seasonality")
    axes[2].plot(decomposition.resid)
    axes[2].set_title("Residual")
    plt.tight_layout()
    plt.savefig(f"{period}.png")
    plt.plot()
    print(ADF(decomposition.resid.dropna()))
    return decomposition.resid.dropna()


def plot_acf_subplots(timeseries_list, titles_list):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    for i, timeseries in enumerate(timeseries_list):
        ax = axs[i]
        plot_acf(timeseries, ax=ax, title=titles_list[i], lags=len(timeseries)-1)

    plt.tight_layout()


def autocovariance(series, lag):
    len_series = len(series)
    mean = np.mean(series)

    covarianc_acc=0
    for index in range(lag, len_series):
        summand = (series[index]-mean) * (series[index-lag]-mean)
        covarianc_acc+=summand

    autocovariance =  covarianc_acc / len_series
    return autocovariance


def plot_acvf_subplots(timeseries_list, titles_list):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    for i, timeseries in enumerate(timeseries_list):
        ax = axs[i]
        acvf = [autocovariance(timeseries, lag) for lag in range(len(timeseries))]
        ax.vlines([i for i in range(len(acvf))], ymax=acvf, ymin=0)
        ax.set_title(titles_list[i])

    plt.tight_layout()
    plt.show()