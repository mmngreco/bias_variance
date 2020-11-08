import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split

rcParams["figure.figsize"] = 8, 6
plt.ion()
np.random.seed(10)


def polyfit(degree, plot=True):
    p = np.polyfit(curve.x, curve.y, deg=degree)
    val = np.polyval(p, curve.x)
    if plot:
        plt.figure()
        sn.regplot(curve.x, curve.y, fit_reg=False)
        plt.plot(curve.x, val, label="fit")
    return p


def get_rmse(y, y_fit):
    return np.sqrt(metrics.mean_squared_error(y, y_fit))


def fit_range(train_X, train_y, test_X, test_y, n):
    rmse_list = []
    for i in range(1, n):
        p = polyfit(i)
        rmse_train = get_rmse(train_y, np.polyval(p, train_X))
        rmse_test = get_rmse(test_y, np.polyval(p, test_X))
        rmse_list.append([i, rmse_train, rmse_test])

    rmse_df = pd.DataFrame(
        rmse_list, columns=["degree", "rmse_train", "rmse_test"]
    )
    rmse_df["rmse_total"] = rmse_df.rmse_train + rmse_df.rmse_test
    return rmse_df


def plot_bias_variance(rmse_df):
    plt.figure()
    plt.plot(rmse_df.degree, rmse_df.rmse_train, label="train", color="r")
    plt.plot(rmse_df.degree, rmse_df.rmse_test, label="test", color="g")
    plt.plot(rmse_df.degree, rmse_df.total_error, label="total", color="b")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.tight_layout()


if __name__ == '__main__':

    x = np.arange(0, 500, 6) * np.pi / 180
    y = np.sin(x) + np.random.normal(0, 0.15, len(x))

    curve = pd.DataFrame(np.column_stack([x, y]), columns=["x", "y"])

    plt.plot(curve["x"], curve["y"], ".")

    train_X, test_X, train_y, test_y = train_test_split(
        curve.x, curve.y, test_size=0.40, random_state=100
    )

    rmse_df = fit_range(train_X, train_y, test_X, test_y, 30)
    plot_bias_variance(rmse_df)
