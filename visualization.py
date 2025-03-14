import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import statsmodels.tsa.arima.model as sma
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import Ridge
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

DIRECTORY = "/Users/momonawada/PycharmProjects/comp4949-assignment1/"
FILE_NAME = "dec14-19_foreign_solar_power_generation.csv"
PATH = DIRECTORY + FILE_NAME
TEST_SIZE = 6

def build_ols_and_predict(df):
    df = implement_back_shifting(df)

    model_exp = ExponentialSmoothing(df["Power"], trend="mul").fit()
    df["power_smoothed"] = model_exp.fittedvalues

    df_predictions = pd.DataFrame()

    for i in range(len(df) - TEST_SIZE):
        num_days_ahead = TEST_SIZE - i
        if num_days_ahead <= 0:
            break

        train = get_train_data(i)
        test = get_test_data(i)

        X_train, y_train = get_x_and_y_values(train)
        X_test, y_test = get_x_and_y_values(test)

        model = sm.OLS(y_train, X_train).fit()
        pred = model.predict(X_test)

        pred_t = pred.iloc[0] # get 1st prediction
        actual_t = y_test.iloc[0] # get 1st actual value
        df_predictions = df_predictions._append({
            "date": test.index[0],
            "Prediction": pred_t,
            "Actual": actual_t}, ignore_index=True)
    df_predictions = pd.DataFrame(df_predictions).set_index("date")
    return df_predictions

def build_holt_winters_and_predict(df):
    n = len(df)
    predictions = []

    for i in range(TEST_SIZE):
        train_end = (n - TEST_SIZE) + i
        train_data = df.iloc[:train_end]

        test_idx = (n - TEST_SIZE) + i
        test_data = df.iloc[test_idx : test_idx + 1] # Only one row

        model = ExponentialSmoothing(
            train_data["Power"], trend="mul"
        ).fit()

        y_pred = model.forecast(1).iloc[0]
        y_true = test_data["Power"].iloc[0]

        predictions.append({"date": test_data.index[0],
                            "Prediction": y_pred,
                            "Actual": y_true})
    df_pred = pd.DataFrame(predictions).set_index("date")
    return df_pred

def build_arima_and_predict(df, order=(2, 0, 1)):
    n = len(df)
    predictions = []

    for i in range(TEST_SIZE):
        train_end = (n - TEST_SIZE) + i
        train_data = df.iloc[:train_end]

        test_idx = (n - TEST_SIZE) + i
        test_data = df.iloc[test_idx: test_idx + 1]

        model = sma.ARIMA(train_data["Power"], order=order).fit()

        # predict one step ahead
        y_pred = model.forecast(1).iloc[0]
        y_true = test_data["Power"].iloc[0]

        predictions.append({
            "date": test_data.index[0],
            "Prediction": y_pred,
            "Actual": y_true
        })
    df_pred = pd.DataFrame(predictions).set_index("date")
    return df_pred


def build_arima_and_predict_gridsearch(df):
    p_range = [0, 1, 2]
    d_range = [0, 1]
    q_range = [0, 1, 2]

    best_rmse = float("inf")
    best_order = None

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    df_pred_temp = build_arima_and_predict(df, order=(p, d, q))
                    rmse_temp = np.sqrt(mean_squared_error(df_pred_temp["Prediction"], df_pred_temp["Actual"]))
                    if rmse_temp < best_rmse:
                        best_rmse = rmse_temp
                        best_order = (p, d, q)
                except:
                    pass

    df_best = build_arima_and_predict(df, order=best_order)
    return df_best, best_order, best_rmse

def build_ridge_and_predict(df):
    df = implement_back_shifting(df)
    n = len(df)
    predictions = []

    scaler = StandardScaler() # Standardization for better Ridge performance

    for i in range(TEST_SIZE):
        train_end = (n - TEST_SIZE) + i
        train_data = df.iloc[:train_end]

        test_idx = (n - TEST_SIZE) + i
        test_data = df.iloc[test_idx: test_idx + 1]

        X_train, y_train = get_x_and_y_values(train_data)
        X_test, y_test = get_x_and_y_values(test_data)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)[0]
        y_true = y_test.iloc[0]

        predictions.append({
            "date": test_data.index[0],
            "Prediction": y_pred,
            "Actual": y_true
        })
    df_pred = pd.DataFrame(predictions).set_index("date")
    return df_pred

def implement_back_shifting(df):
    df["power_t-1"] = df["Power"].shift(1)
    df["power_t-2"] = df["Power"].shift(2)
    df.dropna(inplace=True)
    return df

def scale_features(df):
    global scaler
    feature_cols = ["power_t-1", "power_t-2"]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def get_train_data(i):
    start_index = 0
    end_index = i + (len(power_df) - TEST_SIZE)
    train = power_df.iloc[start_index:end_index]
    return train

def get_test_data(i):
    start_index = i + (len(power_df) - TEST_SIZE)
    end_index = i + len(power_df)
    test = power_df.iloc[start_index:end_index]
    return test

def get_x_and_y_values(df):
    feature_cols = ["power_t-1", "power_t-2"]
    X = df[feature_cols]
    y = df["Power"]

    X = sm.add_constant(X)

    if "const" not in X.columns:
        X["const"] = 1
        preferred_order = ["const"] + feature_cols
        X = X[preferred_order]
    return X, y

def plot_histogram():
    plt.hist(power_df["Power"], bins=30)
    plt.title("Distribution of Power")
    plt.xlabel("Power")
    plt.ylabel("Frequency")
    plt.show()

def plot_scatter_map():
    plt.scatter(power_df["power_t-1"], power_df["Power"], alpha=0.5)
    plt.title("Power vs. Power_t-1")
    plt.xlabel("power_t-1")
    plt.ylabel("Power")
    plt.show()

def plot_heatmap(df, target):
    # corr = df[columns + [target]].corr()
    corr = df.corr()
    corr = corr.sort_values(by=target, ascending=False)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr[[target]], annot=True, cmap="coolwarm")
    plt.title("Heatmap")
    plt.show()


power_df = pd.read_csv(PATH, parse_dates=["date"], index_col="date")
power_df.index.freq = "D"

power_df = implement_back_shifting(power_df)
# print(power_df)

tseries = seasonal_decompose(power_df["Power"], model="multiplicative", extrapolate_trend="freq", period=365)
tseries.plot()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot_histogram()

# plot_acf(power_df["Power"], lags=50)
# plt.title("ACF of Power")
# plt.show()
#
# plot_pacf(power_df["Power"], lags=50)
# plt.title("PACF of Power")
# plt.show()

# plot_scatter_map()

# plot_heatmap(power_df, "Power")

# model = "OLS"
# df_predictions = build_ols_and_predict(power_df)

# model = "Holt-Winters
# df_predictions = build_holt_winters_and_predict(power_df)

# model = "ARIMA"
# df_predictions = build_arima_and_predict(power_df, order=(1, 1, 0))

# model = "ARIMA GridSearch"
# df_predictions, best_order, best_rmse = build_arima_and_predict_gridsearch(power_df)
# print(f"Best ARIMA order: {best_order}, RMSE: {best_rmse}")

# model = "Ridge"
# df_predictions = build_ridge_and_predict(power_df)

# print(df_predictions)

# rmse = np.sqrt(mean_squared_error(df_predictions["Prediction"], df_predictions["Actual"]))
# print(f"RMSE: {rmse}")
# print(f"RMSE: (Grid search Final): {rmse}")



def main():
    pass

if __name__ == '__main__':
    main()