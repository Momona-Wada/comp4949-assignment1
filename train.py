import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

DIRECTORY = "/Users/momonawada/PycharmProjects/comp4949-assignment1/"
FILE_NAME = "dec14-19_foreign_solar_power_generation.csv"
PATH = DIRECTORY + FILE_NAME
TEST_SIZE = 6

def build_ols_and_predict(df):
    df = implement_back_shifting(df)
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
        df_predictions = df_predictions._append({"prediction": pred_t,"actual": actual_t}, ignore_index=True)

    return df_predictions

def build_holt_winters_and_predict(df):
    df_predictions = pd.DataFrame()

    for i in range(len(df) - TEST_SIZE):
        num_days_ahead = TEST_SIZE - i
        if num_days_ahead <= 0:
            break

        train = get_train_data(i)
        test = get_test_data(i)

        # Build HWES3 model with multiplicative decomposition
        fitted_model = ExponentialSmoothing(train["Power"], trend="mul").fit()
        test_predictions = fitted_model.forecast(1)

        pred_t = test_predictions.iloc[0] # get 1st prediction
        actual_t = test.iloc[0]["Power"] # get 1st actual value
        df_predictions = df_predictions._append({"prediction": pred_t, "actual": actual_t}, ignore_index=True)

    return df_predictions

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

def show_plot(df_predictions, original_df):
    final_rows = original_df.tail(TEST_SIZE)
    plt.plot(final_rows.index, df_predictions["prediction"], label="Prediction", color="orange")
    plt.plot(final_rows.index, df_predictions["actual"], label="Actual", color="blue")
    plt.xticks(rotation=70)
    plt.legend()
    plt.tight_layout()
    plt.show()


power_df = pd.read_csv(PATH, parse_dates=["date"], index_col="date")
print(power_df)

df_predictions = build_ols_and_predict(power_df)
# df_predictions = build_holt_winters_and_predict(power_df)
print(df_predictions)

rmse = np.sqrt(mean_squared_error(df_predictions["prediction"], df_predictions["actual"]))
print(f"RMSE: {rmse}")

show_plot(df_predictions, power_df)


def main():
    pass

if __name__ == '__main__':
    main()