import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

DIRECTORY = "/Users/momonawada/PycharmProjects/comp4949-assignment1/"
FILE_NAME = "dec14-19_foreign_solar_power_generation.csv"
PATH = DIRECTORY + FILE_NAME
TEST_SIZE = 6
scaler = MinMaxScaler()

def build_and_predict(df):
    df = implement_back_shifting(df)
    df = scale_features(df)
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

def implement_back_shifting(df):
    df["power_t-1"] = df["Power"].shift(1)
    df["power_t-2"] = df["Power"].shift(2)
    df["power_t-3"] = df["Power"].shift(3)
    df.dropna(inplace=True)
    return df

def scale_features(df):
    global scaler
    feature_cols = ["power_t-1", "power_t-2", "power_t-3"]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def get_train_data(time_step):
    start_index = 0
    end_index = time_step + len(power_df)
    train = power_df.iloc[start_index:end_index]
    return train

def get_test_data(time_step):
    start_index = time_step + len(power_df) - TEST_SIZE
    end_index = time_step + len(power_df)
    test = power_df.iloc[start_index:end_index]
    test_indicies = np.array(test.index)
    return test

def get_x_and_y_values(df):
    feature_cols = ["power_t-1", "power_t-2", "power_t-3"]
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

df_predictions = build_and_predict(power_df)
print(df_predictions)

rmse = np.sqrt(mean_squared_error(df_predictions["prediction"], df_predictions["actual"]))
print(f"RMSE: {rmse}")

show_plot(df_predictions, power_df)


def main():
    pass

if __name__ == '__main__':
    main()