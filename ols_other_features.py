import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import statsmodels.tsa.arima.model as sma
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import Ridge

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

DIRECTORY = "/Users/momonawada/PycharmProjects/comp4949-assignment1/"
FILE_NAME = "dec14-19_foreign_solar_power_generation.csv"
PATH = DIRECTORY + FILE_NAME
TEST_SIZE = 6

# 使用する特徴量のカラム。必ず全て使う必要はなく、有用なものを選択しても問題ありません
FEATURE_COLUMNS = [
                   "A",
                   "B",
                   "C",
                   "D",
                   # "E",
                   # "F",
                   # "G",
                   # "H",
                   # "J",
                   # "K",
                   # "L",
                   # "M",
                   # "N",
                   # "O",
                   ]


def implement_back_shifting(df):
    df = df.copy()
    df["power_t-1"] = df["Power"].shift(1)
    df["power_t-2"] = df["Power"].shift(2)
    for col in FEATURE_COLUMNS:
        df[f"{col}_t-1"] = df[col].shift(1)
        df[f"{col}_t-2"] = df[col].shift(2)
    df.dropna(inplace=True)
    return df


def get_x_and_y_values(df):
    # バックシフトしたターゲットと特徴量を使う
    feature_cols = ["power_t-1", "power_t-2"] + \
                   [f"{col}_t-1" for col in FEATURE_COLUMNS] + \
                   [f"{col}_t-2" for col in FEATURE_COLUMNS]
    print(feature_cols)
    X = df[feature_cols]
    y = df["Power"]
    X = sm.add_constant(X)
    # もし定数項が無い場合は追加（sm.add_constantで追加されるはずですが、念のため）
    if "const" not in X.columns:
        X["const"] = 1.0
        preferred_order = ["const"] + feature_cols
        X = X[preferred_order]
    return X, y


def build_ols_and_predict(df):
    df = implement_back_shifting(df)
    df_predictions = []
    total = len(df)
    # テスト期間（最後のTEST_SIZE日）に対して1日ずつ予測する
    for i in range(TEST_SIZE):
        # トレーニングデータは先頭から「全体の行数 - TEST_SIZE + i」まで
        train = df.iloc[: total - TEST_SIZE + i]
        # テストデータは、続く1行を使用
        test = df.iloc[total - TEST_SIZE + i: total - TEST_SIZE + i + 1]

        # 万が一テストデータが空の場合はループを抜ける
        if test.empty:
            break

        X_train, y_train = get_x_and_y_values(train)
        X_test, y_test = get_x_and_y_values(test)

        model = sm.OLS(y_train, X_train).fit()
        pred = model.predict(X_test)

        df_predictions.append({
            "date": test.index[0],
            "Prediction": pred.iloc[0],
            "Actual": y_test.iloc[0]
        })
    df_predictions = pd.DataFrame(df_predictions).set_index("date")
    return df_predictions


def show_plot(df_predictions, model_name):
    plt.plot(df_predictions.index, df_predictions["Prediction"], marker="o", label="Prediction", color="orange")
    plt.plot(df_predictions.index, df_predictions["Actual"], marker="o", label="Actual", color="blue")
    plt.xticks(rotation=70)
    plt.title(f"{model_name} Forecast (last {TEST_SIZE} days)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# データの読み込み
power_df = pd.read_csv(PATH, parse_dates=["date"], index_col="date")

# モデルの実行
model_name = "OLS"
df_predictions = build_ols_and_predict(power_df)

print(df_predictions)
rmse = np.sqrt(mean_squared_error(df_predictions["Prediction"], df_predictions["Actual"]))
print(f"RMSE ({model_name}): {rmse}")
show_plot(df_predictions, model_name)
