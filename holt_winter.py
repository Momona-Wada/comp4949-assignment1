import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 予測する日数
TEST_SIZE = 6
DIRECTORY = "/Users/momonawada/PycharmProjects/comp4949-assignment1/"
FILE_NAME = "foreign_solar_power_generation.csv"
PATH = DIRECTORY + FILE_NAME


# データ読み込み
co2_df = pd.read_csv(PATH, parse_dates=["date"])
co2_df.set_index("date", inplace=True)  # これで co2_df のIndexがDateになる

def build_holt_winters_predictions(df, forecast_days=TEST_SIZE):
    """
    Holt-Winters で最終 'forecast_days' 日をウォークフォワード予測し、
    実際の値と予測値を DataFrame で返す。
    """
    n = len(df)
    predictions = []

    # 最終日から遡って 'forecast_days' 日分を1日ずつ予測
    # i=0→最後から6日目を予測, i=1→最後から5日目を予測,... i=5→最後の日を予測
    # ここでは「先頭～(n - forecast_days + i)」を学習データにする
    for i in range(forecast_days):
        train_end = (n - forecast_days) + i  # 学習の終端インデックス
        train_data = df.iloc[:train_end]     # 先頭行～train_end-1 行目まで

        test_idx = (n - forecast_days) + i   # テスト対象となる行のインデックス
        test_data = df.iloc[test_idx : test_idx + 1]  # 1行だけ

        # Holt-Wintersモデルを学習
        model = ExponentialSmoothing(
            train_data["Power"], trend="mul"
        ).fit()

        # 1ステップ先を予測（＝学習区間の次の行）
        y_pred = model.forecast(1).iloc[0]
        y_true = test_data["Power"].iloc[0]

        predictions.append({"date": test_data.index[0],
                            "Prediction": y_pred,
                            "Actual": y_true})

    # 日付をインデックスにした DataFrame を作成
    df_pred = pd.DataFrame(predictions).set_index("date")
    return df_pred

def plot_predictions(df_pred, original_df):
    """
    最終 'TEST_SIZE' 日の実測値と予測値をプロットする。
    """
    # ウォークフォワードで予測した日付がインデックス
    dates = df_pred.index

    plt.figure(figsize=(8, 4))
    plt.plot(dates, df_pred["Prediction"], marker="o", color="orange", label="Predicted")
    plt.plot(dates, df_pred["Actual"], marker="o", color="blue", label="Actual")

    plt.title(f"Holt-Winters Forecast (last {TEST_SIZE} days)")
    plt.xticks(rotation=70)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- メイン処理 ---
df_pred = build_holt_winters_predictions(co2_df, forecast_days=TEST_SIZE)
print(df_pred)

rmse = np.sqrt(mean_squared_error(df_pred["Actual"], df_pred["Prediction"]))
print(f"RMSE: {rmse}")

plot_predictions(df_pred, co2_df)
