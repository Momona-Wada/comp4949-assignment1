import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpmath import power
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

TEST_SIZE = 6
DIRECTORY = "/Users/momonawada/PycharmProjects/comp4949-assignment1/"
FILE_NAME = "foreign_solar_power_generation.csv"
PATH = DIRECTORY + FILE_NAME


power_df = pd.read_csv(PATH, parse_dates=["date"])
power_df.set_index("date", inplace=True)  # Set 'date' as the index for power_df

def build_holt_winters_predictions(df, forecast_days=TEST_SIZE):
    n = len(df)
    predictions = []

    for i in range(forecast_days):
        train_end = (n - forecast_days) + i  # End index of the training set
        train_data = df.iloc[:train_end]     # Data from the first row to train_end-1

        test_idx = (n - forecast_days) + i   # Index of the test row
        test_data = df.iloc[test_idx : test_idx + 1]  # Only one row


        model = ExponentialSmoothing(train_data["Power"], trend="mul").fit()

        # Predict the next step (the row after the training period)
        y_pred = model.forecast(1).iloc[0]
        y_true = test_data["Power"].iloc[0]

        predictions.append({"date": test_data.index[0],
                            "Prediction": y_pred,
                            "Actual": y_true})

    df_pred = pd.DataFrame(predictions).set_index("date")
    return df_pred

def plot_predictions(df_pred):
    dates = df_pred.index

    plt.plot(dates, df_pred["Prediction"], marker="o", color="orange", label="Predicted")
    plt.plot(dates, df_pred["Actual"], marker="o", color="blue", label="Actual")

    plt.title(f"Holt-Winters Forecast (last {TEST_SIZE} days)")
    plt.xticks(rotation=70)
    plt.legend()
    plt.tight_layout()
    plt.show()

df_pred = build_holt_winters_predictions(power_df, forecast_days=TEST_SIZE)
print(df_pred)

rmse = np.sqrt(mean_squared_error(df_pred["Actual"], df_pred["Prediction"]))
print(f"RMSE: {rmse}")

plot_predictions(df_pred)

df_pred_out = df_pred.copy()

df_pred_out.rename(columns={"Prediction": "Power"}, inplace=True)
df_pred_out = df_pred_out[["Power"]] # keep only Power column

output_file_name = "power_prediction.csv"
df_pred_out.to_csv(output_file_name, index=False)

print(f"Predictions have been saved to {output_file_name}")
