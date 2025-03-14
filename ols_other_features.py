import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

# Load data
DIRECTORY = "/Users/momonawada/PycharmProjects/comp4949-assignment1/"
FILE_NAME = "foreign_solar_power_generation.csv"
PATH = DIRECTORY + FILE_NAME
TEST_SIZE = 6

power_df = pd.read_csv(PATH, parse_dates=["date"], index_col="date")
power_df.index.freq = "D"


# Implement feature selection based on heatmap
def implement_feature_engineering(df):
    df["power_t-1"] = df["Power"].shift(1)
    df["power_t-2"] = df["Power"].shift(2)
    df["F_t-1"] = df["F"].shift(1)
    df["C_t-1"] = df["C"].shift(1)
    df["D_t-1"] = df["D"].shift(1)
    df.dropna(inplace=True)
    return df


power_df = implement_feature_engineering(power_df)


# Function to get train and test data
def get_train_test_data(i, df):
    start_index = 0
    end_index = i + (len(df) - TEST_SIZE)
    train = df.iloc[start_index:end_index]

    start_index_test = i + (len(df) - TEST_SIZE)
    end_index_test = i + len(df)
    test = df.iloc[start_index_test:end_index_test]

    return train, test


# Get X and Y values
def get_x_y_values(df):
    feature_cols = ["power_t-1", "power_t-2", "F_t-1", "C_t-1", "D_t-1"]
    X = df[feature_cols]
    y = df["Power"]

    X = sm.add_constant(X)

    if "const" not in X.columns:
        X["const"] = 1
        preferred_order = ["const"] + feature_cols
        X = X[preferred_order]

    print(f"Used columns: {X.columns.tolist()}")
    return X, y


# Build OLS model and predict
def build_ols_and_predict(df):
    df_predictions = pd.DataFrame()

    for i in range(len(df) - TEST_SIZE):
        num_days_ahead = TEST_SIZE - i
        if num_days_ahead <= 0:
            break

        train, test = get_train_test_data(i, df)

        X_train, y_train = get_x_y_values(train)
        X_test, y_test = get_x_y_values(test)

        model = sm.OLS(y_train, X_train).fit()
        pred = model.predict(X_test)

        df_predictions = df_predictions._append({
            "date": test.index[0],
            "Prediction": pred.iloc[0],
            "Actual": y_test.iloc[0]
        }, ignore_index=True)

    df_predictions = df_predictions.set_index("date")
    return df_predictions

def show_plot(df_predictions, original_df):
    final_rows = original_df.tail(TEST_SIZE)
    plt.plot(final_rows.index, df_predictions["Prediction"], label="Prediction", marker="o", color="orange")
    plt.plot(final_rows.index, df_predictions["Actual"], label="Actual", marker="o",color="blue")
    plt.xticks(rotation=70)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Run OLS model
df_predictions = build_ols_and_predict(power_df)

# Display predictions
print(df_predictions)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(df_predictions["Prediction"], df_predictions["Actual"]))
print(f"RMSE: {rmse}")

show_plot(df_predictions, power_df)