import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

TEST_SIZE = 6
DIRECTORY = "/Users/momonawada/PycharmProjects/comp4949-assignment1/"
FILE_NAME = "dec14-19_foreign_solar_power_generation.csv"
PATH = DIRECTORY + FILE_NAME

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

def build_ridge_and_predict(df, alpha=1.0):
    """
    Builds and predicts using Ridge Regression with a walk-forward approach.

    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataset with 'Power' as the target variable.
    alpha : float
        Regularization strength for Ridge Regression.

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ["date", "Prediction", "Actual"] containing
        predicted and actual values for the test period.
    """
    df = implement_back_shifting(df)  # Create lag features
    n = len(df)
    predictions = []

    scaler = StandardScaler()  # Standardization for better Ridge performance

    for i in range(TEST_SIZE):
        train_end = (n - TEST_SIZE) + i
        train_data = df.iloc[:train_end]

        test_idx = (n - TEST_SIZE) + i
        test_data = df.iloc[test_idx: test_idx + 1]

        # Prepare X and y for training and testing
        X_train, y_train = get_x_and_y_values(train_data)
        X_test, y_test = get_x_and_y_values(test_data)

        # Standardize the features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Ridge Regression model
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)

        # Predict one step ahead
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

def show_plot(df_predictions, original_df):
    dates = df_predictions.index
    plt.plot(dates, df_predictions["Prediction"], marker="o", label="Prediction", color="orange")
    plt.plot(dates, df_predictions["Actual"], marker="o", label="Actual", color="blue")
    plt.xticks(rotation=70)
    plt.title(f"{model} Forecast (last {TEST_SIZE} days)")
    plt.legend()
    plt.tight_layout()
    plt.show()

power_df = pd.read_csv(PATH, parse_dates=["date"], index_col="date")

model = "Ridge"
df_predictions = build_ridge_and_predict(power_df, alpha=1.0)
print(df_predictions)

rmse = np.sqrt(mean_squared_error(df_predictions["Prediction"], df_predictions["Actual"]))
print(f"RMSE: {rmse}")

show_plot(df_predictions, power_df)