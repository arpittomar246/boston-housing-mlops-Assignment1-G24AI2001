
"""
Reusable functions: load_data(), split_data(), preprocessing, train/eval, printing and saving metrics.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
from typing import Tuple, Dict, Any

def load_data() -> pd.DataFrame:
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def get_features_and_target(df: pd.DataFrame, target_col: str = 'MEDV') -> Tuple[np.ndarray, np.ndarray]:
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    return X, y

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, target_col: str = 'MEDV'):
    X, y = get_features_and_target(df, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def preprocess_scale(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, preds)
    return {
        "mse": float(mse),
        "rmse": rmse,
        "r2": float(r2),
        "preds": preds
    }

def pretty_print_results(model_name: str, metrics: Dict[str, Any], y_test: np.ndarray, preds: np.ndarray, n_show: int = 10):
    print("".center(60, "="))
    print(f" Results for model: {model_name}")
    print("".center(60, "="))
    print(f"MSE  : {metrics['mse']:.4f}")
    print(f"RMSE : {metrics['rmse']:.4f}")
    print(f"R^2  : {metrics['r2']:.4f}")
    print("-" * 60)
    n_show = min(n_show, len(y_test))
    df_compare = pd.DataFrame({
        "actual": y_test.flatten(),
        "predicted": preds.flatten()
    })
    df_compare = df_compare.reset_index(drop=True).head(n_show)
    print("First {} predictions (actual vs predicted):".format(n_show))
    print(df_compare.to_string(index=False, float_format="{:0.3f}".format))
    print("".center(60, "="))

def save_metrics(path: str, model_name: str, metrics: Dict[str, Any]):
    """
    Append metrics for the model into a CSV (creates file if doesn't exist).
    metrics should include keys: mse, rmse, r2
    """
    out = {"model": model_name, "mse": metrics["mse"], "rmse": metrics["rmse"], "r2": metrics["r2"]}
    df = pd.DataFrame([out])
    header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=header, index=False)
# PY
