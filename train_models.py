import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, rmse, r2


def main():
    data_path = "crop yield.csv"
    df = pd.read_csv(data_path)

    print("Dataset shape:", df.shape)

    # Standardize text columns by trimming spaces
    for col in ["Crop", "Season", "State"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Define target and features
    target_col = "Yield"
    if target_col not in df.columns:
        raise ValueError("Target column 'Yield' not found in dataset.")

    x = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_cols = [col for col in ["Crop", "Season", "State"] if col in x.columns]
    numeric_cols = [col for col in x.columns if col not in categorical_cols]

    print("Categorical columns:", categorical_cols)
    print("Numeric columns:", numeric_cols)

    # Preprocessing pipelines
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
    }

    results = []
    trained_pipelines = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(x_train, y_train)
        mae, mse, rmse, r2 = evaluate_model(pipeline, x_test, y_test)

        trained_pipelines[model_name] = pipeline
        results.append(
            {
                "Model": model_name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
            }
        )

    results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False).reset_index(drop=True)

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))

    results_df.to_csv("model_comparison.csv", index=False)

    # Best model selection: highest R2, then lowest RMSE for tie-breaking
    best_row = results_df.sort_values(by=["R2", "RMSE"], ascending=[False, True]).iloc[0]
    best_model_name = best_row["Model"]
    best_pipeline = trained_pipelines[best_model_name]

    bundle = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "feature_columns": list(x.columns),
    }

    with open("best_crop_yield_model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nBest model: {best_model_name}")
    print("Saved model to best_crop_yield_model.pkl")
    print("Saved comparison table to model_comparison.csv")


if __name__ == "__main__":
    main()
