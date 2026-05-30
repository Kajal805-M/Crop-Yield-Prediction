import pickle
import pandas as pd


def load_model(model_path="best_crop_yield_model.pkl"):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def predict_yield(
    crop,
    crop_year,
    season,
    state,
    area,
    production,
    annual_rainfall,
    fertilizer,
    pesticide,
    avg_temperature,
    max_temperature,
    min_temperature,
    model_bundle,
):
    feature_columns = model_bundle["feature_columns"]
    pipeline = model_bundle["pipeline"]

    input_dict = {
        "Crop": crop,
        "Crop_Year": crop_year,
        "Season": season,
        "State": state,
        "Area": area,
        "Production": production,
        "Annual_Rainfall": annual_rainfall,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide,
        "Avg_Temperature": avg_temperature,
        "Max_Temperature": max_temperature,
        "Min_Temperature": min_temperature,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns)

    pred = pipeline.predict(input_df)[0]
    return pred


def main():
    model_bundle = load_model()
    print(f"Loaded best model: {model_bundle['model_name']}")

    print("\nEnter values for prediction:")
    crop = input("Crop: ").strip()
    crop_year = int(input("Crop Year (e.g., 2024): "))
    season = input("Season (e.g., Kharif/Rabi/Whole Year): ").strip()
    state = input("State: ").strip()
    area = float(input("Area: "))
    production = float(input("Production: "))
    annual_rainfall = float(input("Annual Rainfall: "))
    fertilizer = float(input("Fertilizer: "))
    pesticide = float(input("Pesticide: "))
    avg_temperature = float(input("Average Temperature: "))
    max_temperature = float(input("Max Temperature: "))
    min_temperature = float(input("Min Temperature: "))

    predicted = predict_yield(
        crop,
        crop_year,
        season,
        state,
        area,
        production,
        annual_rainfall,
        fertilizer,
        pesticide,
        avg_temperature,
        max_temperature,
        min_temperature,
        model_bundle,
    )

    print(f"\nPredicted Crop Yield: {predicted:.4f}")


if __name__ == "__main__":
    main()
