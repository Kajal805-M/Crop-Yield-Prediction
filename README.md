# Crop Yield Prediction - Beginner Friendly

This project trains multiple machine learning models to predict **crop yield** using the dataset `crop yield.csv`.

## What this project does
- Loads data from CSV
- Cleans and preprocesses data
- Handles missing values
- Encodes categorical columns (`Crop`, `Season`, `State`)
- Scales numerical columns for Linear Regression
- Trains 3 models:
  - Linear Regression
  - Random Forest Regressor
  - Decision Tree Regressor
- Evaluates using MAE, MSE, RMSE, R2
- Selects best model and saves it with pickle
- Provides a user input prediction function

## Project files
- `train_models.py`: full training + evaluation + model saving
- `predict_yield.py`: load saved model and make prediction
- `requirements.txt`: dependencies

## How to run

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Train models
```bash
python train_models.py
```

3. Predict yield (interactive)
```bash
python predict_yield.py
```

## Input features used
- Crop
- Crop_Year
- Season
- State
- Area
- Production
- Annual_Rainfall
- Fertilizer
- Pesticide
- Avg_Temperature
- Max_Temperature
- Min_Temperature

Target:
- Yield

## Output artifacts
- `best_crop_yield_model.pkl`
- `model_comparison.csv`
