# 🌾 Crop Yield Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> A machine learning project that predicts **crop yield** using agricultural and environmental factors — helping farmers, researchers, and policymakers make data-driven decisions about crop productivity.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Objective](#-objective)
- [Features](#-features)
- [Dataset](#-dataset)
- [ML Models Used](#-ml-models-used)
- [Evaluation Metrics](#-evaluation-metrics)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Author](#-author)
- [License](#-license)

---

## 🚀 Overview

Crop yield prediction is a critical challenge in agriculture. Accurate yield estimates enable better planning, resource allocation, and risk management for farmers and agricultural bodies.

This project trains and evaluates multiple machine learning regression models on historical crop data, selects the best-performing model automatically, and provides a simple command-line interface to predict the expected yield for any new set of inputs.

---

## 🎯 Objective

To build an accurate machine learning pipeline that:
- Preprocesses agricultural data (handles missing values, encodes categories, scales features)
- Trains and compares multiple regression models
- Automatically selects and saves the best model
- Predicts crop yield from user-provided inputs

---

## ✨ Features

- Accepts 12 agricultural and climate-related input features
- Handles both categorical (Crop, State, Season) and numerical data seamlessly
- Full scikit-learn `Pipeline` with preprocessing and model in one artifact
- Compares three regression models and saves the best one automatically
- Model comparison results exported to `model_comparison.csv`
- CLI-based prediction interface — no UI framework needed
- Clean, modular, and extensible code structure

---

## 📊 Dataset

**File:** `crop yield.csv`

The dataset contains historical crop production data with the following features:

| Feature             | Type        | Description                                  |
|---------------------|-------------|----------------------------------------------|
| `Crop`              | Categorical | Name of the crop                             |
| `Crop_Year`         | Numerical   | Year of cultivation                          |
| `Season`            | Categorical | Growing season (e.g., Kharif, Rabi, Whole Year) |
| `State`             | Categorical | Indian state where the crop is grown         |
| `Area`              | Numerical   | Cultivated area (in hectares)                |
| `Production`        | Numerical   | Total crop production (in tonnes)            |
| `Annual_Rainfall`   | Numerical   | Annual rainfall received (in mm)             |
| `Fertilizer`        | Numerical   | Amount of fertilizer used (in kg)            |
| `Pesticide`         | Numerical   | Amount of pesticide used (in kg)             |
| `Avg_Temperature`   | Numerical   | Average temperature (°C)                     |
| `Max_Temperature`   | Numerical   | Maximum temperature (°C)                     |
| `Min_Temperature`   | Numerical   | Minimum temperature (°C)                     |
| `Yield` *(target)*  | Numerical   | Crop yield (kg/hectare) — what we predict    |

---

## 🧠 ML Models Used

Three regression models are trained and compared:

| Model                     | Description                                                  |
|---------------------------|--------------------------------------------------------------|
| **Linear Regression**     | Baseline linear model for yield estimation                   |
| **Random Forest Regressor** | Ensemble of 200 decision trees (best performer)            |
| **Decision Tree Regressor** | Single tree model for interpretable predictions            |

The best model is selected automatically based on the highest **R² Score**, with **RMSE** as a tiebreaker.

---

## 📈 Evaluation Metrics

Each model is evaluated on a held-out test set (20% of data) using:

| Metric   | Full Name                         | What it measures                                       |
|----------|-----------------------------------|--------------------------------------------------------|
| **R²**   | R-Squared Score                   | Proportion of variance explained by the model (higher = better) |
| **MAE**  | Mean Absolute Error               | Average prediction error in absolute terms             |
| **MSE**  | Mean Squared Error                | Penalizes large errors more heavily                    |
| **RMSE** | Root Mean Squared Error           | Interpretable error in the same unit as the target     |

All comparison results are saved to `model_comparison.csv`.

---

## 🛠️ Tech Stack

| Category              | Tools / Libraries                    |
|-----------------------|--------------------------------------|
| Programming Language  | Python 3.8+                          |
| Data Handling         | Pandas, NumPy                        |
| Machine Learning      | Scikit-learn                         |
| Model Serialization   | Pickle                               |
| IDE / Environment     | VS Code / Jupyter Notebook           |

---

## 📁 Project Structure

```
Crop-Yield-Prediction/
│
├── crop yield.csv              # Raw dataset with historical crop data
├── train_models.py             # Script to train, evaluate & save the best model
├── predict_yield.py            # Script to load saved model and predict yield
├── model_comparison.csv        # Auto-generated model evaluation results
├── best_crop_yield_model.pkl   # Auto-generated: saved best model pipeline
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

> **Note:** `best_crop_yield_model.pkl` and `model_comparison.csv` are generated when you run `train_models.py`.

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kajal805-M/Crop-Yield-Prediction.git
   cd Crop-Yield-Prediction
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

### Step 1 — Train the Models

Run the training script to preprocess data, train all three models, compare them, and save the best one:

```bash
python train_models.py
```

**Output:**
```
Dataset shape: (XXXX, 13)
Categorical columns: ['Crop', 'Season', 'State']
Numeric columns: ['Crop_Year', 'Area', 'Production', ...]

Model Comparison:
            Model       MAE       MSE      RMSE        R2
    Random Forest    XX.XX    XXX.XX     XX.XX      0.XX
    Decision Tree    XX.XX    XXX.XX     XX.XX      0.XX
 Linear Regression    XX.XX    XXX.XX     XX.XX      0.XX

Best model: Random Forest
Saved model to best_crop_yield_model.pkl
Saved comparison table to model_comparison.csv
```

### Step 2 — Predict Crop Yield

Run the prediction script and enter your values when prompted:

```bash
python predict_yield.py
```

**Example Interaction:**
```
Loaded best model: Random Forest

Enter values for prediction:
Crop: Wheat
Crop Year (e.g., 2024): 2024
Season (e.g., Kharif/Rabi/Whole Year): Rabi
State: Punjab
Area: 5000
Production: 15000
Annual Rainfall: 650.5
Fertilizer: 120.0
Pesticide: 3.5
Average Temperature: 22.0
Max Temperature: 35.0
Min Temperature: 10.0

Predicted Crop Yield: 3.0000
```

---

## 🔍 How It Works

```
Raw CSV Data
     │
     ▼
┌─────────────────────────────────────────┐
│          Preprocessing Pipeline          │
│                                         │
│  Numeric Features:                      │
│    SimpleImputer (median)               │
│    → StandardScaler                     │
│                                         │
│  Categorical Features:                  │
│    SimpleImputer (most_frequent)        │
│    → OneHotEncoder (handle_unknown)     │
│                                         │
│  Combined via ColumnTransformer         │
└─────────────────┬───────────────────────┘
                  │
                  ▼
         Model Training (3 models)
                  │
                  ▼
         Model Evaluation (R², MAE, RMSE)
                  │
                  ▼
         Best Model Selected & Saved
         (best_crop_yield_model.pkl)
                  │
                  ▼
         predict_yield.py → User Input → Prediction
```

The entire preprocessing + model is wrapped in a single scikit-learn `Pipeline`, so training and inference use exactly the same transformations — preventing data leakage.

---

## 👩‍💻 Author

**Kajal** — [@Kajal805-M](https://github.com/Kajal805-M)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

> ⭐ If you found this project useful, consider giving it a star on [GitHub](https://github.com/Kajal805-M/Crop-Yield-Prediction)!
