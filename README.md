# 🌾 Crop Yield Prediction using Machine Learning

A machine learning-based web application that predicts **crop yield** based on agricultural and environmental factors such as crop type, state, season, rainfall, fertilizer usage, pesticide usage, temperature, area, and production.

This project helps in understanding how different factors affect crop productivity and provides an intelligent prediction system for agriculture-related decision making.

---

## 🚀 Project Overview

Crop yield prediction is an important problem in agriculture because it helps farmers, researchers, and policymakers estimate production in advance.  
In this project, machine learning models are trained on historical crop data to predict the expected yield of a crop.

The project includes:

- Data preprocessing
- Missing value handling
- Outlier treatment
- Feature engineering
- Model training
- Model evaluation
- User-friendly crop yield prediction interface

---

## 🎯 Objective

The main objective of this project is to build a machine learning model that can predict crop yield accurately using crop-related and climate-related features.

---

## 📌 Features

- Predicts crop yield based on user input
- Handles categorical and numerical data
- Uses machine learning regression models
- Includes feature engineering such as average temperature
- Provides model evaluation using important metrics
- Clean and simple user interface
- Useful for agriculture analytics and decision support

---

## 🧠 Machine Learning Models Used

The following regression models were tested:

- Linear Regression
- Random Forest Regressor
- Decision Tree

The best-performing model was Random Forest . It was selected based on evaluation metrics.

---

## 📊 Evaluation Metrics

The model performance was evaluated using:

- R² Score
- Mean Absolute Error
- Root Mean Squared Error
- Mean Absolute Percentage Error

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Programming Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Saving | Joblib / Pickle |
| IDE | VS Code / Jupyter Notebook |

---

## 📂 Dataset Features

The dataset contains agricultural and environmental features such as:

| Feature | Description |
|---|---|
| Crop | Name of the crop |
| State | State where crop is grown |
| Season | Crop growing season |
| Crop_Year | Year of cultivation |
| Area | Cultivated area |
| Production | Total crop production |
| Annual_Rainfall | Yearly rainfall |
| Fertilizer | Fertilizer used |
| Pesticide | Pesticide used |
| Max_Temperature | Maximum temperature |
| Min_Temperature | Minimum temperature |
| Avg_Temperature | Average temperature |
| Yield | Target variable |

---

## 📁 Project Structure

```bash
Crop-Yield-Prediction/
│
├── data/
│   └── Crop_Yield.csv
│
├── notebooks/
│   └── predict_yeild.py
│   └── train_model.py
├── models/
│   └── model_comparision.pkl
│   
│
├── 
├── requirements.txt
├── README.md
└── model_metadata.json

---

👩‍💻 Author

**Kajal** — [@Kajal805-M](https://github.com/Kajal805-M)

📄 License

This project is open-source and available under the [MIT License](LICENSE).
