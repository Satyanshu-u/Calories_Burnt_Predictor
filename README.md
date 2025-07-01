# 🔥 Calories Burnt Prediction using Machine Learning (Gradio + XGBoost)

This project builds a machine learning model that predicts the number of "calories burned" during physical activity based on personal and workout attributes such as age, gender, duration, and heart rate. It utilizes a robust *XGBoost Regressor* and features an intuitive user interface powered by "Gradio".



## 📌 Features

- Merges two complementary datasets: exercise.csv and calories.csv
- Performs exploratory data analysis and visualization
- Builds a predictive model using "XGBoost Regression"
- Deploys a real-time *Gradio web interface for predictions
- Designed for reproducibility in "Google Colab"



## 📁 Files in This Repository

| File | Description |

| Calories_Burnt_Prediction.ipynb | Main notebook with preprocessing, model training, and Gradio app |
| requirements.txt | Python package dependencies |
| README.md | Project documentation |
| calories.csv  | Calories dataset |
| exercise.csv  | Exercise dataset |



## 📊 Dataset Overview

- Source: [Calories Burnt Dataset - Kaggle](https://www.kaggle.com/datasets/fmendes/fitter)
- exercise.csv: Contains participant data such as Gender, Age, Duration, Heart Rate, etc.
- calories.csv: Provides calories burned for corresponding records
- Merged using pd.concat() to form a comprehensive dataset



## 🧠 Machine Learning Model

- *Model*: XGBRegressor from the XGBoost library
- *Target Variable*: Calories
- *Input Features*:
  - Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp
- *Preprocessing*:
  - Encoded categorical Gender
  - Train/test split using train_test_split
- *Performance Metrics*:
  - Mean Absolute Error (MAE)
  - R-squared Score (R²)
- *Visualization*:
  - Feature distributions
  - Correlation heatmaps



## 🧪 Gradio Interface

A real-time prediction interface built with *Gradio*, allowing users to input features and get predicted calorie burn instantly.

### Sample Code:
```python
gr.Interface(fn=calorie_prediction, inputs=[...], outputs="text").launch()

> Once the notebook is executed in Colab, a public URL will be provided to interact with the app.






🚀 How to Run

1. Open the notebook in Google Colab



2. Install dependencies:

!pip install xgboost gradio pandas numpy seaborn matplotlib scikit-learn


3. Run all cells to:

Preprocess and visualize data

Train the XGBoost model

Launch the prediction interface







📦 Requirements

Install all dependencies with:

pip install -r requirements.txt

Contents of requirements.txt:

xgboost
gradio
pandas
numpy
seaborn
matplotlib
scikit-learn




📄 License

This project is licensed under the MIT License.




👨‍💻 Author

Developed as a hands-on machine learning project focusing on regression modeling and interactive ML deployment using:

XGBoost

Gradio

Google Colab



