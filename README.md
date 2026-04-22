# Engineer Performance Prediction

Project ini dibuat untuk memprediksi performa engineer menggunakan 4 pendekatan model yang berbeda:

1. Classification Model
2. Regression Model
3. ANN Classification Model
4. ANN Regression Model

Selain melakukan prediksi, project ini juga menggunakan MLflow untuk experiment tracking dan model registry, serta Streamlit untuk tampilan dashboard interaktif.

---

## Project Objective

Tujuan project ini adalah:

- memprediksi apakah performa engineer termasuk `Good` atau `Not Good`
- memprediksi nilai `Efficiency`
- membandingkan performa model machine learning tradisional dan artificial neural network
- menampilkan hasil prediksi ke dalam dashboard interaktif berbasis Streamlit

---

## Models Used

### Classification
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### Regression
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor

### Deep Learning
- ANN Classification
- ANN Regression

---

## Project Structure

```text
Global_Project_Eng_Performance/
│
├── Data/
│   ├── civil_engineering_global_project_dataset.csv
│   ├── clean_dataset.csv
│   ├── featured_dataset.csv
│   ├── X_class_train.csv
│   ├── X_class_test.csv
│   ├── y_class_train.csv
│   ├── y_class_test.csv
│   ├── X_reg_train.csv
│   ├── X_reg_test.csv
│   ├── y_reg_train.csv
│   └── y_reg_test.csv
│
├── Models/
│   ├── scaler_class.pkl
│   ├── scaler_reg.pkl
│   ├── scaler_ann_class.pkl
│   ├── scaler_ann_reg.pkl
│   ├── best_classification_model.pkl
│   ├── best_regression_model.pkl
│   ├── best_ann_classification.keras
│   └── best_ann_regression.keras
│
├── Reports/
│   ├── preprocessing_result.txt
│   ├── eda_result.txt
│   ├── feature_result.txt
│   ├── classification_result.txt
│   ├── regression_result.txt
│   ├── ann_classification_result.txt
│   ├── ann_regression_result.txt
│   └── mlflow_registry_result.txt
│
├── Figures/
│
├── mlruns/
│
├── 01_preprocess.py
├── 02_eda.py
├── 03_feature_engineering.py
├── 04_split_data_class.py
├── 05_split_data_reg.py
├── 06_classification_model.py
├── 07_regression_model.py
├── 08_ann_classification.py
├── 09_ann_regression.py
├── 10_register_mlflow.py
├── app.py
├── requirements.txt
└── README.md
Workflow
Urutan pengerjaan project ini adalah:

01_preprocess.py

membersihkan dataset mentah
mengecek missing values dan duplicate
menyimpan clean_dataset.csv
02_eda.py

melakukan exploratory data analysis
membuat visualisasi dan insight awal
03_feature_engineering.py

membuat fitur turunan
menyimpan featured_dataset.csv
04_split_data_class.py

membagi data classification
menyimpan scaler classification
05_split_data_reg.py

membagi data regression
menyimpan scaler regression
06_classification_model.py

baseline semua model classification
tuning semua model classification
memilih best model final
mengecek feature importance
memilih top 8 features
07_regression_model.py

baseline semua model regression
tuning semua model regression
memilih best model final
mengecek feature importance
memilih top 8 features
08_ann_classification.py

melatih model ANN classification
09_ann_regression.py

melatih model ANN regression
10_register_mlflow.py

mendaftarkan model terbaik ke MLflow Registry
app.py

menampilkan hasil 4 model pada dashboard Streamlit
MLflow Registry
Model yang diregister ke MLflow:

EngineerPerformanceClassificationModel
EngineerPerformanceRegressionModel
EngineerPerformanceANNClassificationModel
EngineerPerformanceANNRegressionModel
Input Features in Streamlit
Input mentah yang digunakan pada dashboard:

Certificates
Years of Experience
age
Time Arrival Strafe
Project Cost
Project Proximity
Violation Risk Index
Company PCAB Score
Weekly Overtime Hours
Salary Bracket
Fitur turunan dihitung otomatis di dalam aplikasi:

Experience_Ratio
Punctuality_Score
Burnout_Risk
Salary_Experience_Ratio
Evaluation Metrics
Classification
Accuracy
Precision
Recall
F1-score
AUC-ROC
Regression
R²
MAE
RMSE
How to Run
1. Install dependencies
pip install -r requirements.txt
2. Run preprocessing
python 01_preprocess.py
3. Run EDA
python 02_eda.py
4. Run feature engineering
python 03_feature_engineering.py
5. Split data
python 04_split_data_class.py
python 05_split_data_reg.py
6. Train models
python 06_classification_model.py
python 07_regression_model.py
python 08_ann_classification.py
python 09_ann_regression.py
7. Register models to MLflow
python 10_register_mlflow.py
8. Run Streamlit app
streamlit run app.py
Output
Output utama dari project ini:

model classification terbaik
model regression terbaik
model ANN classification terbaik
model ANN regression terbaik
report evaluasi model
visualisasi feature importance
dashboard Streamlit interaktif
model registry di MLflow
Notes
Classification dan regression menggunakan target yang berbeda:
Classification target: is_good
Regression target: Efficiency
Fitur yang digunakan dalam model final dipilih kembali berdasarkan feature importance.
MLflow digunakan untuk experiment tracking dan model registry.
Streamlit digunakan untuk menampilkan hasil prediksi 4 model dalam satu dashboard.
Author
Final Project Bootcamp
Engineer Performance Prediction