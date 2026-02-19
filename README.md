# Malware Detection using Machine Learning

This project focuses on predicting whether a Windows machine is infected with malware based on system telemetry data. The workflow includes exploratory data analysis (EDA), data preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation using multiple machine learning algorithms.

The final selected model is a **Gradient Boosting Classifier**, which achieved an accuracy of approximately **63%** on the validation dataset.

---

## Project Objectives

* Perform detailed **Exploratory Data Analysis (EDA)** to understand the dataset.
* Handle missing values, redundant columns, and outliers.
* Build preprocessing pipelines for categorical and numerical features.
* Train and compare multiple machine learning models.
* Select the best-performing model and generate predictions for submission.

---

## Dataset

The dataset contains system-level attributes such as:

* OS information
* Security settings
* Hardware specifications
* Device configuration
* Defender status indicators

The target variable indicates whether the machine has detected malware.

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

* Checked missing values and distributions.
* Identified highly correlated feature pairs.
* Removed redundant columns (single-value or near-duplicate features).
* Analyzed categorical feature uniqueness.
* Visualized correlations using heatmaps.

### 2. Data Cleaning & Preprocessing

Key steps:

* Dropped high-cardinality columns (e.g., MachineID).
* Removed unrealistic or noisy columns.
* Handled outliers (e.g., replacing abnormal values with mode).
* Converted scientific notation columns to numeric.
* Separated categorical and numerical features.
* Built preprocessing pipelines using:

  * Imputation
  * Encoding
  * Scaling

---

### 3. Model Training & Hyperparameter Tuning

The following models were trained and evaluated:

* **SGD Classifier**
* **Decision Tree Classifier**
* **Gradient Boosting Classifier** 

Dataset was split into training and validation sets for evaluation.

---

### 4. Model Evaluation

Evaluation metrics used:

* Accuracy
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

The **Gradient Boosting Classifier** performed best with:

> **Accuracy ≈ 63%**

---

### 5. Final Submission

Predictions were generated using the best-performing model and exported for submission.

---

## Key Insights

* Several features were highly correlated and redundant.
* Removing noisy columns significantly improved performance.
* Gradient Boosting handled feature interactions better than linear models.
* Proper preprocessing pipelines were critical for stable results.

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/anuiitm/MLP-Project.git
cd MLP-Project
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

```bash
jupyter notebook Notebook.ipynb
```

---

## Requirements

Typical libraries used:

* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* jupyter

---

## 📁 Project Structure

```
├── Notebook.ipynb          
├── README.md               
├── requirements.txt        
└── System-Threat-forecaster.zip          
```

---

## Future Improvements

* Try advanced ensemble models (XGBoost, LightGBM, CatBoost).
* Feature selection using SHAP or feature importance.
* Hyperparameter tuning with Bayesian optimization.
* Address class imbalance using resampling techniques.
* Deploy the model as a web service.

---

## Author

**Anubhav Agarwal**
