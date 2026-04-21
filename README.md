# 🔍 Fake Job Posting Detector

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![ML](https://img.shields.io/badge/ML-Classification-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Problem Statement
Millions of fake job postings deceive job seekers every year.
This project uses Machine Learning + NLP to automatically detect
whether a job listing is fraudulent or legitimate.

## 📊 Dataset
- Source: [Kaggle — Real/Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- Records: 17,880 job postings
- Features: 18 (text + structured)
- Target: `fraudulent` (0 = Legitimate, 1 = Fake)

## 🔧 Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- TF-IDF (NLP)
- Matplotlib, Seaborn

## ⚙️ Approach
1. EDA — class imbalance, missing values, fraud by location
2. Text Cleaning — removed HTML, URLs, punctuation
3. Feature Engineering — TF-IDF (5000 features) + structured features
4. Trained 3 models — Logistic Regression, Random Forest, XGBoost
5. Evaluated using Accuracy, ROC-AUC, Confusion Matrix

## 📈 Results
| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | 95.2%    | 0.961   |
| Random Forest       | 97.5%    | 0.981   |
| XGBoost ✅ Best     | 98.1%    | 0.989   |

## 📂 Project Structure
fake-job-detector/
├── fake_job_detector.ipynb   # Main notebook
├── README.md
└── requirements.txt

## 🚀 How to Run
1. Open `fake_job_detector.ipynb` in Google Colab
2. Upload `fake_job_postings.csv` from Kaggle
3. Run all cells

## 📦 Requirements
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
