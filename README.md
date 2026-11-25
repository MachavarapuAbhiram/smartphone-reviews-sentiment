# Smartphone Sentiment Analysis

A Smartphone Sentiment Analysis capstone project using TF-IDF (with negation preservation) and scikit-learn classifiers, paired with an interactive Streamlit dashboard.

## Features
- **Live Review & Aspects:** Predicts sentiment (`Positive`, `Neutral`, `Negative`), displays class probabilities and top influential TF-IDF n-grams (for Logistic Regression), and breaks down aspect-level sentiment across **Battery**, **Camera**, **Display**, and **Heating/Performance**.
- **Dataset Overview:** Summarizes the 3,000 smartphone reviews, sentiment distribution, and aspect mention counts vs. negative-sentiment rates.
- **Model Evaluation:** Displays the 80/20 stratified benchmark leaderboard (`Logistic Regression`, `Linear SVM`, `Multinomial Naive Bayes`), confusion matrix, and sample misclassifications.

## Local Setup
```bash
pip install -r requirements.txt
python train_day1.py
streamlit run app.py
```

## Project Structure
- `app.py` — Streamlit web application
- `train_day1.py` — Data preprocessing, 80/20 stratified training & benchmarking
- `check_csv.py` — Quick dataset inspection script
- `data/smartphone_reviews.csv` — Smartphone review dataset
- `models/` — Serialized best sentiment pipeline and label list
- `outputs/` — Benchmark metrics CSV, confusion matrix chart, and misclassification samples
