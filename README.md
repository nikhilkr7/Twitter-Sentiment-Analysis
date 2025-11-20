# Twitter-Sentiment-Analysis

A complete end-to-end Machine Learning project that performs **sentiment analysis on tweets** using the **Sentiment140 dataset**, applying **text preprocessing**, **TF-IDF vectorization**, and **Logistic Regression** for classification.

This README explains the project architecture, workflow, setup instructions, model pipeline, results, and how to run predictions.


##  Project Overview

This project analyzes the sentiment (Positive / Negative) of tweets using classical machine learning techniques.

### **Key Features**

* Downloads the **Sentiment140** dataset via Kaggle API
* Cleans and preprocesses tweets
* Converts text into numerical features using **TF-IDF** (unigrams + bigrams)
* Trains a **Logistic Regression** model
* Evaluates accuracy, precision, recall, F1-score, and confusion matrix
* Saves & loads the trained model with **pickle**
* Predicts sentiment for any new tweet

---

##  Project Architecture

```mermaid
flowchart LR
    A[Load Sentiment140 Dataset] --> B[Preprocessing & Cleaning]
    B --> C[TF-IDF Vectorization]
    C --> D[Model Training (Logistic Regression)]
    D --> E[Evaluation]
    E --> F[Save Model & Vectorizer]
    F --> G[Prediction on New Tweets]
```

---

##  Project Structure

```
Twitter-Sentiment-Analysis/
│── Twitter_Sentiment_Analysis.ipynb   # Main notebook
│── kaggle.json                        # Kaggle API key
│── sentiment_model.pkl                # Saved ML model
│── vectorizer.pkl                     # Saved TF-IDF vectorizer
│── README.md                          # Project documentation
│── sentiment140.zip                   # Dataset (after Kaggle download)
└── training.1600000.processed.noemoticon.csv
```

---

##  Tech Stack

* **Python 3**
* **Pandas, NumPy** — data handling
* **Scikit-learn** — ML model & TF-IDF
* **Zipfile** — dataset extraction
* **Pickle** — model persistence
* **Kaggle API** — dataset download

---

##  Installation & Setup

### **1. Install Dependencies**

```bash
pip install pandas numpy scikit-learn kaggle
```

### **2. Add Kaggle Credentials**

Place your `kaggle.json` file in:

```bash
~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### **3. Download the Dataset**

```bash
kaggle datasets download -d kazanova/sentiment140
unzip sentiment140.zip
```

### **4. Run the Jupyter Notebook**

```bash
jupyter notebook Twitter_Sentiment_Analysis.ipynb
```

---

##  Preprocessing Steps

Each tweet undergoes:

* Lowercasing
* Removing URLs
* Removing mentions (@usernames)
* Removing special characters & numbers
* Optional stopword removal or stemming

A new column `clean_text` stores the cleaned version.

---

##  Feature Extraction — TF-IDF

The project uses:

* **max_features = 20,000**
* **ngram_range = (1, 2)** (unigrams + bigrams)

TF-IDF converts tweets into sparse numerical vectors that work well with linear models.

---

## Model Used

### **Logistic Regression**

Chosen because:

* Works well for high-dimensional sparse text
* Fast training & inference
* Interpretable
* Reliable baseline for NLP tasks

```python
model = LogisticRegression(max_iter=200)
```

---

## Model Evaluation

The notebook outputs:

* **Accuracy**
* **Precision, Recall, F1-score**
* **Confusion Matrix**

This helps observe model performance on the held-out test set.

---

##  Saving & Loading Model

### Save:

```python

## 🔍 Predicting Sentiment

### **Example:**

```python
sample = ["I love this phone", "I hate this product"]
X = vectorizer.transform(sample)
predictions = model.predict(X)
```

Output:

```
I love this phone -> Positive
I hate this product -> Negative
```

---

## 🚀 Future Improvements

* Implement full `clean_text()` function
* Use **NLP deep learning models** (LSTM, BiLSTM, GRU)
* Fine-tune **BERT-based transformers** for State-of-the-art accuracy
* Add Flask/FastAPI backend for real-time inference
* Build UI dashboard with Streamlit

---

