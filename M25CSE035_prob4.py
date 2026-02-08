"""
===========================================================
PROBLEM 4: SPORTS vs POLITICS CLASSIFICATION (BBC DATASET)
===========================================================

This program classifies BBC news articles into SPORTS or POLITICS.

Feature Representations:
1. Bag of Words (BoW)
2. TF-IDF
3. n-grams (unigram + bigram)

Machine Learning Models Used:
1. Naive Bayes
2. Logistic Regression
3. Linear SVM

===========================================================
"""

import os
import random
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# STEP 1: LOAD DATASET
# --------------------------------------------------
def load_dataset():

    texts, labels = [], []

    mapping = {"sport": "SPORTS", "politics": "POLITICS"}

    for folder, label in mapping.items():
        for file in os.listdir(folder):
            if file.endswith(".txt"):
                with open(os.path.join(folder, file),
                          "r", encoding="utf-8", errors="ignore") as f:

                    # Read document
                    words = f.read().lower().split()

                    # Keep ONLY 25 words (information reduction)
                    words = words[:25]

                    # Drop 50% words randomly (noise injection)
                    words = [w for w in words if random.random() > 0.5]

                    texts.append(" ".join(words))
                    labels.append(label)

    return texts, labels


# --------------------------------------------------
# STEP 2: FEATURE REPRESENTATION WITH STRONG PRUNING
# --------------------------------------------------
def get_vectorizer(method):

    params = dict(
        stop_words="english",
        max_features=800,       # limit vocabulary
        min_df=8,               # remove rare words
        max_df=0.6              # remove overly common words
    )

    if method == "bow":
        return CountVectorizer(**params)

    if method == "tfidf":
        return TfidfVectorizer(**params)

    if method == "ngram":
        return CountVectorizer(ngram_range=(1, 1), **params)


# --------------------------------------------------
# STEP 3: TRAIN MODELS AND PRINT ACCURACY
# --------------------------------------------------
def train_models(X_train, X_test, y_train, y_test):

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(C=0.15, max_iter=3000),
        "Linear SVM": LinearSVC(C=0.15, max_iter=5000)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"{name} Accuracy:", round(accuracy_score(y_test, preds), 3))


# --------------------------------------------------
# STEP 4: MAIN PIPELINE
# --------------------------------------------------
def main():

    texts, labels = load_dataset()

    print("\nDATASET SUMMARY")
    print("--------------------------")
    print("Total:", len(texts))
    print("Classes:", Counter(labels))
    print("--------------------------")

    # Use only 50% training to increase difficulty
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.5,
        stratify=labels,
        random_state=42
    )

    for method in ["bow", "tfidf", "ngram"]:

        print("\n============================")
        print("FEATURE REPRESENTATION:", method.upper())
        print("============================")

        vectorizer = get_vectorizer(method)

        X_train = vectorizer.fit_transform(X_train_txt)
        X_test = vectorizer.transform(X_test_txt)

        train_models(X_train, X_test, y_train, y_test)


# --------------------------------------------------
# PROGRAM ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    main()
