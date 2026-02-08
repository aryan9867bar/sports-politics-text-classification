# Sports vs Politics Text Classification

This repository contains an end-to-end machine learning project for classifying news articles into **Sports** or **Politics**.

## Dataset
The dataset is derived from the BBC News corpus. Each document is stored as a separate `.txt` file.
# https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification/data

dataset/
├── sport/
├── politics/


## Feature Representations
- Bag of Words (BoW)
- TF-IDF
- n-grams (unigram + bigram)

## Machine Learning Models
- Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

## How to Run
```bash
python M25CSE035_prob4.py
```
## Results Summary
| Feature Representation | Naive Bayes | Logistic Regression | SVM |
|----------------------|------------|--------------------|-----|
| Bag of Words | 1.00 | 0.99 | 1.00 |
| TF-IDF | 1.00 | 1.00 | 1.00 |
| n-grams | 1.00 | 0.99 | 1.00 |

## ACCURACY COMPARISON TABLE
| Feature Representation | Naive Bayes | Logistic Regression | SVM |
|----------------------|------------|--------------------|-----|
| Bag of Words | 100% | 99% | 100% |
| TF-IDF | 100% | 100% | 100% |
| n-grams | 100% | 99% | 100% |

## Conclusion
Linear models combined with TF-IDF features perform exceptionally well for this task due to strong lexical separation between sports and political news.