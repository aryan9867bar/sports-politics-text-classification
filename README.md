# Sports vs Politics Text Classification

This repository contains an end-to-end machine learning project for classifying BBC news articles into **Sports** or **Politics** using classical Natural Language Processing techniques.
The objective of this project is to study how different feature representations and machine learning models perform on a binary text classification task.

---

## Dataset
The dataset is derived from the BBC News corpus. Each document is stored as a separate `.txt` file.
Kaggle Source:  
https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification/data

Each article is stored as a separate `.txt` file.


dataset/

├── sport/

├── politics/


---

### Dataset Statistics

- Sports articles: 511  
- Politics articles: 417  
- Total documents: 928  

---

## Feature Representations
- Bag of Words (BoW)
- TF-IDF
- n-grams (unigram + bigram)

---

## Machine Learning Models
- Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

---

## Techniques Applied

- Lowercasing  
- Whitespace tokenization  
- Vocabulary pruning  
- Random noise injection  
- Strong regularization  
- Reduced document length  
- Stratified train-test split  

---

## How to Run
```bash
python M25CSE035_prob4.py
```

---

## Results Summary
| Feature Representation | Naive Bayes | Logistic Regression | Linear SVM |
|-----------------------|------------|--------------------|-----------|
| Bag of Words | 79.5% | 79.7% | 80.4% |
| TF-IDF | 80.2% | 80.2% | 80.4% |
| n-grams | 79.5% | 79.7% | 80.4% |

TF-IDF with Linear SVM achieved the highest accuracy.

---

## Conclusion
The experiments show that traditional linear classifiers perform well on this task due to strong lexical separation between sports and political news. TF-IDF with Linear SVM provided the best performance. However, aggressive preprocessing was required to obtain realistic accuracy.