# Fraud-Detection-Machine-Learning
## Overview

This project aims to develop a fraud detection system using a credit card transactions dataset. The primary objective is to identify fraudulent transactions by applying machine learning techniques, specifically Logistic Regression and Random Forest Classifier. The dataset contains various features related to credit card transactions, including transaction amount, time, and other anonymized variables.

## Dataset

The dataset used for this project is a collection of credit card transactions labeled as either fraudulent or legitimate. It includes the following features:

- **Time**: The elapsed time since the first transaction in the dataset.
- **V1-V28**: Anonymized features derived from PCA (Principal Component Analysis) to protect user identities.
- **Amount**: The transaction amount.
- **Class**: Target variable indicating whether the transaction is fraudulent (1) or legitimate (0).

## Installation

To run this project, you will need to have Python installed along with the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Data Preprocessing

Before applying machine learning models, the data needs to be preprocessed:

1. **Loading the Data**: Read the dataset using pandas.
2. **Handling Missing Values**: Check for and handle any missing values in the dataset.
3. **Feature Scaling**: Normalize or standardize features, particularly the 'Amount' column, to improve model performance.
4. **Train-Test Split**: Split the dataset into training and testing sets (e.g., 80% train, 20% test).

## Exploratory Data Analysis (EDA)

An exploratory analysis is conducted to understand the distribution of features, identify patterns, and visualize correlations. Key steps include:

- Visualizing the distribution of transaction amounts.
- Analyzing the class distribution (fraud vs. legitimate).
- Examining correlations between features.

## Model Development

### Logistic Regression

Logistic Regression is a statistical model used for binary classification problems. In this project:

- The model is trained on the training dataset.
- Hyperparameters are tuned using cross-validation.
- Model performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

### Random Forest Classifier

Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training. For this project:

- The model is trained on the same training dataset.
- Feature importance is analyzed to understand which variables contribute most to fraud detection.
- Similar performance metrics as those used for Logistic Regression are calculated.

## Results

The results section will present a comparison of both models based on their performance metrics. Key findings may include:

- Accuracy rates of both models.
- Confusion matrices illustrating true positives, false positives, true negatives, and false negatives.
- ROC curves to visualize model performance.

## Conclusion

This project demonstrates how machine learning techniques can effectively detect fraudulent credit card transactions. Future work may involve exploring additional algorithms, optimizing hyperparameters further, or implementing real-time fraud detection systems.

## 
1. creditcard_csv - Contains the dataset file for the project.
2. Fraud Detection Machine Learning.ipynb - Contains the code documentation.


## License

This project is licensed under the MIT License - see the LICENSE file for details.
