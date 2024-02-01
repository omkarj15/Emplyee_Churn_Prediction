# Employee Behavior Prediction

## Introduction

Employee churn is a critical concern for companies, as high turnover rates can adversely impact productivity, profitability, and institutional knowledge. This project focuses on testing logistic regression, decision tree, and random forest models on an employee dataset to identify the most effective algorithm for predicting churn.

## Problem Statement

Employee churn has far-reaching consequences, negatively affecting productivity, profitability, and institutional knowledge within an organization. This project aims to address this issue by utilizing machine learning algorithms to predict employee churn effectively.

## Key Features

We analyze nine key employee factors such as Age, Experience, etc., to determine whether an employee is likely to churn or not. Understanding and predicting employee churn is vital for optimizing resource allocation and enhancing overall working efficiency.

## Objective

Develop a predictive model leveraging machine learning algorithms to accurately predict whether an employee will churn or not.

## Process Flow

1. Select the Topic
2. Searching Dataset
3. Data Pre-processing & Data Visualization
4. Applying Algorithms
5. Model Building
6. Interpretation & Conclusions

## Tools & Platform Used

- Tools: Python
- Platform: Jupyter Notebook, AWS, Visual Studio Code
- Libraries Used: Scikit-learn, Pandas, NumPy, Seaborn, Matplotlib

## Data Pre-processing (EDA) & Visualization

### Data Description

1. The dataset comprises 4653 entries & 9 columns.
2. It includes 5 categorical, 3 numerical, and 1 date column.

### Data Pre-processing Steps

1. Remove Null values from the dataset.
2. Remove duplicate values from the dataset.
3. Treat outliers in the dataset.
4. Perform data manipulation.
5. Use Label Encoder on categorical variables.
6. Check if the data is unbalanced.

### Details of Dataset

- Duplicate Values: No duplicate values in the dataset.
- Missing Values: No null values in the dataset.
- Outliers: No outliers, visualized using Boxplot.

### Checking Data Normality

1. Age column converted from numerical data into groups (Positive Skewness).
2. Approximately follows Gaussian distribution curve.

### Data Visualization

1. Sunburst Diagram: Depicting relationships between Age, Bench Status, Gender, City, Education, and the target variable (Leave/Not Leave).
2. Bar Graphs: Representing different age categories and their relation to the target variable.

## Models

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Conclusion

Random Forest exhibits the highest accuracy, precision, F1 score, and recall. These results underscore the effectiveness of Random Forest in accurately predicting outcomes, highlighting its potential as a powerful tool for the given task. It is recommended to use the Random Forest model for live predictions of employee churn.
