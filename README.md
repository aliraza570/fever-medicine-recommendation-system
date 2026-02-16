AI Fever Medicine Prediction Project
Complete Machine Learning Pipeline for Healthcare Data Analysis
Project Introduction

This project is based on a healthcare-oriented dataset designed to analyze fever-related patient conditions, environmental factors, and medicine recommendations. The dataset contains detailed information about patient health such as age, temperature, BMI, heart rate, humidity, AQI, symptoms, lifestyle habits, chronic conditions, allergies, previous medication, and recommended medicine.

The dataset includes 1000 records and 20 attributes, providing a comprehensive view of patient health and environmental conditions.
The main objective of this project was to design a complete and professional machine learning pipeline that processes healthcare data, improves data quality, engineers meaningful features, selects important predictors, and trains reliable predictive models for temperature analysis and medical recommendation support.

Each stage of the project was organized into structured folders, ensuring clarity, professional workflow management, and easy understanding for anyone reviewing the project.

DATA CLEANING

The first stage of the project focused on cleaning and preparing the dataset to ensure reliability and consistency. The dataset was loaded and examined to understand its structure, feature names, data types, and overall composition. Initial inspection allowed identification of missing values, duplicate records, and potential inconsistencies that could negatively affect model performance.

During this process, missing values were detected in the Previous_Medication column. Since this feature is categorical in nature, the most appropriate approach was mode imputation. The missing values were replaced with the most frequently occurring value in that column. This method was chosen because it preserves all records, maintains the natural distribution of the data, and avoids unnecessary data loss. After imputation, the dataset was checked again to confirm that no null values remained.

Duplicate records were also examined carefully. Removing duplicate entries ensured that the dataset remained unbiased and that repeated patient records did not influence the learning process of machine learning models.

Outlier detection was then performed on numeric features such as temperature, age, BMI, humidity, AQI, and heart rate. The Interquartile Range (IQR) method was applied to detect extreme values. Instead of removing records containing outliers, extreme values were capped within lower and upper acceptable limits. This approach preserved all patient records while preventing abnormal values from distorting model training. After this process, the dataset became clean, balanced, and suitable for further analysis.

EDA

Exploratory Data Analysis was performed to understand feature behavior, distributions, and relationships within the dataset. This step provided a deeper understanding of how different health and environmental variables interact with each other.

Distribution analysis was conducted to observe whether numeric features followed normal or skewed patterns. Understanding distribution patterns was essential because machine learning algorithms often perform better when data is well structured and consistent.

A Spearman correlation test was applied to evaluate relationships among features. This statistical method was selected because it works effectively even when data does not follow a perfectly normal distribution. The analysis helped identify strong positive and negative relationships among variables such as temperature, symptoms, and environmental conditions. These insights supported later decisions in feature engineering and feature selection by highlighting which features had meaningful influence on the target variable.

This stage provided a clear overview of dataset behavior and ensured that all patterns and dependencies were well understood before moving to model preparation.

TRAIN TEST SPLIT

After completing cleaning and exploratory analysis, the dataset was divided into training and testing sets. An 80/20 split was applied, where 80 percent of the data was used for training and 20 percent for testing. The training dataset contained 800 records, while the testing dataset contained 200 records.

The purpose of this split was to evaluate model performance on unseen data. Training the model on one portion and testing it on another ensures that the model learns general patterns rather than memorizing specific records. This approach improves reliability and prevents overfitting.

All advanced processing steps such as feature engineering were applied only to the training dataset. This was done to avoid data leakage and ensure fair model evaluation. The testing dataset remained untouched until the final evaluation stage.

FEATURE ENGINEERING

Feature engineering was performed to enhance predictive capability and extract deeper insights from the dataset. New features were created by combining existing variables in meaningful ways.

A health index feature was created by combining temperature and heart rate to represent physiological stress. An environmental stress score was developed using temperature, humidity, and AQI values to capture environmental impact on patient health. BMI values were converted into categorized risk levels to simplify interpretation. A symptom severity score was generated by combining fever severity, headache, fatigue, and body ache. Additionally, a lifestyle risk score was created using smoking history, alcohol consumption, and physical activity data.

These engineered features added more meaningful information to the dataset and allowed machine learning models to better understand complex health patterns. Feature engineering significantly improved the predictive strength and depth of the dataset.

FEATURE SELECTION

After feature engineering, feature selection was applied to identify the most important predictors for model training. Spearman correlation analysis with temperature as the target variable was used to determine which features had strong and statistically significant relationships.

Only features with meaningful correlation strength and relevance were selected. Irrelevant or weak features were removed to improve model accuracy and efficiency. This step reduced noise in the dataset and ensured that models were trained using only impactful predictors.

Feature selection helped improve model performance, reduce overfitting risk, and simplify the dataset structure while maintaining predictive power.

CLASSIFIER MODEL

For classification analysis, temperature values were converted into categorical classes representing different fever levels. A classification model was then trained to predict these categories based on patient health and environmental features.

An XGBoost classifier was selected because of its high accuracy, efficiency, and ability to handle complex structured data. The model was trained using selected features from the feature selection stage.

Cross-validation was applied to test model stability across multiple data splits. This ensured that the classifier performed consistently and did not depend on a specific subset of data. Evaluation metrics such as accuracy, precision, recall, and F1-score were analyzed to measure classification performance. The classifier demonstrated strong and balanced performance across all evaluation metrics, indicating reliable prediction capability.

REGRESSOR MODEL

In addition to classification, regression analysis was performed to predict temperature as a continuous numeric value. A Ridge regression model was selected for this task because it handles multicollinearity effectively and produces stable predictions.

The regression model was trained using the selected features and evaluated using performance metrics such as RMSE, MAE, and R² score. These metrics measured prediction accuracy and error consistency. The model achieved low error values and stable performance, indicating reliable temperature prediction capability.

Residual analysis and prediction comparisons confirmed that the regression model performed consistently across different data samples and maintained balanced prediction behavior.