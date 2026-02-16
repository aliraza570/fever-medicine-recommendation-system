AI Fever Medicine Prediction Project – Visual Analysis Report

DATA CLEANING
Bubble Plot

The bubble plot in the cleaning stage visually represents the overall distribution and density of the dataset before further processing. Each bubble reflects the relationship between selected variables and their magnitude within the dataset. Larger bubbles indicate higher concentration or value presence, allowing quick detection of abnormal clusters or outliers. This visualization helped in identifying noisy or inconsistent records and ensured that the dataset structure was balanced and clean before moving to the analysis stage. It provides a first high-level understanding of how features are distributed and whether the dataset is suitable for machine learning modeling.

EXPLORATORY DATA ANALYSIS (EDA)
Violin + Box Plot

The combined violin and box plot provides a deep understanding of feature distribution and spread. The violin shape shows the full density distribution of values, while the box plot inside displays median, quartiles, and potential outliers. This visualization helps in understanding skewness, symmetry, and variability of each feature. It allows viewers to instantly observe whether the data follows a normal pattern or contains extreme variations. This plot is crucial for detecting imbalance and deciding whether normalization or transformation is required.

Heatmap Correlation

The correlation heatmap displays relationships between all numerical features in the dataset. Strong positive or negative correlations are highlighted through color intensity. This visualization helps identify multicollinearity, redundant features, and strong dependencies among variables. By observing this plot, one can clearly understand which features influence each other and which features can be retained or removed before modeling. It provides a foundational understanding of feature relationships within the dataset.

TRAIN–TEST SPLIT ANALYSIS
Heatmap Correlation on Train Data

This heatmap shows feature relationships specifically within the training dataset. It ensures that correlations remain consistent after splitting the dataset. Any major deviation would indicate imbalance between training and testing sets. This plot confirms that the training data maintains realistic relationships necessary for model learning.

Parallel Coordination Plot on Train Data

The parallel coordination plot visualizes multiple feature values simultaneously for training samples. Each line represents a single observation passing through all feature axes. This plot highlights patterns, clusters, and separations between different data ranges. It helps in understanding feature interaction and distribution complexity in the training dataset.

Heatmap Correlation on Test Data

This plot validates that the testing dataset maintains similar correlation structure to the training dataset. Consistency between train and test correlations ensures reliable model evaluation. If correlation patterns differed significantly, model predictions could become unreliable.

Parallel Coordination Plot on Test Data

This visualization confirms that test data follows patterns similar to training data. It allows visual comparison of feature distributions between datasets and ensures that the model will be evaluated on realistic unseen data.

FEATURE ENGINEERING
Heatmap Correlation after Feature Engineering

This heatmap shows how newly engineered features interact with existing ones. It highlights whether engineered features add meaningful information or create redundancy. Strong new correlations indicate successful feature creation that can improve model performance.

Histogram on Feature Engineering

The histogram displays distribution of engineered features. It shows whether new features follow normal distribution, contain skewness, or introduce outliers. This helps validate the usefulness of feature transformations and ensures they improve predictive capability rather than adding noise.

FEATURE SELECTION
Heatmap Correlation

This heatmap is generated after feature selection to confirm that only relevant features remain. Reduced multicollinearity and clearer relationships indicate successful selection of important features.

Ridge Plot

The ridge plot visualizes distribution overlap among selected features. It helps understand how feature values vary across ranges and whether selected features provide distinct information. Overlapping densities indicate shared patterns, while separated curves indicate strong discriminative power.

Bar Plot on Feature Selection (including engineered features)

This bar plot displays importance or contribution of selected features. It clearly shows which features have the strongest influence on prediction. Viewers can instantly understand which variables drive the model and why those features were selected.

CLASSIFIER CROSS VALIDATION
Bar Plot

The bar plot represents accuracy performance across different cross-validation folds. Each bar shows how the classifier performed on a specific fold. Consistency among bars indicates stable model performance and reliability across multiple splits.

Line Plot

The line plot shows the trend of accuracy across folds. It helps identify performance fluctuations and average model stability. A smooth trend with minimal variation indicates that the classifier generalizes well across different subsets.

ECDF Plot

The ECDF plot shows cumulative distribution of accuracy scores across folds. It visually represents how frequently certain accuracy levels are achieved. This helps viewers understand performance consistency and probability of achieving a given accuracy level.

REGRESSION CROSS VALIDATION
Bar Plot

This bar plot displays RMSE values across regression cross-validation folds. It shows error variation across different data splits. Lower and consistent bars indicate stable regression performance.

Line Plot

The line plot highlights RMSE trend across folds. It helps detect performance consistency and any sudden increase in prediction error. A stable line indicates reliable regression modeling.

Swarm Plot

The swarm plot visualizes distribution of RMSE values in a detailed manner. Each point represents a fold result. It provides a clear picture of variability and spread of prediction errors, helping viewers understand model robustness.

CLASSIFIER EVALUATION
Confusion Matrix

The confusion matrix shows classification performance by comparing actual and predicted classes. It highlights correct predictions and misclassifications, allowing clear evaluation of model accuracy per class.

Feature Importance

This plot ranks features based on their contribution to classification decisions. It shows which variables most strongly influence predictions, helping interpret the model.

AUC-ROC Curve

The ROC curve illustrates model ability to distinguish between classes. Higher curve area indicates better classification performance. It shows how well the model separates different categories.

Precision-Recall Curve

This plot demonstrates the balance between precision and recall across thresholds. It is especially useful for evaluating model performance on imbalanced datasets.

Predicted vs Actual Plot

This visualization compares predicted class labels with actual labels. It helps identify where the classifier performs correctly and where errors occur.

Precision, F1, Recall per Class

This plot shows class-wise performance metrics. It provides deeper insight into how well each class is predicted individually.

Learning Curve

The learning curve shows model performance against training size. It indicates whether the model is underfitting or overfitting and how performance improves with more data.

Calibration Curve

The calibration curve shows how predicted probabilities align with actual outcomes. It evaluates reliability of probability estimates generated by the classifier.

REGRESSOR EVALUATION
Actual vs Predicted

This scatter plot compares predicted temperature values with actual values. Points close to the diagonal line indicate accurate predictions.

Residual vs Predicted

This plot shows distribution of residual errors against predicted values. It helps detect bias or systematic prediction errors.

Residual Distribution

This histogram shows distribution of prediction errors. A normal centered distribution indicates good regression performance.

Actual vs Predicted Line Plot

This line plot compares actual and predicted values sequentially, allowing visual inspection of prediction alignment across samples.

Regressor Fit Line

The regression fit line shows overall relationship between actual and predicted values. A tight fit indicates strong predictive capability.

Error Distribution Box Plot

This box plot summarizes error spread and outliers. It helps evaluate consistency of prediction errors.

Homoscedasticity Plot

This plot checks whether residual variance remains constant across predictions. Uniform spread indicates a well-fitted regression model.

Feature Coefficient Plot

This plot displays coefficient values of features used by the Ridge regressor. It shows how strongly each feature influences prediction and whether its impact is positive or negative.

