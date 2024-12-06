##Project Overview
This project uses machine learning to predict election outcomes based on cleaned and processed polling data. The workflow involves data cleaning, feature engineering, model training, evaluation, and prediction aggregation. Models like Random Forest and Logistic Regression are utilized to classify election results and predict outcomes for unseen data.

#Data Loading and Initial Cleaning

#Data from cleaned_condensed_polls_final.csv is loaded.
Non-informative columns (e.g., target, classification_numeric) and columns that directly correspond to the target (classification) are removed to avoid data leakage.

#Feature Engineering

Categorical columns are one-hot encoded using pandas.get_dummies() to convert them into numeric form for model compatibility.
Missing values are handled using imputation (e.g., mean imputation for numeric columns).

#Data Splitting

Rows with known classification values are used as training data.
Rows without classification values are used as test data for predictions.

#Model Training

Random Forest Classifier is trained on the training data (X_train and y_train).
The model learns to classify election outcomes into pre-defined categories based on the feature set.

#Test Data Predictions

The trained model predicts outcomes for the test data (X_test).
Predictions are added to the original dataset under the predicted column.

#State-Level Predictions

Predictions for the test data are aggregated at the state level.
The most frequent prediction for each state is chosen as the state-level prediction.

#Model Evaluation

The model is evaluated on the training data using accuracy and a classification report.
Feature importance is calculated and visualized to understand the most influential features.

#Saving Outputs

Individual row-level predictions for the test data are saved to test_predictions_random_forest.csv.
State-level predictions are saved to test_state_level_predictions_random_forest.csv.
The complete dataset, including all predictions, is saved to full_predictions_random_forest.csv.

#Visualization

Feature importance is visualized using a horizontal bar chart to highlight the top 10 features influencing the mode
