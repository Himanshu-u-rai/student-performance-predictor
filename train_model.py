# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
file_path = r'H:\Devlopment\StudentData\Student_performance_data _.csv'
data = pd.read_csv(file_path)

# Create a binary target variable for classification based on GPA
threshold = 2.0
data['PassStatus'] = data['GPA'].apply(lambda x: 1 if x >= threshold else 0)

# Prepare data for regression model
X = data.drop(['StudentID', 'GPA', 'PassStatus'], axis=1)
y_reg = data['GPA']

# Split the data into training and testing sets for regression
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Standardize the features for regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train the regression model
reg_model = LinearRegression()
reg_model.fit(X_train_scaled, y_train)

# Predict on the test set for regression
y_pred_reg = reg_model.predict(X_test_scaled)

# Evaluate the regression model
rmse = mean_squared_error(y_test, y_pred_reg) ** 0.5
r2 = r2_score(y_test, y_pred_reg)

print(f"Regression Model RMSE: {rmse}")
print(f"Regression Model R-squared: {r2}")

# Prepare data for classification model
X_clf = data.drop(['StudentID', 'GPA', 'PassStatus'], axis=1)
y_clf = data['PassStatus']

# Split the data into training and testing sets for classification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Standardize the features for classification
X_train_clf_scaled = scaler.fit_transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

# Build and train the classification model
clf_model = LogisticRegression()
clf_model.fit(X_train_clf_scaled, y_train_clf)

# Predict on the test set for classification
y_pred_clf = clf_model.predict(X_test_clf_scaled)

# Evaluate the classification model
accuracy = accuracy_score(y_test_clf, y_pred_clf)
precision = precision_score(y_test_clf, y_pred_clf)
recall = recall_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)

print(f"Classification Model Accuracy: {accuracy}")
print(f"Classification Model Precision: {precision}")
print(f"Classification Model Recall: {recall}")
print(f"Classification Model F1-score: {f1}")
