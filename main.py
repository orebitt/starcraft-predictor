import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import numpy as np

# Load csv file into a pandas dataframe
df = pd.read_csv('starcraft_player_data.csv')

# Remove all players containing unknown data values, '?', from the dataframe
df = df.replace('?', pd.NA).dropna()

# Split the dataframe into feature variable x, last 18 columns, and target variable y, LeagueIndex
x = df.drop(columns=['GameID', 'LeagueIndex'])
y = df['LeagueIndex']

# Normalize the data
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# Using a 70/15/15 split for train/validation/test data
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=1)

# Create the RandomForest model
model = RandomForestClassifier(random_state=4)

# Perform hyperparameter tuning using GridSearchCV with cross-validation
parameters = {
    'n_estimators': [100, 200, 300],
    'max_depth': [2, 4, 6, None],
}

print('Model training in progress (takes ~30 seconds)')
grid_search = GridSearchCV(model, parameters, cv=5)
grid_search.fit(x_train, y_train)

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Perform cross-validation on the best model for evaluation
cv_scores = cross_val_score(best_model, x_train, y_train, cv=5)
mean_cv_accuracy = np.mean(cv_scores)

# Predict using the best model on the validation set
val_predictions = best_model.predict(x_val)

# Predict using the best model on the test set
test_predictions = best_model.predict(x_test)

# Calculate accuracy on the test set
test_accuracy = accuracy_score(y_test, test_predictions)


# Print feature importances
feature_importances = best_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': x.columns,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print('Feature Importances:')
print(importance_df)

# Print results
print('Mean Cross-Validation Accuracy:', str(round(mean_cv_accuracy * 100, 2)) + '%')


# Calculate the differences between true and predicted values
differences = test_predictions - y_test

# Separate positive and negative differences
positive_diff = differences[differences > 0]
negative_diff = differences[differences < 0]

# Apply Platt Scaling for probability calibration
calibrated_model = CalibratedClassifierCV(best_model, cv='prefit', method='sigmoid')
calibrated_model.fit(x_val, y_val)

# Predict using the calibrated model on the test set
calibrated_predictions = calibrated_model.predict(x_test)

# Calculate accuracy on the calibrated predictions
calibrated_accuracy = accuracy_score(y_test, calibrated_predictions)

# Print results
print('Test Accuracy (Before Calibration):', str(round(test_accuracy * 100, 2)) + '%')
print('Test Accuracy (After Calibration):', str(round(calibrated_accuracy * 100, 2)) + '%')

# Calculate and print the percentage of predictions that are off by more than 1
within_margin = np.abs(test_predictions - y_test) <= 1
within_margin_percentage = within_margin.mean() * 100
print('Percentage of predictions within ±1 (Before Calibration):', str(round(within_margin_percentage, 2)) + '%')

# Calculate and print the percentage of predictions that are within ±1
within_margin_calibrated = np.abs(calibrated_predictions - y_test) <= 1
within_margin_calibrated_percentage = within_margin_calibrated.mean() * 100
print('Percentage of predictions within ±1 (After Calibration):', str(round(within_margin_calibrated_percentage, 2)) + '%')


# Calculate the calibrated differences between true and predicted values
calibrated_differences = calibrated_predictions - y_test

# Separate positive and negative calibrated differences
positive_diff_calibrated = calibrated_differences[calibrated_differences > 0]
negative_diff_calibrated = calibrated_differences[calibrated_differences < 0]

# Create a histogram of positive and negative calibrated differences
plt.hist([positive_diff_calibrated, negative_diff_calibrated], bins=range(int(calibrated_differences.min()), int(calibrated_differences.max()) + 2),
         align='left', rwidth=0.8, edgecolor='black', label=['Positive Calibrated Differences', 'Negative Calibrated Differences'])
plt.xticks(range(int(calibrated_differences.min()), int(calibrated_differences.max()) + 1))
plt.xlabel('Difference (Predicted - True)')
plt.ylabel('Count')
plt.title('Prediction Discrepancies')
plt.legend()
plt.show()
