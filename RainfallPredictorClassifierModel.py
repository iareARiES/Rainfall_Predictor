import numpy as np
import pandas as pd
import pickle
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
data = pd.read_csv("Rainfall.csv")
data.columns = data.columns.str.strip()
data = data.drop(columns=["day"])  # Removing the 'day' column as it's not useful

# Handling missing values
data["winddirection"] = data["winddirection"].fillna(data["winddirection"].mode()[0])
data["windspeed"] = data["windspeed"].fillna(data["windspeed"].median())

# Encoding categorical variable
data["rainfall"] = LabelEncoder().fit_transform(data["rainfall"])

# Removing highly correlated columns
data = data.drop(columns=['maxtemp', 'temparature', 'mintemp'])

# Handling class imbalance using downsampling
df_minority = data[data["rainfall"] == 0]
df_majority = data[data["rainfall"] == 1]
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=0)
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
df_downsampled = df_downsampled.sample(frac=1, random_state=2).reset_index(drop=True)

# Splitting features and target variable
X = df_downsampled.drop(columns=["rainfall"])
y = df_downsampled["rainfall"]

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the RandomForest Classifier
rf_model = RandomForestClassifier(random_state=0)
param_grid_rf = {
    "n_estimators": [50, 100, 150, 200],
    "max_features": ["sqrt", "log2", None],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search_rf = GridSearchCV(
    estimator=rf_model, param_grid=param_grid_rf,
    n_jobs=-1, cv=5, verbose=2, return_train_score=True
)
grid_search_rf.fit(X_train, y_train)

# Getting the best model
best_rf_model = grid_search_rf.best_estimator_

# Model Evaluation
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5)
y_pred = best_rf_model.predict(X_test)
print("Test set Accuracy:", accuracy_score(y_test, y_pred))
print("Test set Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Saving the trained model
model_data = {"model": best_rf_model, "feature_names": X.columns.tolist()}
with open("rainfall_prediction_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

# Loading the saved model for prediction
with open("rainfall_prediction_model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
feature_names = model_data["feature_names"]

# Making a prediction
input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
input_df = pd.DataFrame([input_data], columns=feature_names)
prediction = model.predict(input_df)
print("Prediction result:", "Rainfall" if prediction[0] == 1 else "No Rainfall")
