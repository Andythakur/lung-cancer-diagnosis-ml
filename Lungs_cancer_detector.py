from calendar import error
from unicodedata import category

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load data
df = pd.read_csv("dataset_med.csv")
print(df.head())
print(df. info())
print(df.isnull().sum())
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'])
df = df.dropna()
categorical_cols = ['gender', 'country', 'cancer_stage', 'family_history', 'smoking_status', 'hypertension', 'asthma', 'cirrhosis', 'other_cancer', 'treatment_type', 'survived']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Prepare data for modeling
df =df.drop(['id', 'diagnosis_date', 'end_treatment_date'], axis=1)
X = df.drop('survived', axis=1)
y = df['survived']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
importance = model.feature_importances_
features = df.drop('survived', axis=1).columns
plt.figure(figsize=(10,6))
sns.barplot(x=importance, y=features)
plt.title("feature Importance")
plt.show()