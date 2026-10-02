import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA_FILE = "data/sign_landmarks_cleaned.csv"
MODEL_FILE = "models/sign_model.joblib"

if not os.path.exists(DATA_FILE):
    print("Dataset not found. Run collect_data.py first.")
    exit()

os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_FILE)
df = df.dropna()

if "label" not in df.columns:
    print("The dataset is missing the label column.")
    exit()

print("Samples per sign:")
print(df["label"].value_counts())
print()

X = df.drop(columns=["label"])
y = df["label"].astype(str)

if len(y.unique()) < 2:
    print("You need at least 2 different signs to train a model.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")
print()
print("Detailed report:")
print(classification_report(y_test, predictions))

joblib.dump(model, MODEL_FILE)

print()
print(f"Model saved to {MODEL_FILE}")