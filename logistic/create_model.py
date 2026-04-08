from datasets import load_dataset
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# Load dataset from https://huggingface.co/datasets/artem9k/ai-text-detection-pile
ds = load_dataset("artem9k/ai-text-detection-pile", split="train")
# Convert to DataFrame
df = pd.DataFrame(ds)

# Some basic text hygiene
df = df[["text", "source"]].dropna()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 100]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["source"],
    test_size=0.2,
    random_state=42,
    stratify=df["source"]
)

# Build pipeline:
# TF-IDF -> Logistic Regression -> Probability calibration
base_clf = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

# Calibrate probabilities using cross-validation
model = CalibratedClassifierCV(
    estimator=base_clf,
    method="sigmoid",
    cv=5
)

# Evaluate model fit
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, pred))
print("ROC AUC:", roc_auc_score(y_test, proba))

# Save the trained model
joblib.dump(model, "ai_text_probability_model.joblib")
