import joblib
from sklearn.metrics import classification_report, roc_auc_score

# Load model
model = joblib.load("ai_text_probability_model.joblib")

# Recreate test split

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

ds = load_dataset("artem9k/ai-text-detection-pile", split="train")
df = pd.DataFrame(ds)

df = df[["text", "source"]].dropna()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 100]

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["source"],
    test_size=0.2,
    random_state=42,
    stratify=df["source"]
)

# Evaluate
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, pred))
print("ROC AUC:", roc_auc_score(y_test, proba))