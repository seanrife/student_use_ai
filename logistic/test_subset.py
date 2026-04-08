from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


ds = load_dataset("artem9k/ai-text-detection-pile")
small = ds['train'].shuffle(seed=42).select(range(20000))
df = pd.DataFrame(small)

human = df[df['source'] == 'human'].sample(5000, random_state=69)
ai = df[df['source'] == 'ai'].sample(5000, random_state=69)

df_balanced = pd.concat([human, ai])

df_balanced['text'] = df_balanced['text'].str.strip()
df_balanced = df_balanced[df_balanced['text'].str.len() > 100]

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_balanced['text'])

X_train, X_test, y_train, y_test = train_test_split(
    X, df_balanced['source'], test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))