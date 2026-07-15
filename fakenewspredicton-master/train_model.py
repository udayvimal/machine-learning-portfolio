import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("news (4).csv")
df = df.dropna(subset=["text", "label"])

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

acc = accuracy_score(y_test, model.predict(X_test_tfidf))
print(f"Test accuracy: {acc:.4f}")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Saved model.pkl and tfidf_vectorizer.pkl")
