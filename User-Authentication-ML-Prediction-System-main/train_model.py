import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("sme_customer_churn.csv")

le_size = LabelEncoder()
le_payment = LabelEncoder()
df["Company_Size"] = le_size.fit_transform(df["Company_Size"])
df["Payment_History"] = le_payment.fit_transform(df["Payment_History"])

feature_cols = ["Company_Size", "Contract_Length", "Monthly_Bill",
                "Payment_History", "Support_Tickets", "Product_Usage"]
X = df[feature_cols]
y = df["Churn"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

with open("decision_tree_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to decision_tree_model.pkl")
print(f"Training accuracy: {model.score(X, y):.4f}")
