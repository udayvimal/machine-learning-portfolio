from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)

# Debugging: Print the current working directory
print("Current working directory:", os.getcwd())

# Path to the model and vectorizer
model_path = 'model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

# Check if model.pkl exists
if not os.path.exists(model_path):
    print(f"{model_path} does not exist in the current directory.")
    raise FileNotFoundError(f"{model_path} not found.")
else:
    loaded_model = pickle.load(open(model_path, 'rb'))

# Check if tfidf_vectorizer.pkl exists
if not os.path.exists(vectorizer_path):
    print(f"{vectorizer_path} does not exist in the current directory.")
    raise FileNotFoundError(f"{vectorizer_path} not found.")
else:
    tfvect = pickle.load(open(vectorizer_path, 'rb'))

# Load and prepare the data
dataframe = pd.read_csv('news.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)
