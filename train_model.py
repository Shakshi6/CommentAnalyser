import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Create a folder for models if it doesn't exist
models_folder = "models"
os.makedirs(models_folder, exist_ok=True)

# Load the dataset
data = pd.read_csv("sentiment_data.csv")

X = data['comment']
y = data['sentiment']

# Using TfidfVectorizer instead of CountVectorizer
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train models
models = {
    "SVM": SVC(kernel='linear', probability=True),
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

model_info = {}

for name, model in models.items():
    # Fit the model
    model.fit(X_train, y_train)
    # Save model and vectorizer
    model_path = os.path.join(models_folder, f"{name.lower().replace(' ', '_')}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    # Store accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    model_info[name] = accuracy

# Save the vectorizer
vectorizer_path = os.path.join(models_folder, "vectorizer.pkl")
with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

# Display accuracies
for model, acc in model_info.items():
    print(f"{model} Model Accuracy: {acc:.2f}")

print(f"Models and vectorizer saved in the '{models_folder}' folder.")
