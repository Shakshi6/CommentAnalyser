from flask import Flask, request, render_template, jsonify
import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Define the models folder path
MODELS_FOLDER = "models"

# Load the models and vectorizer
models = {
    "SVM": pickle.load(open(os.path.join(MODELS_FOLDER, "svm_model.pkl"), "rb")),
    "Naive Bayes": pickle.load(open(os.path.join(MODELS_FOLDER, "naive_bayes_model.pkl"), "rb")),
    "Logistic Regression": pickle.load(open(os.path.join(MODELS_FOLDER, "logistic_regression_model.pkl"), "rb")),
}
vectorizer = pickle.load(open(os.path.join(MODELS_FOLDER, "vectorizer.pkl"), "rb"))

# Global variable to hold chart data
chart_data = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    global chart_data

    if request.method == 'POST':
        # Get user input
        input_type = request.form.get('input_type')  # 'text', 'csv', or 'link'
        algorithm = request.form.get('algorithm')  # Model choice
        user_input = request.form.get('user_input')  # Text or link input
        file = request.files.get('file')  # CSV file

        # Check selected algorithm
        selected_model = models.get(algorithm)
        if not selected_model:
            return "Invalid model selection."

        # Process input
        try:
            if input_type == 'text':
                input_data = [user_input]
            elif input_type == 'csv' and file:
                # Read CSV file
                data = pd.read_csv(file)

                # Check if 'comment' column exists
                if 'comment' not in data.columns:
                    return "CSV file must contain a 'comment' column."

                # Extract comments
                input_data = data['comment'].dropna().tolist()
            else:
                return "Invalid input type or file missing."

            # Vectorize input
            input_vectorized = vectorizer.transform(input_data)

            # Make predictions
            predictions = selected_model.predict(input_vectorized)

            # Count sentiment distribution
            positive_count = sum(1 for p in predictions if p == 'Positive')
            negative_count = sum(1 for p in predictions if p == 'Negative')
            neutral_count = sum(1 for p in predictions if p == 'Neutral')

            # Prepare chart data
            chart_data = {
                'Positive': positive_count,
                'Negative': negative_count,
                'Neutral': neutral_count
            }

            # Format result
            result = {
                "Algorithm": algorithm,
                "Predictions": predictions.tolist(),
            }
            return render_template('result.html', result=result)

        except Exception as e:
            return f"An error occurred while processing the file: {str(e)}"

@app.route('/get_chart_data', methods=['GET'])
def get_chart_data():
    global chart_data
    return jsonify({"chartData": list(chart_data.values())})

if __name__ == '__main__':
    app.run(debug=True)