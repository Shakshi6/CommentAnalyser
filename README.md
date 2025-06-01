# Sentiment Analysis Web Application

This project is a **Machine Learning-based Sentiment Analysis System** designed to classify user comments into categories such as **Positive**, **Negative**, and **Neutral**.

It uses classical ML algorithms:
- **Naive Bayes**
- **Support Vector Machine (SVM)**
- **Logistic Regression**

The system is wrapped in a simple and interactive **Flask web application**.

---

## 🗂️ Project Structure

```
InhouseProject/
├── train_model.py           # Script to train and save the sentiment analysis model
├── app.py                   # Flask web app to predict sentiments from user input
├── sentiment_data.csv       # Dataset used for training
├── requirements.txt         # List of Python dependencies
├── models/                  # Directory to store the trained model
├── static/                  # Contains assets for styling and interaction
│   ├── css/                 # Custom styles for your web interface
│   ├── js/                  # JavaScript files for interactivity
│   └── charts/              # Chart assets for data visualization
└── templates/
    ├── index.html           # Front page for user input
    └── result.html          # Page for displaying prediction results
```

---

## 💻 Setup Instructions

## 1 Clone or Download the Project

```bash
git clone <your-repository-link>
cd InhouseProject
```

---

### 2⃣️ Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows
```

---

### 3⃣️ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️‍ Training the Model

Run the training script to train your model using your labeled dataset:

```bash
python train_model.py
```

This script:
- Reads the `sentiment_data.csv` file.
- Trains using **Naive Bayes, SVM, and Logistic Regression**.
- Saves the best-performing model into the `models/` folder.

---

## 🚀 Running the Web Application

Once the model is trained and saved, launch the web app:

```bash
python app.py
```

Visit:
```
http://127.0.0.1:5000/
```
Enter a comment, click "Submit" — and the app will display the predicted sentiment.

---

## 📂 Dataset Info

- File: `sentiment_data.csv`
- Columns:
    - `comment`: Text input (user-generated or scraped content).
    - `sentiment`: Label for each comment (e.g., Positive, Negative, Neutral).

Make sure the CSV is clean and properly labeled before training.

---

## ⚙️ Technologies Used

- Python
- Flask
- Scikit-learn (`Naive Bayes`, `SVM`, `Logistic Regression`)
- Pandas
- HTML/CSS for frontend styling

---

## 💡 Notes

- You can modify `train_model.py` to switch algorithms or tune hyperparameters.
- `requirements.txt` should be installed before training or running the web app.
- Model performance depends on the quality and size of `sentiment_data.csv`.

---

## 🔍 Example Prediction

Input: 
```
I absolutely love this product! Highly recommended.
```

Predicted Output: 
```
Sentiment: Positive
```

---

## 🖼️ Example Screenshot

![Web App Screenshot](https://via.placeholder.com/800x400.png?text=Sentiment+Analysis+App+Screenshot)

*Replace this link with your actual project screenshot!*

---

## 📄 Sample requirements.txt

```
Flask
pandas
scikit-learn
```

---

## 🏑 Conclusion

This project is a solid hands-on example of how classic **Machine Learning models** can power practical applications like **Sentiment Analysis** when combined with an intuitive web interface.

Through this project, you’ll gain experience across the entire machine learning pipeline — from cleaning and preparing data, to training models using **Naive Bayes**, **SVM**, and **Logistic Regression**, and finally deploying them inside a user-friendly **Flask** application.

Whether you're a beginner seeking to explore the synergy between data science and web development or a developer aiming to understand the building blocks of text classification, this project offers both depth and simplicity. The modular structure makes it easy to experiment with datasets, swap algorithms, and even scale the system further.

In short: it bridges the gap between theory and real-world implementation, encouraging you to dive deeper into Natural Language Processing (NLP) while also sharpening your coding and deployment skills.

