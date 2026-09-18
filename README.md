# Sentiment Analysis Tool

A machine learning based sentiment analysis tool that classifies text as Positive or Negative using TF-IDF and Logistic Regression.

# Features

- Load labeled sentiment data from CSV
- Text cleaning and preprocessing
- Tokenization
- TF-IDF feature extraction
- Logistic Regression classification
- Accuracy and F1 score evaluation
- Interactive command-line sentiment prediction

# Technologies Used

- Python
- Pandas
- Scikit-learn
- Regular Expressions
- TF-IDF
- Logistic Regression

# Project Structure

Syntecxhub_Sentiment_Analysis/
|
├── data/
│   └── sentiment_data.csv
├── main.py
├── README.md
├── requirements.txt
└── .gitignore

# How It Works

1. Load the sentiment dataset.
2. Clean the text by converting it to lowercase and removing special characters.
3. Tokenize the text.
4. Split the dataset into training and testing data.
5. Convert text into numerical features using TF-IDF.
6. Train a Logistic Regression classifier.
7. Evaluate the model using Accuracy and F1 Score.
8. Allow users to enter text through the CLI and receive a sentiment prediction.

# Installation

Install the required Python packages:

pip install -r requirements.txt

# Run the Project

python main.py

# Example

--- Sentiment Analysis CLI ---

Enter a sentence (or type 'exit' to quit): I love this product

Predicted Sentiment: positive

Enter a sentence (or type 'exit' to quit): This product is terrible

Predicted Sentiment: negative

# Model

The project uses:

- TF-IDF for converting text into numerical features.
- Logistic Regression for binary sentiment classification.

# Evaluation

The model is evaluated using:

- Accuracy
- F1 Score

The current dataset contains 40 labeled examples:

- 20 Positive
- 20 Negative

Because this is a small demonstration dataset, the evaluation score should not be treated as representative of real-world sentiment-analysis performance.

# Internship Project

This project was developed as part of the Syntecxhub AI/ML internship.

# Author

Vishnuvardhan Reddy