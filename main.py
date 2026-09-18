import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load the sentiment dataset
data = pd.read_csv("data/sentiment_data.csv")


# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    tokens = text.split()

    return " ".join(tokens)


# Apply preprocessing
data["clean_text"] = data["text"].apply(preprocess_text)


# Features and labels
X = data["clean_text"]
y = data["sentiment"]


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Convert text into TF-IDF features
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train Logistic Regression classifier
model = LogisticRegression(random_state=42)

model.fit(X_train_tfidf, y_train)


# Make predictions
predictions = model.predict(X_test_tfidf)


# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")


# Display results
print("Total samples:", len(data))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)

print("\nModel training completed successfully!")

print("\nModel Evaluation:")
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("\nActual vs Predicted:")

for actual, predicted in zip(y_test, predictions):
    print("Actual:", actual, "| Predicted:", predicted)
    
    
    # Interactive CLI
print("\n--- Sentiment Analysis CLI ---")

while True:
    user_text = input("\nEnter a sentence (or type 'exit' to quit): ")

    if user_text.lower() == "exit":
        print("Exiting Sentiment Analysis.")
        break

    # Preprocess user input
    cleaned_text = preprocess_text(user_text)

    # Convert text to TF-IDF
    user_tfidf = vectorizer.transform([cleaned_text])

    # Predict sentiment
    prediction = model.predict(user_tfidf)[0]

    print("Predicted Sentiment:", prediction)