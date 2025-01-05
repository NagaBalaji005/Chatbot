import os
import pickle
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents from the JSON file
def load_intents(file_path="intents.json"):
    try:
        with open(file_path, 'r') as file:
            intents = json.load(file)
            print(f"Loaded {len(intents['intents'])} intents.")  # Debugging: number of intents loaded
            return intents['intents']  # Ensure we're returning the correct part of the JSON structure
    except Exception as e:
        print(f"Error loading intents: {str(e)}")
        return []

# Train and save the model
def train_and_save_model():
    intents = load_intents()
    if not intents:
        print("Error: No intents loaded, training cannot proceed.")
        return

    # Prepare training data
    patterns = []
    tags = []
    for intent in intents:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])

    # Convert text data into feature vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(patterns)

    # Train the classifier
    classifier = LogisticRegression(max_iter=10000)
    classifier.fit(X, tags)

    # Save the trained model (vectorizer and classifier)
    if not os.path.exists('model'):
        os.makedirs('model')  # Ensure the model directory exists

    with open('model/chatbot_model.pkl', 'wb') as model_file:
        pickle.dump((vectorizer, classifier), model_file)

    print("Model saved successfully!")

# Entry point
if __name__ == "__main__":
    train_and_save_model()
