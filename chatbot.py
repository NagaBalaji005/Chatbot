import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import json
import warnings

warnings.filterwarnings('ignore')

# Configure SSL and NLTK
def configure_nltk():
    try:
        nltk_data_path = os.path.abspath("nltk_data")
        os.makedirs(nltk_data_path, exist_ok=True)

        ssl._create_default_https_context = ssl._create_unverified_context
        nltk.data.path.append(nltk_data_path)
        nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"Warning: Failed to configure NLTK. {str(e)}")

configure_nltk()

# Function to load intents from a JSON file
def load_intents(file_path="intents.json"):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading intents: {str(e)}")
        return []

# ChatBot class definition
class ChatBot:
    def __init__(self):
        self.intents = load_intents()
        if not self.intents:
            raise ValueError("Error: Intents could not be loaded from JSON.")
        self.vectorizer = TfidfVectorizer()
        self.classifier = LogisticRegression(random_state=0, max_iter=10000)
        self.train_model()

    def train_model(self):
        # Prepare data for training
        patterns, tags = [], []
        for intent in self.intents:
            for pattern in intent['patterns']:
                patterns.append(pattern)
                tags.append(intent['tag'])

        # Train vectorizer and classifier
        X = self.vectorizer.fit_transform(patterns)
        y = tags
        self.classifier.fit(X, y)

    def get_response(self, user_input):
        if not user_input:
            return "Please type something for me to respond.", "unknown"

        try:
            # Predict intent
            input_vector = self.vectorizer.transform([user_input])
            predicted_tag = self.classifier.predict(input_vector)[0]

            # Retrieve a response for the intent
            for intent in self.intents:
                if intent['tag'] == predicted_tag:
                    return random.choice(intent['responses']), predicted_tag

            return "I'm not sure how to respond to that.", "unknown"
        except Exception as e:
            return f"An error occurred: {str(e)}", "error"

# Streamlit UI creation
def create_streamlit_ui():
    st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ChatBot()
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display title and instructions
    st.title("💬 AI Chatbot")
    st.write("Welcome! I'm your friendly AI chatbot. Ask me anything!")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input field for user messages
    if user_message := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        # Get chatbot response
        response, intent = st.session_state.chatbot.get_response(user_message)

        # Add chatbot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        # Handle goodbye intent
        if intent == "goodbye":
            st.write("Thank you for chatting! Feel free to restart the conversation anytime.")

# Entry point
if __name__ == "__main__":
    create_streamlit_ui()
