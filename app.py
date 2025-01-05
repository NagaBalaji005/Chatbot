import os
import pickle
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Initialize NLTK downloads with error handling
@st.cache_resource
def initialize_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        with st.spinner("Downloading required NLTK data..."):
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)

# Remove global initialization of NLTK
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load intents from the JSON file
def load_intents(file_path="intents.json"):
    try:
        if not os.path.exists(file_path):
            st.error(f"Intents file not found at {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            intents = data.get('intents', [])
            if not intents:
                st.warning("No intents found in the JSON file")
            return intents
    except json.JSONDecodeError:
        st.error("Invalid JSON file format")
        return []
    except Exception as e:
        st.error(f"Error loading intents: {str(e)}")
        return []

# Preprocess the input text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    try:
        # Tokenize and remove stopwords
        tokens = nltk.word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        
        # Lemmatize the tokens
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmatized_tokens)
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
        return text.lower()

# ChatBot class definition
class ChatBot:
    def __init__(self):
        self.intents = None
        self.vectorizer = None
        self.classifier = None
        self.initialize_bot()

    def initialize_bot(self):
        """Initialize the chatbot with intents and model"""
        self.intents = load_intents()
        if not self.intents:
            st.error("Failed to initialize chatbot: No intents loaded")
            return
        
        self.load_model()
        if not self.model_loaded():
            st.warning("No pre-trained model found. Some features may be limited.")

    def model_loaded(self):
        """Check if the model is properly loaded"""
        return self.vectorizer is not None and self.classifier is not None

    def load_model(self):
        """Load the pre-trained model"""
        try:
            model_path = 'model/chatbot_model.pkl'
            if not os.path.exists('model'):
                os.makedirs('model')
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as model_file:
                    self.vectorizer, self.classifier = pickle.load(model_file)
        except Exception as e:
            st.warning(f"Could not load pre-trained model: {str(e)}")
            # Initialize new model components if loading fails
            self.vectorizer = TfidfVectorizer(max_features=1000)
            self.classifier = LogisticRegression(random_state=42, max_iter=1000)

    def get_response(self, user_input):
        """Generate response for user input"""
        if not user_input or not isinstance(user_input, str):
            return "Please provide a valid input.", "unknown"

        if not self.intents:
            return "I'm not properly initialized yet.", "error"

        try:
            # Preprocess user input
            preprocessed_input = preprocess_text(user_input)
            if not preprocessed_input:
                return "I didn't understand that. Could you rephrase?", "unknown"

            # Try direct pattern matching first
            for intent in self.intents:
                patterns = [preprocess_text(p) for p in intent.get('patterns', [])]
                if preprocessed_input in patterns:
                    return random.choice(intent['responses']), intent['tag']

            # Fall back to classifier if model is loaded
            if self.model_loaded():
                input_vector = self.vectorizer.transform([preprocessed_input])
                predicted_tag = self.classifier.predict(input_vector)[0]
                
                for intent in self.intents:
                    if intent['tag'] == predicted_tag:
                        return random.choice(intent['responses']), intent['tag']

            return "I'm not sure how to respond to that.", "unknown"

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I encountered an error while processing your request.", "error"

def create_streamlit_ui():
    st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
    
    # Initialize NLTK components
    initialize_nltk()
    
    # Initialize session state

    # Initialize session state
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing chatbot..."):
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
        with st.spinner("Thinking..."):
            response, intent = st.session_state.chatbot.get_response(user_message)

        # Add chatbot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        # Handle goodbye intent
        if intent == "goodbye":
            st.write("Thank you for chatting! Feel free to restart the conversation anytime.")

if __name__ == "__main__":
    create_streamlit_ui()