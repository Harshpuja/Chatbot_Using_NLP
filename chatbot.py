import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL issues for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from JSON file
file_path = os.path.abspath("intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Initialize Vectorizer & Classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess Data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train Model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot Response Function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0  # Message counter for unique keys

# Streamlit Chatbot App
def main():
    global counter
    st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

    # Apply Custom Styling with Background
    st.markdown(
        """
        <style>
        /* Background */
        .stApp {
            background-color: #F8F3D9;
            background-size: cover;
            background-position: center;
        }
        /* Sidebar */
        .css-1d391kg {
            background-color: #B9B28A !important;
        }
        /* Chat Bubbles */
        .chat-bubble {
            padding: 15px;
            margin: 10px 0;
            border-radius: 15px;
            max-width: 75%;
            font-size: 16px;
            font-weight: bold;
        }
        .user {
            background-color: #B9B28A;
            color: white;
            align-self: flex-end;
        }
        .bot {
            background-color: #504B38;
            color: white;
            align-self: flex-start;
        }
        /* Input Box */
        .stTextInput>div>div>input {
            border-radius: 15px;
            border: 2px solid #504B38;
            padding: 12px;
        }
        /* Buttons */
        .stButton>button {
            background-color: #504B38;
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Page Title
    st.title("🤖 AI Chatbot")
    st.markdown("A chatbot using **NLP & Logistic Regression**.")

    # Sidebar Menu
    menu = ["💬 Chat", "📜 Conversation History", "ℹ️ About"]
    choice = st.sidebar.radio("Navigation", menu)

    # Chat Interface
    if choice == "💬 Chat":
        st.markdown("### **Chat with the Bot:**")
        st.write("Type a message below and press **Enter** to chat.")

        # Create chat log file if it doesn't exist
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            response = chatbot(user_input)

            st.markdown(f'<div class="chat-bubble user">You: {user_input}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble bot">🤖 Bot: {response}</div>', unsafe_allow_html=True)

            # Save chat history
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            # End chat if user says "bye"
            if response.lower() in ['goodbye', 'bye']:
                st.success("Chat ended. Have a great day! 😊")
                st.stop()

    # Conversation History
    elif choice == "📜 Conversation History":
        st.subheader("📜 **Chat History**")

        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    st.markdown(f'<div class="chat-bubble user">You: {row[0]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chat-bubble bot">🤖 Bot: {row[1]}</div>', unsafe_allow_html=True)
                    st.text(f"🕒 {row[2]}")
                    st.markdown("---")
        else:
            st.warning("No conversation history found.")

    # About Section
    elif choice == "ℹ️ About":
        st.subheader("ℹ️ **About This Chatbot**")
        st.write("""
        - 🤖 Built with **Natural Language Processing (NLP) & Logistic Regression**.
        - 🎨 Designed with **Streamlit** for an interactive UI.
        - 📜 Supports **conversation history storage** using CSV files.
        - 💡 Future improvements: **Deep learning models, more intent categories, and better UI**.
        """)

# Run Streamlit App
if __name__ == '__main__':
    main()
