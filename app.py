import os
import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import tensorflow as tf
import google.generativeai as gen_ai

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Emotion-aware Chatbot",
    page_icon=":brain:",  # Favicon emoji
    layout="centered",  # Page layout option
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up Google Gemini-Pro AI model
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')

# Load Emotion Detection Model
def load_emotion_model(model_path):
    return tf.keras.models.load_model(model_path)

# Preprocess Frame for Emotion Detection
def preprocess_frame(frame, image_size):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

# Predict Emotion from Webcam Frame
def predict_emotion(model, frame, image_size):
    preprocessed_frame = preprocess_frame(frame, image_size)
    predictions = model.predict(preprocessed_frame)
    emotion = np.argmax(predictions)  # Assuming your model returns a probability distribution
    return emotion

# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# Define CSS styles for chat messages
st.markdown("""
<style>
    .message-container {
        max-height: 400px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        margin-bottom: 10px;
    }
    .user-message {
        align-self: flex-end;
        background-color: #DCF8C6;
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 5px;
        max-width: 70%;
        text-align: left;
    }
    .ai-message {
        align-self: flex-start;
        background-color: #E0E0E0;
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 5px;
        max-width: 70%;
        text-align: left;
    }
    .system-message {
        align-self: center;
        background-color: #F5F5F5;
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 5px;
        max-width: 70%;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize conversation history
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        {"role": "system", "content": "Welcome! I'm here to chat with you."}
    ]

# Function to add messages to chat history
def add_message(role, content):
    st.session_state['flowmessages'].append({"role": role, "content": content})

# Display messages
def display_messages():
    st.markdown('<div class="message-container">', unsafe_allow_html=True)
    for message in st.session_state['flowmessages']:
        if message['role'] == 'system':
            st.markdown(f'<div class="system-message">{message["content"]}</div>', unsafe_allow_html=True)
        elif message['role'] == 'user':
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        elif message['role'] == 'bot':
            st.markdown(f'<div class="ai-message">{message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Display previous chat history
display_messages()

# Capture Frame from Webcam
def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame

# Load emotion detection model
emotion_model_path = "model_f.h5"
emotion_model = load_emotion_model(emotion_model_path)

# Emotion dictionary
emotion_dict = {0: 'happiness', 1: 'sadness', 2: 'anger', 3: 'fear'}

# Get Gemini response based on prompt
def get_gemini_response(question, prompt):
    response = model.generate_content([prompt, question])
    return response.text

# Define the base prompt for Gemini-Pro
base_prompt = """
You are an empathetic assistant dedicated to improving the user's mood based on their given detected emotion: happy, sad, fear, or anger 
and input text. Your responses should be friendly and mood-appropriate:
Happy: Celebrate their joy and ask about the reason for their happiness. Provide enthusiastic reinforcement.
Sad: Offer comfort and empathy. Ask if they want to share what's making them sad and offer help.
Fear: Reassure them and acknowledge their fear. Offer practical advice to alleviate their fear.
Anger: Stay calm and listen. Validate their feelings and suggest ways to manage anger.
Your goal is to create a positive and supportive environment, helping users feel understood and valued.
"""

# Get user input
with st.form(key="user_input_form", clear_on_submit=True):
    user_input = st.text_input("You: ", key="input")
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    elif user_input.lower().strip() == "quit":
        st.stop()
    else:
        # Add user message to chat history
        add_message('user', user_input)

        # Capture frame from webcam
        frame = capture_frame()

        # Predict emotion
        emotion = predict_emotion(emotion_model, frame, image_size=(128, 128))
        detected_emotion = emotion_dict.get(emotion, 'neutral')

        # Prepare the prompt
        prompt = f"{base_prompt} The user is currently feeling {detected_emotion} and his input message is {user_input}."
        # Send user's message to Gemini-Pro and get the response
        gemini_response = get_gemini_response(user_input, prompt)

        # Add bot response to chat history
        add_message('bot', gemini_response)

        # Display updated chat history
        display_messages()
