# KokoroBot

Welcome to KokoroBot, an advanced AI-driven application designed to engage users in personalized conversations based on their emotional states. Inspired by the Japanese word "Kokoro," meaning "heart" or "mind", KokoroBot aims to connect with users on a deeper emotional level, providing supportive and empathetic interactions.

KokoroBot features a user-friendly interface with clear message distinction and a seamless chat history. The chatbot captures user input and analyzes emotional states in real-time using webcam frames. Responses are generated to be supportive and empathetic, tailored to the detected emotions such as sadness, fear, anger, or happiness. This allows KokoroBot to create a supportive and engaging user experience, adapting its replies to match the user's emotional needs and maintaining context across interactions.



## Technology Stack

- **Frontend**: Streamlit with custom CSS for styling.
- **Backend**: TensorFlow for emotion detection, OpenCV for webcam frame processing, and Gemini-Pro AI for response generation.
- **Model**: Convolutional Neural Network (CNN) for emotion classification.

## Files and Directories

- `modelbuilding_code.ipynb`: Jupyter notebook for building and training the emotion detection model.
- `app.py`: Main application script for running the chatbot.
- `requirements.txt`: List of dependencies required for the project.
- `.env`: Environment file containing the Gemini API key.
- `model_f.h5`: Model that helps to detect emotion.
-  `kokorobot.pdf`: Contains the detailed report of chatbot

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/emotion-aware-chatbot.git
   cd emotion-aware-chatbot
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the `.env` file**:
   Create a file named `.env` in the project root directory and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   ```

## How to Use

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Interact with the chatbot**:
   - Open your browser and navigate to the local Streamlit server URL provided in the terminal.
   - Allow webcam access when prompted.
   - Enter your message in the text input field and submit.
   - The chatbot will capture a frame from the webcam, detect your emotion, and generate a response tailored to your emotional state.

## Model Building

The `modelbuilding_code.ipynb` notebook contains the code for training the emotion detection model. This involves:

- Preprocessing images using OpenCV.
- Building a CNN using TensorFlow.
- Training and evaluating the model on an emotion-labeled dataset.


## Acknowledgments

- The Streamlit team for their excellent library.
- Google for providing the powerful Gemini-Pro AI model.
- The TensorFlow community for their extensive resources and support.
---

