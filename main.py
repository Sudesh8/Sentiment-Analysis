print("hello")
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


r"D:\CODES\Projects\Github-Projects\sentiment\trained_model"
# Load model and tokenizer
MODEL_PATH = r"D:\CODES\Projects\Github-Projects\sentiment\trained_model"  # Path where the model is saved
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# Emotion label mapping
label_map = {
    0: "Anxiety",
    1: "Normal",
    2: "Depression",
    3: "Suicidal",
    4: "Stress",
    5: "Bipolar",
    6: "Personality Disorder",
}


# Reverse label mapping
reverse_label_map = {v: k for k, v in label_map.items()}


# Emotion label to response mapping 
emotion_responses = {
    0: "I'm here for you. Want to talk about it?",  # Anxiety
    1: "Take a deep breath. What's on your mind?",  # Normal
    2: "I'm sorry to hear you're feeling this way. It's okay to seek help. I'm here to listen.",  # Depression
    3: "I'm really concerned. Please talk to someone close to you or seek professional help.",  # Suicidal
    4: "It sounds like you're under a lot of stress. Take a moment to breathe. I'm here for you.",  # Stress
    5: "It can be overwhelming dealing with bipolar mood swings. You're not alone in this.",  # Bipolar
    6: "Personality disorders can be really challenging. Consider talking to a mental health professional.",  # Personality disorder
}


# Function to predict emotion
def predict_emotion(text):
    encodings = tokenizer(
        [text], truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    logits = model(**encodings).logits
    predicted_label = torch.argmax(logits, axis=1).item()
    return predicted_label


# Streamlit app
st.title("Emotional Chatbot")
st.write("Chat with me, and I'll respond to your emotions.")

user_input = st.text_input("Type ", "")
if st.button("Send"):
    if user_input.strip():
        emotion_label = predict_emotion(user_input)
        emotion_name = label_map[emotion_label]  # Map the label to its name
        response = emotion_responses.get(emotion_label, "How can I assist you?")
        st.write(f"**Detected Emotion:** {emotion_name}")
        st.write(f"**Bot:** {response}")
    else:
        st.write("**Bot:** Please say something!")
