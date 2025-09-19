import cv2 
import numpy as np 
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
from transformers import pipeline

# Load pre-trained emotions model (Hugging Face model preferred)
emotion_model = pipeline("image-classification", model="ElenaRyumina/face_emotion_recognition")
emotion_labels = ['Angry', 'Disguist', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initiate Langchain with HuggingFace LLM
llm = HuggingFacePipeline.from_model_id(model_id="gpt2", task="text-generation", pipeline_kwargs={"max_length": 100})

#Prompt Template 
prompt = PromptTemplate(
    input_variables=["emotion", "user_input"],
    template="You are an empathetic assistance. The user is feeling {emotion}. Responds to: {user_input}"
)
chain = LLMChain(llm=llm, prompt=prompt)

#Face Detection 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Streamlit UI 
st.title("EmotionSync Agent")
video_feed = st.empty()
chat_feed = st.empty()
user_input = st.text_input("Say something:")

#Video Capture 
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))
        prediction = emotion_model(face_resized)
        emotion = prediction[0]['Label'] #Top prediction
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    video_feed.image(frame, channels="BGR")

    if user_input:
        response = chain.run(emotion=emotion, user_input=user_input)
        chat_feed.text(f"Bot: {response}")

    if st.button("Stop"):
        break

cap.release()   
cv2.destroyAllWindows()