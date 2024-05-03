import cv2
import numpy as np
import tensorflow as tf
import google.generativeai as genai
from sklearn.cluster import KMeans
from deepface import DeepFace
import speech_recognition as sr
import os
import sys
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import threading
from elevenlabs import Voice, VoiceSettings, play
from elevenlabs.client import ElevenLabs
import random


def weather(text):
    if text:
        api_key = os.getenv('WEATHER_API_KEY')
        city = text
        units = "imperial"
        url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&units={units}&appid={api_key}'

        response = requests.get(url)

        print(f"Response status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            return f'The temperature in {city} is {temp}° farenheit with {desc}.'
        else:
            print(f"Response text: {response.text}")
            return "Error fetching weather data"
    else:
        return "Please provide a city name."

def speak(text):
    client = ElevenLabs(
        api_key="aae657101a1f1cfa4cf53e2d78ca8338", # Defaults to ELEVEN_API_KEY
    )

    audio = client.generate(
        text=text,
        voice=Voice(
            voice_id='YwylTSkcY1bofasFyjxT',
            settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
        )
    )

    play(audio)

width = 416
height = 416

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load your model
detect_fn = tf.saved_model.load('efficientdet_d1_coco17_tpu-32/saved_model')

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

g_model = genai.GenerativeModel('gemini-pro')
chat = g_model.start_chat(history=[])

chat.send_message("You are now a chatbot, your name is Jacob, and your goal is to assist users with their queries.")

def get_dominant_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(image)

    return kmeans.cluster_centers_[0]

def process_image(image):
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] if isinstance(i, list) else layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    class_ids = []
    confidences = []
    boxes = []
    detected_objects = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = get_dominant_color(image[y:y+h, x:x+w])
            emotion = None
            if label == 'person':
                face_image = image[y:y+h, x:x+w]
                try:
                    result = DeepFace.analyze(face_image, actions=['emotion'])
                    emotion = result['dominant_emotion']
                except ValueError:
                    pass
            detected_objects.append((label, color, emotion))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    return image, detected_objects

def capture_image():
    cap = cv2.VideoCapture(0)
    try:
        ret, frame = cap.read()
        if ret:  # Check if frame was successfully captured
            image_np_with_detections, detected_objects = process_image(frame)
            if image_np_with_detections is not None:
                cv2.imwrite('output.jpg', image_np_with_detections)
                descriptions = []
                for label, color, emotion in detected_objects:
                    if emotion:
                        descriptions.append(f"{label} with color {color} showing {emotion} expression")
                    else:
                        descriptions.append(f"{label} with color {color}")
                generate_text(f"What is this? {', '.join(descriptions)}")
        else:
            print("Failed to capture frame")
    finally:
        cap.release()

def generate_text(input_text):
    try:
        if input_text.startswith("What is this?"):
            response = chat.send_message(f"In one short sentence, this is a picture of {input_text[13:]}", safety_settings={'HARASSMENT':'block_none', 'HATE_SPEECH':'block_none'})
        else:
            response = chat.send_message("In a brief sentence, " + input_text, safety_settings={'HARASSMENT':'block_none', 'HATE_SPEECH':'block_none'})

        # Clean the response
        cleaned_response = response.text.replace("[", "").replace("]", "").replace("*", "").replace("#", "").replace("(", "").replace(")", "")
        if cleaned_response.startswith("Jacob"):
            cleaned_response = cleaned_response[6:]

        print(f"Model: {cleaned_response}\n")
        try:
            speak(cleaned_response)
        except Exception as e:
            print(f"An error occurred while speaking the response: {e}")
    except Exception as e:
        print(f"An error occurred while generating text: {e}")
        try:
            speak("I'm sorry, I didn't understand that command.")
        except Exception as e:
            print(f"An error occurred while speaking the error message: {e}")
def interpret_command(command):
    try:
        nc = command.lower()
        if nc == "take a picture":
            return "take_picture", None
        elif nc == "exit":
            return "exit", None
        elif "weather" in nc or "temperature" in nc:
            city = command.split("in")[-1].strip() if "in" in nc else command.split("temperature")[-1].strip()
            return "weather", city
        elif "play" in nc or "spotify" in nc:
            return "spotify", None
        else:
            return "chat", None
    except Exception as e:
        print(f"An error occurred while interpreting the command: {e}")
        return "chat", None



print("Initialized keys.")

while True:
    command = input("You: ")
    action, city = interpret_command(command)
    if action == "take_picture":
        capture_image()
    elif action == "weather":
        if city:
            response = weather(city)
            print("Model: " + response)
            speak(response)
        else:
            print("Please provide a city name.")
    elif action == "chat":
        generate_text(command)
    elif command == "exit":
        speak("Goodbye")
        print("Model: Goodbye friend!")
        break

    ## ALL COMMANDS:
    # take a picture (uses your camera to detect objects)
    # whats the weather in (replace with your city or state)
    # exit (breaks the loop)