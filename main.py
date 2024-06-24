from fastapi import FastAPI
import requests
from io import BytesIO
from PIL import Image
import json
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sqlalchemy.orm import sessionmaker
import models
from database import engine, SessionLocal
import cv2
import tempfile
import pytube
from transformers import DistilBertTokenizer, DistilBertModel
import torch


app = FastAPI()


base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def extract_features_from_image_url(image_url, model):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img_array = np.array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img).flatten()
    return features


@app.post("/process_image")
async def process_image(image_url: str):
    features = extract_features_from_image_url(image_url, model)

    session = SessionLocal()
    all_image_content = session.query(models.Content).filter(models.Content.content_type == "image").all()

    mse_threshold = 100
    similar_images = []

    for content in all_image_content:
        existing_features = extract_features_from_image_url(content.content_url, model)
        mse = np.mean((features - existing_features) ** 2)
        if mse < mse_threshold:
            similar_images.append(content.content_url)
    new_content = models.Content(content_type="image", content_url=image_url, content_features=json.dumps(features.tolist()))
    session.add(new_content)
    session.commit()
    session.refresh(new_content)
    session.close()
    if similar_images:
        return {"result": "Изображение схоже с " + ', '.join(similar_images)}
    return {"result": "Изображение добавлено в базу данных"}


@app.post("/process_video")
async def process_video(video_url: str):
    video_path = download_video_from_url(video_url)
    features = extract_features_from_video(video_path, model)

    session = SessionLocal()
    all_video_content = session.query(models.Content).filter(models.Content.content_type == "video").all()  # Corrected content_type

    mse_threshold = 100
    similar_videos = []

    for content in all_video_content:
        existing_features = json.loads(content.content_features)  # Corrected loading of features as JSON
        mse = np.mean((features - existing_features) ** 2)
        if mse < mse_threshold:
            similar_videos.append(content.content_url)
    new_content = models.Content(content_type="video", content_url=video_url, content_features=json.dumps(features.tolist()))  # Corrected conversion to JSON
    session.add(new_content)
    session.commit()
    session.refresh(new_content)
    session.close()
    if similar_videos:
        return {"result": "Видео схоже с " + ', '.join(similar_videos)}
    return {"result": "Видео добавлено в базу данных, схожих видео не найдено"}


def download_video_from_url(video_url):
    yt = pytube.YouTube(video_url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    video_path = stream.download(output_path=tempfile.gettempdir())
    return video_path


def extract_features_from_video(video_path, model):
    features = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = preprocess_frame(frame, model)
        frame_features = extract_frame_features(processed_frame, model)
        features.append(frame_features)
    cap.release()
    return np.array(features)


def preprocess_frame(frame, model):
    resized_frame = cv2.resize(frame, (224, 224))
    resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    preprocessed_frame = preprocess_input(resized_frame_rgb)
    return preprocessed_frame


def extract_frame_features(frame, model):
    features = model.predict(np.expand_dims(frame, axis=0))
    return features.flatten()

def extract_features_from_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()


@app.post("/process_text")
async def process_text(text: str):
    features = extract_features_from_text(text, text_model, tokenizer)

    session = SessionLocal()
    all_text_content = session.query(models.Content).filter(models.Content.content_type == "text").all()

    mse_threshold = 100
    similar_texts = []

    for content in all_text_content:
        existing_features = np.array(json.loads(content.content_features))
        mse = np.mean((features - existing_features) ** 2)
        if mse < mse_threshold:
            similar_texts.append(content.content_url)
    new_content = models.Content(content_type="text", content_url=text, content_features=json.dumps(features.tolist()))
    session.add(new_content)
    session.commit()
    session.refresh(new_content)
    session.close()
    if similar_texts:
        return {"result": "Текст схож с " + ', '.join(similar_texts)}
    return {"result": "Текст добавлен в базу данных"}