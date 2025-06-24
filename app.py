import streamlit as st
from PIL import Image
import torch
import numpy as np
import pandas as pd
from models.facenet_model import load_model
from utils.face_db import load_db, save_to_db
from utils.journal import log_recognition
from facenet_pytorch import MTCNN
from webcam.webcam_capture import capture_image

st.set_page_config(page_title="Reconnaissance Faciale", layout="wide")

model = load_model()
mtcnn = MTCNN(image_size=160, margin=20)
db = load_db()

st.sidebar.title("Bienvenue fait votre 🧱 Actions")
if st.sidebar.button("📷 Capturer depuis Webcam"):
    img = capture_image()
    if img is not None:
        st.session_state["image"] = img
        st.sidebar.image(img, caption="Capture réussie")

name_input = st.sidebar.text_input("🆕 Nom pour ajout visage")

if st.sidebar.button("➕ Enregistrer dans la base"):
    img = st.session_state.get("image", None)
    if img is not None and name_input:
        face = mtcnn(img)
        if face is not None:
            emb = model(face.unsqueeze(0)).detach()
            save_to_db(name_input, emb.squeeze(0))
            st.success(f"✅ Visage de {name_input} enregistré avec succès.")
        else:
            st.warning("Aucun visage détecté.")
    else:
        st.warning("Image ou nom manquant.")

if st.sidebar.button("📄 Voir le journal"):
    try:
        df = pd.read_csv("data/journal.csv")
        st.dataframe(df)
    except FileNotFoundError:
        st.info("Aucun journal pour le moment.")

st.title("🔍 Système de Reconnaissance Faciale Ansou Oualy")
st.balloons()
if "image" in st.session_state:
    st.image(st.session_state["image"], caption="Image en cours de reconnaissance")
    face = mtcnn(st.session_state["image"])
    if face is not None:
        emb = model(face.unsqueeze(0)).detach()
        identity = "Inconnu"
        min_dist = float("inf")

        for name, db_emb in db.items():
            dist = 1 - torch.nn.functional.cosine_similarity(emb, db_emb.unsqueeze(0)).item()
            if dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist < 0.9:
            st.success(f"🎯 Reconnu : {identity}")
            log_recognition(identity)
        else:
            st.error("🙈 Visage inconnu")