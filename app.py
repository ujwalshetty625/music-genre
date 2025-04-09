import streamlit as st
import numpy as np
import joblib

# Load your trained model
model = joblib.load('random_forest.pkl')

# Genre labels as per your dataset
genre_labels = ['Movie', 'R&B', 'A Capella', 'Alternative', 'Country']

# Streamlit App Title
st.title("🎵 Music Genre Prediction App")

st.markdown("Enter the audio features below to predict the music genre.")

# Input fields for each feature
popularity = st.slider("Popularity", 0.0, 100.0, 50.0)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
duration_ms = st.number_input("Duration (ms)", min_value=0, max_value=1000000, value=200000)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5)
key = st.slider("Key", 0, 11, 5)
liveness = st.slider("Liveness", 0.0, 1.0, 0.5)
loudness = st.number_input("Loudness (dB)", min_value=-60.0, max_value=0.0, value=-10.0)
mode = st.selectbox("Mode (0 = Minor, 1 = Major)", [0, 1])
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5)
tempo = st.number_input("Tempo (BPM)", min_value=0.0, max_value=300.0, value=120.0)
valence = st.slider("Valence", 0.0, 1.0, 0.5)

# Make prediction
if st.button("Predict Genre"):
    # Collect all input features into array
    input_features = np.array([[popularity, acousticness, danceability, duration_ms, energy,
                                instrumentalness, key, liveness, loudness, mode,
                                speechiness, tempo, valence]])
    
    # Predict using the model
    prediction = model.predict(input_features)
    predicted_genre = genre_labels[prediction[0]]  # Map index to genre

    st.success(f"🎧 Predicted Genre: **{predicted_genre}**")
