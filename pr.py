#!/usr/bin/env python
# coding: utf-8

# In[15]:


('pip install joblib')


# In[8]:


# Import joblib
import joblib

# Save the model
model = "spotify_song_popularity_model.pkl"
joblib.dump(model, model)

# Print confirmation message
print(f"Model saved successfully to: {model}")


# In[9]:


import pickle


# In[11]:


import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("spotify_song_popularity_model.pkl")

# Define features
features = ["danceability", "energy", "key", "loudness", "mode",
            "speechiness", "acousticness", "instrumentalness", "liveness",
            "valence", "tempo"]

def predict_popularity(data):
    # Preprocess data
    data = pd.DataFrame(data, index=[0])

    # Predict popularity
    prediction = model.predict(data[features])[0]

    # Return prediction
    return prediction

# App layout
st.title("Song Popularity Prediction")

# User input fields
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
key = st.selectbox("Key", range(12))
loudness = st.slider("Loudness", -60.0, 0.0, -30.0)
mode = st.radio("Mode", ("Major", "Minor"))

speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5)
liveness = st.slider("Liveness", 0.0, 1.0, 0.5)

valence = st.slider("Valence", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo", 0.0, 250.0, 120.0)

# Model prediction button
if st.button("Predict Song Popularity"):
    user_input = {
        "danceability": danceability,
        "energy": energy,
        "key": key,
        "loudness": loudness,
        "mode": mode,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
    }

    predicted_popularity = predict_popularity(user_input)

    st.success(f"Predicted Popularity: {predicted_popularity:.4f}")


# In[ ]:





# In[ ]:




