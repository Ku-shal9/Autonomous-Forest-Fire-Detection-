import streamlit as st
import requests
import numpy as np
import os
import pandas as pd
from PIL import Image

st.title("Forest Fire Detection")

# Load countries from CSV
countries_df = pd.read_csv("data/countries.csv")
countries = dict(zip(countries_df["country"], countries_df[["lat", "lon"]].to_dict('records')))

# User selects a country
selected_country = st.selectbox("Select a Country", list(countries.keys()))

if selected_country:
    lat = countries[selected_country]["lat"]
    lon = countries[selected_country]["lon"]

    # Zoom map to the selected country
    map_data = st.map(data=[{"lat": lat, "lon": lon}], zoom=4)

    # Fetch or select image based on coordinates (example with local data)
    def get_image_from_data(lat, lon):
        # Placeholder: Match coordinates to a preprocessed image in data/
        # Replace with actual logic to load from data/ or API
        image_path = f"data/image_{int(lat*10)}_{int(lon*10)}.npy"  # Example naming
        if os.path.exists(image_path):
            image = np.load(image_path)
        else:
            image = np.random.rand(224, 224, 3)  # Fallback to random if not found
        return (image * 255).astype(np.uint8)

    if st.button("Check for Fire"):
        image = get_image_from_data(lat, lon)
        image_data = image.tolist()

        # Send to API
        response = requests.post("http://localhost:8000/predict", json={
            "image": image_data,
            "lat": lat,
            "lon": lon
        })
        if response.status_code == 200:
            result = response.json()
            fire_prob = result["fire_probability"]
            st.write(f"Fire Probability in {selected_country}: {fire_prob * 100:.1f}%")
            if fire_prob > 0.5:
                map_data = st.map(data=[{"lat": lat, "lon": lon}], zoom=6)
                st.image(image, caption=f"Image for {selected_country} - Fire Risk: {fire_prob * 100:.1f}%")
        else:
            st.error("Error fetching prediction")

    # Display map
    st.map(map_data)