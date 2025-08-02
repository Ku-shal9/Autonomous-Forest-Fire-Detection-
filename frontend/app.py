import streamlit as st
import numpy as np
import os
from PIL import Image
import pandas as pd

st.title("Forest Fire Detection (Dummy Mode)")

# Load countries from CSV
countries_df = pd.read_csv("data/countries.csv")
countries = dict(zip(countries_df["country"], countries_df[["lat", "lon"]].to_dict('records')))

# Dummy fire probabilities (replace with real API/model later)
dummy_probabilities = {
    "Nepal": 0.3,
    "Brazil": 0.7,
    "Australia": 0.9,
    "Canada": 0.2,
    "India": 0.5
}

# User selects a country
selected_country = st.selectbox("Select a Country", list(countries.keys()))

if selected_country:
    lat = countries[selected_country]["lat"]
    lon = countries[selected_country]["lon"]

    # Zoom map to the selected country
    map_data = st.map(data=[{"lat": lat, "lon": lon}], zoom=4)

    # Fetch or select image based on coordinates (example with local data)
    def get_image_from_data(lat, lon):
        # Placeholder: Generate a dummy image
        image = np.random.rand(224, 224, 3)  # Random image as placeholder
        return (image * 255).astype(np.uint8)

    if st.button("Check for Fire"):
        image = get_image_from_data(lat, lon)
        fire_prob = dummy_probabilities.get(selected_country, 0.0)  # Default to 0 if not found

        st.write(f"Fire Probability in {selected_country}: {fire_prob * 100:.1f}%")
        if fire_prob > 0.5:
            map_data = st.map(data=[{"lat": lat, "lon": lon}], zoom=6)
            st.image(image, caption=f"Image for {selected_country} - Fire Risk: {fire_prob * 100:.1f}%")
        else:
            st.image(image, caption=f"Image for {selected_country} - No Significant Fire Risk")

    # Display map
    st.map(map_data)