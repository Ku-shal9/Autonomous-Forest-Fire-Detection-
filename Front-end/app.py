import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import requests
import os

st.title("Forest Fire Detection Based on Map Data")

# Load countries from CSV with error handling
try:
    countries_df = pd.read_csv("countries.csv")
    countries = dict(zip(countries_df["country"], countries_df[["lat", "lon"]].to_dict('records')))
except FileNotFoundError:
    st.error("Error: 'countries.csv' not found. Please add the file with country data.")
    st.stop()

# Initialize session state for fire probability and map data
if 'fire_prob' not in st.session_state:
    st.session_state.fire_prob = 0.0
if 'map_data' not in st.session_state:
    st.session_state.map_data = pd.DataFrame(columns=['lat', 'lon', 'fire_percentage'])

# User selects a country with callback to trigger API call
def on_country_select():
    selected_country = st.session_state.selected_country
    if selected_country:
        lat = countries[selected_country]["lat"]
        lon = countries[selected_country]["lon"]
        image_array = get_map_image(selected_country)  # Use country-specific image
        try:
            response = requests.post("http://localhost:8000/predict", json={"image": image_array.tolist()})
            st.session_state.fire_prob = response.json()["fire_probability"]
            st.session_state.map_data = pd.DataFrame([{"lat": lat, "lon": lon, "fire_percentage": st.session_state.fire_prob * 100}])
        except Exception as e:
            st.session_state.fire_prob = 0.0
            st.session_state.map_data = pd.DataFrame([{"lat": lat, "lon": lon}])
            st.error(f"API Error: {str(e)}")

selected_country = st.selectbox("Select a Country", list(countries.keys()), key="selected_country", on_change=on_country_select)

if selected_country:
    lat = countries[selected_country]["lat"]
    lon = countries[selected_country]["lon"]
    fire_prob = st.session_state.fire_prob
    map_data = st.session_state.map_data

    # Fetch map-based image from dataset based on country
    def get_map_image(country):
        image_dir = "../data/forest_images/"
        st.write(f"Checking image directory: {image_dir}")  # Debug output
        if os.path.exists(image_dir):
            image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
            st.write(f"Found image files: {image_files}")  # Debug output
            if image_files:
                # Try to match country name in filename
                for image_file in image_files:
                    if country.lower() in image_file.lower().replace('.jpg', '').replace('.png', ''):
                        image_path = os.path.join(image_dir, image_file)
                        st.write(f"Using image: {image_file} for {country}")
                        break
                else:
                    image_path = os.path.join(image_dir, image_files[0])  # Default to first image
                    st.write(f"No country-specific image found, using default: {image_files[0]}")
                image = Image.open(image_path).resize((224, 224)).convert("RGB")
                image_array = np.array(image) / 255.0  # Normalize to [0, 1] for API
                return image_array
        st.warning("Map images not found, using random image as fallback.")
        return np.random.rand(224, 224, 3)  # [0, 1] range as fallback

    # Display map with dynamic zoom and fire percentage
    map = st.map(data=map_data, zoom=4 if fire_prob <= 0.5 else 6)

    # Display image and probability
    image_array = get_map_image(selected_country)
    image_pil = Image.fromarray((image_array * 255).astype(np.uint8))  # For display
    st.write(f"Fire Percentage in {selected_country}: {fire_prob * 100:.1f}% (based on available image data)")
    caption = f"Map Image for {selected_country} - Fire Percentage: {fire_prob * 100:.1f}%"
    st.image(image_pil, caption=caption, use_column_width=True)