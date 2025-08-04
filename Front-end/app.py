import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import requests
from streamlit_folium import st_folium
import folium
import io
import datetime
from functools import lru_cache
import os
import time

# -----------------------------
# Title
# -----------------------------
st.title("🌍 Forest Fire Detection on Interactive Map (NASA Imagery)")

# -----------------------------
# Session state defaults
# -----------------------------
defaults = {
    "lat": None,
    "lon": None,
    "fire_prob": None,
    "map_data": pd.DataFrame(columns=["lat", "lon", "fire_percentage"]),
    "display_image": None,
    "meta": {}
}
for key, val in defaults.items():
    st.session_state.setdefault(key, val)

# -----------------------------
# NASA API key retrieval (env -> fallback)
# -----------------------------
def get_nasa_api_key():
    env_key = os.getenv("NASA_API_KEY")
    if env_key:
        return env_key
    return "W508K2JBak6aCpMWCTMUnYXeoi7XS5hRX4eyvZtO"  # fallback

NASA_API_KEY = get_nasa_api_key()

# Informational (non-fatal)
st.info("Using NASA API key from environment or fallback. To use secrets.toml later, create ~/.streamlit/secrets.toml with `[nasa] api_key = \"...\"`.")

# -----------------------------
# Robust NASA image fetcher with adaptive dim and retries
# -----------------------------
@lru_cache(maxsize=512)
def _raw_nasa_request(lat: float, lon: float, date_str: str, dim: float):
    url = (
        f"https://api.nasa.gov/planetary/earth/imagery"
        f"?lon={lon}&lat={lat}&date={date_str}"
        f"&dim={dim}&api_key={NASA_API_KEY}"
    )
    return requests.get(url, timeout=15)

def get_map_image(lat, lon):
    """
    Try several dims and dates with retries; returns (array, PIL image, meta).
    """
    dates_to_try = [
        datetime.date.today(),
        datetime.date.today() - datetime.timedelta(days=1),
        datetime.date.today() - datetime.timedelta(days=3),
    ]
    dims_to_try = [0.1, 0.07, 0.04]  # fallback to smaller area

    for date in dates_to_try:
        date_str = date.strftime("%Y-%m-%d")
        for dim in dims_to_try:
            attempt = 0
            while attempt < 2:  # two retries per (date, dim)
                try:
                    response = _raw_nasa_request(lat, lon, date_str, dim)
                    if response.status_code == 200:
                        img = Image.open(io.BytesIO(response.content)).convert("RGB")
                        img_resized = img.resize((224, 224))
                        img_array = np.array(img_resized, dtype=np.float32) / 255.0
                        meta = {"date": date_str, "dim": dim, "status": "success"}
                        return img_array, img_resized, meta
                    elif response.status_code == 429:
                        meta = {"date": date_str, "dim": dim, "status": "rate_limited"}
                        st.warning("NASA API rate limited (429).")
                        return None, None, meta
                    else:
                        st.warning(f"NASA returned {response.status_code} for date {date_str}, dim {dim}.")
                        break  # try next dim
                except requests.exceptions.RequestException as e:
                    backoff = 2 ** attempt
                    st.warning(f"Attempt {attempt+1} for {date_str} dim {dim} failed: {e}. Retrying in {backoff}s.")
                    time.sleep(backoff)
                    attempt += 1

    # fallback gray image
    st.error("Failed to fetch NASA image after retries; showing placeholder.")
    fallback = np.full((224, 224, 3), 0.5, dtype=np.float32)
    meta = {"status": "fallback"}
    return fallback, Image.fromarray((fallback * 255).astype(np.uint8)), meta

# -----------------------------
# Prediction API call
# -----------------------------
def analyze_coordinates(lat, lon):
    image_array, display_image, meta = get_map_image(lat, lon)
    if image_array is None:
        return None, display_image, meta  # early exit on rate limit or failure

    try:
        resp = requests.post(
            "http://localhost:8000/predict",
            json={"image": image_array.tolist()},
            timeout=10
        )
        if resp.status_code == 200:
            fire_prob = resp.json().get("fire_probability", 0.0)
            return fire_prob, display_image, meta
        else:
            st.error(f"Prediction API error: {resp.status_code} - {resp.text[:200]}")
            return None, display_image, meta
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach prediction API: {e}")
        return None, display_image, meta

# -----------------------------
# Interactive folium map
# -----------------------------
base_map = folium.Map(location=[0, 0], zoom_start=2, tiles="OpenStreetMap")
folium.LatLngPopup().add_to(base_map)
map_click_data = st_folium(base_map, width=700, height=500)

# -----------------------------
# Handle map click
# -----------------------------
if map_click_data and map_click_data.get("last_clicked"):
    lat = map_click_data["last_clicked"]["lat"]
    lon = map_click_data["last_clicked"]["lng"]
    st.session_state.lat = lat
    st.session_state.lon = lon

    fire_prob, display_image, meta = analyze_coordinates(lat, lon)
    st.session_state.fire_prob = fire_prob
    st.session_state.display_image = display_image
    st.session_state.meta = meta

    if fire_prob is not None:
        st.session_state.map_data = pd.DataFrame([{
            "lat": lat,
            "lon": lon,
            "fire_percentage": fire_prob * 100
        }])

# -----------------------------
# Retry button (re-fetch / re-predict)
# -----------------------------
if st.session_state.lat is not None and st.session_state.lon is not None:
    if st.button("Retry Fetch & Predict"):
        fire_prob, display_image, meta = analyze_coordinates(st.session_state.lat, st.session_state.lon)
        st.session_state.fire_prob = fire_prob
        st.session_state.display_image = display_image
        st.session_state.meta = meta
        if fire_prob is not None:
            st.session_state.map_data = pd.DataFrame([{
                "lat": st.session_state.lat,
                "lon": st.session_state.lon,
                "fire_percentage": fire_prob * 100
            }])

# -----------------------------
# Display results
# -----------------------------
if st.session_state.lat is not None and st.session_state.lon is not None:
    st.markdown(f"**Selected Coordinates:** {st.session_state.lat:.6f}, {st.session_state.lon:.6f}")
    if st.session_state.fire_prob is not None:
        st.markdown(f"🔥 **Fire Probability:** {st.session_state.fire_prob * 100:.1f}%")
        if st.session_state.display_image is not None:
            st.image(st.session_state.display_image, caption="Satellite image (NASA Earth Imagery)", use_column_width=True)

        # Show meta debug info
        st.caption(f"Image meta: {st.session_state.meta}")

        # Focused map with marker
        result_map = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=6)
        folium.Marker(
            [st.session_state.lat, st.session_state.lon],
            popup=f"Fire: {st.session_state.fire_prob * 100:.1f}%",
            icon=folium.Icon(color="red" if st.session_state.fire_prob > 0.5 else "green")
        ).add_to(result_map)
        st_folium(result_map, width=700, height=450)
    else:
        st.info("Awaiting a valid prediction; try clicking again or hitting Retry.")
