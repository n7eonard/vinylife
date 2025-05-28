import streamlit as st
import pandas as pd
import openai
import re
import json
from PIL import Image
import io
import base64

# === CONFIGURATION ===
openai.api_key = st.secrets["OPENAI_API_KEY"]  # set this in .streamlit/secrets.toml

# === HELPER FUNCTIONS ===
def extract_vinyl_info_from_image(image_bytes):
    prompt = """
    You are a vinyl record expert. Analyze the following photo of a vinyl cover and extract the most likely artist and album/EP title.
    You must respond with a valid JSON object in this exact format, with no additional text or explanation:
    {
        "primary_guess": {
            "artist": "Artist Name",
            "album": "Album Name",
            "confidence_score": 95
        },
        "alternative_guesses": [
            {
                "artist": "Alternative Artist 1",
                "album": "Alternative Album 1",
                "confidence_score": 85
            },
            {
                "artist": "Alternative Artist 2",
                "album": "Alternative Album 2",
                "confidence_score": 75
            }
        ]
    }
    The response must be a single valid JSON object. Do not include any other text or explanation.
    """
    # Convert image bytes to base64 string
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=500,
        response_format={ "type": "json_object" }
    )
    return response.choices[0].message.content.strip()

def generate_story(artist, album):
    prompt = f"""
    You are a passionate music historian. Write a short and captivating 150-word story about the artist {artist} and their album or EP titled '{album}'.
    Include cultural context, musical style, and legacy.
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return response.choices[0].message.content.strip()

def recommend_similar(artist, album):
    prompt = f"""
    Suggest 3 songs similar to any track from the album '{album}' by {artist}. List them in JSON format with fields: title, artist, and why it's similar.
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def estimate_price(artist, album):
    # Placeholder logic
    return "€15–€30 (mock range based on Discogs/eBay EU)"

# === STREAMLIT UI ===
st.title("🎵 Vinyl AI Storyteller")
st.markdown("Upload a photo of a vinyl cover to identify it and generate a story, recommendations, and price.")

uploaded_image = st.file_uploader("Upload a photo of a vinyl record", type=["jpg", "jpeg", "png"])

# Session state for step management
if 'step' not in st.session_state:
    st.session_state.step = 'guess'
if 'selected_artist' not in st.session_state:
    st.session_state.selected_artist = None
if 'selected_album' not in st.session_state:
    st.session_state.selected_album = None
if 'alternatives' not in st.session_state:
    st.session_state.alternatives = []
if 'primary_guess' not in st.session_state:
    st.session_state.primary_guess = None

if uploaded_image:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(uploaded_image, caption="Uploaded Vinyl Cover", use_container_width=True)
    with col2:
        if st.session_state.step == 'guess':
            with st.spinner("Analyzing image with AI..."):
                try:
                    vinyl_info = extract_vinyl_info_from_image(uploaded_image.read())
                    with st.expander("Debug: Raw AI Response"):
                        st.code(vinyl_info)
                    info_dict = json.loads(vinyl_info)
                    if not all(key in info_dict for key in ["primary_guess", "alternative_guesses"]):
                        raise ValueError("Missing required fields in response")
                    primary_guess = info_dict["primary_guess"]
                    if not all(key in primary_guess for key in ["artist", "album", "confidence_score"]):
                        raise ValueError("Missing required fields in primary_guess")
                    st.session_state.primary_guess = primary_guess
                    st.session_state.alternatives = info_dict["alternative_guesses"]
                except Exception as e:
                    st.error(f"AI analysis failed: {str(e)}")
                    st.stop()
            # Show primary guess and actions
            primary_guess = st.session_state.primary_guess
            st.subheader("🎯 Primary Guess")
            primary_col1, primary_col2 = st.columns([2, 1])
            with primary_col1:
                st.markdown(f"**Artist:** {primary_guess['artist']}")
                st.markdown(f"**Album:** {primary_guess['album']}")
            with primary_col2:
                st.metric("Confidence", f"{primary_guess['confidence_score']}%")
            col1, col2 = st.columns(2)
            if col1.button("✅ Well done!", use_container_width=True, key="well_done_btn"):
                st.session_state.selected_artist = primary_guess['artist']
                st.session_state.selected_album = primary_guess['album']
                st.session_state.step = 'results'
                st.rerun()
            if col2.button("❌ Not exactly", use_container_width=True, key="not_exactly_btn"):
                st.session_state.step = 'alternatives'
                st.rerun()
        elif st.session_state.step == 'alternatives':
            st.subheader("🔄 Alternative Guesses")
            alternatives = st.session_state.alternatives
            for i, alt in enumerate(alternatives, 1):
                with st.expander(f"Option {i} - {alt['artist']} - {alt['album']}"):
                    alt_col1, alt_col2 = st.columns([2, 1])
                    with alt_col1:
                        st.markdown(f"**Artist:** {alt['artist']}")
                        st.markdown(f"**Album:** {alt['album']}")
                    with alt_col2:
                        st.metric("Confidence", f"{alt['confidence_score']}%")
            options = ["Primary Guess"] + [f"Option {i}" for i in range(1, len(alternatives) + 1)]
            selected_option = st.radio(
                "Select the correct identification:",
                options,
                key="alt_radio"
            )
            if selected_option == "Primary Guess":
                artist = st.session_state.primary_guess['artist']
                album = st.session_state.primary_guess['album']
            else:
                option_idx = int(selected_option.split()[-1]) - 1
                artist = alternatives[option_idx]['artist']
                album = alternatives[option_idx]['album']
            if st.button("🔮 Generate Story & Insights", use_container_width=True, key="alt_generate_btn"):
                st.session_state.selected_artist = artist
                st.session_state.selected_album = album
                st.session_state.step = 'results'
                st.rerun()
        elif st.session_state.step == 'results':
            with st.spinner("Story and insights generation..."):
                story = generate_story(st.session_state.selected_artist, st.session_state.selected_album)
                recs = recommend_similar(st.session_state.selected_artist, st.session_state.selected_album)
                price = estimate_price(st.session_state.selected_artist, st.session_state.selected_album)
                st.subheader("✨ Results")
                st.markdown(f"**Story:**\n{story}")
                st.markdown(f"**Similar Songs:**\n{recs}")
                st.markdown(f"**Estimated Price:** {price}")