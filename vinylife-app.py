import streamlit as st
import pandas as pd
import openai
import re
import json
from PIL import Image
import io
import base64
from streamlit_carousel import carousel

def get_price_range_with_gpt_web_search(artist, album):
    prompt = (
        f"Search the web and find the range of current selling prices for the vinyl album '{album}' by '{artist}'. "
        "Only consider offers and listings for used or new vinyls, not CDs or digital versions. "
        "List at least 5 prices from marketplaces like Discogs, eBay, Wallapop, or Vinted, indicating the source. "
        "Then, respond in this exact JSON format:\n"
        '{"min": <min_price>, "average": <average_price>, "max": <max_price>, "examples": [ {"price": <price>, "source": "<site>", "url": "<url>"}, ... ] }'
        "\nOnly output valid JSON, nothing else. All prices in EUR."
    )

    query = f"{artist} {album} vinyl price site:discogs.com OR site:ebay.com OR site:wallapop.com OR site:vinted.es"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-search-preview-2025-03-11",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        max_tokens=500
    )
    import re, json
    raw = response.choices[0].message.content.strip()
    # Remove code blocks if present
    raw = re.sub(r"^```[a-zA-Z0-9]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # Extract JSON
    match = re.search(r"({.*})", raw, re.DOTALL)
    if match:
        raw = match.group(1)
    try:
        price_data = json.loads(raw)
        return price_data
    except Exception:
        return None

# Helper function to clean AI JSON responses
def clean_json_response(response):
    """
    Remove code block markers and extract the first valid JSON object.
    """
    response = response.strip()
    # Remove triple backticks and any leading language labels (e.g., ```json)
    response = re.sub(r"^```[a-zA-Z0-9]*\s*", "", response)
    response = re.sub(r"\s*```$", "", response)
    # Extract JSON
    match = re.search(r"({.*})", response, re.DOTALL)
    if match:
        return match.group(1)
    return response

# Helper function to display similar songs using a carousel
def display_similar_songs(recs):
    st.markdown("### Similar Songs:", unsafe_allow_html=True)
    try:
        recs_json = json.loads(recs)
        if not recs_json:
            st.info("No similar songs recommended.")
            return

        # Create items for the carousel
        carousel_items = []
        for rec in recs_json:
            title = rec.get("title", "Unknown Title")
            artist_ = rec.get("artist", "Unknown Artist")
            why = rec.get("why it's similar", rec.get("why", ""))
            # Using basic HTML for content within the card
            content = f"<h4>{title} by {artist_}</h4><p>{why}</p>"
            # Assuming no images for now, but adding 'img' field as per example
            carousel_items.append({"title": title, "text": content, "img": ""}) # Add img field

        # Display the carousel
        # Add a unique key to the carousel component
        carousel(carousel_items, speed=1.5, key="similar_songs_carousel")

    except json.JSONDecodeError:
        st.error("Could not load recommendations (invalid format).")
        with st.expander("Debug: Raw Recommendations Response"):
            st.code(recs)
    except Exception as e:
        st.error(f"An error occurred while displaying recommendations: {e}")
        with st.expander("Debug: Raw Recommendations Response"):
            st.code(recs)

# === CONFIGURATION ===
openai.api_key = st.secrets["OPENAI_API_KEY"]

def extract_vinyl_info_from_image(uploaded_image):
    image_bytes = uploaded_image.read()
    mime_type = uploaded_image.type  # e.g., 'image/jpeg' or 'image/png'
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    image_url = f"data:{mime_type};base64,{base64_image}"
    prompt = (
        "You are a vinyl record expert. Analyze the following image of a vinyl cover and extract the most likely artist and album/EP title. "
        "Respond only with a valid JSON object in this format: "
        '{"artist": "...", "album": "...", "confidence": 0-100} '
        "If you are unsure, set confidence below 70."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the artist and album on this vinyl cover?"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            max_tokens=500
        )
        vinyl_info = response.choices[0].message.content.strip()

        if not vinyl_info:
            raise ValueError("API returned an empty or invalid response.")

        # Debugging: Print the raw API response to logs
        print(f"Raw API response before JSON load: {vinyl_info}")

        # The calling code in the UI already handles JSONDecodeError
        return vinyl_info

    except Exception as e:
        # Catch any other potential errors during the API call or response processing
        raise RuntimeError(f"Error during OpenAI API call: {e}")

def generate_story(artist, album, search_results=None):
    # Format search results for the prompt (you might want to refine this formatting)
    search_context = ""
    if search_results and search_results.get('results'):
        search_context = "\n\nRelevant information from web search:\n"
        for i, result in enumerate(search_results['results'][:3]): # Limit to top 3 results
            search_context += f"Source {i+1}: {result['snippet']}\n"

    prompt = f"""
    You are a passionate music historian. Write a short and captivating 150-word story about the artist {artist} and their album or EP titled '{album}'.
    Include cultural context, musical style, and legacy.
    
    {search_context}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4", # Using a text-based model for story generation
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return response.choices[0].message.content.strip()

def recommend_similar(artist, album):
    prompt = f'''
    Suggest 3 songs similar to any track from the album '{album}' by {artist}.
    For each song, provide:
    - title
    - artist
    - why it's similar
    Respond in JSON format as a list of objects.
    '''
    response = openai.ChatCompletion.create(
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
st.info("📸 For best results, upload a clear, well-lit, front-facing image of the vinyl cover.")

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
    if st.session_state.step != 'results':
        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(uploaded_image, caption="Uploaded Vinyl Cover", use_container_width=True)
        with col2:
            if st.session_state.step == 'guess':
                with st.spinner("Analyzing image with AI..."):
                    try:
                        vinyl_info = extract_vinyl_info_from_image(uploaded_image)

                        if not vinyl_info or vinyl_info.strip() == "":
                            st.error("AI analysis failed: The AI did not return any data. The image may be unclear or there was an internal error. Please try again or upload a different image.")
                            st.stop()

                        try:
                            cleaned = clean_json_response(vinyl_info)
                            info_dict = json.loads(cleaned)
                        except json.JSONDecodeError:
                            st.error("AI analysis failed: The AI did not return valid JSON. This may be a temporary issue or the model was unable to analyze the image. See raw output below for debugging.")
                            with st.expander("Show AI raw response (debug)"):
                                st.code(vinyl_info)
                            st.stop()

                        required_fields = {"artist", "album", "confidence"}
                        if not required_fields.issubset(info_dict.keys()):
                            st.error(f"AI analysis failed: Missing fields in AI response. Expected fields: {required_fields}. See raw output below for debugging.")
                            with st.expander("Show AI raw response (debug)"):
                                st.code(vinyl_info)
                            st.stop()

                        primary_guess = info_dict
                        if primary_guess['confidence'] < 70:
                            st.warning("Confidence is below 70%, please verify the identification.")
                        st.session_state.primary_guess = primary_guess
                        st.session_state.alternatives = []

                    except Exception as e:
                        st.error(f"AI analysis failed: {str(e)}")
                        st.stop()
                # Columns for action buttons
                button_col1, button_col2 = st.columns(2)

                # Show primary guess and actions
                primary_guess = st.session_state.primary_guess
                st.subheader("🎯 Primary Guess")
                primary_col1, primary_col2 = st.columns([2, 1])
                with primary_col1:
                    st.markdown(f"**Artist:** {primary_guess['artist']}")
                    st.markdown(f"**Album:** {primary_guess['album']}")
                with primary_col2:
                    st.metric("Confidence", f"{primary_guess['confidence']}")
                if button_col1.button("✅ Well done!", use_container_width=True, key="well_done_btn"):
                    st.session_state.selected_artist = primary_guess['artist']
                    st.session_state.selected_album = primary_guess['album']
                    st.session_state.step = 'results'
                if button_col2.button("❌ Not exactly", use_container_width=True, key="not_exactly_btn"):
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
                            st.metric("Confidence", f"{alt['confidence']}%")
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
    elif st.session_state.step == 'results':
        with st.spinner("Story and insights generation..."):
            artist = st.session_state.selected_artist
            album = st.session_state.selected_album
            story = generate_story(artist, album)
            recs = recommend_similar(artist, album)
            # This is your new web search with price extraction!
            st.markdown(f"<h1 style='margin-bottom: 1.5rem'>{artist} — {album}</h1>", unsafe_allow_html=True)

            col_story, col_insights = st.columns(2)
            with col_story:
                st.markdown("<b>Story:</b> " + story, unsafe_allow_html=True)
            with col_insights:
                st.markdown("<b>Insights:</b>", unsafe_allow_html=True)

                # Display similar songs using the helper function
                display_similar_songs(recs)

                # Web search for price extraction
                st.markdown("### Estimated Price Range (Live Web):", unsafe_allow_html=True)
                price_data = get_price_range_with_gpt_web_search(artist, album)
                if price_data:
                    min_price = price_data.get("min")
                    avg_price = price_data.get("average")
                    max_price = price_data.get("max")
                    
                    # Ensure prices are numbers before casting to int
                    try:
                        min_price_int = int(float(min_price)) if min_price is not None else 0
                        max_price_int = int(float(max_price)) if max_price is not None else 100 # Provide a default max if needed
                        avg_price_int = int(float(avg_price)) if avg_price is not None else min_price_int
                        
                        # Adjust max_value if it's less than min_value or avg_value
                        if max_price_int < min_price_int:
                            max_price_int = min_price_int + 10
                        if max_price_int < avg_price_int:
                             max_price_int = avg_price_int + 10

                        st.slider(
                            "Web Price Range (EUR):",
                            min_value=min_price_int,
                            max_value=max_price_int,
                            value=avg_price_int,
                            step=1,
                            format="%d",
                            disabled=True
                        )
                        st.markdown(f"**Min:** €{min_price}  **Avg:** €{avg_price}  **Max:** €{max_price}")
                        st.markdown("**Sample Listings:**")
                        for ex in price_data.get("examples", []):
                            price = ex.get("price")
                            source = ex.get("source")
                            url = ex.get("url")
                            if price is not None and source is not None:
                                if url:
                                    st.markdown(f"- [€{price} on {source}]({url})", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"- €{price} on {source}", unsafe_allow_html=True)

                    except (ValueError, TypeError) as e:
                         st.warning(f"Could not parse price data: {e}. Raw data below.")
                         with st.expander("Show raw price data (debug)"):
                             st.json(price_data)

                else:
                    st.warning("Could not retrieve price data from the web.")