from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient
import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
# Optional: altair, plotly can be used later if needed
# import altair as alt
# import plotly.express as px

# -------------------------
# Helper function to get dataset path
# -------------------------
def get_dataset_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "customer_reviews.csv")
    return csv_path

# Helper function to clean text
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text    

# -------------------------
# Setup
# -------------------------
st.set_page_config(
    page_title="GenAI Prototype",
    page_icon="🤖",
    layout="wide"
)

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")

# Initialize Hugging Face client
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-alpha",
    token=hf_token
)

# -------------------------
# AI Response Function
# -------------------------
def get_response(prompt, temperature=0.7, max_tokens=200):
    """Send user prompt to Hugging Face model and return response text."""
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message["content"]

# -------------------------
# Sidebar (Dashboard Controls)
# -------------------------
st.sidebar.title("⚙️ Settings")
temperature = st.sidebar.slider(
    "Creativity (Temperature):",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
    help="0 = focused, 1 = very creative"
)
max_tokens = st.sidebar.slider(
    "Max Tokens:",
    min_value=50,
    max_value=1000,
    value=200,
    step=10,
    help="Maximum length of AI response"
)
st.sidebar.markdown("---")
st.sidebar.info("Adjust model parameters here.")

# -------------------------
# Main Chat Interface
# -------------------------
st.title("🤖 GenAI Prototype Dashboard")
st.markdown("Chat with the AI assistant below 👇")

# Session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get AI response
    with st.spinner("AI is thinking..."):
        ai_reply = get_response(user_input, temperature, max_tokens)
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})

# Display chat history with alignment
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style='text-align: right; background-color: #241f1f; color: white;
                        padding: 10px; border-radius: 15px; margin: 5px 0;
                        display: inline-block; float: right; clear: both;'>
                {msg["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='text-align: left; background-color: #3d3c3c; color: white;
                        padding: 10px; border-radius: 15px; margin: 5px 0;
                        display: inline-block; float: left; clear: both;'>
                {msg["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------
# Dataset Section
# -------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Ingest Dataset"):
        try:
            csv_path = get_dataset_path()
            st.session_state["df"] = pd.read_csv(csv_path)
            st.success("Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path.")

with col2:
    if st.button("🧹 Parse Reviews"):
        if "df" in st.session_state:
            st.session_state["df"]["CLEANED_SUMMARY"] = st.session_state["df"]["SUMMARY"].apply(clean_text)
            st.success("Reviews parsed and cleaned!")
        else:
            st.warning("Please ingest the dataset first.")

# -------------------------
# Dataset Display & Visualization
# -------------------------
if "df" in st.session_state:
    # Product filter dropdown
    st.subheader("🔍 Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader(f"📁 Reviews for {product}")

    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]

    st.dataframe(filtered_df)

    # Histogram of sentiment scores
    st.subheader(f"📊 Sentiment Score Distribution for {product}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(filtered_df["SENTIMENT_SCORE"], bins=10, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Sentiment Scores')
    st.pyplot(fig)
    
    
    
    
    
    
    
    
    
    
