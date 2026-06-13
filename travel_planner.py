import streamlit as st
import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from google import genai
from google.genai import types

# ==========================================
# 1. STREAMLIT FRONTEND LAYOUT
# ==========================================
st.set_page_config(page_title="AI Smart Travel Planner", layout="wide")

st.title("✈️ Smart AI Travel Planner & Budget Filter")
st.markdown("Enter any global destination along with your personal travel style. Our system will extract real-time web logistics, filter them using **Machine Learning**, and craft a personalized itinerary via **Gemini 2.5 Flash**.")

# Setup Sidebar for Inputs
st.sidebar.header("🔑 Authentication")

# Secure API Key Input Field (masked as a password for security)
instructor_key = st.sidebar.text_input(
    "Enter your Gemini API Key:", 
    type="password",
    placeholder="AIzaSy..."
)
st.sidebar.caption("Get a free key from Google AI Studio. Your key is processed securely in-memory and never saved permanently.")

st.sidebar.divider()

st.sidebar.header("🛠️ Step 1: Configure Your Trip")
user_destination = st.sidebar.text_input("Where do you want to go?", placeholder="e.g., Kyoto, Rome, Cairo")

st.sidebar.subheader("👤 User Profile & Vibes")
user_food = st.sidebar.text_area("Food Preferences", value="Obsessed with authentic street food, local markets, and cheap local dessert stalls.")
user_vibe = st.sidebar.text_area("Activities & Preferences", value="Enjoys hidden viewpoints, local thrift shops, and photography walking spots.")

# Execution Button
run_app = st.sidebar.button("🚀 Generate Budget Plan")

# ==========================================
# 2. RUN APPLICATION ENGINE
# ==========================================
if run_app:
    # Enforce API Key Validation check before starting the code pipeline
    if not instructor_key:
        st.sidebar.error("❌ API Key Required! Please enter your Gemini API key in the authentication section above to execute the application.")
    elif not user_destination:
        st.warning("Please enter a destination in the sidebar to begin!")
    else:
        # Pass the instructor's custom key directly into the GenAI Client initialization
        try:
            client = genai.Client(api_key=instructor_key)
        except Exception as api_err:
            st.error(f"Failed to initialize Gemini Client: {api_err}")
            st.stop()
        
        with st.spinner(f"🔍 Accessing web search APIs for current data on {user_destination}..."):
            # A. Live Scrape Prompt Configuration
            config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
            
            scraping_prompt = f"""
            Search the live web for budget travel logistics for a trip to {user_destination}.
            Find exactly 7 realistic options total (mix of budget accommodations and transport options like buses, trains, or flights).

            You MUST output your response strictly as a clean CSV text table with these exact column headers:
            item_name,price,type,duration_hours

            Rules:
            1. 'type' must be either 'accommodation' or 'transport'.
            2. 'price' must be a raw number in USD. If you cannot find a price, write None.
            3. 'duration_hours' must be a decimal number (write 0 for accommodations).
            4. Output ONLY the CSV data block. No markdown, no commentary.
            """
            
            # API Call using the custom validated client
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash", contents=scraping_prompt, config=config
                )
                raw_csv_text = response.text.replace("```csv", "").replace("```", "").strip()
            except Exception as e:
                st.error("Authentication Error: The API key provided appears to be invalid or unauthorized by Google AI Studio. Please verify the characters and try again.")
                st.stop()

        # B. DATA SCIENCE PARSING & DATA IMPUTATION
        try:
            df = pd.read_csv(io.StringIO(raw_csv_text))
            df.columns = df.columns.str.strip()
            
            df["price"] = pd.to_numeric(df["price"], errors='coerce')
            df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors='coerce')

            avg_price = df["price"].mean()
            df["price"] = df["price"].fillna(avg_price if not np.isnan(avg_price) else 50.0)
            df["duration_hours"] = df["duration_hours"].fillna(0)
        except Exception as e:
            st.error("There was a structure error reading live data. Please click generate again to refresh the fetch cycle.")
            st.stop()

        # C. MACHINE LEARNING BUDGET GATEKEEPER
        X_train = np.array([[20, 0], [35, 4], [250, 0], [180, 1.5], [15, 10], [500, 0], [45, 0], [130, 1]])
        y_train = np.array([1, 1, 0, 0, 1, 0, 1, 0])

        ml_gatekeeper = LogisticRegression()
        ml_gatekeeper.fit(X_train, y_train)

        X_live = df[["price", "duration_hours"]].values
        df["prediction"] = ml_gatekeeper.predict(X_live)

        filtered_df = df[df["prediction"] == 1].copy()

        # ==========================================
        # 3. INTERACTIVE WEB PRESENTATION SCREEN
        # ==========================================
        st.success("🎉 Data Science & ML Processing Complete!")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Scraped Live Data Processing Summary")
            st.dataframe(df.style.highlight_max(axis=0, subset=['price'], color='#ffcccc'), width='stretch')
            st.caption("Rows highlighting red showcase the most expensive logistics pulled from search results.")
            
        with col2:
            st.subheader("📈 ML Classifier Boundary Map")
            fig, ax = plt.subplots(figsize=(6, 4))
            approved = df[df["prediction"] == 1]
            rejected = df[df["prediction"] == 0]

            ax.scatter(approved["duration_hours"], approved["price"], color="green", label="Approved Budget", s=80)
            ax.scatter(rejected["duration_hours"], rejected["price"], color="red", label="Rejected", s=80)

            for i, txt in enumerate(df["item_name"]):
                ax.annotate(txt, (df["duration_hours"].iloc[i], df["price"].iloc[i]), xytext=(4, 4), textcoords="offset points", fontsize=7)

            ax.set_xlabel("Duration (Hours)")
            ax.set_ylabel("Price ($ USD)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()
            st.pyplot(fig)

        st.divider()

        # D. GENERATIVE AI NARRATIVE INTEGRATION
        with st.spinner("🤖 Writing your tailored travel story with Gemini..."):
            logistics_summary = filtered_df[["item_name", "price", "type", "duration_hours"]].to_string(index=False)
            
            final_prompt = f"""
            You are an expert local youth travel guide. Build a beautiful, fun, highly detailed 2-day itinerary for {user_destination} using ONLY the approved logistics below.

            Approved Budget Options to incorporate:
            {logistics_summary}

            Traveler Profile:
            - Food Interest: {user_food}
            - Vibe & Activities: {user_vibe}

            Formatting Rules:
            1. Base the stay and transportation choices strictly on the Approved Budget Options provided.
            2. Structure the answer clearly into Day 1 and Day 2 with engaging markdown emojis and bullet points.
            3. Keep the tone conversational, energetic, and highly budget-conscious.
            """

            final_response = client.models.generate_content(
                model="gemini-2.5-flash", contents=final_prompt
            )

        st.subheader(f"🗺️ Your Customized Itinerary for {user_destination.title()}")
        st.markdown(final_response.text)