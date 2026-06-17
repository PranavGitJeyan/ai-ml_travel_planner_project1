✈️ Smart AI Travel Planner & Budget Filter

A Streamlit app that scrapes live, real-world travel logistics for any global destination using Gemini 2.5 Flash with Google Search grounding, filters them through a Machine Learning budget classifier, and generates a personalized 2-day itinerary based on the user's food preferences and travel vibe.

Features


Live data ingestion — Uses the Google GenAI SDK with Google Search grounding to fetch real, current pricing and duration data for accommodations and transport, for any destination typed by the user.
Data preprocessing — Converts raw scraped text into a structured Pandas DataFrame, enforces numeric typing, and imputes missing prices using the column mean.
ML budget classification — A Logistic Regression classifier separates budget-friendly options from expensive outliers based on price and duration.
AI-generated itinerary — Gemini 2.5 Flash turns the ML-approved logistics into a conversational, emoji-rich 2-day itinerary tailored to the user's stated food and activity preferences.
Visualization — A scatter plot shows every scraped item color-coded by the classifier's decision (approved vs. rejected), with each point labeled.


Tech Stack

LayerTechnologyFrontendStreamlitLLM & Search GroundingGoogle GenAI SDK (google-genai), Gemini 2.5 FlashData ProcessingPandas, NumPyMachine LearningScikit-learn (LogisticRegression)VisualizationMatplotlib

Setup


Clone the repository and install dependencies:


bash   pip install streamlit pandas numpy matplotlib scikit-learn google-genai


Get a free Gemini API key from Google AI Studio.
Run the app:


bash   streamlit run travel_planner.py


Paste your Gemini API key into the sidebar, enter a destination and your travel preferences, and click Generate Budget Plan.



Your API key is used only in-memory for the current session and is never stored or logged.



How It Works


Scrape — The app sends a search-grounded prompt to Gemini, asking it to return exactly 7 real budget travel logistics options (accommodations and transport) for the chosen destination, formatted strictly as CSV.
Parse & Clean — The CSV response is loaded into a DataFrame; missing or malformed price/duration values are imputed so the pipeline doesn't break on incomplete data.
Classify — A Logistic Regression model predicts whether each item is "budget-friendly" (1) or not (0), based on its price and duration.
Visualize — Approved and rejected items are plotted on a price-vs-duration scatter chart for transparency into what the model filtered out.
Generate — Only the approved items are passed to a second Gemini call along with the user's food and vibe preferences, producing a structured Day 1 / Day 2 itinerary.


Known Limitations


Synthetic training data: The Logistic Regression classifier is trained on a small, illustrative synthetic dataset representing typical budget-travel price/duration ranges — not on a real-world historical pricing dataset. This was a deliberate scope decision for a proof-of-concept; see Future Scope below.
CSV parsing depends on LLM output compliance: While the prompt strictly specifies the output format, occasional formatting drift from the model can cause a parsing failure. Re-running the generation typically resolves this.
Fixed budget threshold: The classifier's decision boundary doesn't currently adapt per destination — what counts as "budget-friendly" in Tokyo and in Hanoi is treated the same way.


Future Scope


Replacing the synthetic training set with real, labeled historical pricing data so the classifier generalizes to real-world data for any place around the globe.
Upgrading to ensemble models (Random Forests, Gradient Boosting) to handle multi-dimensional budget constraints.
Multi-modal input (e.g., image-based destination/vibe matching via computer vision).
Live booking and transaction integration (Stripe, Amadeus).
Retrieval-Augmented Generation with a vector database to cache historical search results and speed up repeated queries.
