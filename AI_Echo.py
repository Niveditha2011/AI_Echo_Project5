# import streamlit as st
# import pandas as pd
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import nltk
# import matplotlib.pyplot as plt

# # Download VADER lexicon (only the first time)
# nltk.download('vader_lexicon')

# # Initialize Sentiment Analyzer
# sid = SentimentIntensityAnalyzer()

# # Function to analyze sentiment
# def get_sentiment(text):
#     score = sid.polarity_scores(str(text))['compound']
#     if score >= 0.05:
#         return "Positive"
#     elif score <= -0.05:
#         return "Negative"
#     else:
#         return "Neutral"

# # Streamlit Dashboard
# st.set_page_config(page_title="📊 Sentiment Analysis Dashboard", layout="wide")

# st.title("📊 Sentiment Analysis Dashboard")
# st.markdown("Upload your dataset and explore sentiment trends easily 🚀")

# # File uploader
# uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type="csv")

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)

#     if 'review' not in df.columns:
#         st.error("CSV must contain a 'review' column ❗")
#     else:
#         # Sentiment Analysis
#         df['Sentiment'] = df['review'].apply(get_sentiment)

#         # Show sample data
#         st.subheader("🔎 Sample Data")
#         st.dataframe(df.head(10))

#         # Sentiment distribution
#         st.subheader("📊 Sentiment Distribution")
#         sentiment_counts = df['Sentiment'].value_counts()

#         fig, ax = plt.subplots(figsize=(4,3))  # Smaller bar chart
#         sentiment_counts.plot(kind='bar', ax=ax)
#         ax.set_xlabel("Sentiment")
#         ax.set_ylabel("Count")
#         ax.set_title("Distribution of Sentiments")
#         st.pyplot(fig)

#         # Pie chart
#         st.subheader("🥧 Sentiment Proportion")
#         fig2, ax2 = plt.subplots(figsize=(4,3))  # Smaller pie chart
#         sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2, startangle=90)
#         ax2.set_ylabel("")
#         st.pyplot(fig2)

#         # Filter reviews
#         st.subheader("🔍 Explore Reviews by Sentiment")
#         sentiment_choice = st.selectbox("Choose sentiment", ["Positive", "Negative", "Neutral"])
#         st.write(df[df['Sentiment'] == sentiment_choice][['review']].head(10))

# else:
#     st.info("👆 Upload a CSV file to start analysis.")
    
    
    
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ------------------------------
# Load Model & Vectorizer
# ------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ------------------------------
# Prediction Function
# ------------------------------
def predict_sentiment(text):
    # Example using your trained model
    vectorized_text = vectorizer.transform([text])
    pred = model.predict(vectorized_text)
    return pred[0]
# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="📊 AI-ECHO : Sentiment Analysis Dashboard", layout="wide")

st.title("📊 Sentiment Analysis of ChatGPT Reviews")
st.markdown(
    """
    This dashboard analyzes user reviews of a ChatGPT application and classifies them as 
    **Positive 😊, Neutral 😐, or Negative 😡**.  
    Goal → Gain insights into customer satisfaction and improve user experience.
    """
)

# User input
st.subheader("✍️ Try it Yourself")
user_input = st.text_area("Enter a review:")

if st.button("Analyze"):
    if user_input.strip() != "":
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{sentiment}** 🎯")
    else:
        st.warning("Please enter some text!")

# File upload for batch processing
st.subheader("📂 Upload CSV for Bulk Analysis")
uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "review" not in df.columns:
        st.error("CSV must contain a 'review' column ❗")
    else:
        df["Predicted_Sentiment"] = df["review"].apply(predict_sentiment)

        st.write("✅ Sample Results:")
        st.dataframe(df.head())

        # Sentiment distribution
        sentiment_counts = df["Predicted_Sentiment"].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(4, 3))
            sentiment_counts.plot(kind="bar", ax=ax, color=["red", "gray", "green"])
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with col2:
            st.subheader("🥧 Sentiment Proportion")
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            sentiment_counts.plot(
                kind="pie", autopct="%1.1f%%", ax=ax2, startangle=90, colors=["red", "gray", "green"]
            )
            ax2.set_ylabel("")
            st.pyplot(fig2)

        # Download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="sentiment_predictions.csv",
            mime="text/csv",
        )
