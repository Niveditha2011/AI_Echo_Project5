import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

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
# ✅ This must come first before any st.* call

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

df = pd.read_csv(r"C:\Users\Python Class\AI_Echo_Project5\cleaned_reviews.csv")
df['date'] = pd.to_datetime(df['date'], errors='coerce')


Analysis_Options = ["select a Question",
               "1. What is the overall sentiment of user reviews?",
               "2. How does sentiment vary by rating?",
               "3. Which keywords or phrases are most associated with each sentiment class?",
               "4. How has sentiment changed over time?",
               "5. Do verified users tend to leave more positive or negative reviews?",
               "6. Are longer reviews more likely to be negative or positive?",
               "7. Which locations show the most positive or negative sentiment?",
               "8. Is there a difference in sentiment across platforms (Web vs Mobile)?",
               "9. Which ChatGPT versions are associated with higher/lower sentiment?",
               "10.What are the most common negative feedback themes?" 
               ]


st.sidebar.image(r"C:\Users\Python Class\AI_Echo_Project5\download (4).png")
sel_option = st.sidebar.selectbox("Sentiment Analysis Questions",Analysis_Options,index=0) 


# -----------------------------
# 1. Overall Sentiment
# -----------------------------
#1
if sel_option == "1. What is the overall sentiment of user reviews?":
    st.header("Overall Sentiment Distribution")
    sentiment_counts = df['feedback_sentiment'].value_counts(normalize=True) * 100
    st.bar_chart(sentiment_counts)

# -----------------------------
# 2. Sentiment vs Rating
# -----------------------------
if sel_option == "2. How does sentiment vary by rating?":
    st.header("Sentiment vs Rating")
    rating_sentiment = pd.crosstab(df['rating'], df['feedback_sentiment'], normalize='index') * 100
    st.dataframe(rating_sentiment.style.format("{:.1f}%"))

    fig, ax = plt.subplots()
    rating_sentiment.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    plt.ylabel("Percentage")
    plt.xlabel("Rating")
    st.pyplot(fig)

# -----------------------------
# 3. Word Clouds by Sentiment
# -----------------------------
if sel_option == "3. Which keywords or phrases are most associated with each sentiment class?":
    st.header("Keywords / Word Clouds per Sentiment")
    for sentiment in df['feedback_sentiment'].unique():
        st.subheader(f"{sentiment} Reviews")
        text = " ".join(df[df['feedback_sentiment'] == sentiment]['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig_wc, ax_wc = plt.subplots(figsize=(10,4))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)

# -----------------------------
# 4. Sentiment Trends Over Time
# -----------------------------
if sel_option == "4. How has sentiment changed over time?":
    st.header("Sentiment Trends Over Time")
    sentiment_time = df.groupby([df['date'].dt.to_period('M'), 'feedback_sentiment']).size().unstack(fill_value=0)
    sentiment_time.plot(figsize=(12,5))
    plt.xlabel("Month")
    plt.ylabel("Number of Reviews")
    plt.title("Sentiment Trends Over Time")
    st.pyplot(plt.gcf())

# -----------------------------
# 5. Verified Purchase vs Sentiment
# -----------------------------
if sel_option == "5. Do verified users tend to leave more positive or negative reviews?":
    st.header("Verified Purchase vs Sentiment")
    verified_sentiment = pd.crosstab(df['verified_purchase'], df['feedback_sentiment'], normalize='index') * 100
    st.dataframe(verified_sentiment.style.format("{:.1f}%"))

    fig, ax = plt.subplots()
    verified_sentiment.plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')
    plt.ylabel("Percentage")
    plt.xlabel("Verified Purchase")
    st.pyplot(fig)

# -----------------------------
# 6. Review Length vs Sentiment
# -----------------------------
if sel_option == "6. Are longer reviews more likely to be negative or positive?":
    st.header("Review Length vs Sentiment")
    df['review_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    length_sentiment = df.groupby('feedback_sentiment')['review_length'].mean()
    st.bar_chart(length_sentiment)

# -----------------------------
# 7. Sentiment by Location
# -----------------------------
if sel_option == "7. Which locations show the most positive or negative sentiment?":
    st.header("Sentiment by Location")
    location_sentiment = pd.crosstab(df['location'], df['feedback_sentiment'], normalize='index') * 100
    st.dataframe(location_sentiment.style.format("{:.1f}%"))

# -----------------------------
# 8. Sentiment by Platform
# -----------------------------
if sel_option == "8. Is there a difference in sentiment across platforms (Web vs Mobile)?":
    st.header("Sentiment by Platform")
    platform_sentiment = pd.crosstab(df['platform'], df['feedback_sentiment'], normalize='index') * 100
    st.dataframe(platform_sentiment.style.format("{:.1f}%"))

    fig, ax = plt.subplots()
    platform_sentiment.plot(kind='bar', stacked=True, ax=ax, colormap='plasma')
    plt.ylabel("Percentage")
    plt.xlabel("Platform")
    st.pyplot(fig)

# -----------------------------
# 9. Sentiment by ChatGPT Version
# -----------------------------
if sel_option == "9. Which ChatGPT versions are associated with higher/lower sentiment?":
    st.header("Sentiment by ChatGPT Version")
    version_sentiment = pd.crosstab(df['version'], df['feedback_sentiment'], normalize='index') * 100
    st.dataframe(version_sentiment.style.format("{:.1f}%"))

# -----------------------------
# 10. Common Negative Feedback Themes
# -----------------------------
if sel_option == "10.What are the most common negative feedback themes?":
    st.header("Common Negative Feedback Themes")
    negative_text = " ".join(df[df['feedback_sentiment'] == 'Negative']['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(negative_text)
    fig_wc, ax_wc = plt.subplots(figsize=(10,4))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)
