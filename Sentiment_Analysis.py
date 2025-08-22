import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# -----------------------------
# 1️⃣ Load Cleaned Data
# -----------------------------
df = pd.read_csv(r"C:\Users\Python Class\AI_Echo_Project5\cleaned_reviews.csv")
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# st.title("ChatGPT Reviews Sentiment Analysis Dashboard")

user_color='white'
original_title = """
<div style="background-color:{};padding:12px">
<h2 style="font-family:Courier;color:black;font-weight: bold;test-align:center;">🌿Sentiment Analysis Dashboard</h2>
</div>""".format(user_color)
st.markdown(original_title, unsafe_allow_html=True)

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



# # -----------------------------
# # 1. Distribution of Review Ratings
# # -----------------------------
# st.header("📊 1. Distribution of Review Ratings")
# rating_counts = df['rating'].value_counts().sort_index()
# st.bar_chart(rating_counts)
# st.write("💡 Insight: This chart helps understand overall sentiment. High 4–5 stars → mostly happy; many 1–2 stars → mostly frustrated.")

# # -----------------------------
# # 2. Helpful Reviews
# # -----------------------------
# st.header("👍👎 2. Reviews Marked as Helpful")
# helpful_threshold = st.slider("Select helpful votes threshold:", 0, 100, 10)
# helpful_counts = {
#     f'> {helpful_threshold} votes': (df['helpful_votes'] > helpful_threshold).sum(),
#     f'≤ {helpful_threshold} votes': (df['helpful_votes'] <= helpful_threshold).sum()
# }
# helpful_df = pd.DataFrame.from_dict(helpful_counts, orient='index', columns=['count'])
# fig, ax = plt.subplots()
# ax.pie(helpful_df['count'], labels=helpful_df.index, autopct='%1.1f%%', startangle=90, colors=['#4CAF50','#FF5722'])
# ax.axis('equal')
# st.pyplot(fig)
# st.write(f"💡 Insight: Shows which reviews users find valuable (above {helpful_threshold} votes).")

# # -----------------------------
# # 3. Keywords in Positive vs Negative Reviews
# # -----------------------------
# st.header("🧭 3. Keywords in Positive vs Negative Reviews")
# positive_text = " ".join(df[df['rating'].isin([4,5])]['cleaned_text'])
# negative_text = " ".join(df[df['rating'].isin([1,2])]['cleaned_text'])
# fig, axes = plt.subplots(1,2,figsize=(15,6))
# axes[0].imshow(WordCloud(width=800, height=400, background_color='white').generate(positive_text))
# axes[0].axis('off')
# axes[0].set_title("4–5 Stars (Positive)")
# axes[1].imshow(WordCloud(width=800, height=400, background_color='white').generate(negative_text))
# axes[1].axis('off')
# axes[1].set_title("1–2 Stars (Negative)")
# st.pyplot(fig)
# st.write("💡 Insight: Discover what users love or complain about.")

# # -----------------------------
# # 4. Average Rating Over Time
# # -----------------------------
# st.header("📆 4. Average Rating Over Time")
# avg_rating_time = df.groupby(df['date'].dt.to_period('W'))['rating'].mean()
# fig, ax = plt.subplots(figsize=(12,5))
# avg_rating_time.plot(ax=ax)
# plt.xlabel("Week")
# plt.ylabel("Average Rating")
# plt.title("Average Rating Over Time")
# st.pyplot(fig)
# st.write("💡 Insight: Track user satisfaction over weeks/months.")

# # -----------------------------
# # 5. Ratings by Location
# # -----------------------------
# st.header("🌍 5. Ratings by User Location")
# location_avg = df.groupby('location')['rating'].mean().sort_values(ascending=False)
# fig, ax = plt.subplots(figsize=(12,6))
# location_avg.plot(kind='bar', ax=ax)
# plt.ylabel("Average Rating")
# plt.xlabel("Location")
# st.pyplot(fig)
# st.write("💡 Insight: Identify regional differences in satisfaction.")

# # -----------------------------
# # 6. Ratings by Platform
# # -----------------------------
# st.header("🧑‍💻 6. Ratings by Platform")
# platform_avg = df.groupby('platform')['rating'].mean()
# fig, ax = plt.subplots()
# platform_avg.plot(kind='bar', color=['#1f77b4','#ff7f0e'], ax=ax)
# plt.ylabel("Average Rating")
# plt.xlabel("Platform")
# st.pyplot(fig)
# st.write("💡 Insight: See which platform gets better reviews.")

# # -----------------------------
# # 7. Verified Users vs Non-Verified
# # -----------------------------
# st.header("✅❌ 7. Verified vs Non-Verified Users")
# verified_avg = df.groupby('verified_purchase')['rating'].mean()
# fig, ax = plt.subplots()
# verified_avg.plot(kind='bar', color=['#2ca02c','#d62728'], ax=ax)
# plt.ylabel("Average Rating")
# plt.xlabel("Verified Purchase")
# st.pyplot(fig)
# st.write("💡 Insight: Indicates whether loyal/verified users are happier.")

# # -----------------------------
# # 8. Review Length per Rating
# # -----------------------------
# st.header("🔠 8. Review Length per Rating")
# fig, ax = plt.subplots(figsize=(10,5))
# sns.boxplot(x='rating', y='review_length', data=df, ax=ax)
# plt.ylabel("Review Length (words)")
# plt.xlabel("Rating")
# st.pyplot(fig)
# st.write("💡 Insight: Longer reviews may indicate very positive or negative experiences.")

# # -----------------------------
# # 9. Most Mentioned Words in 1-Star Reviews
# # -----------------------------
# st.header("💬 9. Most Mentioned Words in 1-Star Reviews")
# text_1star = " ".join(df[df['rating']==1]['cleaned_text'])
# fig, ax = plt.subplots(figsize=(10,5))
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_1star)
# ax.imshow(wordcloud, interpolation='bilinear')
# ax.axis('off')
# st.pyplot(fig)
# st.write("💡 Insight: Spot recurring issues or complaints.")

# # -----------------------------
# # 10. ChatGPT Version vs Average Rating
# # -----------------------------
# st.header("📱🧪 10. ChatGPT Version vs Average Rating")
# version_avg = df.groupby('version')['rating'].mean().sort_values(ascending=False)
# fig, ax = plt.subplots(figsize=(12,6))
# version_avg.plot(kind='bar', ax=ax, color='skyblue')
# plt.ylabel("Average Rating")
# plt.xlabel("ChatGPT Version")
# st.pyplot(fig)
# st.write("💡 Insight: Evaluate improvement or regression across versions.")
