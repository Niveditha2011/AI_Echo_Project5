import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Assume df is preprocessed with columns: ['review', 'sentiment', 'rating', 'date', 'verified_purchase', 'review_length', 'platform', 'version', 'location']

st.title("🔍 Key Questions for Sentiment Analysis")

# Q1: Overall sentiment proportions
st.subheader("1. What is the overall sentiment of user reviews?")
sentiment_counts = df['sentiment'].value_counts(normalize=True)
st.bar_chart(sentiment_counts)

# Q2: Sentiment by rating
st.subheader("2. How does sentiment vary by rating?")
st.pyplot(sns.countplot(data=df, x='rating', hue='sentiment').figure)

# Q3: Keywords associated with each sentiment
st.subheader("3. Which keywords or phrases are most associated with each sentiment class?")
for label in df['sentiment'].unique():
    st.write(f"**{label} WordCloud**")
    text = ' '.join(df[df['sentiment'] == label]['review'])
    wc = WordCloud(width=800, height=300, background_color='white').generate(text)
    st.image(wc.to_array())

# Q4: Sentiment over time
st.subheader("4. How has sentiment changed over time?")
df['date'] = pd.to_datetime(df['date'])
df_time = df.groupby([pd.Grouper(key='date', freq='M'), 'sentiment']).size().unstack().fillna(0)
st.line_chart(df_time)

# Q5: Sentiment by verified purchase
st.subheader("5. Do verified users tend to leave more positive or negative reviews?")
verified_group = df.groupby('verified_purchase')['sentiment'].value_counts(normalize=True).unstack()
st.bar_chart(verified_group)

# Q6: Sentiment vs review length
st.subheader("6. Are longer reviews more likely to be negative or positive?")
sns.boxplot(data=df, x='sentiment', y='review_length')
st.pyplot(plt.gcf())
plt.clf()

# Q7: Sentiment by location
st.subheader("7. Which locations show the most positive or negative sentiment?")
location_sentiment = df.groupby('location')['sentiment'].value_counts(normalize=True).unstack().fillna(0)
st.dataframe(location_sentiment.sort_values('Positive', ascending=False).head(10))

# Q8: Sentiment across platforms
st.subheader("8. Is there a difference in sentiment across platforms (Web vs Mobile)?")
platform_sentiment = df.groupby('platform')['sentiment'].value_counts(normalize=True).unstack()
st.bar_chart(platform_sentiment)

# Q9: Sentiment by ChatGPT version
st.subheader("9. Which ChatGPT versions are associated with higher/lower sentiment?")
version_sentiment = df.groupby('version')['sentiment'].value_counts(normalize=True).unstack().fillna(0)
st.dataframe(version_sentiment)

# Q10: Common negative feedback themes
st.subheader("10. What are the most common negative feedback themes?")
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=30)
neg_reviews = df[df['sentiment'] == 'Negative']['review']
neg_matrix = vectorizer.fit_transform(neg_reviews)
neg_freq = pd.DataFrame(neg_matrix.sum(axis=0), columns=vectorizer.get_feature_names_out()).T.sort_values(0, ascending=False)
st.dataframe(neg_freq.rename(columns={0: 'Frequency'}))
