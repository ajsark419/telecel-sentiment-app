import streamlit as st
import pandas as pd
import re
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from transformers import pipeline

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Telecel AI Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# =============================
# MOBILE RESPONSIVE STYLES
# =============================
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

/* Make text scale better on mobile */
h2 {
    font-size: clamp(18px, 2.2vw, 26px);
}

p {
    font-size: clamp(12px, 1.5vw, 14px);
}

/* Better spacing for cards */
.element-container {
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# =============================
# BRAND HEADER (RESPONSIVE FIXED)
# =============================
col_logo, col_text = st.columns([1.5, 9])  # improved ratio for mobile

with col_logo:
    st.image("telecel-logo.png", width=170,)  # reduced for mobile

with col_text:
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #d40000, #111827);
        padding: 14px;
        border-radius: 12px;
    ">
        <h2 style="color:white; margin:0;">
            Telecel AI Sentiment Dashboard
        </h2>
        <p style="color:#d1d5db; margin:0;">
            Customer Intelligence System powered by BERT AI
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================
# MODEL (UNCHANGED)
# =============================
@st.cache_resource
def get_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

# =============================
# STOPWORDS SAFE LOAD
# =============================
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# =============================
# PREPROCESSING
# =============================
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# =============================
# AI PREDICTION
# =============================
def predict(text):
    model = get_model()
    result = model(text[:512])[0]

    label = result["label"]
    score = result["score"]

    if score < 0.60:
        label = "NEUTRAL"
    else:
        label = "POSITIVE" if label == "POSITIVE" else "NEGATIVE"

    return label, score

# =============================
# SIDEBAR
# =============================
st.sidebar.markdown("## 📡 Telecel AI System")
st.sidebar.caption("Sentiment Intelligence Engine")

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "🧠 Single Tweet", "📂 Dataset Analysis"]
)

# =============================
# DASHBOARD
# =============================
if menu == "🏠 Dashboard":

    st.markdown("---")

    st.subheader("System Overview")

    col1, col2, col3 = st.columns(3)  # stays 3 (better mobile balance)

    col1.markdown("### 🤖 AI Model\nBERT Transformer")
    col2.markdown("### ⚡ Processing\nReal-Time NLP")
    col3.markdown("### 📊 Output\nPositive / Neutral / Negative")

    st.markdown("---")

    st.info("This system analyzes customer sentiment using BERT-based AI and NLP preprocessing pipelines.")

# =============================
# SINGLE TWEET
# =============================
elif menu == "🧠 Single Tweet":

    st.subheader("Tweet Sentiment Analyzer")

    text = st.text_area("Enter Tweet", height=120)

    if st.button("Analyze Tweet"):

        if text.strip() == "":
            st.warning("Please enter a tweet")
        else:

            with st.spinner("🤖 AI is analyzing sentiment..."):

                clean = preprocess(text)
                label, score = predict(clean)

            st.markdown("### Result")

            if label == "POSITIVE":
                st.success("😊 Positive Sentiment")

            elif label == "NEGATIVE":
                st.error("😠 Negative Sentiment")

            else:
                st.info("😐 Neutral Sentiment")

            st.metric("Confidence Score", f"{round(score*100,2)}%")

# =============================
# DATASET ANALYSIS
# =============================
elif menu == "📂 Dataset Analysis":

    st.subheader("Batch Sentiment Intelligence")

    file = st.file_uploader("Upload CSV (must contain 'Tweets')", type=["csv"])

    if file:

        df = pd.read_csv(file)

        if "Tweets" not in df.columns:
            st.error("Missing 'Tweets' column")
        else:

            with st.spinner("🤖 AI is analyzing dataset... please wait"):

                df["Cleaned"] = df["Tweets"].astype(str).apply(preprocess)
                df["Sentiment"] = df["Cleaned"].apply(lambda x: predict(x)[0])
                df["Confidence"] = df["Cleaned"].apply(lambda x: predict(x)[1])

            # =============================
            # KPI CARDS (MOBILE SAFE)
            # =============================
            st.subheader("📊 Key Metrics")

            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)

            c1.metric("Total Tweets", len(df))
            c2.metric("Positive", (df["Sentiment"]=="POSITIVE").sum())
            c3.metric("Neutral", (df["Sentiment"]=="NEUTRAL").sum())
            c4.metric("Negative", (df["Sentiment"]=="NEGATIVE").sum())

            st.markdown("---")

            # =============================
            # CHART
            # =============================
            st.subheader("Sentiment Distribution")

            st.bar_chart(
                df["Sentiment"].value_counts().reindex(
                    ["POSITIVE", "NEUTRAL", "NEGATIVE"]
                )
            )

            # =============================
            # WORD CLOUD (MOBILE FIXED)
            # =============================
            st.subheader("Insight Cloud")

            text = " ".join(df["Cleaned"])

            if text.strip():

                wc = WordCloud(
                    background_color="black",
                    width=600,   # reduced for mobile
                    height=300,  # reduced for mobile
                    colormap="Reds"
                ).generate(text)

                fig, ax = plt.subplots()
                ax.imshow(wc)
                ax.axis("off")
                st.pyplot(fig)

            # =============================
            # DATA VIEW
            # =============================
            with st.expander("View Dataset"):
                st.dataframe(df)