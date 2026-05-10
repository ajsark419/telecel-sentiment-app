import matplotlib
matplotlib.use("Agg")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import re
import torch
from wordcloud import WordCloud
import matplotlib.pyplot as plt
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
# GLOBAL STYLES
# =============================
st.markdown("""
<style>

/* =============================
   MAIN LAYOUT
============================= */
.block-container {
    padding-top: 1rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

.element-container {
    margin-bottom: 0.5rem;
}

/* =============================
   TYPOGRAPHY
============================= */
h2 {
    font-size: clamp(18px, 2.2vw, 26px);
}

p {
    font-size: clamp(12px, 1.5vw, 14px);
}

/* =============================
   METRIC CARDS
============================= */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #2d3748;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 0 12px rgba(212, 0, 0, 0.18);
}

/* =============================
   SIDEBAR
============================= */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #d40000, #111827);
}

/* ALL SIDEBAR TEXT */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* RADIO LABELS */
section[data-testid="stSidebar"] .stRadio label {
    color: white !important;
    font-weight: 500;
}

/* SELECTED RADIO OPTION */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-baseweb="radio"] {
    background: rgba(255,255,255,0.06);
    padding: 10px;
    border-radius: 10px;
    border-left: 3px solid #ff1e1e;
    box-shadow: 0 0 12px rgba(255,0,0,0.18);
}

/* HOVER EFFECT */
section[data-testid="stSidebar"] .stRadio label:hover {
    color: #ff4d4d !important;
}

/* =============================
   BUTTONS
============================= */
.stButton > button {
    background: linear-gradient(90deg, #d40000, #111827);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-weight: bold;
}

.stButton > button:hover {
    opacity: 0.9;
}

/* =============================
   TEXT AREA
============================= */
textarea {
    border-radius: 10px !important;
}

/* =============================
   HEADER
============================= */
@keyframes pulseMove {
    0% {
        background-position: 200% 0;
    }

    100% {
        background-position: -200% 0;
    }
}

.telecel-header {

    background:
        linear-gradient(
            135deg,
            #0b1220,
            #111827,
            #1f2937
        );

    padding: 20px;

    border-radius: 16px;

    border:
        1px solid rgba(255,0,0,0.25);

    box-shadow:
        0 0 20px rgba(255,0,0,0.18);

    color: white;
}

.pulse-bar {

    margin-top: 14px;

    height: 3px;

    background:
        linear-gradient(
            90deg,
            transparent,
            #ff1e1e,
            transparent
        );

    background-size: 200% 100%;

    animation:
        pulseMove 2s linear infinite;
}

/* =============================
   MOBILE RESPONSIVE
============================= */
@media screen and (max-width: 768px) {

    .telecel-header {
        text-align: center;
        padding: 16px;
    }

    .telecel-header h2 {
        font-size: 20px !important;
    }

    .telecel-header p {
        font-size: 12px !important;
    }
}

@media screen and (max-width: 480px) {

    .telecel-header {
        padding: 14px;
        border-radius: 12px;
    }

    .pulse-bar {
        display: none;
    }
}

</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.markdown("""
<div class="telecel-header">

<h2 style="
    margin-bottom:6px;
    color:white;
">
    Telecel AI Sentiment Dashboard
</h2>

<p style="
    color:#cbd5f5;
    margin-top:0;
    margin-bottom:10px;
    font-size:14px;
">
    Customer Intelligence System powered by BERT AI
</p>

<p style="
    color:#ff4d4d;
    font-size:11px;
    letter-spacing:1px;
    font-weight:600;
    text-transform:uppercase;
    margin-bottom:0;
">
    Telecel Sentiment Analytics
</p>

<div class="pulse-bar"></div>

</div>
""", unsafe_allow_html=True)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def get_model():

    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device
    )

# =============================
# GHANAIAN SLANG DICTIONARY
# =============================
slang_dict = {

    # COMMON TERMS
    "chale": "friend",
    "charley": "friend",
    "massa": "man",
    "bossu": "boss",
    "bros": "brother",
    "sis": "sister",

    # EXPRESSIONS
    "dey": "is",
    "de3": "is",
    "go fit": "can",
    "fit": "can",
    "abi": "",
    "ankasa": "personally",
    "aswear": "seriously",
    "waa": "very",
    "paa": "very",
    "paaa": "very",
    "koraa": "",
    "ooo": "",
    "oo": "",
    "err": "",
    "er": "",
    "ah": "",
    "ei": "",
    "eii": "",
    "herr": "wow",
    "hmmm": "concern",
    "hmm": "concern",

    # POSITIVE
    "no bad": "good",
    "hard": "excellent",
    "mad": "excellent",
    "crazy": "excellent",
    "lit": "excellent",
    "solid": "good",
    "sharp": "fast",
    "clean": "good",
    "correct": "good",
    "top": "excellent",
    "wild": "excellent",
    "fire": "excellent",
    "too sure": "excellent",
    "sweet": "good",
    "sweet pass": "better",
    "e choke": "amazing",
    "choke": "amazing",
    "vhim": "energy",
    "vim": "energy",
    "dem force": "good",
    "really force": "excellent",

    # NEGATIVE
    "someway": "bad",
    "weytin be dis": "bad",
    "wtf": "bad",
    "nonsense": "bad",
    "useless": "bad",
    "trash": "bad",
    "borla": "waste",
    "fool": "stupid",
    "foolish": "stupid",
    "wahala": "problem",
    "yawa": "problem",
    "slow oo": "slow",
    "network no good": "bad network",
    "bad oo": "very bad",
    "kai": "frustration",
    "bore": "annoyed",
    "dem no force": "very bad",
    "dem no try": "bad",

    # INTERNET
    "pls": "please",
    "plsss": "please",
    "plz": "please",
    "u": "you",
    "ur": "your",
    "imo": "in my opinion",
    "idk": "i do not know",
    "lol": "funny",
    "lmao": "funny",
    "omg": "surprised",
    "smh": "disappointed",

    # PIDGIN
    "dem": "they",
    "am": "it",
    "naa": "yes",
    "ano": "no",
    "wey": "which",
    "wetin": "what",
    "sef": "self",
    "dey play": "joking",
    "dey worry": "causing problems",
    "dey stress": "causing stress",
    "dey slow": "slow",
    "dey work": "working",
    "dey try": "doing well",

    # TELECOM
    "bundle finish": "data exhausted",
    "dash me": "give me",
    "credit finish": "airtime exhausted",
    "network dey jam": "network issue",
    "network no dey": "network unavailable",
    "internet no fast": "slow internet",
    "call no go": "call failed",
    "data no dey work": "data not working",
    "sim dey worry": "sim issue",
    "telecel dey try": "telecel is good",
    "mtn better": "mtn is better",

    # LOCAL LANGUAGE
    "medaase": "thank you",
    "akwaaba": "welcome",
    "yoo": "okay",
    "yo": "okay",
    "daaabi": "never",
    "herh": "shock",
    "asem oo": "problem",

    # EMOJIS
    "😂": "funny",
    "🤣": "funny",
    "😭": "sad",
    "😡": "angry",
    "🔥": "excellent",
    "💔": "sad",
    "❤️": "love",
    "😍": "love",
    "🥹": "emotional",
    "🙏": "please"
}

# =============================
# PREPROCESSING
# =============================
def preprocess(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    for slang, replacement in slang_dict.items():
        text = text.replace(slang, replacement)

    return text

# =============================
# SENTIMENT PREDICTION
# =============================
def predict(text):

    if len(text.strip()) < 3:
        return "NEUTRAL", 0.50

    model = get_model()

    try:

        result = model(text[:512])[0]

        label = result["label"].lower()
        score = result["score"]

        if label == "positive":
            sentiment = "POSITIVE"

        elif label == "negative":
            sentiment = "NEGATIVE"

        else:
            sentiment = "NEUTRAL"

        return sentiment, score

    except:
        return "NEUTRAL", 0.50

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

    col1, col2, col3 = st.columns(3)

    col1.markdown("""
    ### 🤖 AI Model
    Twitter-RoBERTa NLP
    """)

    col2.markdown("""
    ### ⚡ Processing
    Real-Time Social AI
    """)

    col3.markdown("""
    ### 📊 Output
    Positive / Neutral / Negative
    """)

    st.markdown("---")

    st.info(
        "This system analyzes customer sentiment using "
        "Twitter-trained Transformer AI optimized for "
        "social media and telecom discussions."
    )

# =============================
# SINGLE TWEET
# =============================
elif menu == "🧠 Single Tweet":

    st.subheader("Tweet Sentiment Analyzer")

    text = st.text_area(
        "Enter Tweet",
        height=140,
        placeholder="Paste tweet here..."
    )

    if st.button("Analyze Tweet"):

        if text.strip() == "":
            st.warning("Please enter a tweet.")

        else:

            with st.spinner("🤖 AI analyzing sentiment..."):

                clean = preprocess(text)

                label, score = predict(clean)

            st.markdown("## Result")

            if label == "POSITIVE":
                st.success("😊 Positive Sentiment")

            elif label == "NEGATIVE":
                st.error("😠 Negative Sentiment")

            else:
                st.info("😐 Neutral Sentiment")

            st.metric(
                "Confidence Score",
                f"{round(score * 100, 2)}%"
            )

            with st.expander("Processed Tweet"):
                st.write(clean)

# =============================
# DATASET ANALYSIS
# =============================
elif menu == "📂 Dataset Analysis":

    st.subheader("Batch Sentiment Intelligence")

    file = st.file_uploader(
        "Upload CSV (must contain 'Tweets' column)",
        type=["csv"]
    )

    if file:

        df = pd.read_csv(file)

        if "Tweets" not in df.columns:

            st.error("Missing 'Tweets' column")

        else:

            with st.spinner("🤖 AI analyzing dataset..."):

                df["Cleaned"] = (
                    df["Tweets"]
                    .astype(str)
                    .apply(preprocess)
                )

                results = df["Cleaned"].apply(predict)

                df["Sentiment"] = results.apply(lambda x: x[0])
                df["Confidence"] = results.apply(lambda x: x[1])

            st.subheader("📊 Key Metrics")

            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)

            c1.metric("Total Tweets", len(df))

            c2.metric(
                "Positive",
                (df["Sentiment"] == "POSITIVE").sum()
            )

            c3.metric(
                "Neutral",
                (df["Sentiment"] == "NEUTRAL").sum()
            )

            c4.metric(
                "Negative",
                (df["Sentiment"] == "NEGATIVE").sum()
            )

            st.markdown("---")

            st.subheader("📈 Sentiment Distribution")

            sentiment_counts = (
                df["Sentiment"]
                .value_counts()
                .reindex(
                    ["POSITIVE", "NEUTRAL", "NEGATIVE"],
                    fill_value=0
                )
            )

            st.bar_chart(sentiment_counts)

            st.markdown("---")

            st.subheader("☁️ Insight Cloud")

            combined_text = " ".join(df["Cleaned"])

            if combined_text.strip():

                wc = WordCloud(
                    background_color="black",
                    width=900,
                    height=400,
                    colormap="Reds"
                ).generate(combined_text)

                fig, ax = plt.subplots(figsize=(10, 5))

                ax.imshow(wc)

                ax.axis("off")

                st.pyplot(fig)

            st.markdown("---")

            st.subheader("📋 Sentiment Results")

            st.dataframe(
                df[
                    [
                        "Tweets",
                        "Sentiment",
                        "Confidence"
                    ]
                ],
                use_container_width=True
            )

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Analyzed Dataset",
                data=csv,
                file_name="telecel_sentiment_results.csv",
                mime="text/csv"
            )