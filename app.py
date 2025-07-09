# app.py

import os
import json
import warnings
import streamlit as st
from datetime import datetime
from glob import glob

from news_pipeline_utils import (
    fetch_and_summarize,
    load_summary_db,
    semantic_filter_articles,
    answer_question,
    summarize_text,         # Make sure summarize_text is imported here
)

# --- Suppress warnings ---
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Utility: Date formatting ---
def format_published_date(published_at_str):
    try:
        dt = datetime.strptime(published_at_str, "%a, %d %b %Y %H:%M:%S %Z")
        return dt.strftime("%b %d, %Y")
    except Exception:
        return published_at_str if published_at_str else "N/A"

# --- Session State ---
if 'summary_db' not in st.session_state:
    st.session_state.summary_db = []

if 'summary_file' not in st.session_state:
    st.session_state.summary_file = None

if 'topic' not in st.session_state:
    st.session_state.topic = ""

# --- Streamlit UI Config ---
st.set_page_config(page_title="🧠 AI News Analyst", layout="wide")
st.title("🧠 AI News Analyst")
st.markdown("Summarize and analyze news articles locally — all offline. ✨")

# --- Sidebar ---
with st.sidebar:
    st.header("🛠️ Controls")
    topic = st.text_input("Enter a topic (e.g., AI, Economy, Climate)", value=st.session_state.topic)
    st.session_state.topic = topic

    max_articles = st.slider("Max number of articles to fetch", 1, 10, 5)
    refresh_clicked = st.button("🔄 Fetch & Summarize Fresh Articles")
    load_clicked = st.button("📥 Load Cached Articles")
    show_debug = st.checkbox("🧪 Show debug info")

    selected_summary = None
    summary_files = []
    if topic:
        topic_clean = topic.lower().replace(" ", "_")
        summary_files = sorted(glob(f"summary_db_{topic_clean}.json"), reverse=True)
        if summary_files:
            selected_summary = st.selectbox("📂 Select cached summary file", summary_files)

# --- Load or Refresh ---
summary_db = st.session_state.summary_db
summary_file = st.session_state.summary_file

def deduplicate_by_title(articles):
    seen = set()
    deduped = []
    for a in articles:
        title = a.get("title", "").strip().lower()
        if title and title not in seen:
            seen.add(title)
            deduped.append(a)
    return deduped

if topic:
    topic_clean = topic.lower().replace(" ", "_")
    filename = f"summary_db_{topic_clean}.json"

    if load_clicked and selected_summary:
        st.session_state.summary_file = selected_summary
        st.session_state.summary_db = load_summary_db(topic_clean)
        summary_db = st.session_state.summary_db
        st.success(f"✅ Loaded {len(summary_db)} articles from cache.")

        try:
            ts = os.path.getmtime(selected_summary)
            st.info(f"🕒 Last Updated: {datetime.fromtimestamp(ts).strftime('%b %d, %Y %I:%M %p')}")
        except:
            st.warning("⚠️ Timestamp not available.")

    elif refresh_clicked:
        with st.spinner("🔄 Fetching and summarizing articles..."):
            new_articles = fetch_and_summarize(topic, max_articles=max_articles)

            old_articles = load_summary_db(topic_clean)
            all_articles = old_articles + new_articles
            deduped_articles = deduplicate_by_title(all_articles)

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(deduped_articles, f, indent=2, ensure_ascii=False)

            st.session_state.summary_file = filename
            st.session_state.summary_db = deduped_articles
            summary_db = deduped_articles

        st.success(f"✅ Fetched and saved {len(new_articles)} new articles.")
        try:
            ts = os.path.getmtime(filename)
            st.info(f"🕒 Updated: {datetime.fromtimestamp(ts).strftime('%b %d, %Y %I:%M %p')}")
        except:
            st.warning("⚠️ Timestamp not available.")

    elif not load_clicked and not refresh_clicked:
        st.info("📢 Enter a topic and choose an action (Load or Refresh)")

# --- Show Summaries ---
if summary_db:
    st.header(f"📰 Summaries for Topic: `{topic}`")
    for i, article in enumerate(summary_db):
        with st.expander(f"{i+1}. {article['title']}", expanded=False):
            st.markdown(f"🔗 [Link to article]({article['url']})")

            published = format_published_date(article.get("published_at", "N/A"))
            source = article.get("source", "Unknown")

            st.markdown(f"📰 **Source:** {source}  |  📅 **Published:** {published}")

            text_words = len(article.get("full_text", "").split())
            summary_words = len(article.get("summary", "").split())
            st.markdown(f"📄 **Text Words:** {text_words}  |  📝 **Summary Words:** {summary_words}")

            st.markdown("### 📝 Summary:")
            st.write(article["summary"])

            if st.checkbox(f"📖 Show full text of article {i+1}", key=f"text_{i}"):
                st.markdown("### 📄 Full Text:")
                st.write(article["full_text"])

            # --- Resummarize button ---
            if st.button(f"🔄 Resummarize Article {i+1}", key=f"resum_{i}"):
                with st.spinner(f"Resummarizing '{article['title']}'..."):
                    new_summary = summarize_text(article["full_text"])
                    st.success("✅ Resummarization complete.")

                    # Update summary in session_state
                    st.session_state.summary_db[i]["summary"] = new_summary

                    # Update cache file immediately
                    try:
                        with open(filename, "w", encoding="utf-8") as f:
                            json.dump(st.session_state.summary_db, f, indent=2, ensure_ascii=False)
                        st.info(f"💾 Cache file updated: {filename}")
                    except Exception as e:
                        st.error(f"❌ Failed to update cache file: {e}")

            if show_debug:
                st.code(json.dumps(article, indent=2, ensure_ascii=False), language="json")
else:
    if topic and load_clicked:
        st.error("❌ No summary file found or file is empty.")
    elif topic and refresh_clicked:
        st.warning("⚠️ No usable articles were extracted.")

# --- Q&A ---
if summary_db:
    st.divider()
    st.header("❓ Ask a Question About These Articles")

    user_question = st.text_input("Type your question below:")
    if st.button("🧠 Get Answer") and user_question.strip():
        with st.spinner("Thinking..."):
            top_articles = semantic_filter_articles(user_question, summary_db)
            answers = answer_question(user_question, top_articles)

        st.subheader("📚 Top Relevant Answers")
        for ans in answers:
            st.markdown(f"### 📌 {ans['title']}")
            st.markdown(f"🔗 [Source]({ans['url']})")
            st.success(ans["answer"])
            if show_debug:
                st.info("Context was derived from summary and full text.")
else:
    st.info("📢 Load or fetch articles to enable the Q&A section.")
