import os
import json
import re
import time
from urllib.parse import quote_plus, urlparse
from datetime import datetime

import feedparser
from newspaper import Article
import trafilatura

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# === Load Models ===
summarizer = pipeline("text2text-generation", model="google/flan-t5-base")
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Domain Blocking ===
def is_valid_news_domain(url):
    bad_domains = [
        "cloudflare.com", "facebook.com", "linkedin.com", "instagram.com",
        "forbes.com/sites", "subscribe.bloomberg.com", "t.co", "youtube.com",
        "msn.com"
    ]
    parsed = urlparse(url)
    full_domain = parsed.netloc + parsed.path
    return not any(bad in full_domain for bad in bad_domains)

# === Relevance Filtering ===
def is_content_relevant(text, title, min_words=100, keyword_match_threshold=0.2):
    if not text or len(text.split()) < min_words:
        return False
    if title:
        title_words = set(re.findall(r'\b\w+\b', title.lower()))
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        common = title_words & text_words
        score = len(common) / len(title_words) if title_words else 0
        return score >= keyword_match_threshold
    return True

# === RSS Fetching ===
def get_bing_news_rss(topic, max_articles=5):
    query = quote_plus(topic)
    rss_url = f"https://www.bing.com/news/search?q={query}&format=rss"
    feed = feedparser.parse(rss_url)

    articles = []
    seen = set()

    for entry in feed.entries:
        match = re.search(r'url=(https?%3[a-zA-Z0-9%._\-\/]+)', entry.link)
        if not match:
            continue
        url = re.sub(r'%3a', ':', match.group(1), flags=re.IGNORECASE)
        url = re.sub(r'%2f', '/', url, flags=re.IGNORECASE)
        if url in seen:
            continue
        seen.add(url)
        articles.append({
            "title": entry.title,
            "url": url,
            "published_at": entry.get("published", ""),
            "source": "Bing",
            "topic": topic
        })
        if len(articles) >= max_articles:
            break
    return articles

def get_yahoo_rss_url(topic):
    curated = {
        "tech": "https://news.yahoo.com/rss/tech",
        "world": "https://news.yahoo.com/rss/world",
        "science": "https://news.yahoo.com/rss/science",
        "business": "https://news.yahoo.com/rss/business",
        "health": "https://news.yahoo.com/rss/health",
        "us": "https://news.yahoo.com/rss/us",
        "politics": "https://news.yahoo.com/rss/politics",
        "sports": "https://news.yahoo.com/rss/sports"
    }

    topic_clean = topic.lower().strip()
    if topic_clean in curated:
        return curated[topic_clean]  # Use curated RSS

    # Fallback to search-based RSS feed
    query = quote_plus(topic)
    return f"https://news.search.yahoo.com/rss?p={query}"

def get_yahoo_news_rss(topic, max_articles=5):
    rss_url = get_yahoo_rss_url(topic)
    feed = feedparser.parse(rss_url)

    print(f"📡 Yahoo RSS feed entries count: {len(feed.entries)}")

    articles = []
    seen = set()

    for entry in feed.entries:
        url = entry.link
        if url in seen or not is_valid_news_domain(url):
            continue
        seen.add(url)
        articles.append({
            "title": entry.title,
            "url": url,
            "published_at": entry.get("published", ""),
            "source": "Yahoo",
            "topic": topic
        })
        if len(articles) >= max_articles:
            break

    print(f"✅ Yahoo RSS filtered articles count: {len(articles)}")
    return articles


def fetch_articles_for_topic(topic, max_articles=10):
    bing_raw = get_bing_news_rss(topic, max_articles)
    bing = [a for a in bing_raw if is_valid_news_domain(a["url"])]
    blocked_bing = len(bing_raw) - len(bing)

    remaining = max_articles - len(bing)
    if remaining > 0:
        print(f"🔁 Bing returned {len(bing)} valid articles. Fetching {remaining} more from Yahoo.")
        yahoo_raw = get_yahoo_news_rss(topic, remaining)
        yahoo = [a for a in yahoo_raw if is_valid_news_domain(a["url"])]
        print(f"🔁 Yahoo returned {len(yahoo)} valid articles.")
    else:
        print(f"✅ Bing provided enough valid articles. Skipping Yahoo.")
        yahoo = []

    combined = bing + yahoo

    # Deduplicate by URL
    seen_urls = set()
    deduped = []
    for art in combined:
        if art["url"] not in seen_urls:
            deduped.append(art)
            seen_urls.add(art["url"])
        if len(deduped) >= max_articles:
            break

    print(f"\n🔍 Topic: {topic}")
    print(f"✅ Articles fetched (deduplicated): {len(deduped)}")
    if blocked_bing:
        print(f"🚫 Filtered out {blocked_bing} blocked Bing articles.")
    print()
    for i, art in enumerate(deduped, 1):
        print(f"{i}. {art['title']}\n🔗 {art['url']}\n📰 Source: {art['source']} | 📅 {art['published_at']} | 📌 Topic: {art['topic']}\n")

    return deduped

# === Extraction ===
def extract_article_text(url, title_hint=None):
    try:
        if not is_valid_news_domain(url):
            print(f"🚫 Blocked domain: {url}")
            return "Blocked Domain", None

        title = title_hint or "Untitled Article"
        np_text = ""
        tf_text = ""

        try:
            article = Article(url)
            article.download()
            article.parse()
            title = article.title or title_hint or "Untitled Article"
            np_text = article.text.strip()
            if np_text:
                print(f"📄 Newspaper3k: {len(np_text.split())} words")
        except Exception as e:
            print(f"❌ Newspaper3k failed: {e}")

        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                tf_text = trafilatura.extract(downloaded) or ""
                if tf_text:
                    print(f"📄 Trafilatura: {len(tf_text.split())} words")
        except Exception as e:
            print(f"❌ Trafilatura failed: {e}")

        if np_text and is_content_relevant(np_text, title):
            return title, np_text
        if tf_text and len(tf_text.split()) > 150:
            return title, tf_text

        print(f"⚠️ Both tools returned poor content for: {url}")
        return title, None

    except Exception as e:
        print(f"❌ Extraction error for {url}:\n{e}")
        return "Failed to fetch title", None

# === Summarization Helpers ===
def chunk_text(text, max_words=380):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence.split())
        if current_len + sentence_len <= max_words:
            current_chunk.append(sentence)
            current_len += sentence_len
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = sentence_len
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def clean_repetitions(text):
    lines = text.splitlines()
    seen = set()
    deduped = []
    for line in lines:
        stripped = line.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            deduped.append(stripped)
    return "\n".join(deduped)

# === Summarization ===

def summarize_text(text, min_words=100, max_words=380):
    chunks = chunk_text(text, max_words=max_words)
    summaries = []

    for chunk in chunks:
        chunk = clean_repetitions(chunk)
        prompt = (
            "You are a professional news analyst. Write a concise, factual summary of the article below "
            "in around 200 words. Avoid repeating any sentences or phrases. Focus on the most important "
            "facts, events, and implications.\n\n"
            f"{chunk}"
        )

        try:
            result = summarizer(
                prompt,
                max_new_tokens=300,
                min_new_tokens=min_words,
                do_sample=False,
                num_beams=4,
                repetition_penalty=1.3,
                temperature=0.7,
                top_p=0.9,
                early_stopping=True,
            )[0]["generated_text"]
            summaries.append(result.strip())
        except Exception as e:
            print(f"⚠️ Summarization error: {e}")

    combined_summary = " ".join(summaries)

    # Optional compression pass
    if len(combined_summary.split()) > 250:
        final_prompt = (
            "You are a professional news analyst. Summarize the following content in 200–250 words, "
            "avoiding repetition. Highlight the key facts and insights clearly and concisely.\n\n"
            f"{combined_summary}"
        )
        try:
            combined_summary = summarizer(
                final_prompt,
                max_new_tokens=300,
                min_new_tokens=150,
                do_sample=False,
                num_beams=4,
                repetition_penalty=1.3,
                temperature=0.7,
                top_p=0.9,
                early_stopping=True,
            )[0]["generated_text"].strip()
        except Exception as e:
            print(f"⚠️ Final summarization error: {e}")

    return combined_summary

# === Orchestration ===
def fetch_and_summarize(topic, max_articles=5, save_path=None):
    articles = fetch_articles_for_topic(topic, max_articles)
    summary_db = []

    for a in articles:
        url = a["url"]
        title = a["title"]
        print(f"🔍 Extracting: {title}")
        title, full_text = extract_article_text(url, title_hint=title)

        if not full_text:
            print(f"❌ Skipped (no usable content): {url}")
            continue

        summary = summarize_text(full_text)
        summary_db.append({
            "title": title,
            "topic": topic,
            "url": url,
            "summary": summary,
            "full_text": full_text,
            "source": a.get("source", "Unknown"),
            "published_at": a.get("published_at", "N/A")
        })

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary_db, f, indent=2, ensure_ascii=False)

    return summary_db

def load_summary_db(topic):
    fname = f"summary_db_{topic.lower()}.json"
    if os.path.exists(fname):
        with open(fname, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# === QA ===
def semantic_filter_articles(question, summary_db, top_n=3, min_score=0.3):
    entries = [f"{a['title']}. {a['summary']}" for a in summary_db]
    embeddings = semantic_model.encode([question] + entries, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(embeddings[0], embeddings[1:])[0]

    top_indices = [i.item() for i in scores.argsort(descending=True)[:top_n] if scores[i] > min_score]
    return [summary_db[i] for i in top_indices]

def answer_question(question, articles):
    answers = []

    for article in articles:
        summary = article.get("summary", "")
        full_text = article.get("full_text", "")

        # Use summary only, unless it's too short
        context = summary
        if len(summary.split()) < 50 and full_text:
            context = summary + "\n\n" + full_text

        # Truncate to max 800 words to fit in model context
        if len(context.split()) > 800:
            context = " ".join(context.split()[:800])

        prompt = f"""You are a concise and factual news analyst.
Avoid repeating sentences. If the context does not contain enough information to answer, say "Insufficient data."

### Question:
{question}

### Context:
{context}

### Answer:"""

        try:
            result = qa_model(
                prompt,
                max_new_tokens=200,         # Reduced max
                do_sample=False,
                repetition_penalty=1.2,     # Added to discourage repeats
                temperature=0.7             # Slight creativity
            )[0]['generated_text']
        except Exception as e:
            print(f"❌ QA error: {e}")
            result = "(Error while answering)"

        answers.append({
            "title": article["title"],
            "url": article["url"],
            "answer": result.strip()
        })

    return answers
