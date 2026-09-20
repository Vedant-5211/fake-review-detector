"""
Fake Review Detector — Streamlit App
Run with:  streamlit run app.py
Loads the Logistic Regression model & TF-IDF vectorizer saved by the notebook.
"""

import re
import streamlit as st
import pandas as pd
import joblib, nltk
from nltk.corpus import stopwords

# ── Setup ────────────────────────────────────────────────────────────────
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
STOP_WORDS = set(stopwords.words("english"))

# Load the pre-trained model and vectorizer once at startup.
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ── Text-cleaning function (must match the notebook's pipeline) ──────────
def clean_text(text):
    """Lowercase, strip punctuation/numbers, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)

# ── Word-contribution explainability (reuses LR coefficients) ────────────
def get_word_contributions(review_text):
    """Return list of (word, contribution_score) for words in the vocab.
    Positive → pushes toward fake; negative → pushes toward genuine."""
    cleaned = clean_text(review_text)
    vocab = vectorizer.vocabulary_
    coefs = model.coef_[0]
    tfidf_vec = vectorizer.transform([cleaned])
    contributions = []
    for word in cleaned.split():
        if word in vocab:
            idx = vocab[word]
            score = float(tfidf_vec[0, idx] * coefs[idx])
            contributions.append((word, round(score, 4)))
    return contributions

# ── Authenticity score for a batch of reviews ────────────────────────────
def get_authenticity_score(reviews):
    """Return the percentage of reviews predicted as genuine."""
    cleaned = [clean_text(r) for r in reviews]
    vecs = vectorizer.transform(cleaned)
    preds = model.predict(vecs)
    return (preds == 0).sum() / len(preds) * 100

def predict_review(text):
    """Return (label_str, confidence_%, proba_array) for a single review."""
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    return ("Fake" if pred == 1 else "Genuine"), proba[pred] * 100, proba

# ── Page config & header ─────────────────────────────────────────────────
st.set_page_config(page_title="Fake Review Detector", page_icon="🔍")
st.title("🔍 Fake Review Detector")
st.caption("Paste a product review and find out if it looks genuine or computer-generated.")

# ── Two modes via tabs ───────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["Single Review", "Check a Product's Reviews (Batch)"])

# ═══════════════════════════  SINGLE REVIEW  ═════════════════════════════
with tab_single:
    review_input = st.text_area("Paste a review here:", height=120, key="single")
    if st.button("Check Review", key="btn_single"):
        if not review_input.strip():
            st.warning("Please paste a review first.")
        else:
            label, confidence, proba = predict_review(review_input)

            # 1 — Verdict with colour
            if label == "Genuine":
                st.success(f"✅ **Verdict: Genuine** — {confidence:.1f}% confidence")
            else:
                st.error(f"🚩 **Verdict: Fake** — {confidence:.1f}% confidence")

            # 2 — Confidence breakdown bar
            st.markdown("**Confidence breakdown**")
            col1, col2 = st.columns(2)
            col1.metric("Genuine", f"{proba[0]*100:.1f}%")
            col2.metric("Fake", f"{proba[1]*100:.1f}%")
            st.progress(float(proba[0]))  # bar length = genuine probability

            # 3 — Word-highlighted review
            contribs = get_word_contributions(review_input)
            contrib_dict = dict(contribs)
            # Build HTML: colour each word by its contribution direction.
            html_parts = []
            for word in review_input.split():
                clean_w = re.sub(r"[^a-z]", "", word.lower())
                if clean_w in contrib_dict:
                    score = contrib_dict[clean_w]
                    if score > 0:
                        bg = "rgba(255,80,80,0.25)"   # light red  → fake
                    else:
                        bg = "rgba(80,200,80,0.25)"    # light green → genuine
                    html_parts.append(
                        f'<span style="background:{bg};padding:2px 4px;border-radius:3px">{word}</span>'
                    )
                else:
                    html_parts.append(word)
            st.markdown("**Word-level highlights** (🟢 genuine · 🔴 fake)")
            st.markdown(" ".join(html_parts), unsafe_allow_html=True)

            # 4 — Top-5 words table (fake + genuine)
            sorted_contribs = sorted(contribs, key=lambda x: x[1], reverse=True)
            top_fake    = sorted_contribs[:5]
            top_genuine = sorted_contribs[-5:][::-1]  # most negative = most genuine
            table_df = pd.DataFrame({
                "Top Fake-Leaning Words":    [f"{w} ({s:+.4f})" for w, s in top_fake],
                "Top Genuine-Leaning Words": [f"{w} ({s:+.4f})" for w, s in top_genuine],
            })
            st.table(table_df)

            # 5 — Details expander with heuristic side-notes
            with st.expander("Details"):
                wc = len(review_input.split())
                has_nums = bool(re.search(r"\d", review_input))
                st.write(f"**Word count:** {wc} · **Contains numbers/specifics:** {'Yes' if has_nums else 'No'}")
                if wc < 15: st.write("ℹ️ Very short review — harder to judge reliably.")
                if not has_nums: st.write("ℹ️ No numbers/quantities — review relies on generic adjectives.")

# ═══════════════════════════  BATCH MODE  ════════════════════════════════
with tab_batch:
    batch_input = st.text_area("Paste multiple reviews, one per line:", height=180, key="batch")
    if st.button("Check All", key="btn_batch"):
        lines = [l.strip() for l in batch_input.strip().splitlines() if l.strip()]
        if not lines:
            st.warning("Please paste at least one review (one per line).")
        else:
            auth_score = get_authenticity_score(lines)
            st.markdown(f"### Review Authenticity Score: **{auth_score:.1f}% genuine**")
            st.progress(float(auth_score / 100))

            # Per-review table
            rows = []
            for review in lines:
                label, confidence, _ = predict_review(review)
                rows.append({
                    "Review": review[:100] + ("…" if len(review) > 100 else ""),
                    "Prediction": label,
                    "Confidence": f"{confidence:.1f}%",
                })
            st.table(pd.DataFrame(rows))
