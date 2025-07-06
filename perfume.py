import streamlit as st
import pandas as pd
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load your custom perfume dataset
perfume_df = pd.read_csv("fra_raw_data.csv", encoding="ISO-8859-1", quoting=csv.QUOTE_MINIMAL)
perfume_df['scent_description'] = perfume_df[[
    'mainaccord1', 'mainaccord2', 'mainaccord3', 'mainaccord4', 'mainaccord5']
].fillna('').agg(' '.join, axis=1)

# Load your custom personality mapping dataset
personality_df = pd.read_csv("personality_mapping.csv", encoding="ISO-8859-1", quoting=csv.QUOTE_MINIMAL)

# TF-IDF vectorizer setup for perfumes
vectorizer = TfidfVectorizer()
perfume_matrix = vectorizer.fit_transform(perfume_df["scent_description"])

# Streamlit UI
# Streamlit UI
st.title("Perfume Recommender")

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'traits' not in st.session_state:
    st.session_state.traits = []
if 'accords' not in st.session_state:
    st.session_state.accords = []
if 'last_recommendations' not in st.session_state:
    st.session_state.last_recommendations = []

user_input = st.chat_input("Describe the person you're gifting a perfume to or type a follow-up question...")

if user_input:
    st.session_state.conversation.append({"role": "user", "content": user_input})

    # Inject context-aware prompt
    messages = [
        {"role": "system", "content": "You are a friendly, expert scent stylist and gift advisor. Based on a personality description, help refine suggestions with each user message. Always follow up with smart questions like: 'Would you like it to be gender-specific?', 'Should it be more long-lasting?','Any budget limits?', 'Do you want a floral or woody profile?', etc. Your goal is to guide them to a better match."},
    ] + st.session_state.conversation

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        reply = response['choices'][0]['message']['content']
        st.session_state.conversation.append({"role": "assistant", "content": reply})

        # Display conversation
        for msg in st.session_state.conversation:
            if msg['role'] == 'user':
                st.chat_message("user").write(msg['content'])
            else:
                st.chat_message("assistant").write(msg['content'])

        # Extract scent accords (if any) and match
        if "Accords:" in reply:
            accords_start = reply.find("Accords:") + len("Accords:")
            accords_str = reply[accords_start:].strip().strip("[]")
            accords = [a.strip().lower() for a in accords_str.replace("'", "").split(",")]
            query_text = ' '.join(accords)
            query_vec = vectorizer.transform([query_text])
            sims = cosine_similarity(query_vec, perfume_matrix).flatten()
            perfume_df["similarity"] = sims
            top_matches = perfume_df.sort_values("similarity", ascending=False).head(5)

            st.session_state.last_recommendations = top_matches.to_dict(orient='records')

            st.markdown("### 🎁 Recommended Perfumes")
            for row in st.session_state.last_recommendations:
                st.write(f"**{row['Perfume']}** by *{row['Brand']}*")
                st.write(f"Main Accords: {row['scent_description']}")
                st.write(f"URL: {row['url']}")
                st.markdown("---")

            # Smart follow-up options after recommendation
            st.markdown("**Would you like to...**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Refine by gender"):
                    st.session_state.conversation.append({"role": "user", "content": "I'd like to refine by gender."})
            with col2:
                if st.button("Set price range"):
                    st.session_state.conversation.append({"role": "user", "content": "I'd like to filter by price range."})
            with col3:
                if st.button("Adjust longevity"):
                    st.session_state.conversation.append({"role": "user", "content": "I want to choose a long-lasting or subtle scent."})

    except Exception as e:
        st.error(f"Error: {e}")
