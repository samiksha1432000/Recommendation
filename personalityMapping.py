import os
import streamlit as st
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text
from langchain_community.chat_models import ChatOpenAI

# ——————— 1. Core Personality Features ———————
core_personality = Object(
    id="core_personality",
    description="Core features needed to understand a user's personality type",
    attributes=[
        Text(
            id="personality_traits",
            description="Explicit traits such as moody, introverted, playful",
            many=True,
        ),
        Text(
            id="interests",
            description="Hobbies like music, travel, journaling",
            many=True,
        ),
        Text(
            id="tone",
            description="Emotional tone of the prompt (e.g. casual, thoughtful)",
        ),
        Text(
            id="communication_style",
            description="Style of expression: poetic, sarcastic, formal",
        ),
    ],
)

# ——————— 2. Enhanced (Good‑to‑Have) Features ———————
enhanced_personality = Object(
    id="enhanced_personality",
    description="Additional features that help refine personality insights",
    attributes=[
        Text(
            id="decision_style",
            description="Impulsive, deliberate, intuitive, etc.",
        ),
        Text(
            id="aesthetic_style",
            description="Minimalist, quirky, elegant",
        ),
        Text(
            id="lifestyle_keywords",
            description="Routine-oriented, chaotic, mindful",
            many=True,
        ),
        Text(
            id="emotional_keywords",
            description="Emotion words like anxious, grateful, excited",
            many=True,
        ),
        Text(
            id="values",
            description="Core values like loyalty, freedom, family, ambition",
            many=True,
        ),
        Text(
            id="energy_keywords",
            description="Indicators like energetic, calm, laid-back",
            many=True,
        ),
    ],
)

# ——————— 3. Optional Contextual Fields ———————
optional_context = Object(
    id="optional_context",
    description="Optional context for enrichment",
    attributes=[
        Text(
            id="zodiac_sign",
            description="Zodiac sign or DOB if mentioned",
        ),
        Text(
            id="age_group",
            description="Teen, adult, senior, etc.",
        ),
        Text(
            id="relationship_to_recipient",
            description="Friend, partner, sibling, etc.",
        ),
        Text(
            id="location",
            description="User or recipient's location",
        ),
        Text(
            id="social_context",
            description="Public, private, solo, surprise, etc.",
        ),
    ],
)

# ——————— 4. Combine into Main Schema ———————
personality_schema = Object(
    id="personality_feature_extraction",
    description="Extract structured personality & context features from user prompt",
    attributes=[core_personality, enhanced_personality, optional_context],
)

# ——————— 5. Initialize LLM + Kor Chain ———————
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = create_extraction_chain(llm, personality_schema, encoder_or_encoder_class="json")

# ——————— 6. Streamlit UI ———————
st.set_page_config(page_title="🧠 Personality Feature Extractor", layout="wide")
st.title("🧠 Personality Feature Extractor")
st.caption("Enter any prompt to extract structured personality insights.")

user_input = st.text_area(
    "✍️ User prompt",
    height=150,
    placeholder="e.g. Looking for something calming and cozy for my introverted best friend who loves painting and journaling."
)

if st.button("Extract Features"):
    if not user_input.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Extracting…"):
            result = chain.invoke(user_input)
            data = result["data"]

        st.success("Done!")

        st.subheader("Full Extracted Data")
        st.json(data)

