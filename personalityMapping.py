import os
import csv
import json
import streamlit as st
from datetime import datetime
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text
from langchain_community.chat_models import ChatOpenAI
import tiktoken  # for token counting

# ——————— 0. Setup history CSV ———————
HISTORY_FILE = "history.csv"
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "user_prompt", "extracted_output"])

# ——————— 1. Utility: Token & Cost Calculation ———————
def calculate_cost(model_name, tokens_used):
    pricing_per_1k = {
        "gpt-3.5-turbo": 0.0015,
        "gpt-4": 0.03,
        "gpt-4-32k": 0.06,
    }
    return (tokens_used / 1000) * pricing_per_1k.get(model_name, 0.0015)

def count_tokens(text: str, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# ——————— 2. Define Schema ———————
core_personality = Object(
    id="core_personality",
    description="Core features needed to understand a user's personality type",
    attributes=[
        Text(id="personality_traits", description="Explicit traits such as moody, introverted, playful", many=True),
        Text(id="interests",        description="Hobbies like music, travel, journaling",    many=True),
        Text(id="tone",             description="Emotional tone (e.g. casual, thoughtful)"),
        Text(id="communication_style", description="Style of expression: poetic, sarcastic, formal"),
    ],
)

enhanced_personality = Object(
    id="enhanced_personality",
    description="Additional features that help refine personality insights",
    attributes=[
        Text(id="decision_style",     description="Impulsive, deliberate, intuitive, etc."),
        Text(id="aesthetic_style",    description="Minimalist, quirky, elegant"),
        Text(id="lifestyle_keywords", description="Routine-oriented, chaotic, mindful", many=True),
        Text(id="emotional_keywords", description="Emotion words like anxious, grateful, excited", many=True),
        Text(id="values",             description="Core values like loyalty, freedom, family, ambition", many=True),
        Text(id="energy_keywords",    description="Indicators like energetic, calm, laid-back", many=True),
    ],
)

optional_context = Object(
    id="optional_context",
    description="Optional context for enrichment",
    attributes=[
        Text(id="zodiac_sign",               description="Zodiac sign or DOB if mentioned"),
        Text(id="age_group",                 description="Teen, adult, senior, etc."),
        Text(id="relationship_to_recipient", description="Friend, partner, sibling, etc."),
        Text(id="location",                  description="User or recipient's location"),
        Text(id="social_context",            description="Public, private, solo, surprise, etc."),
    ],
)

personality_schema = Object(
    id="personality_feature_extraction",
    description="Extract structured personality & context features from user prompt",
    attributes=[core_personality, enhanced_personality, optional_context],
)

# ——————— 3. Slang‐Interpretation Instruction ———————
slang_instruction = (
    "Interpret Gen Z/internet slang (e.g. 'blehh', 'lowkey', 'rizz') as personality signals—"
    "map them to mood, tone, interests, values, etc."
)

# ——————— 4. Initialize LLM + Kor Chain ———————
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = create_extraction_chain(
    llm,
    personality_schema,
    encoder_or_encoder_class="json"
)

# ——————— 5. Streamlit UI ———————
st.set_page_config(page_title="🧠 Personality Extractor", layout="wide")
st.title("🧠 Personality Feature Extractor")
st.caption("Enter any prompt to extract structured personality insights.")

user_input = st.text_area(
    "✍️ User prompt",
    height=150,
    placeholder="e.g. Looking for something chill for my blehh, lowkey artsy friend."
)

if st.button("Extract Features"):
    if not user_input.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Extracting…"):
            # inject slang instruction
            prompt = f"{slang_instruction}\n\nUser Input: {user_input}"
            result = chain.invoke(prompt)
            data = result["data"]

            # token & cost calc
            inp_toks = count_tokens(prompt)
            out_toks = count_tokens(str(result))
            tot_toks = inp_toks + out_toks
            cost     = calculate_cost("gpt-3.5-turbo", tot_toks)

            # append to CSV
            with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    user_input,
                    json.dumps(data, ensure_ascii=False)
                ])

        st.success("Done!")

        st.subheader("🔍 Extracted Data")
        st.json(data)

        st.subheader("💰 Token Usage & Cost")
        st.write(f"- **Input tokens:** {inp_toks}")
        st.write(f"- **Output tokens (approx):** {out_toks}")
        st.write(f"- **Total tokens:** {tot_toks}")
        st.write(f"- **Estimated cost:** ${cost:.5f} USD")
