import streamlit as st
import openai

import os
openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM = {
    "role":"system",
    "content":(
        "You are an agentic,inventive, psychology-driven gift recommender. "
        "When the user gives you a prompt, you must decide if you know enough to recommend gifts. "
        "—If not, ask exactly follow-up questions required to comeup on result- not more than 5  (focus on anything that matters: occasion, personality, relationship, budget, preferences, etc.) \n"
        "Infer at least THREE underlying personality traits or core values from the user's description.\n"
        "—Once you’re ready, stop asking questions and output 3–5 gift ideas with a one-sentence rationale each."
    )
}

def chat(history):
    resp = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages=history,
        temperature=0.8,
        top_p=0.8,
    )
    return resp.choices[0].message["content"].strip()

if "history" not in st.session_state:
    st.session_state.history = [SYSTEM]

if "history" not in st.session_state:
    st.session_state.history = [SYSTEM]

st.title("🤖 AI Gift Recommender — Personality-Driven")

# --- Render full chat history -------------------------------
for msg in st.session_state.history[1:]:  # skip system prompt
    role = "user" if msg["role"] == "user" else "assistant"
    st.chat_message(role).write(msg["content"])

# --- Determine user prompt label ----------------------------
last_content = st.session_state.history[-1]["content"]
is_question = last_content.strip().endswith("?")
if is_question:
    prompt_text = "Your answer:"
else:
    prompt_text = "Ask for a refinement or another gift prompt:"

# --- Accept new user input ----------------------------------
user_input = st.chat_input(prompt_text)
if user_input:
    # append user message
    st.session_state.history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # get assistant reply
    assistant_reply = chat(st.session_state.history)
    st.session_state.history.append({"role": "assistant", "content": assistant_reply})
    st.chat_message("assistant").write(assistant_reply)

    # rerun to update UI
    st.rerun()

# --- Reset button -------------------------------------------
if st.button("Start Over"):
    st.session_state.clear()
    st.rerun()