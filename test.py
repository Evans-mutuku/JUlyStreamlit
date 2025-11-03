import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="GPT-2 Chatbot", page_icon="🤖", layout="centered")

@st.cache_resource(show_spinner=True)
def load_generator():
    # Smallest simplest model; downloads once and reuses
    gen = pipeline("text-generation", model="openai-community/gpt2")
    # ensure padding token is valid for generation
    gen.tokenizer.pad_token = gen.tokenizer.eos_token
    return gen

SYSTEM = (
    "You are a helpful assistant for software engineering. "
    "Answer concisely and give short code examples when useful. "
    "If unsure, say you are unsure.\n\n"
)

def build_prompt(history, user_msg):
    # Very light chat-style prompt for gpt2 (not instruction-tuned)
    convo = []
    for u, a in history:
        convo.append(f"Question: {u}\nAnswer: {a}\n")
    convo.append(f"Question: {user_msg}\nAnswer:")
    return SYSTEM + "\n".join(convo)

st.title("🤖 GPT-2 Chatbot (Hugging Face)")
st.caption("Simple local demo using `openai-community/gpt2`. For better quality, swap to a small instruct model later.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    max_new = st.slider("Max new tokens", 20, 300, 120, 10)
    temp = st.slider("Temperature", 0.1, 1.0, 0.5, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    rep_pen = st.slider("Repetition penalty", 1.0, 2.0, 1.15, 0.05)
    if st.button("Clear chat"):
        st.session_state.history = []

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Chat display
for i, (user, bot) in enumerate(st.session_state.history):
    st.chat_message("user").markdown(user)
    st.chat_message("assistant").markdown(bot)

# Input box
user_msg = st.chat_input("Ask about software engineering…")
if user_msg:
    st.chat_message("user").markdown(user_msg)

    with st.spinner("Generating…"):
        generator = load_generator()
        prompt = build_prompt(st.session_state.history, user_msg)

        out = generator(
            prompt,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=rep_pen,
            pad_token_id=generator.tokenizer.eos_token_id,
            eos_token_id=generator.tokenizer.eos_token_id,
        )[0]["generated_text"]

        # Extract the assistant's answer after the last "Answer:"
        answer = out.split("Answer:")[-1].strip()
        # If model starts another "Question:", cut it off
        if "Question:" in answer:
            answer = answer.split("Question:")[0].strip()

    st.chat_message("assistant").markdown(answer)
    st.session_state.history.append((user_msg, answer))
