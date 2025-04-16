with open("virtual_sai_debug.log", "a") as f:
    f.write("[BOOT] Streamlit app.py started\n")
import streamlit as st
import os
from rag_brain import load_rag_engine, get_answer, log

log("🚀 Streamlit app starting...")

st.set_page_config(page_title="Virtual Sai", layout="centered")
st.title("💬 sAI")

# 🔹 Persona Selection
persona = st.radio("Choose how Virtual Sai should respond:", [
    "🧑 Speak as Sai (First-person)",
    "📄 Speak about Sai (Third-person)"
])

# 🧠 Load Brain
if "rag" not in st.session_state:
    log("🔁 Initializing RAG brain...")
    st.session_state.rag, st.session_state.llm, st.session_state.embed = load_rag_engine()


# 📌 Suggestions
st.markdown("#### 👋 Hi, I’m Virtual Sai. Ask me anything — or try one of these:")
suggestions = {
    "🛠️ My Walmart Ads Project": "Can you describe your project on Meta campaign automation at Walmart?",
    "☁️ My Work at AWS": "Tell me about your work on AWS RDS and DB2 migration.",
    "🏆 Capgemini Innovation Award": "What was the Capgemini project that got nominated for the Aegis Graham Bell Award?",
}

for label, prompt in suggestions.items():
    if st.button(label):
        with st.spinner(f"Fetching: {label}..."):
            log(f"📌 Button clicked: {label}")
            answer = get_answer(
                st.session_state.rag,
                prompt,
                persona,
                st.session_state.llm,
                st.session_state.embed
            )
            st.markdown(f"**You asked:** {prompt}")
            st.write("💬 " + answer)

# 🔍 Custom Query
st.markdown("### Ask your own question:")
question = st.text_input("Ask me anything about Sai's work experience:")

if question:
    with st.spinner("Thinking..."):
        log(f"🔍 Custom Question: {question}")
        answer = get_answer(
            st.session_state.rag,
            question,
            persona,
            st.session_state.llm,
            st.session_state.embed
        )
        st.markdown(f"**You asked:** {question}")
        st.write("💬 " + answer)
