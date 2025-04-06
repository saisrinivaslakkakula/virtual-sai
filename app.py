with open("virtual_sai_debug.log", "a") as f:
    f.write("[BOOT] Streamlit app.py started\n")
import streamlit as st
import os
from rag_brain import load_rag_engine, get_answer, log

log("🚀 Streamlit app starting...")

st.set_page_config(page_title="Virtual Sai", layout="centered")
st.title("💬 Ask Virtual Sai")

# 🔹 Persona Selection
persona = st.radio("Choose how Virtual Sai should respond:", [
    "🧑 Speak as Sai (First-person)",
    "📄 Speak about Sai (Third-person)"
])

# 🧠 Load Brain
if "rag" not in st.session_state:
    log("🔁 Initializing RAG brain...")
    st.session_state.rag, st.session_state.llm, st.session_state.embed = load_rag_engine()

# 📂 File Upload
st.markdown("### 📂 Upload New Resume or Work Journals")
uploaded_files = st.file_uploader(
    "Drop files here (PDF, DOCX, or TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join("data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        log(f"📁 Uploaded: {uploaded_file.name}")
    st.success(f"Uploaded {len(uploaded_files)} file(s) to `data/` folder.")

    if st.button("🔄 Refresh Virtual Sai’s Brain"):
        st.session_state.rag, st.session_state.llm, st.session_state.embed = load_rag_engine()
        st.success("🧠 Brain refreshed with latest documents.")
        log("🧠 Brain refreshed with uploaded documents.")

# 📌 Suggestions
st.markdown("#### 👋 Hi, I’m Sai. Ask me anything — or try one of these:")
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
