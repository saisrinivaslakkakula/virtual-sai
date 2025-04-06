import streamlit as st
from rag_brain import load_rag_engine, get_answer

st.set_page_config(page_title="Virtual Sai", layout="centered")
st.title("💬 Ask Virtual Sai")

persona = st.radio("Choose how Virtual Sai should respond:", [
    "🧑 Speak as Sai (First-person)",
    "📄 Speak about Sai (Third-person)"
])

if "rag" not in st.session_state:
    st.session_state.rag = load_rag_engine()

st.markdown("#### 👋 Hi, I’m Sai. Ask me anything — or try one of these:")

suggestions = {
    "🛠️ My Walmart Ads Project": "Can you describe your project on Meta campaign automation at Walmart?",
    "☁️ My Work at AWS": "Tell me about your work on AWS RDS and DB2 migration.",
    "🏆 Capgemini Innovation Award": "What was the Capgemini project that got nominated for the Aegis Graham Bell Award?",
}

for label, prompt in suggestions.items():
    if st.button(label):
        with st.spinner(f"Fetching: {label}..."):
            answer = get_answer(st.session_state.rag, prompt, persona)
            st.markdown(f"**You asked:** {prompt}")
            st.write("💬 " + answer)

st.markdown("### Ask your own question:")
question = st.text_input("Ask me anything about Sai's work experience:")

if question:
    with st.spinner("Thinking..."):
        answer = get_answer(st.session_state.rag, question, persona)
        st.markdown(f"**You asked:** {question}")
        st.write("💬 " + answer)
