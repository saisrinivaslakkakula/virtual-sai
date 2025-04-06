import streamlit as st
import os
import sys
import traceback
from rag_brain import load_rag_engine, get_answer

st.set_page_config(page_title="Virtual Sai", layout="centered")
st.title("💬 Ask Virtual Sai")

print("🚀 Streamlit app launched successfully.", flush=True)

# 🔹 Voice toggle
persona = st.radio("Choose how Virtual Sai should respond:", [
    "🧑 Speak as Sai (First-person)",
    "📄 Speak about Sai (Third-person)"
])

# 🧠 Initialize or refresh brain
if "rag" not in st.session_state:
    print("🧠 Brain not found in session. Initializing now...", flush=True)
    try:
        st.session_state.rag, st.session_state.llm, st.session_state.embed = load_rag_engine()
    except Exception as e:
        print("❌ Failed to initialize brain:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
        st.error("Virtual Sai failed to load. Please check logs.")

# 📤 Upload files
st.markdown("### 📂 Upload New Resume or Work Journals")
uploaded_files = st.file_uploader(
    "Drop files here (PDF, DOCX, or TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    print(f"📥 Detected {len(uploaded_files)} uploaded file(s). Saving to `data/`...", flush=True)
    try:
        for uploaded_file in uploaded_files:
            save_path = os.path.join("data", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} file(s) to `data/` folder.")
        print("✅ Files saved successfully.", flush=True)
    except Exception as e:
        print("❌ ERROR saving uploaded files:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
        st.error("Failed to save uploaded files.")

    if st.button("🔄 Refresh Virtual Sai’s Brain"):
        print("🔁 Refresh button clicked. Reloading brain...", flush=True)
        try:
            st.session_state.rag, st.session_state.llm, st.session_state.embed = load_rag_engine()
            st.success("🧠 Virtual Sai has been refreshed with the latest documents.")
        except Exception as e:
            print("❌ ERROR reloading brain:", e, flush=True)
            traceback.print_exc(file=sys.stdout)
            st.error("Brain reload failed. Check terminal output.")

# 📌 Smart Suggestions
st.markdown("#### 👋 Hi, I’m Sai. Ask me anything — or try one of these:")
suggestions = {
    "🛠️ My Walmart Ads Project": "Can you describe your project on Meta campaign automation at Walmart?",
    "☁️ My Work at AWS": "Tell me about your work on AWS RDS and DB2 migration.",
    "🏆 Capgemini Innovation Award": "What was the Capgemini project that got nominated for the Aegis Graham Bell Award?",
}

for label, prompt in suggestions.items():
    if st.button(label):
        print(f"👉 Suggestion clicked: {label}", flush=True)
        with st.spinner(f"Fetching: {label}..."):
            try:
                answer = get_answer(
                    st.session_state.rag,
                    prompt,
                    persona,
                    st.session_state.llm,
                    st.session_state.embed
                )
                st.markdown(f"**You asked:** {prompt}")
                st.write("💬 " + answer)
                print("✅ Suggestion answered successfully.", flush=True)
            except Exception as e:
                print(f"❌ Error while answering suggestion {label}:", e, flush=True)
                traceback.print_exc(file=sys.stdout)
                st.error("Failed to get a response.")

# 💬 Custom prompt
st.markdown("### Ask your own question:")
question = st.text_input("Ask me anything about Sai's work experience:")

if question:
    print(f"📝 Received custom question: {question}", flush=True)
    with st.spinner("Thinking..."):
        try:
            answer = get_answer(
                st.session_state.rag,
                question,
                persona,
                st.session_state.llm,
                st.session_state.embed
            )
            st.markdown(f"**You asked:** {question}")
            st.write("💬 " + answer)
            print("✅ Custom question answered successfully.", flush=True)
        except Exception as e:
            print("❌ ERROR answering custom question:", e, flush=True)
            traceback.print_exc(file=sys.stdout)
            st.error("Failed to get a response.")
