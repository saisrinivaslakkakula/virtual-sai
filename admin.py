import streamlit as st
import os
from rag_brain import load_rag_engine

st.set_page_config(page_title="Virtual Sai Admin", layout="centered")
st.title("🛡️ Virtual Sai Admin Panel")

st.markdown("This interface allows you to upload new documents and refresh Virtual Sai’s brain.")

uploaded_files = st.file_uploader(
    "Upload new resumes, journals, or documents:",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join("data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"✅ Uploaded {len(uploaded_files)} file(s) to the brain.")

# 🧠 Manual Refresh
if st.button("🔄 Refresh Virtual Sai's Brain"):
    rag = load_rag_engine()
    st.success("🧠 Virtual Sai has been refreshed with the latest data.")

# 📃 Current Files
st.markdown("### 🗂️ Current Files in `data/`")
data_files = os.listdir("data")
if data_files:
    for file in data_files:
        st.markdown(f"• `{file}`")
else:
    st.info("No documents found yet.")
