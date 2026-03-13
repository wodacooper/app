st.set_page_config(page_title="PDF Contract Q&A", layout="wide")

from io import BytesIO
import streamlit as st
from pathlib import Path
from rag import PDFRagSystem

st.write("✅ Streamlit app started")


st.title("📄 PA Answering (Local)")

st.markdown("""
Upload a folder of PA's and automatically extract key details.
Runs fully locally using Ollama.
""")

pdf_path = st.text_input("📁 Path to folder containing PA's", value=str(Path("PA's").resolve()))

if "rag" not in st.session_state:
    st.session_state.rag = None

if st.button("🔄 Index Documents"):
    if not Path(pdf_path).exists():
        st.error("Folder not found")
    else:
        with st.spinner("Indexing documents..."):
            st.session_state.rag = PDFRagSystem(pdf_path)
        st.success("Documents indexed successfully!")

if st.session_state.rag:
    st.divider()

    st.header("📋 Run Standard Contract Questions")

    if st.button("▶ Run Questions"):
        rag = st.session_state.rag

        with st.spinner("Running standard questions and generating Excel..."):
            df = rag.ask_suggested_questions_per_document()

        st.session_state.standard_results = df
        
        st.success("Extraction complete!")

        if st.session_state.standard_results is not None:
            df = st.session_state.standard_results
            output = BytesIO()
            df.to_excel(output, index=False, engine="openpyxl")
            output.seek(0)

            st.download_button(
                label="📥 Download Extracted Excel",
                data=output,
                file_name="to_excel.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.divider()

    st.header("❓ Ask Your Own Question")

    user_q = st.text_input("Type a question")

    if st.button("Ask"):
        answer = st.session_state.rag.ask(user_q, "")
        st.markdown("### ✅ Answer")
        st.markdown(answer)
