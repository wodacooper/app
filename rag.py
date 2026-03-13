
import sys
from pathlib import Path
from typing import List, Tuple
from altair import value
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd
import re
import streamlit as st
import json


try:
    import PyPDF2
    from sentence_transformers import SentenceTransformer
    import ollama
    import chromadb
except ImportError as e:
    print(f"Missing required package: {e}")
    print("\nPlease install required packages:")
    sys.exit(1)


class PDFRagSystem:
    def __init__(self, pdf_dir: str, model_name: str = "llama3.2"):

        self.pdf_dir = Path(pdf_dir)
        self.model_name = model_name

        if not self.pdf_dir.exists() or not self.pdf_dir.is_dir():
            raise ValueError(f"Invalid PDF directory: {pdf_dir}")

        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        print("Initializing Vector Database...")
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="pdf_docs")

        self._load_and_index_pdfs()

        
    def _load_and_index_pdfs(self):
        pdf_files = list(self.pdf_dir.glob("*.pdf"))

        if not pdf_files:
            raise ValueError("No PDF files found in directory")

        print(f"Found {len(pdf_files)} PDF files")

        all_chunks = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []

        chunk_id = 0

        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path.name}")

            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if not page_text:
                        continue

                    chunks = self._chunk_text(page_text)

                    embeddings = self.embedder.encode(chunks)

                    for i, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        all_embeddings.append(embeddings[i].tolist())
                        all_metadatas.append({
                            "source": pdf_path.name,
                            "page": page_num + 1
                        })
                        all_ids.append(f"chunk_{chunk_id}")
                        chunk_id += 1

        print(f"Indexing {len(all_chunks)} chunks...")
        self.collection.add(
            documents=all_chunks,
            embeddings=all_embeddings,
            metadatas=all_metadatas,
            ids=all_ids
        )

        print("Indexing complete!")
        
    def _chunk_text(self, text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def retrieve_relevant_chunks(
        self,
        query: str,
        n_results: int = 3,
        source: str | None = None
    ):
        query_embedding = self.embedder.encode([query])

        where_filter = {"source": source} if source else None

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=where_filter
        )

        return results


    
    def ask(
        self,
        question: str,
        field_name: str,
        source: str | None = None,
        n_context_chunks: int = 3
    ) -> str:

        results = self.retrieve_relevant_chunks(
            question,
            n_results=n_context_chunks,
            source=source
        )

        context_blocks = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            context_blocks.append(
                f"Source: {meta['source']} (Page {meta['page']})\n{doc}"
            )

        context = "\n\n".join(context_blocks)

        source_note = f'from "{source}"' if source else "from all documents"

        prompt = f"""
    Based on the following context {source_note}, answer the question. If the answer is not in the context, say so.
    After the answer have valid JSON in this exact format:
    {{"{field_name}": "value"}}
    
    Context:
    {context}

    Question: {question}

    Answer:
    """

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]


    
    def get_suggested_questions(self) -> List[str]:
        
        return [
            "What is the \"purchase price\" of the property?",
            "What is the \"address\" of the property being sold?",
            "What is the \"initial earnest deposit\" amount?",
            "What are the \"non-refundable amounts\" and dates mentioned?",
            "What is the \"award deposit amount\" upon award?",
            "What is the \"closing date\" for the property?",
            "What is the \"extensions\" to the contract and how much do they cost? (additional payments?)",
            "Any \"NEPA extensions\" mentioned? If yes, what are the details?"
        ]

    def get_document_names(self):
        return sorted(
            {meta["source"] for meta in self.collection.get()["metadatas"]}
        )

    def ask_suggested_questions_per_document(self, output="to_excel.xlsx"):
        questions = self.get_suggested_questions()
        documents = self.get_document_names()

        rows = []
        
        for doc in documents:
            st.markdown(f"**Document:** {doc}")
            print(f"\n📄 Document: {doc}")
            print("=" * 60)

            row = {"Document": doc}

            for q in questions:
                matches = re.findall(r'"(.*?)"', q)
                print(f"\n❓ {q}")
                answer = self.ask(q, field_name=matches[0] if matches else "", source=doc)
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {answer}")
                st.markdown("---")
                print(f"✅ {answer}\n")
                key, value = self.extract_json_field(answer)
                if key:  
                    row[key] = value
                
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_excel(output, index=False)
        print(f"\n📊 Data saved to {output}")
        return df

    def extract_json_field(self, answer: str):

        match = re.search(r'\{.*?\}', answer, re.DOTALL)
        if not match:
            return (None, "")
        try:
            data = json.loads(match.group())
            if data:
                key = next(iter(data.keys()))
                value = data[key]
                return key, value
        except Exception:
            pass
        
        return (None, "")