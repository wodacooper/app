"""
Fixed PDF Chatbot - Handles Empty Answers
==========================================
This version has better error handling and fallbacks
"""

import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')


class FixedPDFChatbot:
    def __init__(self, pdf_path):
        print("🤖 Initializing PDF Chatbot...")
        
        # Load models
        print("📥 Loading models (this may take a minute first time)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=-1
        )
        
        # Process PDF
        print(f"📄 Processing PDF: {pdf_path}")
        self.text = self._extract_text(pdf_path)
        
        if not self.text.strip():
            raise ValueError("⚠️ No text extracted! PDF might be image-based. Use OCR.")
        
        print(f"   Extracted {len(self.text)} characters")
        
        self.chunks = self._chunk_text(self.text)
        print(f"   Created {len(self.chunks)} chunks")
        
        # Create vector store
        self.embeddings = self._create_embeddings(self.chunks)
        self.index = self._build_index(self.embeddings)
        
        print("✅ Chatbot ready!\n")
    
    def _extract_text(self, pdf_path):
        """Extract text from PDF"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def _chunk_text(self, text, chunk_size=800, overlap=150):
        """Chunk text with better boundaries"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to end at sentence boundary
            if end < len(text):
                # Look for period, question mark, or exclamation
                for punct in ['. ', '! ', '? ', '\n\n']:
                    last_punct = text[max(start, end-200):end].rfind(punct)
                    if last_punct != -1:
                        end = max(start, end-200) + last_punct + len(punct)
                        break
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 50:  
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _create_embeddings(self, chunks):
        """Generate embeddings"""
        return self.embedding_model.encode(
            chunks,
            show_progress_bar=False,
            convert_to_numpy=True
        )
    
    def _build_index(self, embeddings):
        """Build FAISS index"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        return index
    
    def _search_chunks(self, question, top_k=10):
        """Find relevant chunks"""
        q_embedding = self.embedding_model.encode([question], convert_to_numpy=True)
        distances, indices = self.index.search(q_embedding.astype('float32'), top_k)
        return [self.chunks[i] for i in indices[0]]
    
    def ask(self, question):
        """Ask a question with better error handling"""
        # Get relevant chunks
        relevant_chunks = self._search_chunks(question, top_k=5)
        
        # Try with different amounts of context
        for num_chunks in [3, 5, 1]:  # Try 3 first, then 5, then just 1
            context = " ".join(relevant_chunks[:num_chunks])
            
            # Limit context length
            if len(context) > 3000:
                context = context[:3000]
                # End at last complete sentence
                last_period = context.rfind('.')
                if last_period > 2500:
                    context = context[:last_period + 1]
            
            try:
                result = self.qa_pipeline(
                    question=question,
                    context=context,
                    handle_impossible_answer=True,
                    max_answer_len=150
                )
                
                answer = result['answer'].strip()
                
                # If we got a good answer, return it
                if answer and len(answer) > 2 and result['score'] > 0.1:
                    return {
                        'answer': answer,
                        'confidence': result['score'],
                        'source': context[:300] + '...',
                        'num_chunks_used': num_chunks
                    }
            
            except Exception as e:
                continue  # Try next configuration
        
        # If all attempts failed, return a helpful message
        # Show the user what context was searched
        return {
            'answer': f"I couldn't find a clear answer. Here's what I found: {context[:200]}...",
            'confidence': 0.0,
            'source': context[:300] + '...',
            'num_chunks_used': 5,
            'note': 'Try rephrasing your question or asking about specific details in the document.'
        }
    
    def ask_extractive(self, question):
        """Alternative: Just return the most relevant chunk"""
        chunks = self._search_chunks(question, top_k=1)
        return {
            'answer': chunks[0][:500],
            'type': 'extractive',
            'note': 'This is the most relevant section from the document'
        }
    
    def chat(self):
        """Interactive chat"""
        print("\n" + "="*70)
        print("💬 Chat Mode")
        print("Commands: 'quit', 'extract' (show relevant text), 'help'")
        print("="*70 + "\n")
        
        extract_mode = False
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'extract':
                extract_mode = not extract_mode
                mode = "EXTRACT" if extract_mode else "ANSWER"
                print(f"\n✓ Switched to {mode} mode\n")
                continue
            
            if question.lower() == 'help':
                print("\n💡 Tips:")
                print("  • Ask specific questions about the document")
                print("  • Use 'extract' mode to see relevant text directly")
                print("  • Example questions:")
                print("    - What is this document about?")
                print("    - Who is mentioned in the document?")
                print("    - What are the main points?\n")
                continue
            
            if not question:
                continue
            
            print("\n🤖 Chatbot: ", end="")
            
            if extract_mode:
                result = self.ask_extractive(question)
                print(f"\n{result['answer']}\n")
            else:
                result = self.ask(question)
                print(f"{result['answer']}")
                
                if result['confidence'] > 0:
                    confidence_emoji = "✅" if result['confidence'] > 0.5 else "⚠️"
                    print(f"   {confidence_emoji} Confidence: {result['confidence']:.2%}")
                
                if 'note' in result:
                    print(f"   💡 {result['note']}")
            
            print()


def main():
    """Run the chatbot"""
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Enter PDF path: ").strip()
    
    try:
        chatbot = FixedPDFChatbot(pdf_path)
        
        # Ask a few test questions
        print("="*70)
        print("Test Questions:")
        print("="*70)
        
        test_questions = [
            "What is this document about?",
            "What are the main topics covered?"
        ]
        
        for q in test_questions:
            print(f"\n❓ {q}")
            result = chatbot.ask(q)
            print(f"💡 {result['answer']}")
            if result['confidence'] > 0:
                print(f"   Confidence: {result['confidence']:.2%}")
        
        # Start chat
        print()
        start_chat = input("\nStart interactive chat? (y/n): ").strip().lower()
        if start_chat == 'y':
            chatbot.chat()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTry running diagnose_chatbot.py to identify the issue.")


if __name__ == "__main__":
    main()