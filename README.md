# 🧠 Knowledge Assistant

A RAG-powered personal knowledge assistant that answers questions from your own notes using ChromaDB, Sentence Transformers, and Google Gemini.

---

## 📌 What This Does

Upload your personal notes and ask questions about them. The assistant:
1. Cleans and chunks your notes into searchable pieces
2. Embeds them using `sentence-transformers` and stores in ChromaDB
3. When you ask a question, retrieves the most relevant chunks
4. Sends them to Gemini as context to generate an accurate answer
5. Remembers the conversation history for follow-up questions

---

## 📁 Files

```
knowledge-assistant/
├── assistant.py     # Main RAG pipeline — embed, store, retrieve, answer
├── notes.txt        # Sample knowledge base (AI/ML concepts)
├── .env.example     # API key template
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/Vishaal1409/knowledge-assistant.git
cd knowledge-assistant
pip install google-genai sentence-transformers chromadb python-dotenv
cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

---

## 🚀 Run

```bash
python assistant.py
```

Then ask questions like:
```
You: What is RAG?
You: How does ChromaDB work?
You: What are agents?
```

---

## 🛠 Tech Stack

- **Google Gemini** (`gemini-2.0-flash`) — answer generation
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — text embeddings
- **ChromaDB** — persistent vector database
- **python-dotenv** — secure API key loading

---

## 🔄 How the RAG Pipeline Works

```
notes.txt → clean → chunk → embed → ChromaDB
                                        │
User question → embed → search ChromaDB → top 3 chunks
                                                │
                                    Gemini (context + question)
                                                │
                                           Answer ✅
```

---

## 💡 What I Learned

- How to build a RAG pipeline from scratch
- Text cleaning, chunking, and embedding
- Storing and querying vectors with ChromaDB
- Grounding Gemini responses with retrieved context
- Managing conversation history for multi-turn chat

---

*Part of my Data Science (Agent) learning journey — hands-on before the capstone project.*
