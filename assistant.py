import os
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

# ── Load API key ──────────────────────────────────────────
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── M1: Clean text ────────────────────────────────────────
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── M4: Chunk the notes ───────────────────────────────────
def chunk_text(text, chunk_size=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# ── M2+M5: Embed and store in ChromaDB ───────────────────
print("Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path="./memory_db")
collection = chroma_client.get_or_create_collection("my_notes")

# Load and process notes only if DB is empty
if collection.count() == 0:
    print("Reading and storing your notes...")
    with open("notes.txt", "r") as f:
        raw_text = f.read()

    # Clean + chunk
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned, chunk_size=30)

    # Embed + store
    embeddings = embed_model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"Stored {len(chunks)} chunks into ChromaDB ✓")
else:
    print(f"Loaded existing knowledge base with {collection.count()} chunks ✓")

# ── M9: Session memory ────────────────────────────────────
conversation_history = []

# ── M7/M8: Search tool ───────────────────────────────────
def search_notes(query):
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    return results['documents'][0]

# ── M3/M6: Gemini answers using retrieved context ─────────
def ask_assistant(user_question):
    # Step 1 — retrieve relevant chunks
    relevant_chunks = search_notes(user_question)
    context = "\n".join(relevant_chunks)

    # Step 2 — add to memory
    conversation_history.append({
        "role": "user",
        "content": user_question
    })

    # Step 3 — ask Gemini with context
    prompt = f"""
You are a helpful assistant that answers questions based on the user's personal notes.

Relevant notes:
{context}

Conversation so far:
{conversation_history}

Answer the user's question using only the notes provided.
If the answer isn't in the notes, say "I couldn't find that in your notes."

Question: {user_question}
"""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful study assistant. Answer only from the provided notes."
        ),
        contents=prompt
    )

    answer = response.text

    # Step 4 — store answer in memory
    conversation_history.append({
        "role": "assistant",
        "content": answer
    })

    return answer

# ── Main chat loop ────────────────────────────────────────
print("\n🤖 Knowledge Assistant ready!")
print("Ask questions about your notes. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye Love!")
        break
    if not user_input:
        continue
    answer = ask_assistant(user_input)
    print(f"\nAssistant: {answer}\n")