import os
import json
import math
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ------------- CONFIG -------------
KB_PATH = Path("kb.json")
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
SIM_THRESHOLD = 0.60 
# ---------------------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global KB Cache (so we don't reload it every single query)
_KB_CACHE = None

def get_kb():
    """Singleton to load KB only once."""
    global _KB_CACHE
    if _KB_CACHE is None:
        if not KB_PATH.exists():
            raise FileNotFoundError(f"{KB_PATH} not found. Run build_kb.py first.")
        with KB_PATH.open("r", encoding="utf-8") as f:
            _KB_CACHE = json.load(f)
    return _KB_CACHE

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def embed_query(query: str):
    resp = client.embeddings.create(model=EMBED_MODEL, input=query)
    return resp.data[0].embedding

def refine_query(query: str, history: list) -> str:
    if not history:
        return query
    
    formatted_history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:]])
    system_prompt = "Refine the user's query into a single, standalone search query based on history."
    
    user_content = f"HISTORY:\n{formatted_history}\n\nQUERY:\n{query}\n\nREFINED QUERY:"
    
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    return resp.choices[0].message.content.strip()

def search_kb(query: str, kb: list, top_k: int = TOP_K):
    q_emb = embed_query(query)
    scored = []
    for rec in kb:
        sim = cosine_similarity(q_emb, rec["embedding"])
        scored.append((sim, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

# --- MAIN LOGIC FUNCTION (Used by Test Script) ---
def query_vault(user_input: str, history: list = []):
    """
    The main brain. Takes a question, returns (answer, relevant_context).
    """
    kb = get_kb()
    
    # 1. Refine Query
    refined_query = refine_query(user_input, history)
    
    # 2. Search KB
    top_results = search_kb(refined_query, kb, TOP_K)
    
    if not top_results:
        # Fallback if KB empty
        return "No relevant data found in vault.", []
    
    best_sim, _ = top_results[0]
    
    # 3. Decision Gate
    relevant_context = []
    if best_sim >= SIM_THRESHOLD:
        # RAG Mode
        context_chunks = []
        for sim, rec in top_results:
            fname = rec.get("metadata", {}).get("filename", "unknown")
            text = rec["text"]
            context_chunks.append(f"SOURCE: {fname} (Score: {sim:.2f})\n{text}")
            relevant_context.append(rec)
            
        joined_context = "\n\n---\n\n".join(context_chunks)
        
        system_prompt = (
            "You are an expert Resume Assistant. Use the provided context to answer the question.\n"
            "If the answer isn't in the context, say so."
        )
        user_prompt = f"CONTEXT:\n{joined_context}\n\nQUESTION:\n{user_input}"
        
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        answer = resp.choices[0].message.content.strip()
        
    else:
        # Fallback Mode (General Chat)
        relevant_context = []
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]
        )
        answer = resp.choices[0].message.content.strip()
        
    return answer, relevant_context

# --- CLI ENTRY POINT (Used when you type 'python vault_query.py') ---
if __name__ == "__main__":
    print(f"Loaded {len(get_kb())} chunks. Ask me anything (or 'exit').")
    history = []
    
    while True:
        try:
            user_in = input("\n>> ")
            if user_in.lower() in ('exit', 'quit'):
                break
            
            # Call the unified function
            ans, ctx = query_vault(user_in, history)
            
            print(f"\n--- ANSWER ---\n{ans}")
            
            # Update History
            history.append({"role": "user", "content": user_in})
            history.append({"role": "assistant", "content": ans})
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break