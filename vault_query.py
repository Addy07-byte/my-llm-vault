import os
import json
import math
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ------------- CONFIG -------------
KB_PATH = Path("kb.json")
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # or another chat model you have access to
TOP_K = 5
SIM_THRESHOLD = 0.50  # tune this
# ---------------------------------

# load API key from env
load_dotenv() 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_kb(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run build_kb.py first.")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def cosine_similarity(a, b):
    # assumes len(a) == len(b)
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def embed_query(query: str):
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    )
    return resp.data[0].embedding


def refine_query(query: str, history: list) -> str:
    # If there is no history, no refinement is needed
    if not history:
        return query
        
    # 1. Format history into a clean, readable string
    formatted_history = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in history
    )
    
    system_prompt = (
        "You are a helpful assistant that specializes in query refinement and query compression. "
        "Your task is to analyze the provided conversation history and the latest user query. "
        "Based on the history, rewrite the user's query into a single, clear, standalone search query "
        "that contains all necessary context. Respond only with the refined query and nothing else."
    )
    
    # 2. Construct the user input with the clean history
    user_content = (
        f"--- CONVERSATION HISTORY ---\n{formatted_history}\n"
        f"--- LATEST USER QUERY ---\n{query}\n\n"
        f"REFINED SEARCH QUERY:"
    )

    # Call LLM to get the refined query
    refined_query = call_llm(user_content, system_prompt=system_prompt, history=[])
    
    return refined_query.strip()

def search_kb(query: str, kb: list, top_k: int = TOP_K):
    q_emb = embed_query(query)
    scored = []
    for rec in kb:
        sim = cosine_similarity(q_emb, rec["embedding"])
        scored.append((sim, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def answer_with_vault(query: str, kb: list, history: list =[]):
    
    # 1. Sliding Window (Memory Management) 🪟
    if len(history) > 10:
        history = history[-10:]
        
    # 2. Query Refinement (Fixes RAG context loss) 🧠
    refined_query = refine_query(query, history)
    
    # 3. RAG Search (Uses the new, clear query)
    top = search_kb(refined_query, kb, TOP_K) 

    # If no data is found at all, fall back to generic LLM
    if not top:
        return call_llm(query, system_prompt="You are a helpful assistant.", history=history)

    best_sim, _ = top[0]
    print(f"\n🔎 Best similarity score: {best_sim:.2f} (from refined query: {refined_query})")

    if best_sim >= SIM_THRESHOLD:
        # Use personal context (RAG)
        context_chunks = [rec["text"] for _, rec in top]
        context = "\n\n---\n\n".join(context_chunks)
        system = (
            "You are Adi's personal assistant. "
            "Answer using ONLY the provided personal context. "
            "If something is not in the context, say you don't know."
        )
        # We use the ORIGINAL query in the prompt, since the LLM already knows the context from the history.
        user_content = (
            f"Context from Adi's vault:\n{context}\n\n"
            f"Question: {query}"
        )
    else:
        # Fall back to general LLM
        system = "You are a helpful assistant."
        user_content = query

    # Both paths pass the full history to the LLM for conversation context
    return call_llm(user_content, system_prompt=system, history=history)

def call_llm(user_content: str, system_prompt: str, history: list = []):
    
    # 1. Start the messages list with the System Prompt (always first)
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # 2. Add the conversation history
    messages += history # This adds all previous Q&A turns
    
    # 3. Add the new User Message (always last)
    messages.append({"role": "user", "content": user_content})

    # Now, pass the complete list to the API
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages # <--- The key change!
    )
    return resp.choices[0].message.content.strip()

def main():
    kb = load_kb(KB_PATH)
    print(f"Loaded {len(kb)} chunks from {KB_PATH}\n")

    # 1. Initialize the memory list
    conversation_history = [] 

    while True:
        try:
            query = input("\nAsk LLM Vault (or type 'exit'): ").strip()
            if query.lower() in ("exit", "quit"):
                break

            # 2. Pass the history list to the function
            answer = answer_with_vault(query, kb, history=conversation_history) 
            print("\n--- ANSWER ---")
            print(answer)
            
            # 3. Update the history list
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": answer})

        except KeyboardInterrupt:
            break

def jd_gap_analysis(jd_text: str, kb: list, top_k: int = 10) -> str:
    """
    Compare a job description against Aditya's KB.
    Return a structured gap analysis.
    """
    top = search_kb(jd_text, kb, top_k=top_k)
    if not top:
        return "I couldn't find any relevant information in the knowledge base to compare with this job description."

    context_chunks = [rec["text"] for sim, rec in top]
    context = "\n\n-----\n\n".join(context_chunks)

    system_prompt = (
        "You are a careful career assistant. Compare the job description with the "
        "candidate's resume context. DO NOT invent skills. Your output must identify:\n"
        "1. Key JD skills & responsibilities.\n"
        "2. Candidate's matching skills from context.\n"
        "3. Missing or weak areas.\n"
        "4. Fit score (0–100).\n"
        "5. Role suggestions.\n"
        "Be truthful and grounded ONLY in the provided candidate context."
    )

    user_prompt = (
        f"JOB DESCRIPTION:\n{jd_text}\n\n"
        f"CANDIDATE RESUME CONTEXT:\n{context}"
    )

    #  CORRECT: Use the 'client' object defined at the top of the file
    response = client.chat.completions.create(
        model="gpt-4o-mini", # or CHAT_MODEL variable
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    main