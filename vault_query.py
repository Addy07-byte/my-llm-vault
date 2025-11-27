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

def search_kb(query: str, kb: list, top_k: int = TOP_K):
    q_emb = embed_query(query)
    scored = []
    for rec in kb:
        sim = cosine_similarity(q_emb, rec["embedding"])
        scored.append((sim, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def answer_with_vault(query: str, kb: list, history: list =[]):
    top = search_kb(query, kb, TOP_K)

    # 1. IMPLEMENT SLIDING WINDOW HERE
    if len(history) > 10:
        history = history[-10:]

    if not top:
        # no data at all, just fall back to generic
        return call_llm(query, system_prompt="You are a helpful assistant.", history=history)

    best_sim, _ = top[0]
    print(f"\n🔎 Best similarity score: {best_sim:.2f}")

    if best_sim >= SIM_THRESHOLD:
        # Use personal context
        context_chunks = [rec["text"] for _, rec in top]
        context = "\n\n---\n\n".join(context_chunks)
        system = (
            "You are Adi's personal assistant. "
            "Answer using ONLY the provided personal context. "
            "If something is not in the context, say you don't know."
        )
        user_content = (
            f"Context from Adi's vault:\n{context}\n\n"
            f"Question: {query}"
        )
    else:
        # Fall back to general LLM
        system = "You are a helpful assistant."
        user_content = query

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

    while True:
        try:
            query = input("\nAsk LLM Vault (or type 'exit'): ").strip()
            if query.lower() in ("exit", "quit"):
                break

            answer = answer_with_vault(query, kb)
            print("\n--- ANSWER ---")
            print(answer)

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
    main()
