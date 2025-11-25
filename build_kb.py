import os
import json
import openai
from pathlib import Path
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# ------------- CONFIG -------------
DATA_DIR = Path("resumes")   #pointing to resumes folder
OUTPUT_FILE = "kb.json"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 400  # characters per chunk (tune later)
# ---------------------------------

def extract_text_from_file(file_path: Path) -> str:
    """Extract plain text depending on file type."""
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        text = ""
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    

    elif ext in [".txt", ".md"]:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    

    elif ext == ".docx":
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)        


    else:
        print(f"⚠️ Skipping unsupported file type: {file_path}")
        return ""

#chunks paragraph based,section coherent, meaning tight
def chunk_text(text, chunk_size=400):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) < chunk_size:
            current += p + "\n"
        else:
            chunks.append(current.strip())
            current = p + "\n"
    if current:
        chunks.append(current.strip())

    return chunks



def embed_text(text: str):
    """Get embedding vector for given text using OpenAI API."""
    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def main():
    records = []

    # Traverse your vault_data folder
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            file_path = Path(root) / fname
            text = extract_text_from_file(file_path)
            if not text.strip():
                continue

            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks):
                try:
                    emb = embed_text(chunk)
                    record = {
                        "id": f"{fname}_chunk_{idx}",
                        "source": str(file_path),
                        "text": chunk,
                        "embedding": emb,
                        "metadata": {
                            "category": file_path.parent.name,  # resumes / certs / projects
                            "filename": fname
                        }
                    }
                    records.append(record)
                    print(f"✅ Embedded: {fname} | chunk {idx+1}/{len(chunks)}")
                except Exception as e:
                    print(f"❌ Error embedding {fname}: {e}")

    # Save all embeddings to kb.json
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Done! Created {len(records)} chunks → saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
