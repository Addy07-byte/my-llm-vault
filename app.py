from fastapi import FastAPI
from pydantic import BaseModel
from vault_query import answer_with_vault, load_kb
from pathlib import Path
from vault_query import jd_gap_analysis
import openai
from fastapi import FastAPI, Form 
from fastapi.responses import HTMLResponse

app = FastAPI()
KB_PATH = Path("kb.json")
kb = load_kb(KB_PATH)

class Query(BaseModel):
    question: str

class JDRequest(BaseModel):
    jd_text:str

@app.post("/query")
def query_endpoint(q: Query):
    answer = answer_with_vault(q.question, kb)
    return {"answer": answer}

@app.post("/jd-gap-analysis")
def jd_gap_endpoint(req: JDRequest):
    analysis = jd_gap_analysis(req.jd_text, kb)
    return {"analysis": analysis}

@app.get("/", response_class=HTMLResponse)
def home_page():
    # Simple HTML form UI
    return """
    <html>
      <head>
        <title>LLM Vault – JD Gap Analysis</title>
        <style>
          body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; }
          textarea { width: 100%; height: 300px; font-family: monospace; }
          button { padding: 10px 18px; font-size: 16px; margin-top: 10px; }
          .result-box { margin-top: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 6px; background: #f9f9f9; white-space: pre-wrap; }
          h1 { font-size: 24px; }
          label { font-weight: bold; }
        </style>
      </head>
      <body>
        <h1>JD Gap Analysis – LLM Vault</h1>
        <p>Paste a job description below and click <b>Analyze JD</b>. The system will compare it against Aditya's knowledge base and show:</p>
        <ul>
          <li>Key JD skills & responsibilities</li>
          <li>Your matching skills</li>
          <li>Missing or weak areas</li>
          <li>Fit score (0–100)</li>
          <li>Suggested roles</li>
        </ul>

        <form method="post" action="/jd-gap-ui">
          <label for="jd_text">Job Description</label><br/>
          <textarea name="jd_text" id="jd_text" placeholder="Paste the full JD here..."></textarea>
          <br/>
          <button type="submit">Analyze JD</button>
        </form>
      </body>
    </html>
    """


@app.post("/jd-gap-ui", response_class=HTMLResponse)
def jd_gap_ui(jd_text: str = Form(...)):
    # Use your existing gap-analysis function
    from vault_query import jd_gap_analysis  # or keep at top if already imported

    analysis = jd_gap_analysis(jd_text, kb)

    # Return HTML with the analysis rendered nicely
    return f"""
    <html>
      <head>
        <title>JD Gap Analysis – Result</title>
        <style>
          body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; }}
          .result-box {{ margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 6px; background: #f9f9f9; white-space: pre-wrap; }}
          a {{ display: inline-block; margin-top: 15px; }}
        </style>
      </head>
      <body>
        <h1>JD Gap Analysis – Result</h1>
        <div class="result-box">{analysis}</div>
        <a href="/">← Analyze another JD</a>
      </body>
    </html>
    """
