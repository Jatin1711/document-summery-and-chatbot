# document-summery-and-chatbot
in this it contain chatbot which can work on the file which has been upload file or image and what ever you ask related to that it will answer that question on the bases of that file or image data not more than that
import io
import os

import requests
import pymupdf
import docx
import pandas as pd
from pptx import Presentation
from PIL import Image
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Document Chatbot")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

session = {
    "filename": None,
    "text": "",
    "summary": "",
    "messages": [],
}


def read_any_file(filename: str, content: bytes) -> str:
    name = (filename or "").lower()

    if name.endswith(".pdf"):
        pdf = pymupdf.open(stream=content, filetype="pdf")
        try:
            return "\n".join(page.get_text() for page in pdf)
        finally:
            pdf.close()

    if name.endswith(".docx"):
        document = docx.Document(io.BytesIO(content))
        return "\n".join(p.text for p in document.paragraphs)

    if name.endswith(".pptx"):
        prs = Presentation(io.BytesIO(content))
        lines = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    lines.append(shape.text)
        return "\n".join(lines)

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content)).to_string()

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content)).to_string()

    if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
        return pytesseract.image_to_string(Image.open(io.BytesIO(content)))

    return content.decode(errors="ignore")


def ollama_ready():
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        return True, models
    except requests.RequestException:
        return False, []


def ollama_chat(messages: list[dict]) -> str:
    running, models = ollama_ready()
    if not running:
        raise HTTPException(status_code=503, detail="Ollama is not running")
    if OLLAMA_MODEL not in models:
        raise HTTPException(status_code=503, detail=f"Run: ollama pull {OLLAMA_MODEL}")

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 300},
            },
            timeout=180,
        )
        response.raise_for_status()
        text = response.json().get("message", {}).get("content", "").strip()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Ollama LLM failed: {exc}") from exc

    if not text:
        raise HTTPException(status_code=502, detail="Ollama returned an empty answer")
    return text


class ChatRequest(BaseModel):
    message: str


@app.get("/status")
def status():
    running, models = ollama_ready()
    return {
        "status": "ok",
        "llm": "ollama",
        "ollama_running": running,
        "ollama_model": OLLAMA_MODEL,
        "model_ready": OLLAMA_MODEL in models,
        "loaded_file": session["filename"],
    }


@app.post("/v1/summarize")
async def summarize(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        raw_text = read_any_file(file.filename, content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}") from exc

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Could not read text from file")

    text = raw_text[:8000]
    summary = ollama_chat([
        {"role": "system", "content": "Summarize documents in 3-5 lines of very simple English. No jargon."},
        {"role": "user", "content": text},
    ])

    session["filename"] = file.filename
    session["text"] = text
    session["summary"] = summary
    session["messages"] = [{
        "role": "system",
        "content": (
            "You are a document chatbot powered by Ollama. "
            "Answer only from the document. If it is not there, say you cannot find it. "
            "Use very simple English.\n\n"
            f"File: {file.filename}\n\nDocument:\n{text}"
        ),
    }]
    return {
        "filename": file.filename,
        "simple_summary": summary,
        "chars_read": len(raw_text),
        "model_used": OLLAMA_MODEL,
        "llm": "ollama",
    }


@app.post("/v1/chat")
def chat(body: ChatRequest):
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty")
    if not session["text"]:
        raise HTTPException(status_code=400, detail="Upload a document first")

    session["messages"].append({"role": "user", "content": message})
    answer = ollama_chat(session["messages"])
    session["messages"].append({"role": "assistant", "content": answer})
    return {"answer": answer, "model_used": OLLAMA_MODEL, "llm": "ollama"}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Ollama document chatbot</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Segoe UI, Arial, sans-serif;
      color: #f7f3ee;
      background:
        radial-gradient(900px 520px at 8% -10%, rgba(255, 122, 69, 0.55), transparent 60%),
        radial-gradient(700px 480px at 100% 0%, rgba(64, 156, 255, 0.35), transparent 55%),
        radial-gradient(800px 500px at 80% 110%, rgba(46, 196, 182, 0.28), transparent 50%),
        linear-gradient(160deg, #1b1030 0%, #12182b 42%, #0c1c24 100%);
    }
    body:before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
      background-size: 42px 42px;
      mask-image: radial-gradient(circle at center, black, transparent 75%);
    }
    .wrap { position: relative; max-width: 820px; margin: 0 auto; padding: 36px 16px 48px; }
    .card {
      background: rgba(16, 18, 32, 0.72);
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 24px;
      padding: 26px;
      backdrop-filter: blur(16px);
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255,255,255,0.12);
    }
    h2 { margin: 0 0 6px; font-size: 30px; }
    .sub { color: #d7c7ea; margin: 0 0 18px; }
    .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    input[type="file"] { color: #f0e7ff; }
    input[type="text"] {
      flex: 1;
      min-width: 220px;
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(8, 10, 20, 0.72);
      color: white;
      border-radius: 14px;
      padding: 12px 14px;
      font-size: 16px;
    }
    button {
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      color: #1b1020;
      background: linear-gradient(135deg, #ffb35c, #ff6b6b 55%, #c084fc);
    }
    button:hover { filter: brightness(1.08); }
    #meta { color: #eadcff; min-height: 22px; }
    #log {
      margin-top: 8px;
      background: rgba(6, 8, 16, 0.72);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      min-height: 320px;
      max-height: 460px;
      overflow: auto;
      padding: 14px;
    }
    .msg { margin: 10px 0; padding: 12px 14px; border-radius: 16px; line-height: 1.45; white-space: pre-wrap; }
    .you { background: linear-gradient(135deg, #5b2a4a, #3a245c); margin-left: 18%; }
    .ollama, .summary { background: linear-gradient(135deg, #123c46, #1d3f68); margin-right: 18%; }
    .system, .error { background: rgba(255,255,255,0.08); color: #f3e9ff; }
    .who { display: block; font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase; opacity: 0.7; margin-bottom: 4px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h2>Document chatbot</h2>
      <p class="sub">Ollama llama3.2:3b · upload a file, then ask questions</p>
      <div class="row">
        <input id="file" type="file">
        <button onclick="upload()">Upload and summarize</button>
      </div>
      <p id="meta"></p>
      <div id="log"></div>
      <div class="row">
        <input id="msg" type="text" placeholder="Ask about the document" onkeydown="if(event.key==='Enter')send()">
        <button onclick="send()">Send</button>
      </div>
    </div>
  </div>
  <script>
    const log = document.getElementById("log");
    function add(who, text) {
      const box = document.createElement("div");
      box.className = "msg " + who;
      box.innerHTML = "<span class='who'>" + who + "</span>" + text;
      log.appendChild(box);
      log.scrollTop = log.scrollHeight;
    }
    async function upload() {
      const file = document.getElementById("file").files[0];
      if (!file) return alert("Choose a file");
      const data = new FormData();
      data.append("file", file);
      add("system", "Sending file to Ollama...");
      const res = await fetch("/v1/summarize", { method: "POST", body: data });
      const json = await res.json();
      if (!res.ok) return add("error", JSON.stringify(json));
      document.getElementById("meta").textContent = json.filename + " | model " + json.model_used;
      add("summary", json.simple_summary);
    }
    async function send() {
      const box = document.getElementById("msg");
      const message = box.value.trim();
      if (!message) return;
      add("you", message);
      box.value = "";
      const res = await fetch("/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
      });
      const json = await res.json();
      add(res.ok ? "ollama" : "error", res.ok ? json.answer : JSON.stringify(json));
    }
  </script>
</body>
</html>
"""
