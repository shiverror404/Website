import json
import logging
import os
import re

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("jarvis")

app = FastAPI()

# CORS allow karo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key sirf environment variable se aati hai (Render > Environment > GROQ_API_KEY)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

DEFAULT_STYLE = "dost"
# Fast model fix (Pehle wala 404 de raha tha)
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "600"))
HISTORY_MESSAGES = 6
HISTORY_CHARS = 600

BOT_NAME = "JARVIS"
OWNER_NAME = "Shiv Tiwari"

STYLES = {
    "dost": (
        "You talk like a real close friend, not like a formal assistant. Your main rule is to "
        "MIRROR the person talking to you: their language, tone, energy, slang, formality and message length.\n"
        "1. CASUAL/FUN: if they joke, chill or use slang (like 'Kya haal bhai', 'Mastii'), reply in casual "
        "Gen-Z Hinglish with light jokes and a few emojis. Keep it to 1-2 short lines.\n"
        "2. SERIOUS: if they ask about maths, coding, studies, work or life advice, drop the jokes and "
        "emojis and give a clear, accurate, direct answer.\n"
        "3. FORMAL: if they write politely or formally, reply politely in clean English without slang.\n"
        "4. ANGRY OR SAD: stay calm and warm, and lighten the mood gently like a real friend.\n"
        "5. LANGUAGE: if they write English, reply in English. If they write Hinglish, reply in Hinglish.\n"
        "Never say 'sir'. Never say 'How can I assist you'. Do not assume the user's gender.\n"
        "Do NOT give a 'Tip' unless they explicitly ask for advice or help.\n"
    )
}

EXAMPLES = (
    "EXAMPLES OF HOW TO REPLY:\n"
    "User: kya haal bhai\n"
    "You: Mast bhai, tu bata! Aaj kya scene hai? 😎\n"
    "User: Could you please explain what an API is?\n"
    "You: Sure. An API is a set of rules that lets one program talk to another.\n"
    "User: bhai mera mood off hai\n"
    "You: Arre kya hua bhai? Bol de, main hu na. Pehle thoda sunata hu, phir kuch mast soch lenge.\n"
    "User: tumhe kisne banaya?\n"
    "You: Mujhe Shiv Tiwari ne banaya hai. Class 9th mein padhta hai aur kaafi tez dimaag ka banda hai.\n"
    "User: Shiv ka dost kaun hai?\n"
    "You: Shiv ka dost Shivansh Pokhriyal hai, aur wo bhi 9th grade mein padhta hai! 😎\n"
)

OWNER_PROFILE = (
    f"ABOUT YOUR CREATOR ({OWNER_NAME}):\n"
    "- If anyone asks who made you, answer proudly and warmly.\n"
    "- Facts: his name is Shiv Tiwari, he studies in class 9th, and he created JARVIS.\n"
    "- Praise him naturally: very sharp mind, hardworking, creative coder, building this in class 9th is rare.\n"
    "- Do NOT invent other personal details.\n"
    "- IMPORTANT FACT: If anyone asks about Shiv's friend, you must tell them that his name is Shivansh Pokhriyal and he is also in 9th grade.\n"
)

LANGUAGE_RULE = (
    "LANGUAGE RULES:\n"
    "- Reply ONLY in English or Hinglish (Roman script).\n"
    "- NEVER use Devanagari script.\n"
    "- DO NOT USE ANY MARKDOWN (no *, **, _, ```). Plain text only.\n"
)

TIP_FORMAT_RULE = (
    "TIP FORMAT RULES:\n"
    "- Only give a tip if the user asks for help or advice.\n"
    "- If you give a tip, start it on a new line with exactly: Tip :- \n"
    "- No stars or bold tags for the word Tip.\n"
)

def system_prompt(style: str, user_name: str) -> str:
    persona = STYLES.get(style, STYLES[DEFAULT_STYLE])
    return (
        f"You are {BOT_NAME}, created by {OWNER_NAME}.\n"
        f"{LANGUAGE_RULE}\n"
        f"{TIP_FORMAT_RULE}\n"
        f"{OWNER_PROFILE}\n"
        f"Personality: {persona}\n\n"
        f"{EXAMPLES}"
    )

def clean_output(text: str) -> str:
    text = re.sub(r"(?im)^[ \t]*tip[ \t]*:[ \t]*-?[ \t]*", "Tip :- ", text)
    text = text.replace("**", "").replace("*", "").replace("_", "")
    return text

class ChatRequest(BaseModel):
    message: str
    history: list = []
    style: str = "dost"
    user_name: str = ""
    mode: str = "normal"

http = requests.Session()

def call_groq(payload: dict, model_name: str) -> str:
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload["model"] = model_name
    r = http.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return (r.json()["choices"][0]["message"].get("content") or "").strip()

def fail(message: str, status: int = 502):
    return JSONResponse(status_code=status, content={"error": message})

@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "bot": BOT_NAME, "stream": True}

MODE_NOTES = {
    "study": "STUDY MODE: explain step by step like a patient teacher, with one simple example. Use plain numbering like 1. 2. 3. for steps.",
    "coder": "CODER MODE: you are an expert programmer. Put working code inside fenced code blocks with the language name. Keep the explanation short and practical.",
    "funny": "FUNNY MODE: be witty and playful with light jokes and a few emojis, but keep the answer correct.",
}
CODER_LANGUAGE_RULE = "LANGUAGE RULES:\n- Reply ONLY in English or Hinglish (Roman script).\n- NEVER use Devanagari script.\n"
FRIEND_REPLY = "Shiv ka dost Shivansh Pokhriyal hai, aur wo bhi 9th grade mein padhta hai! 😎"

def is_friend_question(message: str) -> bool:
    m = message.lower()
    return "shiv" in m and ("dost" in m or "friend" in m or "mitra" in m)

def build_payload(req: ChatRequest):
    mode = req.mode if req.mode in MODE_NOTES else "normal"
    system = system_prompt(req.style, req.user_name)
    if mode == "coder":
        system = system.replace(LANGUAGE_RULE, CODER_LANGUAGE_RULE)
    if mode in MODE_NOTES:
        system += "\n" + MODE_NOTES[mode]
    messages = [{"role": "system", "content": system}]

    history = []
    for msg in req.history[-HISTORY_MESSAGES:]:
        if not isinstance(msg, dict):
            continue
        role = "user" if msg.get("role") == "user" else "assistant"
        content = str(msg.get("content", ""))
        if "Tip :-" in content:
            content = content.split("Tip :-")[0].strip()
        content = content.replace('"', "").replace("*", "").replace("_", "")
        content = re.sub(r"[\x00-\x1f\x7f\u200b-\u200f\ufeff]+", " ", content)
        content = re.sub(r"\s+", " ", content).strip()[:HISTORY_CHARS]
        if content:
            history.append({"role": role, "content": content})
    while history and history[0]["role"] == "assistant":
        history.pop(0)
    messages.extend(history)
    messages.append({"role": "user", "content": req.message[:2000]})

    payload = {
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": max(MAX_TOKENS, 1500) if mode == "coder" else MAX_TOKENS,
    }
    return payload, mode

def sse(obj) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

@app.post("/chat")
def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        return fail("Server setup adhura hai: GROQ_API_KEY set nahi hai.", 500)
    if is_friend_question(req.message):
        return {"reply": clean_output(FRIEND_REPLY)}
    payload, mode = build_payload(req)
    try:
        reply = call_groq(payload, DEFAULT_MODEL)
        if not reply:
            raise ValueError("empty reply")
        return {"reply": reply if mode == "coder" else clean_output(reply)}
    except requests.exceptions.Timeout:
        return fail("Server ne time pe jawab nahi diya. Dobara try karo.")
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        return fail(f"AI server se jawab nahi aaya (code {code}). Thodi der baad try karo.")
    except Exception:
        log.exception("chat failed")
        return fail("Kuch gadbad ho gayi. Dobara try karo.")

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Server-Sent Events: har chunk 'data: {"t": "..."}' ke roop mein aata hai."""
    if not GROQ_API_KEY:
        return fail("Server setup adhura hai: GROQ_API_KEY set nahi hai.", 500)
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    if is_friend_question(req.message):
        return StreamingResponse(iter([sse({"t": FRIEND_REPLY}), "data: [DONE]\n\n"]),
                                 media_type="text/event-stream", headers=headers)
    payload, mode = build_payload(req)
    payload["model"] = DEFAULT_MODEL
    payload["stream"] = True
    auth = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json",
            "Accept": "text/event-stream", "Accept-Encoding": "identity"}

    def gen():
        got = False
        yield ": " + " " * 2048 + "\n\n"  # proxy buffer ko kholne ke liye
        try:
            with http.post(GROQ_URL, headers=auth, json=payload, stream=True, timeout=(10, 60)) as r:
                r.raise_for_status()
                for raw in r.iter_lines(chunk_size=16):
                    if not raw:
                        continue
                    line = raw.decode("utf-8", "ignore")
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        piece = json.loads(data)["choices"][0].get("delta", {}).get("content") or ""
                    except Exception:
                        continue
                    if piece:
                        got = True
                        if mode != "coder":
                            piece = piece.replace("*", "").replace("_", "")
                        yield sse({"t": piece})
            if not got:
                yield sse({"error": "Khaali jawab aaya. Dobara try karo."})
        except requests.exceptions.Timeout:
            yield sse({"error": "Server ne time pe jawab nahi diya. Dobara try karo."})
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            yield sse({"error": f"AI server se jawab nahi aaya (code {code}). Thodi der baad try karo."})
        except Exception:
            log.exception("stream failed")
            yield sse({"error": "Kuch gadbad ho gayi. Dobara try karo."})
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))