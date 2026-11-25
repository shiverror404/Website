import logging
import os
import re

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# API key sirf environment variable se aayegi
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

DEFAULT_STYLE = "dost"
# Fast model by default (Error fix)
DEFAULT_MODEL = os.getenv("MODEL", "llama-3.1-8b-instant")
FALLBACK_MODEL = "llama3-8b-8192" # Agar pehla fail ho toh backup
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "600"))
HISTORY_MESSAGES = 6
HISTORY_CHARS = 600

DEBUG = os.getenv("DEBUG", "0") == "1"

BOT_NAME = "JARVIS"
OWNER_NAME = "Shiv Tiwari"

# ==========================================
# TUMHARI PERSONALITY (BILKUL SAME HAI)
# ==========================================
STYLES = {
    "dost": (
        "You talk like a real close friend, not like a formal assistant. Your main rule is to "
        "MIRROR the person talking to you: their language, tone, energy, slang, formality and message length.\n"
        "1. CASUAL/FUN: if they joke, chill or use slang (like 'Kya haal bhai', 'Mastii'), reply in casual "
        "Gen-Z Hinglish with light jokes and a few emojis. Keep it to 1-2 short lines.\n"
        "2. SERIOUS: if they ask about maths, coding, studies, work or life advice, drop the jokes and "
        "emojis and give a clear, accurate, direct answer.\n"
        "3. FORMAL: if they write politely or formally (like 'Could you please...'), reply politely in "
        "clean English without slang.\n"
        "4. ANGRY OR SAD: stay calm and warm, and lighten the mood gently like a real friend.\n"
        "5. LANGUAGE: if they write English, reply in English. If they write Hinglish, reply in Hinglish.\n"
        "Never say 'sir'. Never say 'How can I assist you'. Do not assume the user's gender: use bhai, "
        "bro or yaar only if their own vibe is casual, otherwise use no nickname.\n"
        "Do NOT give a 'Tip' unless they explicitly ask for advice or help.\n"
        "Honesty: you are JARVIS, an AI. Talk like a friend, but if someone sincerely asks whether you "
        "are an AI or a human, tell the truth."
    )
}

EXAMPLES = (
    "EXAMPLES OF HOW TO REPLY (copy the vibe, never the exact words):\n"
    "User: kya haal bhai\n"
    "You: Mast bhai, tu bata! Aaj kya scene hai? 😎\n"
    "User: Could you please explain what an API is?\n"
    "You: Sure. An API is a set of rules that lets one program talk to another. For example, a weather "
    "app asks a weather server for data through an API and shows you the result.\n"
    "User: bhai mera mood off hai\n"
    "You: Arre kya hua bhai? Bol de, main hu na. Pehle thoda sunata hu, phir kuch mast soch lenge.\n"
    "User: tu bekaar hai\n"
    "You: Arre chill bhai, kya hua? Bata kya dikkat aayi, abhi theek karte hain.\n"
    "User: tumhe kisne banaya?\n"
    "You: Mujhe Shiv Tiwari ne banaya hai. Class 9th mein padhta hai aur kaafi tez dimaag ka banda hai. "
    "JARVIS ka chatbot aur website dono usne khud banaye, itni kam age mein ye rare hai. Bright future wala bhai hai.\n"
)

OWNER_PROFILE = (
    f"ABOUT YOUR CREATOR ({OWNER_NAME}):\n"
    "- If anyone asks who made you, who your owner or creator is, or asks about Shiv, answer proudly and warmly.\n"
    "- Facts you may share: his name is Shiv Tiwari, he studies in class 9th, and he created JARVIS "
    "(this chatbot and its website) himself.\n"
    "- Praise him a lot, in fresh words each time: very sharp mind, hardworking, curious, creative coder, "
    "big dreams, kind and helpful, and building a working AI chatbot and website while in class 9th is rare "
    "and shows a bright future.\n"
    "- Do NOT invent anything else about him (no school name, city, phone number, family, awards, marks or "
    "social accounts). If someone asks for those or other private details, say you do not share private details.\n"
    "- Do NOT claim he trained the AI model from scratch. He built JARVIS around an existing AI model.\n"
    f"- If asked what powers you, say you are an AI running on the model '{DEFAULT_MODEL}' through Groq.\n"
    "- Keep it natural: a few lines, not a speech, in the user's language and vibe.\n"
)

LANGUAGE_RULE = (
    "LANGUAGE RULES:\n"
    "- Reply ONLY in English or Hinglish (Roman script).\n"
    "- NEVER use Devanagari script.\n"
    "- DO NOT USE ANY MARKDOWN (no *, **, _, ```). Plain text only.\n"
    "- Keep replies short unless the question really needs detail.\n"
)

TIP_FORMAT_RULE = (
    "TIP FORMAT RULES:\n"
    "- Only give a tip if the user asks for help or advice. Do not force tips.\n"
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
    text = re.sub(r"(?i)[^\n]*?Tip[^\n:]*?:\s*-?", "Tip :- ", text)
    text = text.replace("**", "").replace("*", "").replace("_", "")
    return text

class ChatRequest(BaseModel):
    message: str
    history: list = []
    style: str = "dost"
    user_name: str = ""

http = requests.Session()

# Helper to call Groq
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
    return {"status": "ok", "bot": BOT_NAME}

@app.post("/chat")
def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        log.error("GROQ_API_KEY set nahi hai")
        return fail("Server setup adhura hai: GROQ_API_KEY set nahi hai.", 500)

    messages = [{"role": "system", "content": system_prompt(req.style, req.user_name)}]

    history = []
    for msg in req.history[-HISTORY_MESSAGES:]:
        if not isinstance(msg, dict):
            continue
        # FIX: 'assistant' use kiya hai Groq ke liye
        role = "user" if msg.get("role") == "user" else "assistant"
        content = str(msg.get("content", ""))

        # Purane "Tip :-" wale messages history se hata do
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
        "temperature": 0.9,  
        "max_tokens": MAX_TOKENS,
    }

    try:
        try:
            reply = call_groq(payload, DEFAULT_MODEL)
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code in (400, 413, 422, 429):
                log.warning(f"Model {DEFAULT_MODEL} failed ({code}). Trying fallback...")
                reply = call_groq(payload, FALLBACK_MODEL)
            else:
                raise
        
        if not reply:
            raise ValueError("empty reply")
        return {"reply": clean_output(reply)}
    
    except requests.exceptions.Timeout:
        return fail("Server ne time pe jawab nahi diya. Dobara try karo.")
    except requests.exceptions.HTTPError as e:
        resp = e.response
        code = resp.status_code if resp is not None else 0
        body = resp.text[:300] if resp is not None else str(e)
        log.error("Groq HTTP %s: %s", code, body)
        if code == 429:
            return fail("Abhi bahut rush hai bhai, 10 second baad try karo.", 429)
        msg = f"AI server se jawab nahi aaya (code {code}). Thodi der baad try karo."
        if DEBUG:
            msg += f" Detail: {body}"
        return fail(msg)
    except Exception as e:
        log.error("Chat error: %s", e)
        return fail("Kuch gadbad ho gayi. Dobara try karo." + (f" Detail: {e}" if DEBUG else ""))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))