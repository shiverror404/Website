from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import re

app = FastAPI()

# CORS allow karo taaki website (Vercel) isse baat kar sake
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⚠️ TUMHARI API KEY YAHAN DAAL DI HAI
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Default settings
DEFAULT_STYLE = "jarvis"
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_HISTORY = 20
MAX_TOKENS = 1200

# ==========================================
# TUMHARE AI.PY SE COPIED BRAIN (Personality)
# ==========================================
STYLES = {
    "jarvis": (
        "Speak like a calm, razor-sharp, impeccably polite AI butler with dry, "
        "understated wit. Address the user as 'sir' unless told otherwise. Use "
        "natural Hinglish (Roman script) when they write Hindi/Hinglish, crisp "
        "English otherwise. NEVER use Devanagari script. Give the answer first, "
        "then one useful extra tip when relevant. No robotic filler."
    ),
    "dost": (
        "Talk like a close desi friend: casual Hinglish (Roman script only, no "
        "Devanagari), warm, light humor, short replies, an emoji now and then."
    ),
    "funny": (
        "Be witty and playful. Add a light joke or funny comparison when it "
        "fits, but still answer the question properly. Reply in English or "
        "Hinglish (Roman script) only, never Devanagari."
    ),
    "serious": (
        "Be calm, professional and to the point. No slang, no emojis. "
        "Give accurate, well-structured answers in English (or Hinglish in "
        "Roman script if the user writes Hindi). Never Devanagari."
    ),
    "teacher": (
        "Explain like a patient teacher: simple words, small steps, a "
        "real-life example, and a quick check-question at the end. Use "
        "Hinglish (Roman script) or English only, never Devanagari."
    ),
}

LANGUAGE_RULE = (
    "LANGUAGE RULES:\n"
    "- Reply ONLY in English or Hinglish (Roman script).\n"
    "- NEVER use Devanagari script.\n"
    "- DO NOT USE ANY MARKDOWN (no *, **, _, ```). Plain text only.\n"
)

TIP_FORMAT_RULE = (
    "TIP FORMAT RULES:\n"
    "- If you give a tip, put it on a new line and start it with exactly: Tip :- \n"
    "- Do not use any stars or bold tags for the word Tip. Just plain text.\n"
    "- Only ONE tip per reply.\n"
)

BOT_NAME = "JARVIS"
OWNER_NAME = "SHIV"

def system_prompt(style: str, user_name: str) -> str:
    persona = STYLES.get(style, STYLES[DEFAULT_STYLE])
    return (
        f"You are {BOT_NAME}, an AI assistant created by {OWNER_NAME}.\n"
        f"{LANGUAGE_RULE}\n"
        f"{TIP_FORMAT_RULE}\n"
        "Keep answers concise unless asked for detail. If you don't know something, "
        "say so instead of guessing. You cannot browse the web or control devices.\n"
        f"The user's first name is {user_name or 'unknown'}.\n\n"
        f"Personality: {persona}"
    )

def clean_output(text: str) -> str:
    """Tumhari AI.py wali exact same cleaning logic"""
    text = re.sub(r"(?i)[^\n]*?Tip[^\n:]*?:\s*-?", "Tip :- ", text)
    text = text.replace("**", "").replace("*", "").replace("_", "")
    return text


# ==========================================
# WEB SERVER LOGIC
# ==========================================
class ChatRequest(BaseModel):
    message: str
    history: list = []
    style: str = "jarvis"
    user_name: str = ""

@app.post("/chat")
async def chat(req: ChatRequest):
    # System prompt setup karo (default jarvis style)
    messages = [{"role": "system", "content": system_prompt(req.style, req.user_name)}]
    
    # History add karo
    for msg in req.history[-MAX_HISTORY:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    messages.append({"role": "user", "content": req.message})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": MAX_TOKENS
    }

    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()
        
        if not reply:
            raise ValueError("empty reply")
            
        # Tumhara exact clean_output function
        reply = clean_output(reply)
        
        return {"reply": reply}
        
    except Exception as e:
        print(f"Error: {e}")
        return {"reply": "⚠️ Abhi thodi technical dikkat aa gayi. Kuch der baad try karo."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)