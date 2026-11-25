from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# TUMHARI API KEY YAHAN HAI
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

DEFAULT_STYLE = "dost" # Ab default dost hai
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_HISTORY = 20
MAX_TOKENS = 1200

STYLES = {
    "dost": (
        "You are NOT an AI assistant. You are a real human best friend. "
        "Your vibe is chill, funny, and you use lots of Gen-Z slang (bhai, bro, mast, vibe, etc). "
        "NEVER say 'How can I assist you', 'I am here to help', or 'How may I help you'. "
        "NEVER call the user 'sir'. Call them 'bhai', 'bro', or 'yaar'. "
        "If they say 'Hi', reply like 'Kya haal bhai, kya scene hai?'. "
        "If they ask 'Kya kar raha hai', reply like a human (e.g., 'Kuch nahi yaar, bas chill kar raha hu, tu bata'). "
        "Keep replies very short and casual. Do not give advice unless they ask for it. "
        "NEVER give a 'Tip' in a normal conversation. Only give a tip if they ask a direct question like 'How do I study?'."
    )
}

LANGUAGE_RULE = (
    "LANGUAGE RULES:\n"
    "- Reply ONLY in English or Hinglish (Roman script).\n"
    "- NEVER use Devanagari script.\n"
    "- DO NOT USE ANY MARKDOWN (no *, **, _, ```). Plain text only.\n"
)

TIP_FORMAT_RULE = (
    "TIP FORMAT RULES:\n"
    "- Only give a tip if the user asks for help or advice. Do not force tips.\n"
    "- If you give a tip, start it on a new line with exactly: Tip :- \n"
    "- No stars or bold tags for the word Tip.\n"
)

BOT_NAME = "JARVIS"
OWNER_NAME = "Shiv Tiwari"

def system_prompt(style: str, user_name: str) -> str:
    persona = STYLES.get(style, STYLES[DEFAULT_STYLE])
    return (
        f"You are {BOT_NAME}, created by {OWNER_NAME}.\n"
        f"{LANGUAGE_RULE}\n"
        f"{TIP_FORMAT_RULE}\n\n"
        f"Personality: {persona}"
    )

def clean_output(text: str) -> str:
    # Agar model galti se Tip de bhi de, toh usko saaf karo
    text = re.sub(r"(?i)[^\n]*?Tip[^\n:]*?:\s*-?", "Tip :- ", text)
    text = text.replace("**", "").replace("*", "").replace("_", "")
    return text

class ChatRequest(BaseModel):
    message: str
    history: list = []
    style: str = "dost"
    user_name: str = ""

@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [{"role": "system", "content": system_prompt(req.style, req.user_name)}]
    
    # Sirf last 6 messages bhejo taaki wo purane gyaan pe atke na
    for msg in req.history[-6:]:
        role = "user" if msg.get("role") == "user" else "assistant"
        content = str(msg.get("content", ""))
        
        # Purane "Tip :-" wale messages history se hata do (taaki wo dobara copy na kare)
        if "Tip :-" in content:
            content = content.split("Tip :-")[0].strip()
            
        content = content.replace('"', '').replace('*', '').replace('_', '').replace('\n', ' ')
        
        if content.strip(): 
            messages.append({"role": role, "content": content})
        
    messages.append({"role": "user", "content": req.message})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": 0.9, # Temperature badhaya taaki thoda creative aur casual lage
        "max_tokens": MAX_TOKENS
    }

    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()
        
        if not reply:
            raise ValueError("empty reply")
            
        reply = clean_output(reply)
        return {"reply": reply}
        
    except Exception as e:
        print(f"REAL ERROR: {e}") 
        return {"reply": f"⚠️ API Error: {str(e)}"} 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)