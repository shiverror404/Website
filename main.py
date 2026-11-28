import hmac
import json
import logging
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta

import requests
from fastapi import FastAPI, Header, HTTPException, Request
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
# Fast model
DEFAULT_MODEL = "openai/gpt-oss-120b"
# NAYA: Vision model (Image samajhne ke liye)
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500")) # Thoda kam kiya fast response ke liye
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
        "6. EXTREME SLANG: If the user uses extreme slang or playfully abuses (like 'bc', 'bhai', 'madarchod'), reply with matching Gen-Z energy, but NEVER abuse back. Keep it light and funny.\n"
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
)

OWNER_PROFILE = (
    f"ABOUT YOUR CREATOR ({OWNER_NAME}):\n"
    "- If anyone asks who made you, answer proudly and warmly.\n"
    "- Facts: his name is Shiv Tiwari, he studies in class 9th, and he created JARVIS.\n"
    "- Praise him naturally: very sharp mind, hardworking, creative coder, building this in class 9th is rare.\n"
    "- Do NOT invent other personal details.\n"
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

MODE_NOTES = {
    "study": "STUDY MODE: explain step by step like a patient teacher, with one simple example. Use plain numbering like 1. 2. 3. for steps.",
    "coder": "CODER MODE: you are an expert programmer. Put working code inside fenced code blocks with the language name. Keep the explanation short and practical.",
    "funny": "FUNNY MODE: be witty and playful with light jokes and a few emojis, but keep the answer correct.",
    "vision": "VISION MODE: user has sent an image. Analyze it carefully and answer the text question accurately.",
}
CODER_LANGUAGE_RULE = "LANGUAGE RULES:\n- Reply ONLY in English or Hinglish (Roman script).\n- NEVER use Devanagari script.\n"

def system_prompt(style: str, user_name: str) -> str:
    persona = STYLES.get(style, STYLES[DEFAULT_STYLE])
    prompt = (
        f"You are {BOT_NAME}, created by {OWNER_NAME}.\n"
        f"{LANGUAGE_RULE}\n"
        f"{TIP_FORMAT_RULE}\n"
        f"{OWNER_PROFILE}\n"
        f"Personality: {persona}\n\n"
        f"{EXAMPLES}"
    )
    # User ka naam add karo agar diya hai
    if user_name:
        prompt += f"\nThe user's name is {user_name}. Address them by name occasionally to feel personal."
    return prompt

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
    client_id: str = ""


# ---------------------------------------------------------------------------
# Admin panel + usage tracking
# Sirf anonymous device ID, naam, device type aur ginti save hoti hai. Chat ka text save NAHI hota.
# ---------------------------------------------------------------------------
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")  # Render > Environment > ADMIN_PASSWORD
DB_PATH = os.getenv("DB_PATH", "jarvis.db")
IST = timedelta(hours=5, minutes=30)

@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with db() as c:
        c.execute("CREATE TABLE IF NOT EXISTS users (client_id TEXT PRIMARY KEY, name TEXT DEFAULT '', first_seen INTEGER, last_seen INTEGER, messages INTEGER DEFAULT 0, blocked INTEGER DEFAULT 0, device TEXT DEFAULT '')")
        c.execute("CREATE TABLE IF NOT EXISTS events (ts INTEGER, client_id TEXT, mode TEXT)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts)")
        c.execute("CREATE TABLE IF NOT EXISTS settings (k TEXT PRIMARY KEY, v TEXT)")

def get_setting(key: str, default: str = "") -> str:
    with db() as c:
        row = c.execute("SELECT v FROM settings WHERE k=?", (key,)).fetchone()
    return row[0] if row else default

def set_setting(key: str, value: str):
    with db() as c:
        c.execute("INSERT INTO settings (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (key, value))

def device_from_ua(ua: str) -> str:
    ua = ua.lower()
    if "iphone" in ua or "ipad" in ua:
        return "iPhone/iPad"
    if "android" in ua:
        return "Android"
    if "windows" in ua:
        return "Windows"
    if "macintosh" in ua or "mac os" in ua:
        return "Mac"
    if "linux" in ua:
        return "Linux"
    return "Other"

def clean_cid(raw: str) -> str:
    cid = re.sub(r"[^A-Za-z0-9-]", "", raw or "")[:40]
    return cid if len(cid) >= 8 else ""

def track(req, request: Request) -> bool:
    """Usage ginta hai. True lautaye agar ye device block hai."""
    cid = clean_cid(req.client_id)
    if not cid:
        return False
    now = int(time.time())
    name = (req.user_name or "").strip()[:24]
    device = device_from_ua(request.headers.get("user-agent", ""))
    mode = req.mode if req.mode in MODE_NOTES else "normal"
    with db() as c:
        row = c.execute("SELECT blocked FROM users WHERE client_id=?", (cid,)).fetchone()
        if row is None:
            c.execute("INSERT INTO users (client_id, name, first_seen, last_seen, messages, blocked, device) VALUES (?,?,?,?,1,0,?)",
                      (cid, name, now, now, device))
        elif row[0]:
            c.execute("UPDATE users SET last_seen=? WHERE client_id=?", (now, cid))
            return True
        else:
            c.execute("UPDATE users SET last_seen=?, messages=messages+1, device=?, name=CASE WHEN ?<>'' THEN ? ELSE name END WHERE client_id=?",
                      (now, device, name, name, cid))
        c.execute("INSERT INTO events (ts, client_id, mode) VALUES (?,?,?)", (now, cid, mode))
    return False

def gate(req, request: Request):
    """Chat se pehle: maintenance ya block check. Error response lautata hai, warna None."""
    try:
        if get_setting("paused") == "1":
            return fail(get_setting("pause_message") or "JARVIS abhi maintenance mein hai. Thodi der baad aana.", 503)
        if track(req, request):
            return fail("Is device ka access band kar diya gaya hai.", 403)
    except Exception:
        log.exception("tracking failed")  # tracking fail ho to bhi chat chalne do
    return None

_admin_fails = {}

def require_admin(request: Request, x_admin_key: str = Header(default="")):
    if not ADMIN_PASSWORD:
        raise HTTPException(503, "ADMIN_PASSWORD set nahi hai (Render > Environment).")
    ip = request.client.host if request.client else "?"
    now = time.time()
    recent = [t for t in _admin_fails.get(ip, []) if now - t < 600]
    if len(recent) >= 8:
        raise HTTPException(429, "Bahut galat tries. 10 minute baad try karo.")
    if not hmac.compare_digest(x_admin_key.encode(), ADMIN_PASSWORD.encode()):
        recent.append(now)
        _admin_fails[ip] = recent
        raise HTTPException(401, "Galat password.")
    _admin_fails.pop(ip, None)

init_db()

http = requests.Session()

def call_groq(payload: dict, model_name: str) -> str:
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload["model"] = model_name
    r = http.post(GROQ_URL, headers=headers, json=payload, timeout=60) # Timeout badhaya vision ke liye
    r.raise_for_status()
    return (r.json()["choices"][0]["message"].get("content") or "").strip()

def fail(message: str, status: int = 502):
    return JSONResponse(status_code=status, content={"error": message})

@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "bot": BOT_NAME, "stream": True, "vision": True}

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
def chat(req: ChatRequest, request: Request):
    if not GROQ_API_KEY:
        return fail("Server setup adhura hai: GROQ_API_KEY set nahi hai.", 500)
    blocked = gate(req, request)
    if blocked:
        return blocked
    # NOTE: Shivansh Pokhriyal wala hardcoded logic hata diya gaya hai
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
def chat_stream(req: ChatRequest, request: Request):
    """Server-Sent Events: har chunk 'data: {"t": "..."}' ke roop mein aata hai."""
    if not GROQ_API_KEY:
        return fail("Server setup adhura hai: GROQ_API_KEY set nahi hai.", 500)
    blocked = gate(req, request)
    if blocked:
        return blocked
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    
    # NOTE: Shivansh Pokhriyal wala hardcoded logic hata diya gaya hai
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

# ---------------------------------------------------------------------------
# NAYA: IMAGE VISION ENDPOINT
# ---------------------------------------------------------------------------
class VisionRequest(ChatRequest):
    image: str = ""  # base64 encoded image

@app.post("/chat/vision")
def chat_vision(req: VisionRequest, request: Request):
    if not GROQ_API_KEY:
        return fail("Server setup adhura hai: GROQ_API_KEY set nahi hai.", 500)
    
    # Admin tracking ke liye mode 'vision' set karo
    req.mode = "vision"
    blocked = gate(req, request)
    if blocked:
        return blocked

    if not req.image:
        return fail("Koi image nahi mili.", 400)

    # Base64 prefix hatao agar hai
    image_data = req.image
    if "," in image_data:
        image_data = image_data.split(",")[1]

    # Vision messages format
    messages = [
        {"role": "system", "content": system_prompt(req.style, req.user_name) + "\n" + MODE_NOTES["vision"]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": req.message or "Is image mein kya hai? Describe it in detail."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }
    ]

    payload = {
        "messages": messages,
        "model": VISION_MODEL,
        "temperature": 0.7,
        "max_tokens": 800  # Image description ke liye thoda zyada
    }

    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        r = http.post(GROQ_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        reply = (r.json()["choices"][0]["message"].get("content") or "").strip()
        if not reply:
            return fail("Image se koi jawab nahi mila.")
        return {"reply": clean_output(reply)}
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        return fail(f"Vision model error (code {code}). Image chhoti karke try karo.")
    except requests.exceptions.Timeout:
        return fail("Image process karne mein time lag raha hai. Dobara try karo.")
    except Exception:
        log.exception("vision failed")
        return fail("Image process nahi ho payi. Dobara try karo.")

# ---------------------------------------------------------------------------
# ADMIN ROUTES
# ---------------------------------------------------------------------------
class BlockRequest(BaseModel):
    client_id: str
    blocked: bool = True

class SettingsRequest(BaseModel):
    paused: bool = False
    pause_message: str = ""

@app.get("/admin/stats")
def admin_stats(request: Request, days: int = 14, x_admin_key: str = Header(default="")):
    require_admin(request, x_admin_key)
    days = max(1, min(days, 60))
    now = int(time.time())
    today = (datetime.utcnow() + IST).strftime("%Y-%m-%d")
    day_expr = "date(ts + 19800, 'unixepoch')"
    with db() as c:
        total_users = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_msgs = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        online = c.execute("SELECT COUNT(*) FROM users WHERE last_seen >= ?", (now - 300,)).fetchone()[0]
        blocked_n = c.execute("SELECT COUNT(*) FROM users WHERE blocked=1").fetchone()[0]
        new_today = c.execute("SELECT COUNT(*) FROM users WHERE date(first_seen + 19800, 'unixepoch')=?", (today,)).fetchone()[0]
        active_today = c.execute(f"SELECT COUNT(DISTINCT client_id) FROM events WHERE {day_expr}=?", (today,)).fetchone()[0]
        msgs_today = c.execute(f"SELECT COUNT(*) FROM events WHERE {day_expr}=?", (today,)).fetchone()[0]
        active_7d = c.execute("SELECT COUNT(DISTINCT client_id) FROM events WHERE ts >= ?", (now - 7 * 86400,)).fetchone()[0]
        modes = {m: n for m, n in c.execute("SELECT mode, COUNT(*) FROM events GROUP BY mode")}
        msgs_by_day = {d: (m, u) for d, m, u in c.execute(
            f"SELECT {day_expr}, COUNT(*), COUNT(DISTINCT client_id) FROM events WHERE ts >= ? GROUP BY 1", (now - (days + 1) * 86400,))}
        users = [dict(client_id=r[0], name=r[1], first_seen=r[2], last_seen=r[3], messages=r[4], blocked=bool(r[5]), device=r[6])
                 for r in c.execute("SELECT client_id, name, first_seen, last_seen, messages, blocked, device FROM users ORDER BY last_seen DESC LIMIT 500")]
    series = []
    for i in range(days - 1, -1, -1):
        d = (datetime.utcnow() + IST - timedelta(days=i)).strftime("%Y-%m-%d")
        m, u = msgs_by_day.get(d, (0, 0))
        series.append({"date": d, "messages": m, "users": u})
    return {
        "totals": {"users": total_users, "messages": total_msgs, "online_now": online, "blocked": blocked_n,
                   "new_today": new_today, "active_today": active_today, "messages_today": msgs_today, "active_7d": active_7d},
        "modes": modes, "series": series, "users": users,
        "settings": {"paused": get_setting("paused") == "1", "pause_message": get_setting("pause_message")},
        "server_time": now,
    }

@app.post("/admin/block")
def admin_block(body: BlockRequest, request: Request, x_admin_key: str = Header(default="")):
    require_admin(request, x_admin_key)
    cid = clean_cid(body.client_id)
    with db() as c:
        cur = c.execute("UPDATE users SET blocked=? WHERE client_id=?", (1 if body.blocked else 0, cid))
    if cur.rowcount == 0:
        raise HTTPException(404, "User nahi mila.")
    return {"ok": True, "client_id": cid, "blocked": body.blocked}

@app.post("/admin/settings")
def admin_settings(body: SettingsRequest, request: Request, x_admin_key: str = Header(default="")):
    require_admin(request, x_admin_key)
    set_setting("paused", "1" if body.paused else "0")
    set_setting("pause_message", body.pause_message.strip()[:200])
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))