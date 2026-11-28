import base64
import hmac
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional

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
# Photo samajhne wale (vision) models. Render > Environment se badal sakte ho, code badalne ki zarurat nahi.
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
VISION_FALLBACK = os.getenv("VISION_FALLBACK", "qwen/qwen3.6-27b")
MAX_IMAGES = 3
MAX_IMAGE_CHARS = 3_000_000  # ek photo (base64 text) ki limit, ~2.2 MB
IMAGE_RE = re.compile(r"^data:image/(jpeg|png|webp);base64,[A-Za-z0-9+/=]+$")
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
        "You CAN create images. If someone asks whether you can, say yes and tell them to just write something like 'ek sher ki image banao' or tap the palette button near the photo button. Never claim you cannot make images.\n"
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
    images: list = []


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

VISION_NOTE = (
    "User ne photo bheji hai. Photo dhyan se dekh kar user ke sawal ka jawab do. "
    "Photo mein likha text ho to padh kar batao. Kisi insaan ko uske chehre se pehchaan kar naam mat batao, "
    "bas jo dikh raha hai wo batao. Jawab Hinglish mein, chhota aur saaf."
)

def clean_images(req) -> list:
    """Sirf sahi data-URL photos (max 3) rakhta hai. Galat ho to ValueError."""
    imgs = req.images if isinstance(req.images, list) else []
    imgs = [i for i in imgs if isinstance(i, str) and i]
    if len(imgs) > MAX_IMAGES:
        raise ValueError(f"Ek baar mein max {MAX_IMAGES} photos bhej sakte ho.")
    for i in imgs:
        if len(i) > MAX_IMAGE_CHARS or not IMAGE_RE.match(i):
            raise ValueError("Photo sahi format mein nahi hai. JPG, PNG ya WEBP bhejo.")
    return imgs

def models_and_variants(has_images: bool):
    """(models, extra-params) ki list. Pehle extras ke saath try, 400/404 aaye to bina extras ke."""
    if has_images:
        models = [VISION_MODEL] + ([VISION_FALLBACK] if VISION_FALLBACK and VISION_FALLBACK != VISION_MODEL else [])
        return models, [{"reasoning_effort": "none"}, {}]
    return [DEFAULT_MODEL], [{}]

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)

def strip_think(text: str) -> str:
    text = THINK_BLOCK.sub("", text)
    return re.sub(r"<think>.*", "", text, flags=re.S).strip()

class ThinkFilter:
    """Stream ke beech se <think>...</think> hissa hata deta hai (tag do chunks mein tut sakta hai)."""
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buf, self.inside = "", False

    def feed(self, s: str) -> str:
        self.buf += s
        out = ""
        while True:
            if self.inside:
                i = self.buf.find(self.CLOSE)
                if i == -1:
                    self.buf = self.buf[-(len(self.CLOSE) - 1):]
                    return out
                self.buf = self.buf[i + len(self.CLOSE):]
                self.inside = False
            else:
                i = self.buf.find(self.OPEN)
                if i == -1:
                    cut = len(self.buf)
                    for k in range(min(len(self.OPEN) - 1, len(self.buf)), 0, -1):
                        if self.OPEN.startswith(self.buf[-k:]):
                            cut = len(self.buf) - k
                            break
                    out += self.buf[:cut]
                    self.buf = self.buf[cut:]
                    return out
                out += self.buf[:i]
                self.buf = self.buf[i + len(self.OPEN):]
                self.inside = True

    def flush(self) -> str:
        out = "" if self.inside else self.buf
        self.buf = ""
        return out

def open_groq(payload: dict, models: list, variants: list, stream: bool, timeout):
    """Groq ko call karta hai. Model/param reject (400/404) ho to agla combination try karta hai."""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    if stream:
        headers.update({"Accept": "text/event-stream", "Accept-Encoding": "identity"})
    last = None
    for model in models:
        for extra in variants:
            body = dict(payload, model=model, **extra)
            r = http.post(GROQ_URL, headers=headers, json=body, stream=stream, timeout=timeout)
            if r.status_code in (400, 404) and (len(models) > 1 or len(variants) > 1):
                log.warning("groq %s rejected model=%s extra=%s: %s", r.status_code, model, list(extra), r.text[:200])
                last = r
                continue
            return r
    return last

def call_groq(payload: dict, models: list, variants: list = None) -> str:
    r = open_groq(payload, models, variants or [{}], False, 45)
    r.raise_for_status()
    return strip_think((r.json()["choices"][0]["message"].get("content") or "").strip())

def fail(message: str, status: int = 502):
    return JSONResponse(status_code=status, content={"error": message})

@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "bot": BOT_NAME, "stream": True, "image": bool(image_provider())}

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
    images = clean_images(req)
    user_text = req.message[:2000]
    if images:
        messages[0]["content"] += "\n" + VISION_NOTE
        messages.append({"role": "user", "content": [{"type": "text", "text": user_text}] +
                         [{"type": "image_url", "image_url": {"url": u}} for u in images]})
    else:
        messages.append({"role": "user", "content": user_text})

    tokens = max(MAX_TOKENS, 1500) if mode == "coder" else MAX_TOKENS
    if images:
        tokens = max(tokens, 1200)  # soch-vichar ke tokens ke baad bhi jawab ke liye jagah bache
    payload = {"messages": messages, "temperature": 0.8, "max_tokens": tokens}
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
    try:
        payload, mode = build_payload(req)
    except ValueError as e:
        return fail(str(e), 400)
    models, variants = models_and_variants(bool(req.images))
    try:
        reply = call_groq(payload, models, variants)
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
    try:
        payload, mode = build_payload(req)
    except ValueError as e:
        return fail(str(e), 400)
    payload["stream"] = True
    models, variants = models_and_variants(bool(req.images))

    def gen():
        got = False
        think = ThinkFilter()
        yield ": " + " " * 2048 + "\n\n"  # proxy buffer ko kholne ke liye
        try:
            r = open_groq(payload, models, variants, True, (10, 60))
            try:
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
                    piece = think.feed(piece) if piece else ""
                    if piece:
                        got = True
                        if mode != "coder":
                            piece = piece.replace("*", "").replace("_", "")
                        yield sse({"t": piece})
                tail = think.flush()
                if tail.strip():
                    got = True
                    yield sse({"t": tail if mode == "coder" else tail.replace("*", "").replace("_", "")})
            finally:
                r.close()
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
# Image generation (FLUX). Groq image nahi banata, isliye alag provider chahiye:
#   TOGETHER_API_KEY -> Together AI (recommended, sasta aur fast)
#   HF_TOKEN         -> Hugging Face Inference Providers
# Model badalne ke liye Render > Environment: IMAGE_MODEL_FAST / IMAGE_MODEL_HD
# Hum image par koi watermark ya text overlay nahi lagate. Output model ka original hota hai.
# ---------------------------------------------------------------------------
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "").strip().lower()      # together | huggingface (khaali = auto)
IMAGE_MODEL_FAST = os.getenv("IMAGE_MODEL_FAST", "black-forest-labs/FLUX.1-schnell")
IMAGE_MODEL_HD = os.getenv("IMAGE_MODEL_HD", "black-forest-labs/FLUX.1-dev")
IMAGE_DAILY_LIMIT = int(os.getenv("IMAGE_DAILY_LIMIT", "12"))          # ek device ki roz ki limit
IMAGE_GLOBAL_DAILY = int(os.getenv("IMAGE_GLOBAL_DAILY", "300"))       # poori site ki roz ki limit
TOGETHER_URL = "https://api.together.xyz/v1/images/generations"
HF_URL = "https://router.huggingface.co/hf-inference/models/"
_img_sem = threading.BoundedSemaphore(int(os.getenv("IMAGE_CONCURRENCY", "3")))

IMAGE_RATIOS = {"1:1": (1024, 1024), "3:4": (864, 1152), "16:9": (1344, 768), "9:16": (768, 1344)}
IMAGE_STYLES = {
    "auto": "",
    "photo": "ultra realistic photograph, shot on a 35mm lens, natural lighting, tack sharp focus, fine natural detail, true-to-life colors",
    "cinematic": "cinematic film still, dramatic lighting, anamorphic lens, shallow depth of field, rich color grading, epic composition",
    "anime": "high quality anime illustration, clean line art, vibrant colors, detailed painterly background, studio quality",
    "3d": "polished 3D render, soft global illumination, smooth materials, highly detailed, Pixar-like charm",
    "art": "digital concept art, intricate details, painterly brush work, striking composition, professional illustration",
    "oil": "classical oil painting, visible brush strokes, rich textures, warm museum-quality finish",
    "minimal": "minimal flat vector illustration, clean geometric shapes, limited harmonious palette, modern design",
}

BAD_PROMPT = re.compile(
    r"\b(nude|nudes|naked|nsfw|porn|porno|pornography|xxx|hentai|topless|erotic|sexual|sex|nangi|nanga|"
    r"chudai|boobs|nipples|genitals|blowjob|onlyfans|gore|beheading|decapitat\w*|loli|lolita)\b", re.I)
BAD_MESSAGE = "Is tarah ki image main nahi bana sakta. Kuch aur creative try karo, jaise koi mast scene ya character."

class ImageError(Exception):
    def __init__(self, user_msg: str, status: int = 502):
        super().__init__(user_msg)
        self.user_msg, self.status = user_msg, status

class ImageRequest(BaseModel):
    prompt: str
    style: str = "auto"
    ratio: str = "1:1"
    quality: str = "fast"
    enhance: bool = True
    seed: Optional[int] = None
    client_id: str = ""
    user_name: str = ""

def image_provider() -> str:
    want = {"hf": "huggingface"}.get(IMAGE_PROVIDER, IMAGE_PROVIDER)
    if want == "together" and TOGETHER_API_KEY:
        return "together"
    if want == "huggingface" and HF_TOKEN:
        return "huggingface"
    if TOGETHER_API_KEY:
        return "together"
    if HF_TOKEN:
        return "huggingface"
    return ""

def steps_for(model: str):
    m = model.lower()
    if "schnell" in m:
        return 4
    if "dev" in m and "klein" not in m:
        return 28
    return None  # baaki models (pro, klein...) ke apne default steps sahi hote hain

def sniff_mime(b: bytes) -> str:
    if b.startswith(b"\x89PNG"):
        return "image/png"
    if b.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    return ""

def provider_error(r):
    """Provider ke error ko user-friendly Hinglish message mein badalta hai."""
    code = r.status_code
    if code < 400:
        return
    body = (r.text or "")[:300]
    log.warning("image provider %s: %s", code, body)
    if "nsfw" in body.lower() or "safety" in body.lower() or "moderation" in body.lower():
        raise ImageError("Ye prompt image server ko safe nahi laga. Kuch aur try karo.", 400)
    if code in (401, 403):
        raise ImageError("Image server ki key ya permission galat hai. Owner ko batao.", 502)
    if code == 402:
        raise ImageError("Image credits khatam ho gaye hain. Owner ko batao.", 503)
    if code == 404:
        raise ImageError("Ye image model available nahi hai. Owner ko IMAGE_MODEL check karna hoga.", 502)
    if code == 429:
        raise ImageError("Image server abhi busy hai. Thodi der baad try karo.", 429)
    if code in (400, 422):
        raise ImageError("Is prompt ya size ke saath image nahi ban payi. Prompt thoda badal ke try karo.", 400)
    raise ImageError("Image server ne jawab nahi diya. Dobara try karo.", 502)

def together_generate(model, prompt, w, h, steps, seed) -> bytes:
    body = {"model": model, "prompt": prompt, "width": w, "height": h, "n": 1, "seed": seed, "response_format": "b64_json"}
    if steps:
        body["steps"] = steps
    r = http.post(TOGETHER_URL, headers={"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"},
                  json=body, timeout=(10, 110))
    provider_error(r)
    item = (r.json().get("data") or [{}])[0]
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    if item.get("url"):
        rr = http.get(item["url"], timeout=40)
        rr.raise_for_status()
        return rr.content
    raise ImageError("Image server ne khaali jawab diya. Dobara try karo.", 502)

def hf_generate(model, prompt, w, h, steps, seed) -> bytes:
    params = {"width": w, "height": h, "seed": seed}
    if steps:
        params["num_inference_steps"] = steps
    body = {"inputs": prompt, "parameters": params}
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json", "Accept": "image/png"}
    for attempt in (0, 1):
        r = http.post(HF_URL + model, headers=headers, json=body, timeout=(10, 110))
        if r.status_code == 503 and attempt == 0:      # model warm ho raha hai
            try:
                wait = float(r.json().get("estimated_time", 12))
            except Exception:
                wait = 12
            time.sleep(min(max(wait, 3), 25))
            continue
        provider_error(r)
        return r.content
    raise ImageError("Image model abhi warm ho raha hai. 30 second baad try karo.", 503)

def generate_image_bytes(model, prompt, w, h, seed) -> bytes:
    steps = steps_for(model)
    fn = together_generate if image_provider() == "together" else hf_generate
    data = fn(model, prompt, w, h, steps, seed)
    if not sniff_mime(data):
        log.warning("image provider returned non-image bytes (%d)", len(data or b""))
        raise ImageError("Image server se sahi image nahi mili. Dobara try karo.", 502)
    return data

ENHANCER_SYSTEM = (
    "You are the prompt writer and safety filter for an AI image generator (FLUX). "
    "The user's request may be in Hindi, Hinglish (Roman script) or English. "
    "Reply with ONLY one JSON object, no markdown: {\"ok\": true or false, \"reason\": \"...\", \"prompt\": \"...\"}.\n"
    "Set ok=false (and put a short, kind Hinglish sentence in reason) if the request asks for: sexual or nude content, "
    "graphic gore, any sexualised minor, hate symbols or hateful content, self-harm, or realistic fake images of real named people "
    "doing harmful, shameful or deceptive things. Everything else is allowed, including cartoons, fantasy, products, places, animals, "
    "posters and logos.\n"
    "If ok=true, prompt must be ONE English paragraph under 90 words. Keep the user's subject and every detail they asked for. "
    "Drop command words like 'image bana do' or 'draw'. If the user wants words inside the image, keep that exact text in quotes; "
    "otherwise never add text, captions, signatures or logos to the scene. "
)
ENHANCE_ON = ("Expand it into a rich, vivid description: subject, setting, composition, lighting, mood, camera or lens, "
              "colour palette and textures, so the result looks stunning.")
ENHANCE_OFF = "Translate it faithfully into clear English and add almost nothing."

def craft_prompt(raw: str, enhance: bool):
    """(ok, prompt_ya_reason). Groq se prompt chamkata hai aur safety check bhi karta hai."""
    if BAD_PROMPT.search(raw):
        return False, BAD_MESSAGE
    prompt = raw
    if GROQ_API_KEY:
        try:
            payload = {"messages": [{"role": "system", "content": ENHANCER_SYSTEM + (ENHANCE_ON if enhance else ENHANCE_OFF)},
                                    {"role": "user", "content": raw}],
                       "temperature": 0.7, "max_tokens": 900}
            text = call_groq(payload, [DEFAULT_MODEL], [{"reasoning_effort": "low"}, {}])
            m = re.search(r"\{.*\}", text, re.S)
            obj = json.loads(m.group(0)) if m else {}
            if obj.get("ok") is False:
                return False, (str(obj.get("reason") or "").strip()[:200] or BAD_MESSAGE)
            p = str(obj.get("prompt") or "").strip()
            if p:
                prompt = p
        except Exception:
            log.warning("prompt enhancer failed, raw prompt use kar rahe hain", exc_info=True)
    prompt = re.sub(r"\s+", " ", prompt).strip()[:900]
    if BAD_PROMPT.search(prompt):
        return False, BAD_MESSAGE
    return True, prompt

def image_gate(req, request: Request):
    """(error_response, client_id, used_today). Limits + block + maintenance check."""
    if get_setting("paused") == "1":
        return fail(get_setting("pause_message") or "JARVIS abhi maintenance mein hai. Thodi der baad aana.", 503), "", 0
    if get_setting("images_off") == "1":
        return fail("Image banana abhi band hai. Thodi der baad try karo.", 503), "", 0
    cid = clean_cid(req.client_id)
    if not cid:
        return fail("Device ID nahi mili. Page refresh karke dobara try karo.", 400), "", 0
    today = (datetime.utcnow() + IST).strftime("%Y-%m-%d")
    day_expr = "date(ts + 19800, 'unixepoch')"
    with db() as c:
        row = c.execute("SELECT blocked FROM users WHERE client_id=?", (cid,)).fetchone()
        if row and row[0]:
            return fail("Is device ka access band kar diya gaya hai.", 403), cid, 0
        used = c.execute(f"SELECT COUNT(*) FROM events WHERE client_id=? AND mode='image' AND {day_expr}=?", (cid, today)).fetchone()[0]
        total = c.execute(f"SELECT COUNT(*) FROM events WHERE mode='image' AND {day_expr}=?", (today,)).fetchone()[0]
    if used >= IMAGE_DAILY_LIMIT:
        return fail(f"Aaj ki {IMAGE_DAILY_LIMIT} images ki limit poori ho gayi. Kal phir banana!", 429), cid, used
    if total >= IMAGE_GLOBAL_DAILY:
        return fail("Aaj ke liye JARVIS ki image limit poori ho gayi. Kal try karo.", 429), cid, used
    return None, cid, used

def record_image(cid: str, name: str, request: Request):
    """Sirf kaamyaab image par ginti badhti hai (fail hone par user ki limit nahi katti)."""
    now = int(time.time())
    name = (name or "").strip()[:24]
    device = device_from_ua(request.headers.get("user-agent", ""))
    with db() as c:
        if c.execute("SELECT 1 FROM users WHERE client_id=?", (cid,)).fetchone() is None:
            c.execute("INSERT INTO users (client_id, name, first_seen, last_seen, messages, blocked, device) VALUES (?,?,?,?,1,0,?)",
                      (cid, name, now, now, device))
        else:
            c.execute("UPDATE users SET last_seen=?, messages=messages+1, device=?, name=CASE WHEN ?<>'' THEN ? ELSE name END WHERE client_id=?",
                      (now, device, name, name, cid))
        c.execute("INSERT INTO events (ts, client_id, mode) VALUES (?,?,?)", (now, cid, "image"))

@app.post("/image")
def make_image(req: ImageRequest, request: Request):
    if not image_provider():
        return fail("Image feature ka setup adhura hai. Owner ko Render mein TOGETHER_API_KEY (ya HF_TOKEN) daalni hogi.", 503)
    try:
        blocked, cid, used = image_gate(req, request)
    except Exception:
        log.exception("image gate failed")
        return fail("Kuch gadbad ho gayi. Dobara try karo.", 500)
    if blocked:
        return blocked
    raw = (req.prompt or "").strip()[:600]
    if len(raw) < 2:
        return fail("Batao kaisi image banani hai.", 400)
    style = req.style if req.style in IMAGE_STYLES else "auto"
    w, h = IMAGE_RATIOS.get(req.ratio, IMAGE_RATIOS["1:1"])
    quality = "hd" if req.quality == "hd" else "fast"
    seed = req.seed if isinstance(req.seed, int) and 0 < req.seed < 2**31 else random.randint(1, 2**31 - 1)

    if not _img_sem.acquire(timeout=45):
        return fail("Abhi bahut log image bana rahe hain. Ek minute baad try karo.", 429)
    try:
        ok, text = craft_prompt(raw, bool(req.enhance))
        if not ok:
            return fail(text, 400)
        final = text + (", " + IMAGE_STYLES[style] if IMAGE_STYLES[style] else "")
        model, fell_back = (IMAGE_MODEL_HD if quality == "hd" else IMAGE_MODEL_FAST), False
        try:
            data = generate_image_bytes(model, final, w, h, seed)
        except ImageError as e:
            if quality == "hd" and IMAGE_MODEL_FAST != model and e.status in (429, 502, 503):
                log.warning("HD fail (%s), fast model try kar rahe hain", e.user_msg)
                model, fell_back = IMAGE_MODEL_FAST, True
                data = generate_image_bytes(model, final, w, h, seed)
            else:
                raise
        try:
            record_image(cid, req.user_name, request)
        except Exception:
            log.exception("image tracking failed")
        mime = sniff_mime(data)
        return JSONResponse(
            content={"image": f"data:{mime};base64," + base64.b64encode(data).decode("ascii"),
                     "prompt": text, "seed": seed, "model": model.split("/")[-1], "hd_fallback": fell_back,
                     "remaining": max(0, IMAGE_DAILY_LIMIT - used - 1), "limit": IMAGE_DAILY_LIMIT},
            headers={"Cache-Control": "no-store"})
    except ImageError as e:
        return fail(e.user_msg, e.status)
    except requests.exceptions.Timeout:
        return fail("Image banne mein bahut time lag gaya. Dobara try karo.", 504)
    except Exception:
        log.exception("image failed")
        return fail("Kuch gadbad ho gayi. Dobara try karo.", 500)
    finally:
        _img_sem.release()

class BlockRequest(BaseModel):
    client_id: str
    blocked: bool = True

class SettingsRequest(BaseModel):
    paused: bool = False
    pause_message: str = ""
    images_enabled: Optional[bool] = None   # None = ise mat chhedo

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
        images_total = c.execute("SELECT COUNT(*) FROM events WHERE mode='image'").fetchone()[0]
        images_today = c.execute(f"SELECT COUNT(*) FROM events WHERE mode='image' AND {day_expr}=?", (today,)).fetchone()[0]
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
                   "new_today": new_today, "active_today": active_today, "messages_today": msgs_today, "active_7d": active_7d,
                   "images_today": images_today, "images_total": images_total},
        "modes": modes, "series": series, "users": users,
        "settings": {"paused": get_setting("paused") == "1", "pause_message": get_setting("pause_message"),
                     "images_enabled": get_setting("images_off") != "1", "image_provider": image_provider() or "none"},
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
    if body.images_enabled is not None:
        set_setting("images_off", "0" if body.images_enabled else "1")
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))