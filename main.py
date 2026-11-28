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
from urllib.parse import quote

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
    mode = req.mode if (req.mode in MODE_NOTES or req.mode == "image") else "normal"
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
    return {"status": "ok", "bot": BOT_NAME, "stream": True, "image": image_providers_ready()}

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
# IMAGE GENERATION (free, watermark-free setup)
#
# Providers ek ke baad ek try hote hain (IMAGE_PROVIDERS env se order badal sakte ho).
# Jo provider ki key set nahi hai wo automatically skip ho jata hai.
#
#   pollinations_key : POLLINATIONS_API_KEY  (free key: https://enter.pollinations.ai) -> FLUX, NO watermark
#   huggingface      : HF_TOKEN              (free token: huggingface.co/settings/tokens) -> FLUX.1-schnell
#   cloudflare       : CF_ACCOUNT_ID + CF_API_TOKEN (free daily quota) -> FLUX.1-schnell
#   pollinations     : bina key ka last backup (anonymous tier: slow, kabhi kabhi watermark aa sakta hai)
# ---------------------------------------------------------------------------
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY", "")
POLLINATIONS_MODEL = os.getenv("POLLINATIONS_MODEL", "flux")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_IMAGE_URL = os.getenv("HF_IMAGE_URL", "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell")
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
CF_IMAGE_MODEL = os.getenv("CF_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell")
IMAGE_PROVIDERS = [p.strip() for p in os.getenv("IMAGE_PROVIDERS", "pollinations_key,huggingface,cloudflare,pollinations").split(",") if p.strip()]
IMAGE_DAILY_LIMIT = int(os.getenv("IMAGE_DAILY_LIMIT", "30"))   # ek device ki 24 ghante ki limit (free quota bachane ke liye)
IMAGE_GAP_SECONDS = int(os.getenv("IMAGE_GAP_SECONDS", "6"))    # do images ke beech minimum gap
IMAGE_SLOTS = threading.BoundedSemaphore(int(os.getenv("IMAGE_CONCURRENCY", "3")))

IMAGE_SIZES = {"square": (1024, 1024), "portrait": (768, 1344), "landscape": (1344, 768)}
IMAGE_STYLES = {
    "auto": "",
    "realistic": "photorealistic, ultra detailed, natural lighting, sharp focus, shot on 85mm lens, high dynamic range",
    "cinematic": "cinematic still, dramatic lighting, anamorphic lens, subtle film grain, color graded, epic composition",
    "anime": "high quality anime illustration, vibrant colors, clean line art, detailed background, studio quality",
    "digital": "digital art, concept art, highly detailed, rich vivid colors, polished professional illustration",
    "3d": "3D render, stylized Pixar-like look, soft global illumination, smooth materials, high detail",
    "sketch": "detailed pencil sketch, hand-drawn linework, cross-hatching shading, paper texture",
}

ENHANCER_SYSTEM = (
    "You write prompts for an AI image generator (FLUX). The user's request may be in Hindi, Hinglish or English. "
    "Rewrite it as ONE detailed English image prompt of 40 to 70 words: subject, setting, composition, lighting, colors, mood, camera or medium. "
    "Keep the user's intent exactly. Do not put any text, captions, logos or watermarks inside the image. "
    "If the request is sexual, sexualizes or endangers minors, makes a deepfake of a real person, is graphic gore, or promotes hate, "
    "reply with exactly the single word REFUSE. "
    "Output only the final prompt. No quotes, no explanation."
)

class ImageRequest(BaseModel):
    prompt: str
    style: str = "auto"
    ratio: str = "square"
    seed: int = 0
    client_id: str = ""
    user_name: str = ""
    mode: str = "image"

class ImageError(Exception):
    pass

_img_last = {}  # client -> last image request time

def image_providers_ready():
    return [n for n in IMAGE_PROVIDERS if provider_ready(n)]

def provider_ready(name: str) -> bool:
    if name == "pollinations_key":
        return bool(POLLINATIONS_API_KEY)
    if name == "huggingface":
        return bool(HF_TOKEN)
    if name == "cloudflare":
        return bool(CF_ACCOUNT_ID and CF_API_TOKEN)
    return name == "pollinations"

def sniff_image(data: bytes):
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None

def enhance_prompt(user_prompt: str) -> str:
    """Hinglish/short idea ko detailed English image prompt banata hai. Fail ho to original prompt chalta hai."""
    if not GROQ_API_KEY:
        return user_prompt
    try:
        payload = {
            "messages": [{"role": "system", "content": ENHANCER_SYSTEM}, {"role": "user", "content": user_prompt}],
            "temperature": 0.7,
            "max_tokens": 400,
        }
        out = call_groq(payload, [DEFAULT_MODEL]).strip().strip('"').strip()
        if out.upper().startswith("REFUSE"):
            raise ImageError("refused")
        return out[:900] if len(out) > 10 else user_prompt
    except ImageError:
        raise
    except Exception:
        log.warning("prompt enhance failed, original prompt use ho raha hai")
        return user_prompt

def read_image_response(r) -> tuple:
    if r.status_code != 200:
        raise RuntimeError(f"http {r.status_code}")
    data = r.content
    mime = sniff_image(data)
    if not mime or len(data) < 2000:
        raise RuntimeError("response image nahi hai")
    return data, mime

def gen_pollinations(prompt, w, h, seed, keyed):
    q = {"model": POLLINATIONS_MODEL, "width": w, "height": h, "seed": seed, "nologo": "true", "enhance": "false", "safe": "true"}
    if keyed:
        url = "https://gen.pollinations.ai/image/" + quote(prompt, safe="")
        q["key"] = POLLINATIONS_API_KEY
    else:
        url = "https://image.pollinations.ai/prompt/" + quote(prompt, safe="")
    return read_image_response(http.get(url, params=q, timeout=(10, 100)))

def gen_huggingface(prompt, w, h, seed):
    r = http.post(HF_IMAGE_URL,
                  headers={"Authorization": f"Bearer {HF_TOKEN}", "Accept": "image/jpeg"},
                  json={"inputs": prompt, "parameters": {"width": w, "height": h, "seed": seed}},
                  timeout=(10, 100))
    return read_image_response(r)

def gen_cloudflare(prompt, w, h, seed):
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_IMAGE_MODEL}"
    r = http.post(url, headers={"Authorization": f"Bearer {CF_API_TOKEN}"},
                  json={"prompt": prompt, "steps": 8}, timeout=(10, 100))
    if r.status_code != 200:
        raise RuntimeError(f"http {r.status_code}")
    b64 = (r.json().get("result") or {}).get("image")
    if not b64:
        raise RuntimeError("image missing")
    data = base64.b64decode(b64)
    mime = sniff_image(data)
    if not mime:
        raise RuntimeError("response image nahi hai")
    return data, mime

def run_providers(prompt, w, h, seed):
    for name in IMAGE_PROVIDERS:
        if not provider_ready(name):
            continue
        try:
            if name == "pollinations_key":
                data, mime = gen_pollinations(prompt, w, h, seed, True)
            elif name == "huggingface":
                data, mime = gen_huggingface(prompt, w, h, seed)
            elif name == "cloudflare":
                data, mime = gen_cloudflare(prompt, w, h, seed)
            else:
                data, mime = gen_pollinations(prompt, w, h, seed, False)
            return data, mime, name
        except Exception as e:
            # URL/key kabhi log mein nahi jaati, sirf error ka type aur message
            log.warning("image provider %s failed: %s %s", name, type(e).__name__, str(e)[:120])
    raise ImageError("all providers failed")

@app.post("/image")
def make_image(req: ImageRequest, request: Request):
    prompt = re.sub(r"\s+", " ", (req.prompt or "")).strip()[:500]
    if len(prompt) < 2:
        return fail("Batao kya image banani hai.", 400)
    ip = request.client.host if request.client else "?"
    cid = clean_cid(req.client_id) or ip
    now = time.time()
    if now - _img_last.get(cid, 0) < IMAGE_GAP_SECONDS:
        return fail("Thoda ruko, pichhli image abhi ban rahi hai.", 429)
    try:
        with db() as c:
            used = c.execute("SELECT COUNT(*) FROM events WHERE client_id=? AND mode='image' AND ts>=?", (cid, int(now) - 86400)).fetchone()[0]
        if used >= IMAGE_DAILY_LIMIT:
            return fail(f"Aaj ki {IMAGE_DAILY_LIMIT} images ki limit poori ho gayi. Kal phir try karo.", 429)
    except Exception:
        log.exception("image limit check failed")
    req.mode = "image"
    blocked = gate(req, request)
    if blocked:
        return blocked
    _img_last[cid] = now
    if len(_img_last) > 5000:
        for k in [k for k, t in _img_last.items() if now - t > 300]:
            _img_last.pop(k, None)

    try:
        final = enhance_prompt(prompt)
    except ImageError:
        return fail("Ye wali image main nahi bana sakta. Kuch aur idea bolo!", 400)
    style = IMAGE_STYLES.get(req.style, "")
    if style:
        final = f"{final}, {style}"
    w, h = IMAGE_SIZES.get(req.ratio, IMAGE_SIZES["square"])
    seed = req.seed if req.seed > 0 else random.randint(1, 999_999)

    if not IMAGE_SLOTS.acquire(timeout=25):
        return fail("Abhi bahut log image bana rahe hain. Thodi der baad try karo.", 503)
    try:
        data, mime, provider = run_providers(final, w, h, seed)
    except ImageError:
        return fail("Image abhi nahi ban payi. Thodi der baad dobara try karo.", 502)
    except Exception:
        log.exception("image failed")
        return fail("Kuch gadbad ho gayi. Dobara try karo.", 502)
    finally:
        IMAGE_SLOTS.release()
    log.info("image ok via %s (%d KB)", provider, len(data) // 1024)
    return {"image": f"data:{mime};base64," + base64.b64encode(data).decode("ascii"),
            "prompt": final, "seed": seed, "width": w, "height": h}

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