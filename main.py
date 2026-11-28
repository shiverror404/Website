import asyncio
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
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timedelta
from urllib.parse import quote

import requests
from fastapi import FastAPI, Header, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

try:
    import edge_tts  # pip install edge-tts  (natural Hindi/English neural awaaz)
except Exception:  # package na ho to /tts band rahega, app browser voice par chalega
    edge_tts = None

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

NO_FAKE_IMAGE_RULE = (
    "IMAGE ABILITY:\n"
    "- JARVIS CAN create images. When the user asks what you can do, mention image creation too.\n"
    "- The app creates the image automatically when the user clearly asks for one, for example: ek robot ki image banao.\n"
    "- If the user asks whether you can make images, say yes and tell them to just describe what they want, with one short example.\n"
    "- You never draw inside chat text. NEVER write markdown images, image links or fake attachments.\n"
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

VOICE_NOTE = (
    "VOICE CALL MODE: the user is talking to you by voice and your reply will be SPOKEN ALOUD.\n"
    "- Reply in 1 to 3 short, natural spoken sentences, like a friend on a phone call.\n"
    "- No emojis, no lists, no numbering, no code, no markdown, no 'Tip'.\n"
    "- Use simple everyday words. Write Hinglish in Roman letters (never Devanagari).\n"
    "- If the answer is long, give the short answer first and offer to explain more.\n"
    "- If code is needed, say you are putting it in the chat instead of reading it out.\n"
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
        f"{NO_FAKE_IMAGE_RULE}\n"
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
    voice: bool = False  # True = voice mode, jawab bolkar sunaya jayega


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

def models_and_variants(has_images: bool, mode: str = "normal", voice: bool = False):
    """(models, extra-params) ki list. Pehle extras ke saath try, 400/404 aaye to bina extras ke."""
    if has_images:
        models = [VISION_MODEL] + ([VISION_FALLBACK] if VISION_FALLBACK and VISION_FALLBACK != VISION_MODEL else [])
        return models, [{"reasoning_effort": "none"}, {}]
    # gpt-oss reasoning model hai: bina limit ke wo jawab se pehle kai second "sochta" hai (yahi bada delay tha).
    # Chat/voice ke liye 'low' kaafi hai; sirf coder mode mein thoda zyada sochne do.
    effort = "medium" if (mode == "coder" and not voice) else "low"
    return [DEFAULT_MODEL], [{"reasoning_effort": effort}, {}]

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
    return {"status": "ok", "bot": BOT_NAME, "stream": True, "image": image_providers_ready(),
            "stt": bool(GROQ_API_KEY), "tts": edge_tts is not None}

def build_payload(req: ChatRequest):
    mode = req.mode if req.mode in MODE_NOTES else "normal"
    system = system_prompt(req.style, req.user_name)
    if mode == "coder":
        system = system.replace(LANGUAGE_RULE, CODER_LANGUAGE_RULE)
    if mode in MODE_NOTES:
        system += "\n" + MODE_NOTES[mode]
    if req.voice:
        system += "\n" + VOICE_NOTE
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
    if req.voice and not images:
        tokens = min(tokens, 260)  # bolne wala jawab chhota hi chahiye
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
    models, variants = models_and_variants(bool(req.images), mode, req.voice)
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
    models, variants = models_and_variants(bool(req.images), mode, req.voice)

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


# ---------------------------------------------------------------------------
# VOICE: /stt (awaaz -> text, Groq Whisper) aur /tts (text -> natural awaaz, Edge neural voices)
# Browser ka SpeechRecognition (jo har baar Google wali "ting" bajata tha) ab use nahi hota.
# ---------------------------------------------------------------------------
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
STT_MAX_BYTES = 4_000_000
# Roman-script Hinglish ka namuna: Whisper ko batata hai ki kis tarah ka text chahiye
STT_PROMPT_HI = "Haan bhai, kya haal hai? Mujhe batao, aaj ka plan kya hai? Theek hai, chalo shuru karte hain."
STT_HALLUCINATIONS = ("thanks for watching", "thank you for watching", "subtitles by", "amara.org", "please subscribe")
_voice_hits = {}

def voice_rate_ok(ip: str, limit: int = 90) -> bool:
    """Ek IP se 1 minute mein limit se zyada voice calls nahi."""
    now = time.time()
    recent = [t for t in _voice_hits.get(ip, []) if now - t < 60]
    if len(recent) >= limit:
        _voice_hits[ip] = recent
        return False
    recent.append(now)
    _voice_hits[ip] = recent
    if len(_voice_hits) > 5000:
        for k in [k for k, v in _voice_hits.items() if not v or now - v[-1] > 120]:
            _voice_hits.pop(k, None)
    return True

def voice_gate(request: Request, cid_raw: str):
    """Maintenance / blocked device check (message count nahi badhata). Error response ya None."""
    ip = request.client.host if request.client else "?"
    if not voice_rate_ok(ip):
        return fail("Bahut jaldi jaldi bol rahe ho. Thoda ruko.", 429)
    try:
        if get_setting("paused") == "1":
            return fail(get_setting("pause_message") or "JARVIS abhi maintenance mein hai. Thodi der baad aana.", 503)
        cid = clean_cid(cid_raw)
        if cid:
            with db() as c:
                row = c.execute("SELECT blocked FROM users WHERE client_id=?", (cid,)).fetchone()
            if row and row[0]:
                return fail("Is device ka access band kar diya gaya hai.", 403)
    except Exception:
        log.exception("voice gate failed")
    return None

def _stt_sync(data: bytes, mime: str, lang: str) -> str:
    ext = "wav" if "wav" in mime else "webm" if "webm" in mime else "mp4" if ("mp4" in mime or "m4a" in mime) else "ogg" if "ogg" in mime else "wav"
    form = {"model": STT_MODEL, "response_format": "verbose_json", "temperature": "0"}
    if lang in ("hi", "en"):
        form["language"] = lang
    if lang == "hi":
        form["prompt"] = STT_PROMPT_HI
    r = http.post(STT_URL, headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                  files={"file": (f"speech.{ext}", data, mime or "audio/wav")}, data=form, timeout=(8, 30))
    r.raise_for_status()
    j = r.json()
    segs = j.get("segments") or []
    if segs:
        # jin tukdon mein Whisper ko khud "yahan awaaz nahi hai" lage unhe hata do (khamoshi par fake text aata hai)
        text = " ".join(str(sg.get("text", "")) for sg in segs
                        if not (sg.get("no_speech_prob", 0) > 0.6 and sg.get("avg_logprob", 0) < -0.8))
    else:
        text = j.get("text", "")
    text = re.sub(r"\s+", " ", text).strip()
    low = text.lower()
    if any(h in low for h in STT_HALLUCINATIONS):
        return ""
    return text

@app.post("/stt")
async def speech_to_text(request: Request, lang: str = "hi", x_client_id: str = Header(default="")):
    """Body = raw audio (wav/webm/mp4). Jawab: {"text": "..."}"""
    if not GROQ_API_KEY:
        return fail("Server setup adhura hai: GROQ_API_KEY set nahi hai.", 500)
    blocked = voice_gate(request, x_client_id)
    if blocked:
        return blocked
    data = await request.body()
    if len(data) < 1500:
        return {"text": ""}
    if len(data) > STT_MAX_BYTES:
        return fail("Audio bahut lamba hai. Chhota bolo.", 413)
    mime = (request.headers.get("content-type") or "audio/wav").split(";")[0].strip().lower()
    try:
        text = await run_in_threadpool(_stt_sync, data, mime, lang if lang in ("hi", "en") else "hi")
        return {"text": text[:1000]}
    except requests.exceptions.Timeout:
        return fail("Awaaz samajhne mein der lag gayi. Dobara bolo.", 504)
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        log.warning("stt http error %s", code)
        return fail(f"Awaaz samajhne wala server busy hai (code {code}). Thodi der baad try karo.", 502)
    except Exception:
        log.exception("stt failed")
        return fail("Awaaz samajh nahi aayi. Dobara bolo.", 502)


TTS_VOICES = {
    "madhur": "hi-IN-MadhurNeural",    # Hindi male
    "swara": "hi-IN-SwaraNeural",      # Hindi female
    "prabhat": "en-IN-PrabhatNeural",  # Indian-English male
    "neerja": "en-IN-NeerjaNeural",    # Indian-English female
}
TTS_DEFAULT_VOICE = os.getenv("TTS_VOICE", "madhur")
TTS_RATE = os.getenv("TTS_RATE", "+0%")
TRANSLIT_MODEL = os.getenv("TRANSLIT_MODEL", "llama-3.1-8b-instant")
TRANSLIT_SYSTEM = (
    "You prepare Hinglish text for a Hindi text-to-speech engine. Rewrite the text so that every Hindi word is in "
    "Devanagari script, while every English word stays in English (Latin) letters. Keep punctuation, numbers, names "
    "and meaning exactly the same. Do not translate, add, remove or explain anything. Output only the rewritten text."
)
# Roman Hindi ke aam shabd (English se na takrane wale). Inme se koi mile to text Hinglish maana jata hai.
HINDI_HINTS = set((
    "hai hain nahi nahin kya kyu kyun kaise kaisa kaisi kaun kab kahan aap aapka aapki tum tumhe tumhara mujhe mera meri mere "
    "tera teri tere bhai yaar haan hoon hu hoga hogi karo karna karte kar raha rahi rahe wala wali bahut bohot accha acha "
    "theek thik aur lekin kuch koi abhi chalo batao bolo bol dost arre arrey sab kal aaj mat sakta sakte sakti chahiye "
    "matlab samjha samjhe liye isme usme yeh woh iska uska ke ki ka mein pe hota hote dena lena jao aao dekho suno sunao "
    "bata bataya banaya shukriya dhanyavad namaste kitna kitni kitne kaam padhai ghar paisa mast maza"
).split())
_tts_cache = OrderedDict()
_translit_cache = {}
_translit_lock = threading.Lock()
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]")

def clean_tts_text(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " code chat mein hai. ", text or "")
    text = re.sub(r"https?://\S+", " ", text)
    text = EMOJI_RE.sub("", text)
    text = re.sub(r"(?im)^\s*tip\s*:-?\s*", "Tip. ", text)
    text = re.sub(r"[*_`#>~|]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def needs_translit(text: str) -> bool:
    return any(w in HINDI_HINTS for w in re.findall(r"[a-z]+", text.lower()))

def to_devanagari(text: str) -> str:
    """Roman Hinglish -> Hindi shabd Devanagari mein (English shabd waise hi). Fail ho to original text."""
    with _translit_lock:
        if text in _translit_cache:
            return _translit_cache[text]
    try:
        payload = {"messages": [{"role": "system", "content": TRANSLIT_SYSTEM}, {"role": "user", "content": text}],
                   "temperature": 0, "max_tokens": 600}
        r = open_groq(payload, [TRANSLIT_MODEL, DEFAULT_MODEL], [{}], False, (5, 10))
        r.raise_for_status()
        out = strip_think((r.json()["choices"][0]["message"].get("content") or "").strip()).strip("\"' ")
        if not out or len(out) > len(text) * 3 + 20 or "\n\n" in out:
            return text
    except Exception as e:
        log.warning("translit failed: %s", type(e).__name__)
        return text
    with _translit_lock:
        if len(_translit_cache) > 300:
            _translit_cache.clear()
        _translit_cache[text] = out
    return out

class TTSRequest(BaseModel):
    text: str
    voice: str = ""
    client_id: str = ""

async def _edge_synth(text: str, voice: str) -> bytes:
    comm = edge_tts.Communicate(text, voice, rate=TTS_RATE)
    out = bytearray()
    async for ch in comm.stream():
        if ch.get("type") == "audio":
            out += ch["data"]
    return bytes(out)

@app.post("/tts")
async def text_to_speech(req: TTSRequest, request: Request):
    """Text -> mp3 (natural neural awaaz). Fail ho to JSON error; app browser voice par chala jata hai."""
    if edge_tts is None:
        return fail("TTS install nahi hai (requirements.txt mein edge-tts add karo).", 501)
    blocked = voice_gate(request, req.client_id)
    if blocked:
        return blocked
    text = clean_tts_text(req.text)[:600]
    if not re.search(r"\w", text):
        return Response(status_code=204)
    key_name = req.voice if req.voice in TTS_VOICES else TTS_DEFAULT_VOICE
    voice = TTS_VOICES.get(key_name, TTS_VOICES["madhur"])
    spoken = text
    if voice.startswith("hi-IN") and re.search(r"[A-Za-z]", text) and needs_translit(text):
        spoken = await run_in_threadpool(to_devanagari, text)
    ck = (voice, spoken)
    audio = _tts_cache.get(ck)
    if audio is None:
        last = None
        for attempt in range(2):
            try:
                audio = await asyncio.wait_for(_edge_synth(spoken, voice), timeout=15)
                if audio:
                    break
            except Exception as e:
                last = e
                await asyncio.sleep(0.3)
        if not audio:
            log.warning("tts failed: %s %s", type(last).__name__, str(last)[:120])
            return fail("Awaaz abhi nahi ban payi.", 502)
        _tts_cache[ck] = audio
        if len(_tts_cache) > 80:
            _tts_cache.popitem(last=False)
    else:
        _tts_cache.move_to_end(ck)
    return Response(content=audio, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})

INTENT_MODEL = os.getenv("INTENT_MODEL", "llama-3.1-8b-instant")
INTENT_SYSTEM = (
    "You are the router for JARVIS, a chat app that can also create images. "
    "Decide if the user's LATEST message is a request to CREATE, GENERATE, DRAW or EDIT a picture right now "
    "(image, photo, logo, poster, wallpaper, DP, art and similar). Messages may be Hindi, Hinglish or English with typos. "
    "Answer image ONLY when the user clearly wants a new picture made. "
    "Answer chat for: questions about whether or how you can make images (for example: image bana sakte ho, Image create ???, image kaise banate hain), "
    "questions about existing images, code or design help, and normal conversation. "
    "Use the recent chat for follow-ups like: isko anime style mein bana do. Then write a complete standalone prompt. "
    "Reply with ONLY one JSON object: "
    '{"intent":"image" or "chat","prompt":"standalone image description, only if image","style":"auto|realistic|cinematic|anime|digital|3d|sketch","ratio":"square|portrait|landscape"}. '
    "ratio: phone wallpaper, story or poster means portrait; desktop wallpaper, banner or thumbnail means landscape; logo, DP or profile picture means square; otherwise square. "
    "style: pick one only if the user asked for it or clearly implies it, else auto."
)

class IntentRequest(BaseModel):
    message: str
    history: list = []

@app.post("/intent")
def image_intent(req: IntentRequest):
    """Message image banwana chahta hai ya sirf baat kar raha hai? Fail ho to hamesha 'chat'."""
    if not GROQ_API_KEY:
        return {"intent": "chat"}
    try:
        ctx = "\n".join(f"{'user' if m.get('role') == 'user' else 'assistant'}: {str(m.get('content', ''))[:200]}"
                        for m in (req.history or [])[-4:] if isinstance(m, dict))
        payload = {
            "messages": [{"role": "system", "content": INTENT_SYSTEM},
                         {"role": "user", "content": f"Recent chat:\n{ctx}\n\nLatest user message:\n{(req.message or '')[:500]}"}],
            "temperature": 0, "max_tokens": 600,
        }
        raw = call_groq(payload, [INTENT_MODEL, DEFAULT_MODEL])
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        if data.get("intent") != "image":
            return {"intent": "chat"}
        prompt = str(data.get("prompt") or "").strip()[:500]
        return {"intent": "image", "prompt": prompt,
                "style": data.get("style") if data.get("style") in IMAGE_STYLES else "auto",
                "ratio": data.get("ratio") if data.get("ratio") in IMAGE_SIZES else "square"}
    except Exception:
        log.warning("intent check failed")
        return {"intent": "chat"}


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