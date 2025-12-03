import os
import logging
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ------------- Config ------------- #

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "x-ai/grok-4.1-fast:free")

# Optional: configure these in env if your OpenRouter key expects them
OPENROUTER_REFERRER = os.getenv("OPENROUTER_REFERRER")  # e.g. https://your-site.com
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "Telegram Grok Vision Bot")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Per-process in-memory storage
user_api_keys: dict[int, str] = {}   # telegram_user_id -> openrouter_api_key
waiting_for_key: set[int] = set()    # users who just ran /set_api_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()


# ------------- Telegram helpers ------------- #

def send_message(chat_id: int, text: str):
    """Send a message to a Telegram chat, splitting if too long."""
    MAX_LEN = 4000
    if len(text) <= MAX_LEN:
        _send_message_raw(chat_id, text)
    else:
        for i in range(0, len(text), MAX_LEN):
            _send_message_raw(chat_id, text[i:i + MAX_LEN])


def _send_message_raw(chat_id: int, text: str):
    resp = None
    try:
        resp = requests.post(
            TELEGRAM_API_URL + "sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=20,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.error("Error sending Telegram message: %s - resp=%s", e, getattr(resp, "text", ""))


def get_file_url(file_id: str) -> str:
    """Get a public HTTPS URL for a Telegram file (photo)."""
    resp = requests.get(
        TELEGRAM_API_URL + "getFile",
        params={"file_id": file_id},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram getFile error: {data}")
    file_path = data["result"]["file_path"]
    return f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"


# ------------- OpenRouter / Grok helpers ------------- #

def _openrouter_headers(api_key: str) -> dict:
    """Build headers for OpenRouter, with optional referrer/title."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_REFERRER:
        headers["HTTP-Referer"] = OPENROUTER_REFERRER
    if OPENROUTER_TITLE:
        headers["X-Title"] = OPENROUTER_TITLE
    return headers


def call_grok_text(api_key: str, user_text: str) -> str:
    """Text-only chat with Grok."""
    headers = _openrouter_headers(api_key)

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that can also analyze images."
            },
            {
                "role": "user",
                "content": user_text,
            },
        ],
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
        if not resp.ok:
            # Show full error body to user for easier debugging
            return f"❌ OpenRouter error {resp.status_code}: {resp.text}"
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        logger.exception("Error calling Grok (text): %s", e)
        return f"❌ Error talking to Grok: {e}"


def analyze_image_with_grok(api_key: str, prompt: str, image_url: str) -> str:
    """Send an image + text prompt to Grok for vision analysis."""
    headers = _openrouter_headers(api_key)

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
        if not resp.ok:
            return f"❌ OpenRouter error {resp.status_code}: {resp.text}"
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        logger.exception("Error calling Grok (vision): %s", e)
        return f"❌ Error while analyzing the image: {e}"


# ------------- Update handling ------------- #

def handle_update(update: dict):
    """Process a single Telegram update dict (text + photo support)."""
    if "message" not in update:
        return

    message = update["message"]
    chat_id = message["chat"]["id"]
    from_user = message.get("from", {})
    user_id = from_user.get("id")

    if user_id is None:
        return

    text = message.get("text")
    photo = message.get("photo")
    caption = message.get("caption")

    # 1) Commands and text
    if text:
        text = text.strip()

        # Commands
        if text.startswith("/start"):
            handle_start(chat_id)
            return

        if text.startswith("/set_api_key"):
            handle_set_api_key_command(chat_id, user_id)
            return

        if text.startswith("/forget_key"):
            handle_forget_key(chat_id, user_id)
            return

        # If waiting for API key, treat this text as the key
        if user_id in waiting_for_key:
            user_api_keys[user_id] = text
            waiting_for_key.remove(user_id)
            send_message(chat_id, "✅ Your OpenRouter API key has been saved.")
            return

        # Normal text chat with Grok
        api_key = user_api_keys.get(user_id)
        if not api_key:
            send_message(
                chat_id,
                "⚠️ You haven’t set an OpenRouter API key yet.\nUse /set_api_key first."
            )
            return

        reply = call_grok_text(api_key, text)
        send_message(chat_id, reply)
        return

    # 2) Photo (vision analysis)
    if photo:
        api_key = user_api_keys.get(user_id)
        if not api_key:
            send_message(
                chat_id,
                "⚠️ You haven’t set an OpenRouter API key yet.\nUse /set_api_key first."
            )
            return

        try:
            # largest size is last item
            file_id = photo[-1]["file_id"]
            image_url = get_file_url(file_id)
        except Exception as e:
            logger.exception("Error getting Telegram file URL: %s", e)
            send_message(chat_id, "❌ Couldn’t fetch the image from Telegram.")
            return

        # Use caption as prompt if present, otherwise a default prompt
        if caption:
            prompt = caption.strip()
        else:
            prompt = "Describe this image in detail and point out anything interesting or unusual."

        reply = analyze_image_with_grok(api_key, prompt, image_url)
        send_message(chat_id, reply)
        return

    # Ignore other update types for now (video, stickers, documents, etc.)


def handle_start(chat_id: int):
    text = (
        "👋 Hi! I’m a Grok-powered bot via OpenRouter.\n\n"
        "I can:\n"
        "• Chat with you using text\n"
        "• Analyze images you send (photos)\n\n"
        "To use me, you need *your own* OpenRouter API key:\n"
        "1️⃣ Get an API key from OpenRouter.\n"
        "2️⃣ Use /set_api_key and send me your key.\n"
        "3️⃣ Then send text or photos and I’ll use Grok to respond.\n\n"
        "You can remove your key with /forget_key."
    )
    send_message(chat_id, text)


def handle_set_api_key_command(chat_id: int, user_id: int):
    waiting_for_key.add(user_id)
    text = (
        "🔑 Please send me your *OpenRouter API key* as the **next message**.\n\n"
        "It will be kept only in memory in this simple version "
        "(if the bot restarts, you’ll need to set it again).\n\n"
        "You can clear it later with /forget_key."
    )
    send_message(chat_id, text)


def handle_forget_key(chat_id: int, user_id: int):
    user_api_keys.pop(user_id, None)
    waiting_for_key.discard(user_id)
    send_message(chat_id, "✅ Your stored API key has been removed.")


# ------------- FastAPI routes ------------- #

@app.get("/")
async def root():
    return {"status": "ok", "message": "Grok vision bot is running"}


@app.post("/webhook")
async def telegram_webhook(request: Request):
    """Telegram will POST updates here."""
    update = await request.json()
    logger.info("Received update: %s", update)
    try:
        handle_update(update)
    except Exception as e:
        logger.exception("Error handling update: %s", e)
    # Telegram just needs a quick 200 OK
    return JSONResponse(content={"ok": True})
