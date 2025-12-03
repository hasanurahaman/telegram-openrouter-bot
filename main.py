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

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# In-memory storage (per process)
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
    try:
        resp = requests.post(
            TELEGRAM_API_URL + "sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=20,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.error("Error sending Telegram message: %s - resp=%s", e, getattr(resp, "text", ""))


# ------------- OpenRouter helper ------------- #

def call_openrouter(api_key: str, user_text: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # optional but recommended:
        "HTTP-Referer": "https://example.com",  # change to your site/repo if you have one
        "X-Title": "Telegram OpenRouter Bot",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_text},
        ],
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        logger.exception("Error talking to OpenRouter")
        return f"❌ Error talking to OpenRouter: {e}"


# ------------- Update handling ------------- #

def handle_update(update: dict):
    """Process a single Telegram update dict."""
    if "message" not in update:
        return

    message = update["message"]
    chat_id = message["chat"]["id"]
    from_user = message.get("from", {})
    user_id = from_user.get("id")
    text = message.get("text")

    if user_id is None or text is None:
        return

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

    # If we’re waiting for this user's key, treat message as the key
    if user_id in waiting_for_key:
        user_api_keys[user_id] = text
        waiting_for_key.remove(user_id)
        send_message(chat_id, "✅ Your OpenRouter API key has been saved.")
        return

    # Normal chat
    api_key = user_api_keys.get(user_id)
    if not api_key:
        send_message(
            chat_id,
            "⚠️ You haven’t set an OpenRouter API key yet.\nUse /set_api_key first."
        )
        return

    reply = call_openrouter(api_key, text)
    send_message(chat_id, reply)


def handle_start(chat_id: int):
    text = (
        "👋 Hi! I’m an AI chat bot using OpenRouter.\n\n"
        "To use me, you need *your own* OpenRouter API key:\n"
        "1️⃣ Get an API key from OpenRouter.\n"
        "2️⃣ Use /set_api_key and send me your key.\n"
        "3️⃣ Then just chat with me normally.\n\n"
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
    return {"status": "ok"}


@app.post("/webhook")
async def telegram_webhook(request: Request):
    """Telegram will POST updates here."""
    update = await request.json()
    logger.info("Received update: %s", update)
    try:
        handle_update(update)
    except Exception as e:
        logger.exception("Error handling update: %s", e)
    # Telegram requires a 200 OK quickly
    return JSONResponse(content={"ok": True})
