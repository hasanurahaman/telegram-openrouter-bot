import os
import time
import logging
import requests
from dotenv import load_dotenv

# ------------- Config ------------- #

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "x-ai/grok-4.1-fast:free")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# In-memory storage
user_api_keys: dict[int, str] = {}   # telegram_user_id -> openrouter_api_key
waiting_for_key: set[int] = set()    # users who just ran /set_api_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ------------- Telegram helpers ------------- #

def get_updates(offset=None, timeout=30):
    """Poll updates from Telegram using getUpdates."""
    params = {
        "timeout": timeout,
    }
    if offset is not None:
        params["offset"] = offset

    resp = requests.get(TELEGRAM_API_URL + "getUpdates", params=params, timeout=timeout+5)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data}")
    return data["result"]


def send_message(chat_id: int, text: str):
    """Send a message to a Telegram chat, splitting if too long."""
    MAX_LEN = 4000
    if len(text) <= MAX_LEN:
        _send_message_raw(chat_id, text)
    else:
        for i in range(0, len(text), MAX_LEN):
            _send_message_raw(chat_id, text[i:i + MAX_LEN])


def _send_message_raw(chat_id: int, text: str):
    resp = requests.post(
        TELEGRAM_API_URL + "sendMessage",
        json={
            "chat_id": chat_id,
            "text": text,
        },
        timeout=20,
    )
    try:
        resp.raise_for_status()
    except Exception as e:
        logger.error("Error sending message: %s - response: %s", e, resp.text)


# ------------- OpenRouter helper ------------- #

def call_openrouter(api_key: str, user_text: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional but recommended:
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


# ------------- Update handler ------------- #

def handle_update(update: dict):
    """Process a single Telegram update."""
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

    # If we’re waiting for this user's key, treat text as API key
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
            "⚠️ You haven’t set an OpenRouter API key yet.\n"
            "Use /set_api_key first."
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


# ------------- Main loop ------------- #

def main():
    logger.info("🤖 Bot is running (raw Telegram Bot API + requests). Press Ctrl+C to stop.")

    last_update_id = None

    while True:
        try:
            updates = get_updates(offset=last_update_id, timeout=30)
            for update in updates:
                last_update_id = update["update_id"] + 1
                handle_update(update)
        except KeyboardInterrupt:
            print("\nStopping bot.")
            break
        except Exception as e:
            logger.error("Error in update loop: %s", e)
            # small sleep so we don't spin like crazy on errors
            time.sleep(5)


if __name__ == "__main__":
    main()
