import os
import sys
import time
import requests

def main():
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        print("❌ Error: TELEGRAM_TOKEN environment variable is not set!")
        print("Please set it first using: set TELEGRAM_TOKEN=your_token_here")
        return

    print("🔍 Fetching bot information...")
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
        r.raise_for_status()
        bot_info = r.json()
    except Exception as e:
        print(f"❌ Failed to connect to Telegram API: {e}")
        return

    if not bot_info.get("ok"):
        print(f"❌ Telegram API returned an error: {bot_info}")
        return

    bot_user = bot_info["result"]["username"]
    bot_name = bot_info["result"]["first_name"]
    
    print("\n==================================================")
    print(f"🤖 BOT FOUND: {bot_name} (@{bot_user})")
    print("==================================================")
    print(f"👉 Click this link to open your exact bot directly:")
    print(f"   https://t.me/{bot_user}")
    print("👉 Click START (or send a message like '/start' or 'hi')")
    print("==================================================")
    
    print("\n⏳ Waiting for your message... (Send a message now to get your CHAT ID)")
    
    while True:
        try:
            r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=10)
            r.raise_for_status()
            updates = r.json()
        except Exception as e:
            print(f"\r❌ API Error: {e}", end="")
            time.sleep(2)
            continue

        results = updates.get("result", [])
        if not results:
            print("\r⏳ Still waiting... (No messages received yet)", end="", flush=True)
            time.sleep(2)
            continue

        # Get the latest message
        latest_update = results[-1]
        if "message" in latest_update:
            msg = latest_update["message"]
            chat_id = msg["chat"]["id"]
            user_name = msg["from"].get("first_name", "User")
            text = msg.get("text", "")
            
            print("\n\n==================================================")
            print("🎉 SUCCESS! MESSAGE RECEIVED!")
            print("==================================================")
            print(f"👤 From: {user_name}")
            print(f"💬 Text: \"{text}\"")
            print(f"🆔 YOUR CORRECT CHAT ID IS: {chat_id}")
            print("==================================================")
            print("\nTo set this on your VPS, run:")
            print(f"   set TELEGRAM_CHAT_ID={chat_id}")
            print("==================================================")
            break
        else:
            time.sleep(2)

if __name__ == "__main__":
    main()