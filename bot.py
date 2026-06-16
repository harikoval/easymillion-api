import os
import json
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MINI_APP_URL = os.getenv("MINI_APP_URL")
OWNER_TELEGRAM_IDS = [
    int(x.strip())
    for x in os.getenv("OWNER_TELEGRAM_IDS", "").split(",")
    if x.strip()
]

STORAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "access_requests.json")


def load_requests():
    if not os.path.exists(STORAGE_FILE):
        return {}
    with open(STORAGE_FILE, "r") as f:
        return json.load(f)


def save_requests(data):
    with open(STORAGE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def mini_app_markup():
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("Open EASY MILLION 🚀", web_app=WebAppInfo(url=MINI_APP_URL))
    ]])


def build_notification_text(record, uid, outcome_line=""):
    text = (
        f"🔔 <b>New Access Request</b>\n\n"
        f"Name: {record['first_name']} {record['last_name']}\n"
        f"Username: {record['username']}\n"
        f"User ID: <code>{uid}</code>\n"
        f"Requested: {record['requested_at']}"
    )
    if outcome_line:
        text += f"\n\n{outcome_line}"
    return text


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    uid = str(user.id)
    data = load_requests()

    if uid in data and data[uid]["status"] == "approved":
        await update.message.reply_text(
            "Welcome back to <b>EASY MILLION</b> 🌱\n\nYou're approved! Tap below to open the app.",
            parse_mode="HTML",
            reply_markup=mini_app_markup()
        )
        return

    if uid in data and data[uid]["status"] == "pending":
        await update.message.reply_text(
            "⏳ Your request is awaiting approval. We'll notify you once approved."
        )
        return

    # New visitor or previously denied — show request button
    keyboard = [[InlineKeyboardButton("🔓 Request Access", callback_data="request_access")]]
    await update.message.reply_text(
        "Welcome to <b>EASY MILLION</b> 🌱\n\nTap below to request access.",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def request_access(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user = query.from_user
    uid = str(user.id)
    data = load_requests()

    if uid in data and data[uid]["status"] == "approved":
        await query.edit_message_text(
            "You're already approved! Use /start to open the app.",
            reply_markup=mini_app_markup()
        )
        return

    if uid in data and data[uid]["status"] == "pending":
        await query.edit_message_text(
            "⏳ Your request is awaiting approval. We'll notify you once approved."
        )
        return

    # Save as pending
    username = f"@{user.username}" if user.username else "no username"
    first_name = user.first_name or ""
    last_name = user.last_name or ""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    data[uid] = {
        "status": "pending",
        "first_name": first_name,
        "last_name": last_name,
        "username": username,
        "requested_at": timestamp,
        "owner_notification_message_ids": {}
    }
    save_requests(data)

    await query.edit_message_text(
        "✅ Request sent! Please wait for approval. You'll be notified here once approved."
    )

    # Notify both owners
    owner_keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Approve", callback_data=f"approve_{uid}"),
        InlineKeyboardButton("❌ Deny", callback_data=f"deny_{uid}")
    ]])
    notification_text = build_notification_text(data[uid], uid)

    for owner_id in OWNER_TELEGRAM_IDS:
        try:
            msg = await context.bot.send_message(
                chat_id=owner_id,
                text=notification_text,
                parse_mode="HTML",
                reply_markup=owner_keyboard
            )
            data[uid]["owner_notification_message_ids"][str(owner_id)] = msg.message_id
        except Exception as e:
            print(f"Failed to notify owner {owner_id}: {e}")

    save_requests(data)


async def handle_decision(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    acting_owner_id = query.from_user.id
    if acting_owner_id not in OWNER_TELEGRAM_IDS:
        return

    action, uid = query.data.split("_", 1)
    data = load_requests()

    if uid not in data:
        await query.edit_message_text("⚠️ This request no longer exists.")
        return

    record = data[uid]

    # Race condition: already decided by the other owner
    if record["status"] in ("approved", "denied"):
        await context.bot.send_message(
            chat_id=acting_owner_id,
            text=f"ℹ️ This request was already {record['status']} by the other admin."
        )
        return

    new_status = "approved" if action == "approve" else "denied"
    record["status"] = new_status
    save_requests(data)

    acting_name = query.from_user.first_name or str(acting_owner_id)
    outcome_label = "✅ APPROVED" if new_status == "approved" else "❌ DENIED"
    result_text = build_notification_text(record, uid, f"— {outcome_label} by {acting_name}")

    # Edit the acting owner's message (removes buttons)
    try:
        await query.edit_message_text(result_text, parse_mode="HTML")
    except Exception as e:
        print(f"Failed to edit acting owner message: {e}")

    # Sync the other owner's notification
    for owner_id in OWNER_TELEGRAM_IDS:
        if owner_id == acting_owner_id:
            continue
        other_msg_id = record["owner_notification_message_ids"].get(str(owner_id))
        try:
            if other_msg_id:
                await context.bot.edit_message_text(
                    chat_id=owner_id,
                    message_id=other_msg_id,
                    text=result_text,
                    parse_mode="HTML"
                )
            else:
                raise ValueError("No stored message_id")
        except Exception as e:
            print(f"Failed to edit other owner message: {e}")
            try:
                await context.bot.send_message(
                    chat_id=owner_id,
                    text=f"ℹ️ The access request from user <code>{uid}</code> was {new_status} by {acting_name}.",
                    parse_mode="HTML"
                )
            except Exception as e2:
                print(f"Fallback message to owner {owner_id} also failed: {e2}")

    # Notify the requesting user
    int_uid = int(uid)
    if new_status == "approved":
        try:
            await context.bot.send_message(
                chat_id=int_uid,
                text="🎉 Access approved! Tap below to open the app.",
                reply_markup=mini_app_markup()
            )
        except Exception as e:
            print(f"Failed to notify user {uid} of approval: {e}")
    else:
        try:
            await context.bot.send_message(
                chat_id=int_uid,
                text="Your access request was not approved at this time."
            )
        except Exception as e:
            print(f"Failed to notify user {uid} of denial: {e}")


def main():
    if not BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")
    if not MINI_APP_URL:
        raise ValueError("MINI_APP_URL not set in .env")
    if not OWNER_TELEGRAM_IDS:
        raise ValueError("OWNER_TELEGRAM_IDS not set in .env")

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(request_access, pattern="^request_access$"))
    app.add_handler(CallbackQueryHandler(handle_decision, pattern="^(approve|deny)_"))

    print("Bot is running... Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
