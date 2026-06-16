import os
import sys
import json
from datetime import datetime
from contextlib import contextmanager
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Ensure every print() flushes immediately — required for Railway log visibility
sys.stdout.reconfigure(line_buffering=True)

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MINI_APP_URL = os.getenv("MINI_APP_URL")
OWNER_TELEGRAM_IDS = [
    int(x.strip())
    for x in os.getenv("OWNER_TELEGRAM_IDS", "").split(",")
    if x.strip()
]
# Railway injects DATABASE_URL as postgres:// — psycopg2 requires postgresql://
DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgres://", "postgresql://", 1)


@contextmanager
def get_db():
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS access_requests (
                    user_id   BIGINT PRIMARY KEY,
                    status    TEXT NOT NULL,
                    first_name TEXT,
                    last_name  TEXT,
                    username   TEXT,
                    requested_at TEXT,
                    owner_notification_message_ids JSONB NOT NULL DEFAULT '{}'
                )
            """)


def db_get(uid: str) -> dict | None:
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM access_requests WHERE user_id = %s",
                (int(uid),)
            )
            row = cur.fetchone()
    if row is None:
        return None
    return {
        "status": row["status"],
        "first_name": row["first_name"] or "",
        "last_name": row["last_name"] or "",
        "username": row["username"] or "",
        "requested_at": row["requested_at"] or "",
        "owner_notification_message_ids": dict(row["owner_notification_message_ids"] or {}),
    }


def db_save(uid: str, record: dict):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO access_requests
                    (user_id, status, first_name, last_name, username,
                     requested_at, owner_notification_message_ids)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (user_id) DO UPDATE SET
                    status     = EXCLUDED.status,
                    first_name = EXCLUDED.first_name,
                    last_name  = EXCLUDED.last_name,
                    username   = EXCLUDED.username,
                    requested_at = EXCLUDED.requested_at,
                    owner_notification_message_ids = EXCLUDED.owner_notification_message_ids
            """, (
                int(uid),
                record["status"],
                record.get("first_name", ""),
                record.get("last_name", ""),
                record.get("username", ""),
                record.get("requested_at", ""),
                json.dumps(record.get("owner_notification_message_ids", {})),
            ))


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
    uid = str(update.effective_user.id)
    record = db_get(uid)

    if record and record["status"] == "approved":
        await update.message.reply_text(
            "Welcome back to <b>EASY MILLION</b> 🌱\n\nYou're approved! Tap below to open the app.",
            parse_mode="HTML",
            reply_markup=mini_app_markup()
        )
        return

    if record and record["status"] == "pending":
        await update.message.reply_text(
            "⏳ Your request is awaiting approval. We'll notify you once approved."
        )
        return

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
    record = db_get(uid)

    if record and record["status"] == "approved":
        await query.edit_message_text(
            "You're already approved! Use /start to open the app.",
            reply_markup=mini_app_markup()
        )
        return

    if record and record["status"] == "pending":
        await query.edit_message_text(
            "⏳ Your request is awaiting approval. We'll notify you once approved."
        )
        return

    username = f"@{user.username}" if user.username else "no username"
    first_name = user.first_name or ""
    last_name = user.last_name or ""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    new_record = {
        "status": "pending",
        "first_name": first_name,
        "last_name": last_name,
        "username": username,
        "requested_at": timestamp,
        "owner_notification_message_ids": {},
    }
    db_save(uid, new_record)

    await query.edit_message_text(
        "✅ Request sent! Please wait for approval. You'll be notified here once approved."
    )

    owner_keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Approve", callback_data=f"approve_{uid}"),
        InlineKeyboardButton("❌ Deny", callback_data=f"deny_{uid}"),
    ]])
    notification_text = build_notification_text(new_record, uid)

    for owner_id in OWNER_TELEGRAM_IDS:
        try:
            msg = await context.bot.send_message(
                chat_id=owner_id,
                text=notification_text,
                parse_mode="HTML",
                reply_markup=owner_keyboard,
            )
            new_record["owner_notification_message_ids"][str(owner_id)] = msg.message_id
        except Exception as e:
            print(f"Failed to notify owner {owner_id}: {e}")

    # Persist message_ids so handle_decision can edit both owner messages later
    db_save(uid, new_record)


async def handle_decision(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    acting_owner_id = query.from_user.id
    if acting_owner_id not in OWNER_TELEGRAM_IDS:
        return

    action, uid = query.data.split("_", 1)
    record = db_get(uid)

    if record is None:
        await query.edit_message_text("⚠️ This request no longer exists.")
        return

    # Race condition: other owner already decided
    if record["status"] in ("approved", "denied"):
        await context.bot.send_message(
            chat_id=acting_owner_id,
            text=f"ℹ️ This request was already {record['status']} by the other admin."
        )
        return

    new_status = "approved" if action == "approve" else "denied"
    record["status"] = new_status
    db_save(uid, record)

    acting_name = query.from_user.first_name or str(acting_owner_id)
    outcome_label = "✅ APPROVED" if new_status == "approved" else "❌ DENIED"
    result_text = build_notification_text(record, uid, f"— {outcome_label} by {acting_name}")

    try:
        await query.edit_message_text(result_text, parse_mode="HTML")
    except Exception as e:
        print(f"Failed to edit acting owner message: {e}")

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
                    parse_mode="HTML",
                )
            else:
                raise ValueError("No stored message_id")
        except Exception as e:
            print(f"Failed to edit other owner message: {e}")
            try:
                await context.bot.send_message(
                    chat_id=owner_id,
                    text=f"ℹ️ The access request from user <code>{uid}</code> was {new_status} by {acting_name}.",
                    parse_mode="HTML",
                )
            except Exception as e2:
                print(f"Fallback message to owner {owner_id} also failed: {e2}")

    int_uid = int(uid)
    if new_status == "approved":
        try:
            await context.bot.send_message(
                chat_id=int_uid,
                text="🎉 Access approved! Tap below to open the app.",
                reply_markup=mini_app_markup(),
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
        raise ValueError("TELEGRAM_BOT_TOKEN not set")
    if not MINI_APP_URL:
        raise ValueError("MINI_APP_URL not set")
    if not OWNER_TELEGRAM_IDS:
        raise ValueError("OWNER_TELEGRAM_IDS not set")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL not set")

    print("Connecting to database...")
    init_db()
    print("Database ready.")

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(request_access, pattern="^request_access$"))
    app.add_handler(CallbackQueryHandler(handle_decision, pattern="^(approve|deny)_"))

    print("Bot is running... Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
