# check_db.py
from db.session import SessionLocal
from db.models import SessionModel, MessageModel

print("=" * 50)
print("📊 DATABASE CONTENTS CHECK")
print("=" * 50)

db = SessionLocal()

# Count sessions
sessions = db.query(SessionModel).all()
print(f"\n Total Sessions: {len(sessions)}")

# Count messages
messages = db.query(MessageModel).all()
print(f"💬 Total Messages: {len(messages)}")

# Show recent messages
print("\n" + "=" * 50)
print("📝 RECENT MESSAGES (Last 10):")
print("=" * 50)

recent_msgs = db.query(MessageModel).order_by(MessageModel.timestamp.desc()).limit(10).all()

for msg in recent_msgs:
    role_emoji = "👤" if msg.role == "user" else "🤖" if msg.role == "assistant" else "⚙️"
    print(f"\n{role_emoji} [{msg.role.upper()}] at {msg.timestamp.strftime('%H:%M:%S')}")
    print(f"   {msg.content[:100]}...")

db.close()
print("\n" + "=" * 50)
print("✅ Database check complete!")