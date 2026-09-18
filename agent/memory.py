# agent/memory.py
from db.session import SessionLocal, engine, Base
from db.models import SessionModel, MessageModel
import logging

logger = logging.getLogger(__name__)

# Automatically create the database tables if they don't exist yet
Base.metadata.create_all(bind=engine)


class MemoryManager:
    def __init__(self):
        self.current_session_id = self._create_new_session()
        logger.info(f"📂 Started new conversation session: {self.current_session_id}")

    def _create_new_session(self) -> int:
        db = SessionLocal()
        try:
            new_session = SessionModel()
            db.add(new_session)
            db.commit()
            db.refresh(new_session)
            return new_session.id
        finally:
            db.close()

    def add_message(self, role: str, content: str):
        """Saves a message to the database."""
        db = SessionLocal()
        try:
            msg = MessageModel(session_id=self.current_session_id, role=role, content=content)
            db.add(msg)
            db.commit()
        finally:
            db.close()

    def get_recent_history(self, limit: int = 10) -> list:
        """Retrieves the last N messages for the current session."""
        db = SessionLocal()
        try:
            # Get messages, ordered by newest first, then limit
            messages = db.query(MessageModel).filter(
                MessageModel.session_id == self.current_session_id
            ).order_by(MessageModel.timestamp.desc()).limit(limit).all()

            # Reverse the list so it's in chronological order (oldest to newest)
            messages.reverse()

            return [{"role": msg.role, "content": msg.content} for msg in messages]
        finally:
            db.close()