# db/models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from db.session import Base


class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to messages
    messages = relationship("MessageModel", back_populates="session", cascade="all, delete-orphan")


class MessageModel(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    role = Column(String)  # "system", "user", or "assistant"
    content = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship back to session
    session = relationship("SessionModel", back_populates="messages")