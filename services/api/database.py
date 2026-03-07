"""Database setup and models for conversation storage.

Uses SQLAlchemy async with Postgres for persistent conversation history.
"""

import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://clsa:clsa@localhost:5432/clsa",
)

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True)
    title = Column(String, default="New Conversation")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    messages = relationship(
        "Message", back_populates="conversation", order_by="Message.timestamp"
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Deliberation metadata (nullable, only for assistant messages)
    delib_steps = Column(Float, nullable=True)
    delib_entropy = Column(Float, nullable=True)
    delib_converged = Column(String, nullable=True)
    precision_logic = Column(Float, nullable=True)
    precision_eq = Column(Float, nullable=True)

    conversation = relationship("Conversation", back_populates="messages")


async def init_db():
    """Create all tables. Called on app startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Dependency that yields a database session."""
    async with async_session() as session:
        yield session
