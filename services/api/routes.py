"""API routes for chat and conversation management."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from services.api.cache import get_cached_response, set_cached_response
from services.api.database import Conversation, Message, get_session
from services.api.model_client import call_model_service
from services.api.schemas import (
    ChatMessage,
    ChatMessageRequest,
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    DeliberationMetadata,
)

router = APIRouter()


@router.post("/conversations", response_model=ConversationSummary)
async def create_conversation(
    session: AsyncSession = Depends(get_session),
):
    """Create a new conversation."""
    conv = Conversation(
        id=str(uuid.uuid4()),
        title="New Conversation",
    )
    session.add(conv)
    await session.commit()
    return ConversationSummary(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        message_count=0,
    )


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    session: AsyncSession = Depends(get_session),
):
    """List all conversations, most recent first."""
    stmt = (
        select(
            Conversation.id,
            Conversation.title,
            Conversation.created_at,
            func.count(Message.id).label("message_count"),
        )
        .outerjoin(Message)
        .group_by(Conversation.id)
        .order_by(Conversation.created_at.desc())
    )
    result = await session.execute(stmt)
    return [
        ConversationSummary(
            id=row.id,
            title=row.title,
            created_at=row.created_at,
            message_count=row.message_count,
        )
        for row in result.all()
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get a conversation with all its messages."""
    stmt = (
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages))
    )
    result = await session.execute(stmt)
    conv = result.scalar_one_or_none()

    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = []
    for msg in conv.messages:
        delib = None
        if msg.delib_steps is not None:
            delib = DeliberationMetadata(
                steps=int(msg.delib_steps),
                module_precisions={
                    "logic": msg.precision_logic or 1.0,
                    "eq": msg.precision_eq or 1.0,
                },
                converged=msg.delib_converged == "true",
                final_entropy=msg.delib_entropy or 0.0,
            )
        messages.append(
            ChatMessage(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp,
                deliberation=delib,
            )
        )

    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        messages=messages,
        created_at=conv.created_at,
    )


@router.post("/conversations/{conversation_id}/messages", response_model=ChatResponse)
async def send_message(
    conversation_id: str,
    request: ChatMessageRequest,
    session: AsyncSession = Depends(get_session),
):
    """Send a message and get a CLSA response."""
    # Verify conversation exists
    conv = await session.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    now = datetime.now(timezone.utc)

    # Save user message
    user_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role="user",
        content=request.content,
        timestamp=now,
        precision_logic=request.precision.logic,
        precision_eq=request.precision.eq,
    )
    session.add(user_msg)

    # Check cache
    cached = await get_cached_response(
        request.content, request.precision.logic, request.precision.eq
    )

    if cached is not None:
        response_text = cached["text"]
        delib_data = cached.get("deliberation")
    else:
        # Call model service
        model_response = await call_model_service(
            prompt=request.content,
            precision_logic=request.precision.logic,
            precision_eq=request.precision.eq,
            return_deliberation=request.return_deliberation,
        )
        response_text = model_response["text"]
        delib_data = model_response.get("deliberation")

        # Cache the response
        await set_cached_response(
            request.content,
            request.precision.logic,
            request.precision.eq,
            model_response,
        )

    # Save assistant message
    assistant_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role="assistant",
        content=response_text,
        timestamp=datetime.now(timezone.utc),
    )

    if delib_data:
        assistant_msg.delib_steps = delib_data["steps"]
        assistant_msg.delib_entropy = delib_data["final_entropy"]
        assistant_msg.delib_converged = str(delib_data["converged"]).lower()
        assistant_msg.precision_logic = delib_data["module_precisions"].get("logic", 1.0)
        assistant_msg.precision_eq = delib_data["module_precisions"].get("eq", 1.0)

    session.add(assistant_msg)

    # Update conversation title from first message
    msg_count = await session.scalar(
        select(func.count(Message.id)).where(
            Message.conversation_id == conversation_id
        )
    )
    if msg_count <= 1:
        conv.title = request.content[:50]

    await session.commit()

    delib_meta = None
    if delib_data:
        delib_meta = DeliberationMetadata(**delib_data)

    return ChatResponse(
        message=ChatMessage(
            id=assistant_msg.id,
            role="assistant",
            content=response_text,
            timestamp=assistant_msg.timestamp,
            deliberation=delib_meta,
        ),
        conversation_id=conversation_id,
    )
