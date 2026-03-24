"""Memory trigger configuration schema."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class MemoryTriggerConfig(BaseModel):
    """Configuration for when to check conversation for memory items."""

    mode: Literal["phrase", "interval", "ai_determined"]
    # For "phrase" mode: user says this to trigger memory check
    trigger_phrase: Optional[str] = Field(default=None)
    # For "interval" mode: check every N messages
    message_interval: Optional[int] = Field(default=None, gt=0)

    # Note: Message counter is always per-session (in-memory), resets on agent restart
