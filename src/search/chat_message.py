"""
Data structures for the chat window message model.

Defines the message types, sender roles, content types, and chat modes
used by the ChatWindow to render a conversational interface.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Any
import time


class MessageSender(Enum):
    USER = auto()
    SYSTEM = auto()


class MessageContentType(Enum):
    TEXT = auto()
    PREVIEW = auto()
    ERROR = auto()
    STATUS = auto()


class ChatMode(Enum):
    HUMAN_AGENT = auto()  # Agent plans edits, human selects from 5 options at each step


@dataclass
class PreviewItem:
    """A single selectable preview image within a chat message."""

    texture_name: str  # TextureManager key, e.g. "preview_0"
    label: str  # Display label
    item_data: Any = None  # LUT result dict or effect instance
    item_index: int = 0  # Index within the preview set


@dataclass
class ChatMessage:
    """A single message in the chat history."""

    sender: MessageSender
    content_type: MessageContentType
    text: str
    timestamp: float = field(default_factory=time.time)
    preview_items: List[PreviewItem] = field(default_factory=list)
    preview_effect_name: Optional[str] = None  # For effect preview messages
    preview_source: Optional[str] = None  # "lut" or "effect"
    is_selectable: bool = False  # Whether preview images are still clickable
    selected_index: Optional[int] = None  # Index auto-selected by vision (for display)
