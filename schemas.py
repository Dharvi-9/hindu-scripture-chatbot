from pydantic import BaseModel
from typing import Optional, List

# DEFINE DATA MODELS

# What we receive from user
class ChatRequest(BaseModel):
    message: str      # users question
    session_id: Optional[str] = None  # Optional: for multi-user support

# what we send back to user
class ChatResponse(BaseModel):
    response: str     #  Bot's answer
    type: str     ="answer"    #  Type: "rag", "greeting", "decline"
    session_id: str   #  Session ID generated if not provided
    sources: list = []  # ← Add sources
    confidence: float = 0.0  # ← Add confidence
print("Data models defined")
