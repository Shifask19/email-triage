"""
Typed Pydantic models for the Email Triage OpenEnv environment.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ── Email domain types ────────────────────────────────────────────────────────

Priority = Literal["urgent", "high", "normal", "low"]
Category = Literal[
    "billing", "technical_support", "sales", "hr", "legal",
    "spam", "internal", "customer_complaint", "general_inquiry"
]
ResponseAction = Literal[
    "reply_now", "delegate", "archive", "escalate", "delete", "schedule_followup"
]


class EmailMessage(BaseModel):
    """A single email in the inbox."""
    id: str
    subject: str
    sender: str
    body: str
    timestamp: str
    has_attachment: bool = False
    thread_length: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── OpenEnv core models ───────────────────────────────────────────────────────

class Observation(BaseModel):
    """What the agent sees at each step."""
    inbox: List[EmailMessage]
    current_email: Optional[EmailMessage] = None
    step_number: int = 0
    emails_processed: int = 0
    emails_remaining: int = 0
    task_description: str = ""
    context: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    """Agent action: triage a single email."""
    email_id: str = Field(..., description="ID of the email being triaged")
    priority: Priority = Field(..., description="Assigned priority level")
    category: Category = Field(..., description="Email category/topic")
    response_action: ResponseAction = Field(..., description="What to do with this email")
    summary: str = Field(..., description="One-sentence summary of the email", max_length=200)
    reasoning: str = Field(default="", description="Optional reasoning for the decision")


class Reward(BaseModel):
    """Reward signal for a single step."""
    value: float = Field(..., ge=0.0, le=1.0, description="Reward in [0, 1]")
    priority_score: float = Field(default=0.0, ge=0.0, le=1.0)
    category_score: float = Field(default=0.0, ge=0.0, le=1.0)
    action_score: float = Field(default=0.0, ge=0.0, le=1.0)
    summary_score: float = Field(default=0.0, ge=0.0, le=1.0)
    penalty: float = Field(default=0.0, ge=0.0, le=1.0, description="Penalty applied")
    breakdown: Dict[str, Any] = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    """Full serializable state of the environment."""
    task_id: str
    step: int
    done: bool
    inbox: List[EmailMessage]
    processed: List[Dict[str, Any]]
    total_reward: float
    episode_rewards: List[float]
