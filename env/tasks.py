"""
Task definitions and graders for the Email Triage environment.

Tasks:
  easy_triage   — 5 emails with obvious signals (spam, urgent billing, etc.)
  medium_triage — 6 emails requiring body comprehension and context
  hard_triage   — 8 emails with subtle cues, conflicting signals, nuanced judgment
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple

from .data import GROUND_TRUTH, TASK_EMAILS
from .models import Action, Reward

# ── Scoring weights per task ──────────────────────────────────────────────────
# (priority_w, category_w, action_w, summary_w)
# Hard task heavily weights action correctness — missing an escalation is costly.
TASK_WEIGHTS: Dict[str, Tuple[float, float, float, float]] = {
    "easy_triage":   (0.35, 0.30, 0.25, 0.10),
    "medium_triage": (0.30, 0.30, 0.30, 0.10),
    "hard_triage":   (0.25, 0.25, 0.40, 0.10),
}

# Priority adjacency — partial credit for close misses
PRIORITY_ORDER = ["urgent", "high", "normal", "low"]
PRIORITY_PARTIAL: Dict[Tuple[str, str], float] = {}
for i, p1 in enumerate(PRIORITY_ORDER):
    for j, p2 in enumerate(PRIORITY_ORDER):
        dist = abs(i - j)
        PRIORITY_PARTIAL[(p1, p2)] = max(0.0, 1.0 - dist * 0.4)

# Category partial credit groups (semantically related categories)
CATEGORY_GROUPS = [
    {"billing", "legal"},
    {"technical_support", "internal"},
    {"sales", "general_inquiry"},
    {"hr", "internal"},
    {"customer_complaint", "general_inquiry"},
]


def _priority_score(predicted: str, truth: str) -> float:
    """Partial credit for priority: exact=1.0, adjacent=0.6, two-off=0.2, else=0."""
    return PRIORITY_PARTIAL.get((predicted, truth), 0.0)


def _category_score(predicted: str, truth: str) -> float:
    """Partial credit for category: exact=1.0, same group=0.4, else=0."""
    if predicted == truth:
        return 1.0
    for group in CATEGORY_GROUPS:
        if predicted in group and truth in group:
            return 0.4
    return 0.0


def _action_score(predicted: str, truth: str, task_id: str) -> float:
    """
    Partial credit for response action.
    Hard task penalizes wrong escalation decisions more.
    """
    if predicted == truth:
        return 1.0
    # Partial credit map: (predicted, truth) -> score
    partial = {
        ("escalate", "reply_now"): 0.5,
        ("reply_now", "escalate"): 0.5,
        ("delegate", "escalate"): 0.3,
        ("escalate", "delegate"): 0.3,
        ("schedule_followup", "reply_now"): 0.3,
        ("reply_now", "schedule_followup"): 0.3,
        ("archive", "delete"): 0.5,
        ("delete", "archive"): 0.5,
        ("delegate", "reply_now"): 0.2,
        ("reply_now", "delegate"): 0.2,
    }
    base = partial.get((predicted, truth), 0.0)
    # Hard task: escalation errors cost more
    if task_id == "hard_triage" and truth == "escalate" and predicted != "escalate":
        base *= 0.5
    return base


def _summary_score(summary: str, email_body: str) -> float:
    """
    Heuristic summary quality: checks length and keyword overlap with body.
    Penalizes verbatim copying (summary too close to full body length).
    Returns 0.0–1.0.
    """
    if not summary or len(summary.strip()) < 10:
        return 0.0

    summary_stripped = summary.strip()
    length = len(summary_stripped)

    # Penalize verbatim body copying: if summary is >60% of body length, it's not a summary
    body_len = len(email_body.strip())
    if body_len > 0 and length / body_len > 0.6:
        return 0.1

    # Length check: 20–180 chars is ideal for a one-sentence summary
    if length < 20:
        length_score = 0.3
    elif length <= 180:
        length_score = 1.0
    elif length <= 200:
        length_score = 0.7
    else:
        length_score = 0.3  # too verbose, not a summary

    # Keyword overlap with body (simple bag-of-words, excluding stop words)
    body_words = set(email_body.lower().split())
    summary_words = set(summary_stripped.lower().split())
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "i", "we",
        "you", "it", "to", "of", "and", "or", "in", "on", "for",
        "with", "this", "that", "be", "have", "has", "had", "will",
        "would", "could", "should", "my", "our", "your", "their",
    }
    body_content = body_words - stop_words
    summary_content = summary_words - stop_words
    if not body_content or not summary_content:
        overlap_score = 0.5
    else:
        overlap = len(body_content & summary_content) / len(summary_content)
        overlap_score = min(1.0, overlap * 2)

    return round((length_score * 0.5 + overlap_score * 0.5), 3)


def grade_action(
    action: Action,
    task_id: str,
    email_body: str,
) -> Reward:
    """
    Grade a single triage action against ground truth.
    Returns a Reward with partial credit breakdown.
    """
    truth = GROUND_TRUTH.get(action.email_id)
    if truth is None:
        return Reward(
            value=1e-6,
            breakdown={"error": f"Unknown email_id: {action.email_id}"},
        )

    pw, cw, aw, sw = TASK_WEIGHTS[task_id]

    p_score = _priority_score(action.priority, truth["priority"])
    c_score = _category_score(action.category, truth["category"])
    a_score = _action_score(action.response_action, truth["response_action"], task_id)
    s_score = _summary_score(action.summary, email_body)

    # Penalty: if agent marks urgent spam as urgent (catastrophic error)
    penalty = 0.0
    if truth["category"] == "spam" and action.response_action in ("reply_now", "escalate"):
        penalty = 0.3
    # Penalty: deleting a legal/urgent email
    if truth["priority"] == "urgent" and action.response_action == "delete":
        penalty = 0.5

    raw = pw * p_score + cw * c_score + aw * a_score + sw * s_score
    raw_value = max(0.0, round(raw - penalty, 4))
    # Clamp strictly inside (0, 1) as required by the grading platform
    value = max(1e-6, min(1.0 - 1e-6, raw_value))

    return Reward(
        value=value,
        priority_score=p_score,
        category_score=c_score,
        action_score=a_score,
        summary_score=s_score,
        penalty=penalty,
        breakdown={
            "truth_priority": truth["priority"],
            "truth_category": truth["category"],
            "truth_action": truth["response_action"],
            "predicted_priority": action.priority,
            "predicted_category": action.category,
            "predicted_action": action.response_action,
        },
    )


TASK_DESCRIPTIONS: Dict[str, str] = {
    "easy_triage": (
        "Triage 5 emails with clear, unambiguous signals. "
        "Identify obvious spam, an urgent billing suspension notice, a general product inquiry, "
        "and a production outage alert. Assign correct priority, category, and response action."
    ),
    "medium_triage": (
        "Triage 6 emails that require reading and understanding the body content. "
        "Includes an escalating customer refund complaint, a legal cease-and-desist, "
        "an HR scheduling notice, a sales partnership inquiry, a low-priority facilities "
        "reminder, and a critical database corruption incident blocking a release."
    ),
    "hard_triage": (
        "Triage 8 emails with deliberate traps designed to fool frontier models. "
        "Includes: a positive expansion email hiding unauthorized plan usage, "
        "a routine security alert hiding an active intrusion attempt, "
        "an aggressive refund demand that is actually fraud, "
        "a calm customer email hiding imminent churn, "
        "a contract renewal hiding a legal compliance crisis, "
        "and a viral PR crisis disguised as an internal monitoring alert. "
        "Surface tone, sender domain, and subject line all actively mislead. "
        "Read every detail of the body carefully."
    ),
}
