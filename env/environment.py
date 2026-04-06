"""
EmailTriageEnv — OpenEnv-compliant email triage environment.

API:
    env = EmailTriageEnv(task_id="easy_triage")
    obs  = env.reset()
    obs, reward, done, info = env.step(action)
    state = env.state()
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from .data import EMAIL_BY_ID, TASK_EMAILS
from .models import (
    Action, EmailMessage, EnvironmentState, Observation, Reward
)
from .tasks import TASK_DESCRIPTIONS, grade_action

VALID_TASKS = list(TASK_EMAILS.keys())
MAX_STEPS_PER_TASK = 30  # safety ceiling


class EmailTriageEnv:
    """
    Email Triage OpenEnv environment.

    The agent receives an inbox of emails and must triage each one by
    assigning a priority, category, and response action, plus a brief summary.

    Episode ends when all emails in the inbox have been processed or
    MAX_STEPS_PER_TASK is reached.
    """

    def __init__(self, task_id: str = "easy_triage") -> None:
        if task_id not in VALID_TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {VALID_TASKS}")
        self.task_id = task_id
        self._step = 0
        self._done = False
        self._inbox: List[EmailMessage] = []
        self._processed: List[Dict[str, Any]] = []
        self._episode_rewards: List[float] = []
        self._total_reward: float = 0.0
        self._pending_ids: List[str] = []

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self._step = 0
        self._done = False
        self._processed = []
        self._episode_rewards = []
        self._total_reward = 0.0

        email_ids = TASK_EMAILS[self.task_id]
        self._inbox = [
            EmailMessage(**EMAIL_BY_ID[eid]) for eid in email_ids
        ]
        self._pending_ids = [e.id for e in self._inbox]

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Process one triage action.

        Args:
            action: An Action specifying how to triage one email.

        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step += 1
        info: Dict[str, Any] = {"step": self._step}

        # Validate email_id
        if action.email_id not in self._pending_ids:
            # Penalize invalid action (already processed or unknown)
            reward = Reward(
                value=0.0,
                breakdown={"error": f"email_id '{action.email_id}' not in pending inbox"},
            )
            self._episode_rewards.append(0.0)
            info["error"] = reward.breakdown["error"]
            obs = self._build_observation()
            return obs, reward, self._done, info

        # Grade the action
        email_data = EMAIL_BY_ID[action.email_id]
        reward = grade_action(action, self.task_id, email_data["body"])

        # Mark email as processed
        self._pending_ids.remove(action.email_id)
        self._processed.append({
            "email_id": action.email_id,
            "action": action.model_dump(),
            "reward": reward.model_dump(),
            "step": self._step,
        })
        self._episode_rewards.append(reward.value)
        self._total_reward += reward.value

        # Check termination
        if not self._pending_ids or self._step >= MAX_STEPS_PER_TASK:
            self._done = True

        obs = self._build_observation()
        return obs, reward, self._done, info

    def state(self) -> EnvironmentState:
        """Return the full serializable environment state."""
        return EnvironmentState(
            task_id=self.task_id,
            step=self._step,
            done=self._done,
            inbox=self._inbox,
            processed=self._processed,
            total_reward=round(self._total_reward, 4),
            episode_rewards=self._episode_rewards,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        pending_emails = [e for e in self._inbox if e.id in self._pending_ids]
        current = pending_emails[0] if pending_emails else None
        return Observation(
            inbox=pending_emails,
            current_email=current,
            step_number=self._step,
            emails_processed=len(self._processed),
            emails_remaining=len(self._pending_ids),
            task_description=TASK_DESCRIPTIONS[self.task_id],
            context={
                "task_id": self.task_id,
                "total_emails": len(self._inbox),
                # NOTE: no reward leakage — agent does not see past rewards mid-episode
            },
        )

    def close(self) -> None:
        """Clean up resources. No-op for this environment; included for spec compliance."""
        pass

    @property
    def total_reward(self) -> float:
        return self._total_reward

    @property
    def done(self) -> bool:
        return self._done

    def final_score(self) -> float:
        """Normalized episode score in [0, 1]."""
        n = len(self._inbox)
        if n == 0:
            return 0.0
        return round(self._total_reward / n, 4)
