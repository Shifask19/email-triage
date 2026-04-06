"""
FastAPI application exposing the EmailTriageEnv via HTTP.
Serves as the HuggingFace Space endpoint — satisfies OpenEnv HTTP spec.

Endpoints:
  GET  /           — environment metadata
  GET  /health     — liveness probe
  GET  /tasks      — list all tasks with metadata
  POST /reset      — start/restart an episode
  POST /step       — submit one triage action
  GET  /state      — full serialized environment state
"""
from __future__ import annotations
import os
import threading
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from env import EmailTriageEnv
from env.models import Action
from env.tasks import TASK_DESCRIPTIONS
from env.data import TASK_EMAILS

app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "An OpenEnv environment for email triage — a real-world task for AI agents. "
        "Agents must read, prioritize, categorize, and decide response actions for "
        "a corporate inbox across three difficulty levels."
    ),
    version="1.0.0",
)

VALID_TASKS = ["easy_triage", "medium_triage", "hard_triage"]

# Thread-safe per-task environment store
_envs: Dict[str, EmailTriageEnv] = {}
_locks: Dict[str, threading.Lock] = {t: threading.Lock() for t in VALID_TASKS}


def _get_or_create_env(task_id: str) -> EmailTriageEnv:
    """Get existing env or create and auto-reset a fresh one."""
    if task_id not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Valid: {VALID_TASKS}")
    if task_id not in _envs:
        env = EmailTriageEnv(task_id=task_id)
        env.reset()  # always start in a valid state
        _envs[task_id] = env
    return _envs[task_id]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Environment metadata — returns 200 for HF Space ping."""
    return {
        "name": "email-triage-env",
        "version": "1.0.0",
        "description": "OpenEnv email triage environment",
        "tasks": VALID_TASKS,
        "endpoints": {
            "reset": "POST /reset?task_id=<task_id>",
            "step":  "POST /step?task_id=<task_id>",
            "state": "GET  /state?task_id=<task_id>",
            "tasks": "GET  /tasks",
            "health": "GET /health",
        },
    }


@app.get("/health")
def health():
    """Liveness probe — always returns 200 if server is up."""
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """List all tasks with descriptions and difficulty metadata."""
    difficulty_map = {"easy_triage": "easy", "medium_triage": "medium", "hard_triage": "hard"}
    return {
        tid: {
            "description": TASK_DESCRIPTIONS[tid],
            "num_emails": len(TASK_EMAILS[tid]),
            "difficulty": difficulty_map[tid],
            "reward_range": [0.0, 1.0],
        }
        for tid in VALID_TASKS
    }


@app.post("/reset")
def reset(task_id: str = Query(default="easy_triage")):
    """
    Reset the environment for the given task and return the initial observation.
    Always produces a clean state — safe to call multiple times.
    """
    if task_id not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Valid: {VALID_TASKS}")

    with _locks[task_id]:
        env = EmailTriageEnv(task_id=task_id)
        obs = env.reset()
        _envs[task_id] = env

    return obs.model_dump()


@app.post("/step")
def step(action: Action, task_id: str = Query(default="easy_triage")):
    """
    Submit one triage action. Returns observation, reward, done flag, and info.
    If the episode is already done, returns 409 — call /reset first.
    """
    if task_id not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Valid: {VALID_TASKS}")

    with _locks[task_id]:
        env = _get_or_create_env(task_id)

        if env.done:
            raise HTTPException(
                status_code=409,
                detail="Episode is done. Call POST /reset to start a new episode.",
            )

        try:
            obs, reward, done, info = env.step(action)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(task_id: str = Query(default="easy_triage")):
    """Return the full serialized environment state."""
    if task_id not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Valid: {VALID_TASKS}")

    with _locks[task_id]:
        env = _get_or_create_env(task_id)
        return env.state().model_dump()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
