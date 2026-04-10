"""
Inference Script — Email Triage OpenEnv
=======================================
Runs an LLM agent against all 3 tasks and emits structured stdout logs.

STDOUT FORMAT (strict):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Mandatory environment variables:
  HF_TOKEN     — HuggingFace / API key  (https://huggingface.co/settings/tokens)
  API_BASE_URL — LLM endpoint           (default: https://router.huggingface.co/v1)
  MODEL_NAME   — Model identifier       (default: Qwen/Qwen2.5-72B-Instruct)

Setup:
  cp .env.example .env   # fill in your values
  python inference.py
"""
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

# Load .env file if present (local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — rely on shell env vars

from openai import OpenAI

from env import EmailTriageEnv
from env.models import Action

# ── Mandatory config ──────────────────────────────────────────────────────────
HF_TOKEN     = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "email-triage-env"
MAX_STEPS    = 30
TEMPERATURE  = 0.2
MAX_TOKENS   = 512
SUCCESS_SCORE_THRESHOLD = 0.5

# Validate all mandatory variables before doing anything else
_missing = []
if not HF_TOKEN:
    _missing.append("HF_TOKEN")
if not API_BASE_URL:
    _missing.append("API_BASE_URL")
if not MODEL_NAME:
    _missing.append("MODEL_NAME")

if _missing:
    print("ERROR: The following mandatory environment variables are not set:", file=sys.stderr)
    for var in _missing:
        print(f"  - {var}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Set them in your shell or copy .env.example to .env and fill in your values:", file=sys.stderr)
    print("  cp .env.example .env", file=sys.stderr)
    print("  # edit .env, then run:", file=sys.stderr)
    print("  python inference.py", file=sys.stderr)
    sys.exit(1)

TASKS = ["easy_triage", "medium_triage", "hard_triage"]

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert email triage assistant for a B2B SaaS company.
    Your job is to triage each email by assigning:
      - priority: one of [urgent, high, normal, low]
      - category: one of [billing, technical_support, sales, hr, legal, spam, internal, customer_complaint, general_inquiry]
      - response_action: one of [reply_now, delegate, archive, escalate, delete, schedule_followup]
      - summary: a single sentence (max 200 chars) describing the email
      - reasoning: brief explanation of your decision

    Guidelines:
      - urgent: requires action within hours (outages, legal threats, viral PR crises)
      - high: requires action today (customer complaints, SLA issues, billing disputes)
      - normal: requires action this week
      - low: no urgency (newsletters, internal chit-chat, vendor follow-ups)
      - escalate: send to senior management or specialized team
      - delegate: assign to appropriate team member
      - reply_now: respond immediately yourself
      - schedule_followup: set a reminder to address later
      - archive: keep for records but no action needed
      - delete: spam or irrelevant

    Respond ONLY with a valid JSON object. No markdown, no explanation outside the JSON.
""").strip()


def build_user_prompt(obs_dict: Dict[str, Any]) -> str:
    email = obs_dict.get("current_email")
    if not email:
        return "No email to process."
    return textwrap.dedent(f"""
        Triage this email:

        ID: {email['id']}
        From: {email['sender']}
        Subject: {email['subject']}
        Timestamp: {email['timestamp']}
        Has Attachment: {email['has_attachment']}
        Thread Length: {email['thread_length']}

        Body:
        {email['body']}

        Respond with JSON:
        {{
          "email_id": "{email['id']}",
          "priority": "<urgent|high|normal|low>",
          "category": "<category>",
          "response_action": "<action>",
          "summary": "<one sentence max 200 chars>",
          "reasoning": "<brief reasoning>"
        }}
    """).strip()


def call_llm(messages: List[Dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def parse_action(raw: str, fallback_email_id: str) -> Action:
    """Parse LLM JSON output into an Action, with safe fallback on error."""
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        data = json.loads(text)
        return Action(**data)
    except Exception:
        return Action(
            email_id=fallback_email_id,
            priority="normal",
            category="general_inquiry",
            response_action="archive",
            summary="Unable to parse email content.",
            reasoning="Parse error - using safe defaults.",
        )


def _action_str(action: Action) -> str:
    """
    Compact action string with no spaces (safe for log parsers).
    Format: triage(id=<id>,priority=<p>,category=<c>,action=<a>)
    """
    return (
        f"triage(id={action.email_id},"
        f"priority={action.priority},"
        f"category={action.category},"
        f"action={action.response_action})"
    )


def run_task(task_id: str) -> Dict[str, Any]:
    """Run one full episode, emit structured logs, return results."""
    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()
    obs_dict = obs.model_dump()

    step_num = 0
    rewards: List[float] = []
    last_error: Optional[str] = None
    done = False
    final_score = 0.0
    success = False

    # [START] — emitted once at episode begin
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while not done and step_num < MAX_STEPS:
            if obs_dict.get("emails_remaining", 0) == 0:
                break

            current_email = obs_dict.get("current_email")
            if not current_email:
                break

            user_prompt = build_user_prompt(obs_dict)
            messages_for_step = messages + [{"role": "user", "content": user_prompt}]

            try:
                raw_response = call_llm(messages_for_step)
                action = parse_action(raw_response, current_email["id"])
                last_error = None
            except Exception as e:
                last_error = str(e).replace("\n", " ")
                action = Action(
                    email_id=current_email["id"],
                    priority="normal",
                    category="general_inquiry",
                    response_action="archive",
                    summary="Error during inference.",
                    reasoning=last_error,
                )

            obs_new, reward, done, info = env.step(action)
            step_num += 1
            rewards.append(reward.value)
            obs_dict = obs_new.model_dump()

            # error field: raw error string or null
            step_error = info.get("error") or last_error or "null"

            # [STEP] — emitted immediately after env.step() returns
            print(
                f"[STEP] step={step_num}"
                f" action={_action_str(action)}"
                f" reward={reward.value:.2f}"
                f" done={str(done).lower()}"
                f" error={step_error}",
                flush=True,
            )

        final_score = env.final_score()
        success = final_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        # Catch-all: still emit [END] below
        last_error = str(e).replace("\n", " ")
        if not rewards:
            rewards = [0.0]
        final_score = env.final_score() if step_num > 0 else 0.0
        success = False

    finally:
        env.close()
        # [END] — always emitted, even on exception
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(
            f"[END] success={str(success).lower()}"
            f" steps={step_num}"
            f" rewards={rewards_str}",
            flush=True,
        )

    return {
        "task_id": task_id,
        "score": final_score,
        "steps": step_num,
        "success": success,
        "rewards": rewards,
    }


def main():
    print(f"Running Email Triage baseline | model={MODEL_NAME}", file=sys.stderr)
    results = []
    for task_id in TASKS:
        result = run_task(task_id)
        results.append(result)

    # Summary to stderr only — keeps stdout clean for log parser
    print("\n=== Baseline Results ===", file=sys.stderr)
    for r in results:
        print(
            f"  {r['task_id']:20s}  score={r['score']:.3f}"
            f"  steps={r['steps']}  success={r['success']}",
            file=sys.stderr,
        )
    avg = sum(r["score"] for r in results) / len(results)
    print(f"  {'AVERAGE':20s}  score={avg:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
