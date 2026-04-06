---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - email
  - triage
  - nlp
  - decision-making
  - real-world
  - agent-benchmark
app_port: 7860
---

# Email Triage OpenEnv

An [OpenEnv](https://github.com/openenv)-compliant environment where AI agents triage a corporate email inbox — a task that knowledge workers perform every day.

---

## Motivation

Email triage is a high-value, real-world task: prioritizing, categorizing, and deciding how to respond to emails requires reading comprehension, domain knowledge, and judgment under ambiguity. It's an ideal benchmark for evaluating LLM agents on practical enterprise tasks.

---

## Environment Description

The agent receives an inbox of emails and must triage each one by producing:

| Field | Type | Description |
|---|---|---|
| `priority` | `urgent \| high \| normal \| low` | How time-sensitive is this email? |
| `category` | enum (9 values) | What is this email about? |
| `response_action` | enum (6 values) | What should be done with it? |
| `summary` | string (≤200 chars) | One-sentence description |
| `reasoning` | string | Optional explanation |

### Action Space

```json
{
  "email_id": "e001",
  "priority": "urgent",
  "category": "billing",
  "response_action": "reply_now",
  "summary": "Invoice overdue, service suspension threatened in 24 hours.",
  "reasoning": "Explicit urgency signal in subject and body, financial impact."
}
```

**Priority values:** `urgent`, `high`, `normal`, `low`

**Category values:** `billing`, `technical_support`, `sales`, `hr`, `legal`, `spam`, `internal`, `customer_complaint`, `general_inquiry`

**Response action values:** `reply_now`, `delegate`, `archive`, `escalate`, `delete`, `schedule_followup`

### Observation Space

```json
{
  "inbox": [...],
  "current_email": { "id": "...", "subject": "...", "sender": "...", "body": "..." },
  "step_number": 1,
  "emails_processed": 0,
  "emails_remaining": 5,
  "task_description": "...",
  "context": { "task_id": "easy_triage", "total_emails": 5 }
}
```

---

## Tasks

| Task | Difficulty | Emails | Description |
|---|---|---|---|
| `easy_triage` | Easy | 5 | Obvious spam, urgent billing, clear support requests, production outage |
| `medium_triage` | Medium | 6 | Customer complaints, legal notices, HR scheduling, DB corruption incident |
| `hard_triage` | Hard | 8 | Deliberate traps: calm-tone churn risk, hidden SLA breach, viral PR crisis |

---

## Reward Function

Each step returns a reward in `[0.0, 1.0]` computed as a weighted sum:

| Component | Easy | Medium | Hard | Description |
|---|---|---|---|---|
| Priority | 35% | 30% | 25% | Partial credit: adjacent levels score 0.6, two-off score 0.2 |
| Category | 30% | 30% | 25% | Partial credit: semantically related categories score 0.4 |
| Action | 25% | 30% | 40% | Partial credit: related actions score 0.2–0.5 |
| Summary | 10% | 10% | 10% | Heuristic: length + keyword overlap, penalizes verbatim copying |

**Penalties:**
- Replying to or escalating spam: −0.3
- Deleting an urgent email: −0.5

The episode score is the mean reward across all emails in the task.

---

## Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Score | Notes |
|---|---|---|
| `easy_triage` | ~0.85 | Clear signals, most models handle well |
| `medium_triage` | ~0.70 | Requires body comprehension, some escalation misses |
| `hard_triage` | ~0.55 | Deliberate traps catch most frontier models |

A perfect agent (correct on all labels, good summaries) scores ~0.96 on all tasks.
A naive agent (always `normal/general_inquiry/archive`) scores ~0.30–0.34.

---

## Setup & Usage

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes (inference) | — | HuggingFace / API key |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `PORT` | No | `7860` | Server port |

### Local (Python)

```bash
pip install -r requirements.txt

# Run the API server
python app.py

# Run baseline inference
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  email-triage-env
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Environment info — returns 200 for HF Space ping |
| GET | `/health` | Liveness probe |
| GET | `/tasks` | List all tasks with metadata |
| POST | `/reset?task_id=easy_triage` | Start/restart an episode |
| POST | `/step?task_id=easy_triage` | Submit one triage action |
| GET | `/state?task_id=easy_triage` | Full serialized state |

### Python SDK

```python
from env import EmailTriageEnv
from env.models import Action

env = EmailTriageEnv(task_id="medium_triage")
obs = env.reset()

while not env.done:
    action = Action(
        email_id=obs.current_email.id,
        priority="high",
        category="customer_complaint",
        response_action="escalate",
        summary="Customer threatening chargeback after 3-week refund delay.",
    )
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward.value:.3f}")

print(f"Final score: {env.final_score():.3f}")
```

### Pre-submission Validation

```bash
python run_checks.py
```

Runs all Phase 1/2/3 checks locally: clean reset, API contracts, grader ranges, score variance, difficulty spread, HTTP endpoint contracts, exploit checks.
