"""
Phase 1/2/3 validation checks — run before submission.
"""
import sys
from env import EmailTriageEnv
from env.models import Action
from env.data import GROUND_TRUTH

def check_clean_reset():
    print("=== PHASE 1 GATE 1: reset() produces clean state ===")
    for task_id in ["easy_triage", "medium_triage", "hard_triage"]:
        env = EmailTriageEnv(task_id=task_id)
        obs = env.reset()
        assert obs.step_number == 0
        assert obs.emails_processed == 0
        assert obs.emails_remaining > 0
        assert obs.current_email is not None
        assert not env.done
        # Reset mid-episode
        env.step(Action(email_id=obs.current_email.id, priority="normal",
                        category="general_inquiry", response_action="archive",
                        summary="test"))
        obs2 = env.reset()
        assert obs2.step_number == 0, "reset() did not clean step counter"
        assert obs2.emails_processed == 0, "reset() did not clean processed list"
        assert not env.done, "reset() left done=True"
        print(f"  {task_id}: clean reset OK")


def check_api_contracts():
    print("\n=== PHASE 1 GATE 2: step()/reset()/state() contracts ===")
    env = EmailTriageEnv(task_id="easy_triage")
    obs = env.reset()
    action = Action(
        email_id=obs.current_email.id, priority="urgent",
        category="billing", response_action="reply_now",
        summary="Urgent billing issue requiring immediate payment.",
    )
    obs2, reward, done, info = env.step(action)
    state = env.state()
    assert 0.0 <= reward.value <= 1.0, f"Reward out of range: {reward.value}"
    assert state.step == 1
    assert len(state.processed) == 1
    # Reward model has all required fields
    assert hasattr(reward, "priority_score")
    assert hasattr(reward, "category_score")
    assert hasattr(reward, "action_score")
    assert hasattr(reward, "breakdown")
    print("  step()/reset()/state() all valid, reward fields present")


def check_graders():
    print("\n=== PHASE 1 GATE 3: 3+ tasks, graders in [0,1] ===")
    for task_id in ["easy_triage", "medium_triage", "hard_triage"]:
        env = EmailTriageEnv(task_id=task_id)
        obs = env.reset()
        scores = []
        while not env.done:
            email = obs.current_email
            gt = GROUND_TRUTH[email.id]
            action = Action(
                email_id=email.id, priority=gt["priority"],
                category=gt["category"], response_action=gt["response_action"],
                summary=f"Summary of email from {email.sender}.",
            )
            obs, reward, done, info = env.step(action)
            scores.append(reward.value)
            assert 0.0 <= reward.value <= 1.0, f"Score out of range: {reward.value}"
        final = env.final_score()
        assert 0.0 <= final <= 1.0
        print(f"  {task_id}: {len(scores)} emails graded, final={final:.4f}")


def check_score_variance():
    print("\n=== PHASE 2: Score variance (graders must NOT always return same score) ===")
    for task_id in ["easy_triage", "medium_triage", "hard_triage"]:
        env = EmailTriageEnv(task_id=task_id)
        obs = env.reset()
        scores = []
        while not env.done:
            email = obs.current_email
            action = Action(
                email_id=email.id, priority="normal",
                category="general_inquiry", response_action="archive",
                summary="Generic archive.",
            )
            obs, reward, done, info = env.step(action)
            scores.append(reward.value)
        variance = max(scores) - min(scores)
        print(f"  {task_id}: scores={[round(s, 3) for s in scores]}, variance={variance:.3f}")
        assert variance > 0.0, f"DISQUALIFICATION: all scores identical in {task_id}!"


def check_difficulty_spread():
    print("\n=== PHASE 2: Difficulty spread (naive agent) ===")
    naive_scores = {}
    for task_id in ["easy_triage", "medium_triage", "hard_triage"]:
        env = EmailTriageEnv(task_id=task_id)
        obs = env.reset()
        while not env.done:
            email = obs.current_email
            action = Action(
                email_id=email.id, priority="normal",
                category="general_inquiry", response_action="archive",
                summary="Generic summary.",
            )
            obs, reward, done, info = env.step(action)
        naive_scores[task_id] = env.final_score()
    e = naive_scores["easy_triage"]
    m = naive_scores["medium_triage"]
    h = naive_scores["hard_triage"]
    print(f"  easy={e:.4f}  medium={m:.4f}  hard={h:.4f}")
    assert e >= h, f"Hard task should be harder than easy for naive agent: easy={e}, hard={h}"


def check_http_endpoints():
    print("\n=== PHASE 1/3: HTTP endpoint contracts ===")
    from fastapi.testclient import TestClient
    from app import app as fastapi_app
    client = TestClient(fastapi_app)

    # Root returns 200
    r = client.get("/")
    assert r.status_code == 200, f"GET / failed: {r.status_code}"
    data = r.json()
    assert "tasks" in data and "endpoints" in data
    print("  GET / returns 200 with tasks and endpoints")

    # Health returns 200
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("  GET /health returns 200")

    # Tasks endpoint
    r = client.get("/tasks")
    assert r.status_code == 200
    tasks = r.json()
    assert len(tasks) == 3
    for tid in ["easy_triage", "medium_triage", "hard_triage"]:
        assert tid in tasks
        assert "difficulty" in tasks[tid]
        assert "num_emails" in tasks[tid]
    print("  GET /tasks returns all 3 tasks")

    # Reset returns valid observation
    r = client.post("/reset?task_id=easy_triage")
    assert r.status_code == 200, f"POST /reset failed: {r.status_code}"
    obs = r.json()
    assert "current_email" in obs
    assert "emails_remaining" in obs
    assert obs["emails_remaining"] > 0
    print("  POST /reset returns valid observation")

    # State returns valid state
    r = client.get("/state?task_id=easy_triage")
    assert r.status_code == 200
    state = r.json()
    assert "task_id" in state and "step" in state and "done" in state
    print("  GET /state returns valid state")

    # Step through full episode
    obs_data = obs
    step_count = 0
    while True:
        email = obs_data.get("current_email")
        if not email:
            break
        payload = {
            "email_id": email["id"], "priority": "normal",
            "category": "general_inquiry", "response_action": "archive",
            "summary": "Test summary for this email.",
        }
        r = client.post("/step?task_id=easy_triage", json=payload)
        assert r.status_code == 200, f"POST /step failed: {r.status_code} {r.text}"
        data = r.json()
        assert "reward" in data
        assert 0.0 <= data["reward"]["value"] <= 1.0
        obs_data = data["observation"]
        step_count += 1
        if data["done"]:
            break
    print(f"  POST /step: completed {step_count} steps, all rewards in [0,1]")

    # Done episode returns 409
    r = client.post("/step?task_id=easy_triage", json=payload)
    assert r.status_code == 409, f"Expected 409 after done, got {r.status_code}"
    print("  POST /step after done returns 409 (not silent auto-reset)")

    # Reset clears done state
    r = client.post("/reset?task_id=easy_triage")
    assert r.status_code == 200
    obs_data = r.json()
    r = client.post("/step?task_id=easy_triage", json={
        "email_id": obs_data["current_email"]["id"], "priority": "normal",
        "category": "general_inquiry", "response_action": "archive",
        "summary": "Post-reset step test.",
    })
    assert r.status_code == 200, f"Step after reset failed: {r.status_code}"
    print("  POST /reset then /step works correctly")

    # Unknown task_id returns 400
    r = client.post("/reset?task_id=nonexistent")
    assert r.status_code == 400
    print("  Unknown task_id returns 400")


def check_no_reward_leakage():
    print("\n=== PHASE 3: No reward leakage in observation ===")
    env = EmailTriageEnv(task_id="hard_triage")
    obs = env.reset()
    assert "episode_rewards_so_far" not in obs.context, "Reward leakage in observation!"
    # Step once and check again
    email = obs.current_email
    action = Action(email_id=email.id, priority="normal",
                    category="general_inquiry", response_action="archive",
                    summary="Test.")
    obs2, _, _, _ = env.step(action)
    assert "episode_rewards_so_far" not in obs2.context, "Reward leakage after step!"
    print("  No reward leakage in observation context")


def check_stdout_format():
    """Verify inference.py stdout matches the mandatory spec format exactly."""
    print("\n=== MANDATORY: stdout log format matches spec ===")
    import re
    import io
    import json
    import inference as inf_mod

    # Patch call_llm with a deterministic mock
    from env.data import GROUND_TRUTH
    def mock_llm(messages):
        user_msg = messages[-1]["content"]
        eid = None
        for line in user_msg.split("\n"):
            stripped = line.strip()
            if stripped.startswith("ID:"):
                eid = stripped[3:].strip()
                break
        gt = GROUND_TRUTH.get(eid, {})
        return json.dumps({
            "email_id": eid,
            "priority": gt.get("priority", "normal"),
            "category": gt.get("category", "general_inquiry"),
            "response_action": gt.get("response_action", "archive"),
            "summary": f"Summary for {eid}.",
            "reasoning": "mock",
        })
    inf_mod.call_llm = mock_llm

    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    for task_id in inf_mod.TASKS:
        inf_mod.run_task(task_id)
    sys.stdout = old_stdout
    output = buf.getvalue()

    lines = [l for l in output.strip().split("\n") if l]

    # Patterns from the spec example (single space after tag)
    start_pat = re.compile(r"^\[START\] task=\S+ env=\S+ model=\S+$")
    step_pat  = re.compile(r"^\[STEP\] step=\d+ action=\S+ reward=\d+\.\d{2} done=(true|false) error=\S+$")
    end_pat   = re.compile(r"^\[END\] success=(true|false) steps=\d+ score=\d+\.\d{2} rewards=[\d.,]+$")

    starts = [l for l in lines if l.startswith("[START]")]
    steps  = [l for l in lines if l.startswith("[STEP]")]
    ends   = [l for l in lines if l.startswith("[END]")]

    assert len(starts) == 3, f"Expected 3 [START] lines, got {len(starts)}"
    assert len(ends) == 3,   f"Expected 3 [END] lines, got {len(ends)}"
    assert len(steps) == 19, f"Expected 19 [STEP] lines, got {len(steps)}"

    for l in lines:
        assert "\n" not in l and "\r" not in l, f"Embedded newline in line: {repr(l)}"
        if l.startswith("[START]"):
            assert start_pat.match(l), f"Bad [START] format: {repr(l)}"
        elif l.startswith("[STEP]"):
            assert step_pat.match(l),  f"Bad [STEP] format: {repr(l)}"
        elif l.startswith("[END]"):
            assert end_pat.match(l),   f"Bad [END] format: {repr(l)}"

    # Scores and rewards must be in [0, 1]
    for l in ends:
        score = float(l.split("score=")[1].split(" ")[0])
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        for r in l.split("rewards=")[1].split(","):
            assert 0.0 <= float(r) <= 1.0, f"Reward out of range: {r}"

    # Field order: START has task= before env= before model=
    for l in starts:
        parts = l.split(" ")
        keys = [p.split("=")[0] for p in parts[1:]]
        assert keys == ["task", "env", "model"], f"Wrong field order in START: {keys}"

    # Field order: STEP has step= action= reward= done= error=
    for l in steps:
        parts = l.split(" ")
        keys = [p.split("=")[0] for p in parts[1:]]
        assert keys == ["step", "action", "reward", "done", "error"], \
            f"Wrong field order in STEP: {keys}"

    # Field order: END has success= steps= score= rewards=
    for l in ends:
        parts = l.split(" ")
        keys = [p.split("=")[0] for p in parts[1:]]
        assert keys == ["success", "steps", "score", "rewards"], \
            f"Wrong field order in END: {keys}"

    print(f"  [START]={len(starts)} [STEP]={len(steps)} [END]={len(ends)} — all valid")
    print("  Field names, order, spacing, and value ranges all match spec")


if __name__ == "__main__":
    try:
        check_clean_reset()
        check_api_contracts()
        check_graders()
        check_score_variance()
        check_difficulty_spread()
        check_http_endpoints()
        check_no_reward_leakage()
        check_stdout_format()
        print("\n" + "="*50)
        print("ALL PHASE 1 / 2 / 3 CHECKS PASSED")
        print("="*50)
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)
