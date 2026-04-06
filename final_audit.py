"""Final pre-submission audit — checks every mandatory requirement."""
import os, sys, re, io, json, ast, pathlib

# Set a dummy token so inference.py's fail-fast guard doesn't block the audit.
# The guard is correct behaviour — this only bypasses it for static analysis.
os.environ.setdefault("HF_TOKEN", "audit-dummy-token")

src = pathlib.Path("inference.py").read_text()

results = {}

# 1. inference.py in root
results["inference_in_root"] = pathlib.Path("inference.py").exists()

# 1b. .env.example exists with all 3 vars
env_example = pathlib.Path(".env.example").read_text() if pathlib.Path(".env.example").exists() else ""
results["env_example_exists"]       = bool(env_example)
results["env_example_hf_token"]     = "HF_TOKEN=" in env_example
results["env_example_api_base_url"] = "API_BASE_URL=" in env_example
results["env_example_model_name"]   = "MODEL_NAME=" in env_example

# 2. HF_TOKEN mandatory — no API_KEY fallback
results["hf_token_no_fallback"] = (
    'HF_TOKEN     = os.getenv("HF_TOKEN")' in src
    and "API_KEY" not in src
)

# 3. Fail-fast on missing vars — checks all 3
results["failfast_hf_token"]     = "if not HF_TOKEN:" in src or "_missing" in src
results["failfast_api_base_url"] = "API_BASE_URL" in src and "_missing" in src
results["failfast_model_name"]   = "MODEL_NAME" in src and "_missing" in src
results["failfast_exits"]        = "sys.exit(1)" in src

# 4. API_BASE_URL from env
results["api_base_url_from_env"] = 'os.getenv("API_BASE_URL"' in src

# 5. MODEL_NAME from env
results["model_name_from_env"] = 'os.getenv("MODEL_NAME"' in src

# 6. OpenAI client uses HF_TOKEN as api_key
tree = ast.parse(src)
openai_ok = False
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "OpenAI":
            for kw in node.keywords:
                if kw.arg == "api_key":
                    openai_ok = isinstance(kw.value, ast.Name) and kw.value.id == "HF_TOKEN"
results["openai_client_uses_hf_token"] = openai_ok

# 7. stdout format — run with mock LLM
import inference as inf_mod
from env.data import GROUND_TRUTH

def mock_llm(messages):
    user_msg = messages[-1]["content"]
    eid = None
    for line in user_msg.split("\n"):
        s = line.strip()
        if s.startswith("ID:"):
            eid = s[3:].strip()
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
old = sys.stdout
sys.stdout = buf = io.StringIO()
for t in inf_mod.TASKS:
    inf_mod.run_task(t)
sys.stdout = old
lines = [l for l in buf.getvalue().strip().split("\n") if l]

start_pat = re.compile(r"^\[START\] task=\S+ env=\S+ model=\S+$")
step_pat  = re.compile(r"^\[STEP\] step=\d+ action=\S+ reward=\d+\.\d{2} done=(true|false) error=\S+$")
end_pat   = re.compile(r"^\[END\] success=(true|false) steps=\d+ score=\d+\.\d{2} rewards=[\d.,]+$")

starts = [l for l in lines if l.startswith("[START]")]
steps  = [l for l in lines if l.startswith("[STEP]")]
ends   = [l for l in lines if l.startswith("[END]")]

results["stdout_3_starts"]  = len(starts) == 3
results["stdout_19_steps"]  = len(steps) == 19
results["stdout_3_ends"]    = len(ends) == 3
results["stdout_start_fmt"] = all(start_pat.match(l) for l in starts)
results["stdout_step_fmt"]  = all(step_pat.match(l) for l in steps)
results["stdout_end_fmt"]   = all(end_pat.match(l) for l in ends)
results["no_embedded_newlines"] = all("\n" not in l and "\r" not in l for l in lines)

# field order
order_ok = True
for l in starts:
    keys = [p.split("=")[0] for p in l.split(" ")[1:]]
    if keys != ["task", "env", "model"]:
        order_ok = False
for l in steps:
    keys = [p.split("=")[0] for p in l.split(" ")[1:]]
    if keys != ["step", "action", "reward", "done", "error"]:
        order_ok = False
for l in ends:
    keys = [p.split("=")[0] for p in l.split(" ")[1:]]
    if keys != ["success", "steps", "score", "rewards"]:
        order_ok = False
results["stdout_field_order"] = order_ok

# scores in [0,1]
results["scores_in_range"] = all(
    0.0 <= float(l.split("score=")[1].split(" ")[0]) <= 1.0
    for l in ends
)
results["rewards_in_range"] = all(
    all(0.0 <= float(r) <= 1.0 for r in l.split("rewards=")[1].split(","))
    for l in ends
)

# 8. 3 tasks
results["three_tasks"] = len(inf_mod.TASKS) == 3

# 9. README has all 3 env vars documented
readme = pathlib.Path("README.md").read_text()
results["readme_hf_token"]     = "HF_TOKEN" in readme
results["readme_api_base_url"] = "API_BASE_URL" in readme
results["readme_model_name"]   = "MODEL_NAME" in readme
results["readme_hf_frontmatter"] = readme.strip().startswith("---")

# 10. Dockerfile
df = pathlib.Path("Dockerfile").read_text()
results["dockerfile_healthcheck"]  = "HEALTHCHECK" in df
results["dockerfile_port_7860"]    = "7860" in df
results["dockerfile_unbuffered"]   = '"-u"' in df
results["dockerfile_nonroot_user"] = "appuser" in df

# 11. openenv.yaml
try:
    import yaml
    oy = yaml.safe_load(pathlib.Path("openenv.yaml").read_text())
    results["yaml_name"]        = "name" in oy
    results["yaml_3_tasks"]     = len(oy.get("tasks", [])) == 3
    results["yaml_entry_point"] = "entry_point" in oy
    results["yaml_endpoints"]   = "endpoints" in oy
    results["yaml_obs_space"]   = "observation_space" in oy
    results["yaml_action_space"] = "action_space" in oy
    results["yaml_reward"]      = "reward" in oy
except Exception as e:
    results["yaml_parse_error"] = False

# 12. HTTP endpoints via TestClient
from fastapi.testclient import TestClient
from app import app as fastapi_app
client = TestClient(fastapi_app)

r = client.get("/")
results["http_root_200"] = r.status_code == 200

r = client.get("/health")
results["http_health_200"] = r.status_code == 200 and r.json().get("status") == "ok"

r = client.post("/reset?task_id=easy_triage")
results["http_reset_200"] = r.status_code == 200
if r.status_code == 200:
    obs = r.json()
    results["reset_has_current_email"] = obs.get("current_email") is not None
    results["reset_clean_state"] = obs.get("step_number") == 0

r = client.get("/state?task_id=easy_triage")
results["http_state_200"] = r.status_code == 200

r = client.get("/tasks")
results["http_tasks_200"] = r.status_code == 200 and len(r.json()) == 3

# step then check done=409
obs_data = client.post("/reset?task_id=easy_triage").json()
while True:
    email = obs_data.get("current_email")
    if not email:
        break
    payload = {"email_id": email["id"], "priority": "normal",
               "category": "general_inquiry", "response_action": "archive",
               "summary": "Test."}
    resp = client.post("/step?task_id=easy_triage", json=payload)
    obs_data = resp.json()["observation"]
    if resp.json()["done"]:
        break
r = client.post("/step?task_id=easy_triage", json=payload)
results["http_done_returns_409"] = r.status_code == 409

# Print
print()
all_pass = True
for k, v in results.items():
    status = "PASS" if v else "FAIL"
    if not v:
        all_pass = False
    print(f"  [{status}] {k}")

print()
if all_pass:
    print("=" * 52)
    print("ALL MANDATORY CHECKS PASSED — ready to submit")
    print("=" * 52)
else:
    print("SOME CHECKS FAILED — fix before submitting")
    sys.exit(1)
