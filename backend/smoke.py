import json, sys, httpx

msg = sys.argv[1] if len(sys.argv) > 1 else "Что есть в этих данных?"
sid = sys.argv[2] if len(sys.argv) > 2 else None

payload = {"message": msg}
if sid:
    payload["session_id"] = sid

with httpx.Client(timeout=180) as c:
    r = c.post("http://localhost:8000/api/chat", json=payload)
    r.raise_for_status()
    data = r.json()

print(f"session_id = {data['session_id']}")
print(f"status     = {data['status']}")
print(f"intent     = {data['trace']['intent']}")
print(f"routed_by  = {data['trace']['routed_by']}")
print(f"llm_calls  = {[c['purpose'] for c in data['trace']['llm_calls']]}")
print(f"tool_calls = {[(t['tool'], t['status']) for t in data['trace']['tool_calls']]}")
for t in data["trace"]["tool_calls"]:
    if t.get("error"):
        print(f"  ERROR [{t['tool']}]: {t['error']}")
    if t.get("args"):
        print(f"  ARGS  [{t['tool']}]: {json.dumps(t['args'], ensure_ascii=False)}")
print(f"warnings   = {data['trace']['warnings']}")
print()
for b in data["blocks"]:
    t = b["type"]
    if t == "text":
        print(f"[text] {b['markdown']}")
        for i, p in enumerate(b.get("points", []), 1):
            print(f"   {i}. {p}")
    elif t == "chart":
        print(f"[chart] type={b['spec']['chart_type']} title={b['spec'].get('title','')}")
    elif t == "table":
        cols = [c["name"] for c in b["data"]["columns"]]
        print(f"[table] {len(b['data']['rows'])} rows, cols={cols}")
        for row in b["data"]["rows"][:5]:
            print(f"   {row}")
    elif t == "clarification":
        print(f"[clarify] {b['question']} options={b['options']}")
    elif t == "error":
        print(f"[error] {b['message']} retryable={b['retryable']}")
    elif t == "dashboard":
        print(f"[dashboard] id={b['dashboard_id']} widgets={b['widget_count']}")
print()
print("follow_ups =", data.get("follow_ups", []))