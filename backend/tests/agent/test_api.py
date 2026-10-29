import json

from fastapi.testclient import TestClient

from app.agent import AgentConfig
from app.agent.factory import build_agent
from app.api.server import create_app
from tests.agent.fakes import ScriptedLLM, intent, narrative
from tests.agent.test_integrations import SALES

TOP_CITY_ARGS = {"column": "Продажи.Город", "limit": 10}


def make_client(llm):
    bundle = build_agent(config=AgentConfig(), llm=llm, data_dir="/nonexistent")
    return TestClient(create_app(bundle)), bundle


def test_stream_sends_status_then_done():
    llm = ScriptedLLM()
    client, _ = make_client(llm)
    with client.stream("POST", "/api/chat/stream", json={"message": "Привет!", "session_id": "s1"}) as r:
        text = "".join(r.iter_text())
    events = [line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: ")]
    assert events[0] == "status" and events[-1] == "done"
    done = json.loads(text.strip().split("data: ")[-1])
    assert done["response"]["status"] == "ok" and done["response"]["session_id"] == "s1"


def test_upload_catalog_chat_history_and_png():
    llm = ScriptedLLM({
        "intent_decision": [intent("chart", "Какие города встречаются чаще всего?", wants_explanation=True)],
        "chart_plan": [{"items": [{"chart_type": "bar", "title": "Города", "purpose": "Частота"}]}],
        "dsl_query": [{
            "steps": [
                {"var": "T", "op": "source", "table": "Продажи"},
                {"var": "A", "op": "aggregate", "input": "T", "group_by": ["Город"],
                 "aggregates": [{"func": "count", "column": "*", "alias": "Заказы"}]},
                {"var": "S", "op": "sort", "input": "A", "by": [{"column": "Заказы", "dir": "desc"}]},
            ],
            "output": "S",
        }],
        "narrative": [narrative("Пермь лидирует.", ["Пермь: 3"])],
    })
    client, _ = make_client(llm)
    assert client.get("/api/health").json()["tables"] == 0

    r = client.post("/api/datasets", files={"file": ("Продажи.csv", SALES, "text/csv")})
    assert r.status_code == 200 and r.json()["row_count"] == 5
    assert client.get("/api/catalog").json()["tables"][0]["name"] == "Продажи"

    r = client.post("/api/chat", json={"message": "Какие города встречаются чаще всего?"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok", body
    chart = next(b for b in body["blocks"] if b["type"] == "chart")
    assert chart["engine_spec"]["type"] == "bar"

    history = client.get(f"/api/sessions/{body['session_id']}").json()
    assert history["turns"][0]["summary"]

    png = client.post("/api/charts/png",
                      json={"engine_spec": chart["engine_spec"], "data": chart["data"]})
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"


def test_dashboard_endpoints():
    llm = ScriptedLLM({
        "intent_decision": [intent("chart", "Дашборд по городам")],
        "chart_plan": [{"items": [
            {"chart_type": "bar", "title": "Топ городов", "purpose": "Частота"},
            {"chart_type": "table", "title": "Все города", "purpose": "Список"},
        ]}],
        "dsl_query": [
            {"steps": [
                {"var": "T", "op": "source", "table": "Продажи"},
                {"var": "A", "op": "aggregate", "input": "T", "group_by": ["Город"],
                 "aggregates": [{"func": "count", "column": "*", "alias": "Заказы"}]},
            ], "output": "A"},
            {"steps": [
                {"var": "T", "op": "source", "table": "Продажи"},
                {"var": "A", "op": "aggregate", "input": "T", "group_by": ["Город"],
                 "aggregates": [{"func": "count", "column": "*", "alias": "Заказы"}]},
            ], "output": "A"},
        ],
    })
    client, bundle = make_client(llm)
    bundle.store.add_csv(SALES, "Продажи.csv")
    body = client.post("/api/chat",
                       json={"message": "Дашборд по городам", "session_id": "s1"}).json()
    dash_id = next(b for b in body["blocks"] if b["type"] == "dashboard")["dashboard_id"]

    listed = client.get("/api/dashboards", params={"session_id": "s1"}).json()
    assert listed[0]["id"] == dash_id and listed[0]["widget_count"] == 2
    assert client.get(f"/api/dashboards/{dash_id}").json()["widgets"][0]["tool"] == "data_query"
    assert client.post(f"/api/dashboards/{dash_id}/refresh").json()["refreshed_at"] is not None
    assert client.get("/api/dashboards/nope").status_code == 404
