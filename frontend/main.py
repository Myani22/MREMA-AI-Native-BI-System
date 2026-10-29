import os
import random
import string
import uuid
import json
from io import BytesIO
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api")
DB_URL = os.getenv("DATABASE_URL") 

TIMEOUT_CHAT = 120
TIMEOUT_CHECK = 15

class ResponseParser:
    def __init__(self, raw_response: Dict[str, Any]):
        self.raw = raw_response
        self.status = raw_response.get("status", "error")
        self.blocks = raw_response.get("blocks", [])
        self.trace = raw_response.get("trace", {})
        self.follow_ups = raw_response.get("follow_ups", [])

    @property
    def is_success(self) -> bool:
        return self.status == "ok"

    @property
    def is_clarification(self) -> bool:
        return self.status == "clarification"

    @property
    def is_error(self) -> bool:
        return self.status == "error"

    def get_clarification_block(self) -> Dict[str, Any] | None:
        for b in self.blocks:
            if b.get("type") == "clarification":
                return b
        return None
        
    def get_error_block(self) -> Dict[str, Any] | None:
        for b in self.blocks:
            if b.get("type") == "error":
                return b
        return None

def generate_chat_name() -> str:
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    return f"Session-{suffix}"

@st.fragment(run_every=f"{TIMEOUT_CHECK}s")
def render_server_status():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            model_name = data.get("model", "Unknown")
            st.button(f"Server: OK | Model: {model_name}", use_container_width=True)
        else:
            st.button("Server: Error", use_container_width=True)
    except Exception:
        st.button("Server: Offline", use_container_width=True)

st.set_page_config(
    page_title="MREMA-AI-Native-Bi", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.chat_name = generate_chat_name()
    st.session_state.messages = []

def fetch_available_tables():
    try:
        cat_resp = requests.get(f"{API_URL}/catalog", timeout=3)
        if cat_resp.status_code == 200:
            return [t["name"] for t in cat_resp.json().get("tables", [])]
    except Exception:
        pass
    return []

if "available_tables" not in st.session_state:
    st.session_state.available_tables = fetch_available_tables()

with st.sidebar:
    render_server_status()
    
    st.header(st.session_state.chat_name)
    
    if st.button("Новый чат", use_container_width=True):
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.chat_name = generate_chat_name()
        st.session_state.messages = []
        st.rerun()
        
    st.divider()
    
    debug_mode = st.toggle("Режим отладки")
    
    st.header("Загрузка CSV")
    
    upload_mode = st.radio(
        "Куда загружать данные:",
        options=["В память", "В базу данных"]
    )
    
    uploaded_files = st.file_uploader("CSV файлы", type=["csv"], accept_multiple_files=True)
    
    if uploaded_files and st.button("Загрузить", use_container_width=True):
        with st.spinner("Загрузка файлов..."):
            endpoint = "/datasets/import" if "базу данных" in upload_mode else "/datasets"
            success_count = 0
            
            for file in uploaded_files:
                files_payload = {"file": (file.name, file.getvalue(), "text/csv")}
                try:
                    resp = requests.post(f"{API_URL}{endpoint}", files=files_payload, timeout=90)
                    if resp.status_code in (200, 201):
                        success_count += 1
                    else:
                        st.error(f"Ошибка загрузки {file.name} (Код {resp.status_code})")
                        if debug_mode:
                            with st.expander(f"Trace ошибки {file.name}"):
                                try:
                                    st.json(resp.json())
                                except Exception:
                                    st.text(resp.text)
                except requests.exceptions.Timeout:
                    st.error(f"Таймаут загрузки: {file.name}")
                except Exception as e:
                    st.error(f"Сетевая ошибка ({file.name}): {e}")
            
            if success_count > 0:
                st.success(f"Загружено файлов: {success_count} из {len(uploaded_files)}")
                
                if "базу данных" in upload_mode:
                    if DB_URL is None:
                        st.warning("Файлы в БД загружены, но DATABASE_URL не задан в env фронта — автоматическая синхронизация пропущена. Откройте отладку и синхронизируйте вручную (или задайте DATABASE_URL).")
                    else:
                        with st.spinner("Автоматическая синхронизация БД..."):
                            current_schema = st.session_state.get("db_schema_input", "data")
                            payload = {"url": DB_URL, "schema": current_schema}
                            
                            try:
                                sync_resp = requests.post(f"{API_URL}/datasets/import-db", json=payload, timeout=30)
                                if sync_resp.status_code == 200:
                                    sync_data = sync_resp.json()
                                    st.success(f"БД обновлена! Всего таблиц: {sync_data.get('tables', 0)}")
                                else:
                                    error_text = sync_resp.text[:200]
                                    try:
                                        error_text = sync_resp.json().get("detail", error_text)
                                    except Exception:
                                        pass
                                    st.warning(f"Ошибка синхронизации (Код {sync_resp.status_code}): {error_text}")
                            except Exception as e:
                                st.warning(f"Файлы загружены, но авто-синхронизация не удалась: {e}")
                            
    if debug_mode:
        st.divider()
        if DB_URL:
            st.header("Синхронизация БД")
            db_schema = st.text_input("Схема БД", value="data", key="db_schema_input")
            
            if st.button("Синхронизировать БД вручную", use_container_width=True):
                with st.spinner("Подключение к базе..."):
                    payload = {
                        "url": DB_URL,
                        "schema": db_schema
                    }
                    try:
                        resp = requests.post(f"{API_URL}/datasets/import-db", json=payload, timeout=30)
                        if resp.status_code == 200:
                            data = resp.json()
                            st.success(f"Подключено! Таблиц: {data.get('tables', 0)}")
                        else:
                            st.error(f"Ошибка синхронизации (Код {resp.status_code})")
                            with st.expander("Trace ошибки"):
                                try:
                                    st.json(resp.json())
                                except Exception:
                                    st.text(resp.text)
                    except requests.exceptions.Timeout:
                        st.error("Время ожидания подключения к БД превышено.")
                    except Exception as e:
                        st.error(f"Сетевая ошибка: {e}")
        else:
            st.info("Синхронизация БД отключена (нет DATABASE_URL)")

st.title("MREMA-AI-Native-Bi")

for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            parser = ResponseParser(msg["raw_data"])
            
            if debug_mode:
                with st.expander("Trace"):
                    st.json(parser.raw)
            
            if parser.is_error:
                err_block = parser.get_error_block()
                err_msg = err_block.get("message", "Ошибка") if err_block else "Ошибка обработки запроса"
                st.error(err_msg)
                
            elif parser.is_clarification:
                clar_block = parser.get_clarification_block()
                if clar_block:
                    st.warning(clar_block.get("question", "Уточните запрос"))
                    for opt_idx, opt in enumerate(clar_block.get("options", [])):
                        if st.button(opt, key=f"clar_{msg_idx}_{opt_idx}"):
                            st.session_state.messages.append({"role": "user", "content": opt})
                            st.rerun()
                            
            elif parser.is_success:
                for block in parser.blocks:
                    btype = block.get("type")
                    
                    if btype == "text":
                        st.markdown(block.get("markdown", ""))
                        points = block.get("points", [])
                        if points:
                            for i, point in enumerate(points, 1):
                                st.markdown(f"{i}. {point}")
                                
                    elif btype == "dashboard":
                        st.info(f"Дашборд «{block.get('title')}» ({block.get('widget_count')} виджетов) успешно спроектирован.")
                        
                    elif btype == "table":
                        cols = [c["name"] for c in block["data"]["columns"]]
                        df = pd.DataFrame(block["data"]["rows"], columns=cols)
                        st.dataframe(df)
                        
                    elif btype == "chart" and not block.get("engine_spec"):
                        if block.get("note"):
                            st.caption(block["note"])
                    elif btype == "chart":
                        if block.get("note"):
                            st.caption(block["note"])
                        with st.spinner("Генерация графика..."):
                            try:
                                png_resp = requests.post(
                                    f"{API_URL}/charts/png", 
                                    json={"engine_spec": block["engine_spec"], "data": block["data"]},
                                    timeout=30
                                )
                                if png_resp.status_code == 200:
                                    st.image(BytesIO(png_resp.content))
                                else:
                                    st.error("Не удалось сгенерировать график (ошибка рендерера)")
                            except Exception:
                                st.error("Таймаут или ошибка при генерации графика")

st.markdown("---")
col_select, col_refresh = st.columns([9, 1])

with col_refresh:
    st.write("") # Небольшой отступ, чтобы кнопка выровнялась по высоте с селектором
    if st.button("🔄", help="Обновить список таблиц из БД", use_container_width=True):
        st.session_state.available_tables = fetch_available_tables()
        st.rerun()

with col_select:
    selected_tables = st.multiselect(
        "Поиск только в этих таблицах:",
        options=st.session_state.available_tables,
        default=st.session_state.get("selected_tables", st.session_state.available_tables),
        key="selected_tables"
    )

if prompt := st.chat_input("Введите запрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.status("Обработка запроса...", expanded=True) as status:
            try:
                # Используем стриминговый эндпоинт
                payload = {
                    "message": prompt, 
                    "session_id": st.session_state.session_id,
                    "tables": selected_tables if selected_tables else None
                }
                
                resp = requests.post(
                    f"{API_URL}/chat/stream", 
                    json=payload,
                    stream=True,
                    timeout=300
                )
                
                final_data = None
                
                if resp.status_code == 200:
                    for line in resp.iter_lines():
                        if line:
                            decoded = line.decode('utf-8')
                            if decoded.startswith("data: "):
                                data_str = decoded[6:]
                                try:
                                    event_data = json.loads(data_str)
                                    if event_data.get("type") == "status":
                                        # Обновляем статус в UI
                                        status.update(label=f"⏳ {event_data.get('message')}")
                                    elif event_data.get("type") == "done":
                                        final_data = event_data.get("response")
                                        status.update(label="Готово!", state="complete", expanded=False)
                                        break
                                except json.JSONDecodeError:
                                    pass
                                    
                    if final_data:
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "raw_data": final_data 
                        })
                        st.rerun()
                    else:
                        st.error("Не удалось получить финальный ответ от сервера.")
                else:
                    st.error(f"Ошибка сервера: {resp.status_code}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "raw_data": {"status": "error", "blocks": [{"type": "error", "message": f"Ошибка сервера", "retryable": True}]}
                    })
                    st.rerun()

            except requests.exceptions.Timeout:
                status.update(label="Таймаут", state="error")
                st.error("Время ожидания истекло. Попробуйте еще раз.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "raw_data": {"status": "error", "blocks": [{"type": "error", "message": "Таймаут сервера", "retryable": True}]}
                })
                st.rerun()
            except Exception as e:
                status.update(label="Ошибка", state="error")
                st.error(f"Ошибка соединения: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "raw_data": {"status": "error", "blocks": [{"type": "error", "message": f"Ошибка: {e}", "retryable": True}]}
                })
                st.rerun()