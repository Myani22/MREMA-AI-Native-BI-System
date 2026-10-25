import os
import random
import string
import uuid
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
    
    uploaded_file = st.file_uploader("CSV файл", type=["csv"])
    
    if uploaded_file and st.button("Загрузить", use_container_width=True):
        with st.spinner("Загрузка..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            
            endpoint = "/datasets/import" if "базу данных" in upload_mode else "/datasets"
            try:
                resp = requests.post(f"{API_URL}{endpoint}", files=files, timeout=60)
                if resp.status_code in (200, 201):
                    st.success("Файл успешно загружен")
                else:
                    st.error(f"Ошибка загрузки (Код {resp.status_code})")
                    if debug_mode:
                        with st.expander("Trace ошибки"):
                            try:
                                st.json(resp.json())
                            except Exception:
                                st.text(resp.text)
            except requests.exceptions.Timeout:
                st.error("Время ожидания загрузки файла превышено.")
            except Exception as e:
                st.error(f"Сетевая ошибка: {e}")
                            
    if debug_mode:
        st.divider()
        if DB_URL:
            st.header("Синхронизация БД")
            db_schema = st.text_input("Схема БД", value="data")
            
            if st.button("Синхронизировать БД", use_container_width=True):
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
                        
                    elif btype == "chart":
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

if prompt := st.chat_input("Введите запрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("Обработка..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat", 
                    json={"message": prompt, "session_id": st.session_state.session_id},
                    timeout=TIMEOUT_CHAT
                )
                
                if resp.status_code == 200:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "raw_data": resp.json() 
                    })
                    st.rerun()
                else:
                    st.error(f"Ошибка сервера (Код {resp.status_code})")
                    if debug_mode:
                        with st.expander("Trace ошибки"):
                            try:
                                st.json(resp.json())
                            except Exception:
                                st.text(resp.text)
                                
            except requests.exceptions.Timeout:
                st.error(f"Время ожидания ответа превышено (таймаут {TIMEOUT_CHAT}). Попробуйте еще раз.")
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка соединения с сервером: {e}")