import os
import random
import string
import uuid
from io import BytesIO
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api")

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

    def get_text_blocks(self) -> List[Dict[str, Any]]:
        return [b for b in self.blocks if b.get("type") == "text"]

    def get_chart_blocks(self) -> List[Dict[str, Any]]:
        return [b for b in self.blocks if b.get("type") == "chart"]

    def get_table_blocks(self) -> List[Dict[str, Any]]:
        return [b for b in self.blocks if b.get("type") == "table"]

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

st.set_page_config(page_title="AI-Native BI", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.chat_name = generate_chat_name()
    st.session_state.messages = []

with st.sidebar:
    st.header(st.session_state.chat_name)
    
    if st.button("Новый чат", use_container_width=True):
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.chat_name = generate_chat_name()
        st.session_state.messages = []
        st.rerun()
        
    st.divider()
    
    debug_mode = st.toggle("Режим отладки")
    
    st.header("Данные")
    uploaded_file = st.file_uploader("CSV файл", type=["csv"])
    
    if uploaded_file and st.button("Загрузить"):
        with st.spinner("Загрузка..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            resp = requests.post(f"{API_URL}/datasets", files=files)
            if resp.status_code == 200:
                st.success(resp.json().get("name"))
            else:
                st.error("Ошибка загрузки")

st.title(st.session_state.chat_name)

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
                for text_block in parser.get_text_blocks():
                    st.markdown(text_block.get("markdown", ""))
                    
                for table_block in parser.get_table_blocks():
                    cols = [c["name"] for c in table_block["data"]["columns"]]
                    df = pd.DataFrame(table_block["data"]["rows"], columns=cols)
                    st.dataframe(df)
                    
                for chart_block in parser.get_chart_blocks():
                    with st.spinner("Генерация графика..."):
                        png_resp = requests.post(
                            f"{API_URL}/charts/png", 
                            json={"engine_spec": chart_block["engine_spec"], "data": chart_block["data"]}
                        )
                        if png_resp.status_code == 200:
                            st.image(BytesIO(png_resp.content))

            if msg_idx == len(st.session_state.messages) - 1 and parser.follow_ups:
                st.write("---")
                for f_idx, f_up in enumerate(parser.follow_ups):
                    if st.button(f_up, key=f"fup_{msg_idx}_{f_idx}"):
                        st.session_state.messages.append({"role": "user", "content": f_up})
                        st.rerun()

if prompt := st.chat_input("Введите запрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("Обработка..."):
            resp = requests.post(
                f"{API_URL}/chat", 
                json={"message": prompt, "session_id": st.session_state.session_id}
            )
            
            if resp.status_code == 200:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "raw_data": resp.json() 
                })
                st.rerun()
            else:
                st.error("Ошибка сервера")
                st.session_state.messages.pop()