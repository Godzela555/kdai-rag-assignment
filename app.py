"""
ห้องทดลอง RAG — หน้าเว็บสำหรับถาม-ตอบ และเทียบระบบเดิม (1) กับระบบที่ปรับปรุง (2)

วิธีรัน (ต้องเปิด Docker / OpenSearch และรัน authenticRAG.py + improved_ingest.py มาก่อน):
    streamlit run app.py
"""
import contextlib
import io
import os
import time

import streamlit as st

from improved_search import ImprovedSearchRAG, IMPROVED_BM25_INDEX

SAMPLE_QUESTIONS = [
    "โรคหัดและโรคหัดเยอรมันแตกต่างกันอย่างไร?",
    "อธิบายสาเหตุของโรคหัดเยอรมันและการป้องกัน",
    "ทำไมโรคหัดเยอรมันจึงมีอันตรายกับหญิงตั้งครรภ์?",
    "ถ้าคนที่ฉีดวัคซีนป้องกันโรคหัดเยอรมันแล้ว จะมีโอกาสติดเชื้อหรือไม่?",
    "คนเป็นกรดไหลย้อนควรเลี่ยงซอสมะเขือเทศกับกระเทียมไหม",
]
SYSTEMS = {
    "(2) ระบบใหม่: Thai BM25 + Reranker (MaxP)": dict(use_thai_bm25=True, use_reranker=True, rerank_mode="maxp"),
    "(1) ระบบเดิม: Standard BM25 + Vector + RRF": dict(use_thai_bm25=False, use_reranker=False),
}

st.set_page_config(page_title="ห้องทดลอง RAG", page_icon="🔎", layout="centered")


@st.cache_resource(show_spinner="กำลังโหลดโมเดล bge-m3 และ reranker (ครั้งแรกประมาณ 1 นาที)...")
def load_rag():
    with contextlib.redirect_stdout(io.StringIO()):
        return ImprovedSearchRAG()


def count(rag, index):
    try:
        return rag.opensearch_client.count(index=index)["count"]
    except Exception:
        return "—"


st.title("ห้องทดลอง RAG")
st.caption("ถามคำถามจากบทความสุขภาพ 4 เรื่อง (หัดเยอรมัน, อหิวาตกโรค, ต้อกระจก, กรดไหลย้อน) แล้วเทียบระบบเดิมกับระบบที่ปรับปรุง")

if "GROQ_API_KEY" not in os.environ:
    st.error("ยังไม่ได้ตั้งค่า GROQ_API_KEY — ปิดหน้านี้ ตั้งค่าแล้วรัน `streamlit run app.py` ใหม่")
    st.stop()

try:
    rag = load_rag()
except Exception as e:
    st.error(f"เชื่อมต่อ OpenSearch ไม่ได้ — เปิด Docker Desktop และ container `opensearch` ก่อน\n\n{e}")
    st.stop()

# ---------- 1. สถานะข้อมูล ----------
st.subheader("1. สถานะข้อมูล")
c1, c2, c3 = st.columns(3)
c1.metric("Chunks ใน vector index", count(rag, rag.vector_index_name))
c2.metric("BM25 เดิม (standard)", count(rag, rag.baseline_bm25_index))
c3.metric("BM25 ใหม่ (thai)", count(rag, IMPROVED_BM25_INDEX))

# ---------- 2. ถาม-ตอบ ----------
st.subheader("2. ถาม-ตอบ")
system = st.radio("เลือกระบบ", list(SYSTEMS), key="system")

if "question" not in st.session_state:
    st.session_state.question = SAMPLE_QUESTIONS[0]
st.caption("ตัวอย่างคำถาม (กดเพื่อใส่ในช่อง)")
cols = st.columns(2)
for i, q in enumerate(SAMPLE_QUESTIONS):
    if cols[i % 2].button(q, key=f"sample{i}", use_container_width=True):
        st.session_state.question = q
question = st.text_area("คำถามของคุณ", key="question", height=80)

if st.button("ค้นเอกสารและตอบ", type="primary", disabled=not question.strip()):
    rag.configure(**SYSTEMS[system])
    t0 = time.time()
    with st.spinner("กำลังค้นหาและสร้างคำตอบ..."):
        with contextlib.redirect_stdout(io.StringIO()):
            bm25_hits = len(rag.sparse_search(question, k=10))
            result = rag.search_for_question(question, k=5)
    elapsed = time.time() - t0

    st.subheader("คำตอบ")
    st.markdown(result["answer"])
    st.caption(f"พบ {len(result['results'])} chunks · BM25 เจอ {bm25_hits} chunks · ใช้เวลา {elapsed:.1f} วินาที · "
               "คำตอบสร้างจาก AI เพื่อการเรียนรู้ โปรดตรวจเทียบเอกสารต้นทาง")
    if bm25_hits == 0:
        st.warning("BM25 หาไม่เจอเลย — ระบบนี้กำลังใช้แค่ vector search (ปัญหาตัดคำภาษาไทยของ standard analyzer)")

    st.subheader("หลักฐานที่ค้นพบ")
    score_name = "Reranker" if SYSTEMS[system].get("use_reranker") else "RRF"
    st.caption(f"เรียงตามคะแนน {score_name} — คะแนนใช้จัดอันดับภายในคำถามนี้เท่านั้น")
    for rank, doc in enumerate(result["results"], 1):
        with st.expander(f"อันดับ {rank} · {doc['doc_id']} · {score_name} {doc['score']:.4f}"):
            if doc.get("context"):
                st.markdown(f"**Context:** {doc['context']}")
            st.text(doc["content"])
