"""
ข้อ (2) ส่วนที่ 1: สร้าง BM25 index ที่ตัดคำภาษาไทยได้ (Thai analyzer)

ปัญหาของระบบเดิม (authenticRAG.py):
    BM25 index ใช้ "standard" analyzer ซึ่งตัดคำด้วยช่องว่าง
    ภาษาไทยไม่มีช่องว่างระหว่างคำ ทั้งประโยคจึงกลายเป็น "คำเดียว"
    เช่น "อาการของหัดเยอรมัน" -> ['อาการของหัดเยอรมัน']
    ทำให้ค้นหาด้วยประโยคภาษาไทยแทบไม่เจออะไรเลย (0 results)

วิธีแก้:
    ใช้ "thai" analyzer ที่มีใน OpenSearch (ตัดคำด้วยพจนานุกรมภาษาไทย)
    "อาการของหัดเยอรมัน" -> ['อาการ', 'หัดเยอรมัน']

วิธีใช้:
    python improved_ingest.py          # คัดลอก chunks + context เดิมจาก index ของระบบเดิม (เร็ว, ไม่เรียก LLM)
    python improved_ingest.py --full   # รัน pipeline เต็มใหม่ทั้งหมด (โหลดไฟล์ -> chunk -> สร้าง context ด้วย LLM -> index)

โหมดปกติใช้ chunks และ context ชุดเดียวกับระบบเดิมทุกตัวอักษร
ทำให้การเปรียบเทียบ (1) กับ (2) ยุติธรรม: ต่างกันแค่วิธีตัดคำเท่านั้น
"""
import sys
from opensearchpy import OpenSearch

BASELINE_BM25_INDEX = "anthropic-bm25-index"
IMPROVED_BM25_INDEX = "improved-bm25-index-thai"

THAI_BM25_SETTINGS = {
    "settings": {
        "analysis": {"analyzer": {"default": {"type": "thai"}}},
        "similarity": {"default": {"type": "BM25"}},
        "index.queries.cache.enabled": False
    },
    "mappings": {
        "properties": {
            "content": {"type": "text", "analyzer": "thai"},
            "contextualized_content": {"type": "text", "analyzer": "thai"},
            "doc_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "original_index": {"type": "integer", "index": False}
        }
    }
}


def create_thai_bm25_index(client, index_name=IMPROVED_BM25_INDEX):
    """ลบ index เก่า (ถ้ามี) แล้วสร้างใหม่ด้วย Thai analyzer"""
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    client.indices.create(index=index_name, body=THAI_BM25_SETTINGS)
    print(f"สร้าง BM25 index {index_name} (thai analyzer) สำเร็จ")


def copy_from_baseline(client):
    """คัดลอกเอกสารจาก BM25 index เดิม -> index ใหม่ (OpenSearch จะตัดคำใหม่ด้วย thai analyzer ระหว่างคัดลอก)"""
    if not client.indices.exists(index=BASELINE_BM25_INDEX):
        sys.exit(f"ไม่พบ {BASELINE_BM25_INDEX} — กรุณารัน python authenticRAG.py ก่อน หรือใช้ --full")
    create_thai_bm25_index(client)
    client.reindex(
        body={"source": {"index": BASELINE_BM25_INDEX}, "dest": {"index": IMPROVED_BM25_INDEX}},
        refresh=True, wait_for_completion=True
    )
    count = client.count(index=IMPROVED_BM25_INDEX)["count"]
    print(f"คัดลอก {count} chunks ไปยัง {IMPROVED_BM25_INDEX} สำเร็จ")


def full_ingest():
    """รัน pipeline เต็มของระบบเดิม แต่เขียน BM25 ลง index ใหม่ที่ใช้ thai analyzer"""
    from authenticRAG import AnthropicStyleContextualRAG

    class ThaiBM25ContextualRAG(AnthropicStyleContextualRAG):
        def _create_or_update_indices(self):
            self.bm25_index_name = IMPROVED_BM25_INDEX
            create_thai_bm25_index(self.opensearch_client)
            super()._create_or_update_indices()  # สร้าง vector index ถ้ายังไม่มี

    md_paths = ["./corpus_input/1.md", "./corpus_input/2.md", "./corpus_input/44.md", "./corpus_input/5555.md"]
    rag = ThaiBM25ContextualRAG()
    docs = rag.load_documents(md_paths)
    rag.add_documents_with_context(docs)


def show_tokenizer_difference(client, text="อาการของหัดเยอรมัน"):
    for analyzer in ["standard", "thai"]:
        tokens = client.indices.analyze(body={"analyzer": analyzer, "text": text})["tokens"]
        print(f"  {analyzer:8s} -> {[t['token'] for t in tokens]}")


if __name__ == "__main__":
    client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False)
    print("ตัวอย่างการตัดคำ:")
    show_tokenizer_difference(client)
    if "--full" in sys.argv:
        full_ingest()
    else:
        copy_from_baseline(client)
