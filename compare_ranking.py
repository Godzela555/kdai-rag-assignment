"""
ดูการจัดอันดับทีละขั้น ของคำถามเดียว: BM25 -> Vector -> RRF -> Reranker
ไม่เรียก LLM และไม่แก้ข้อมูลใน OpenSearch (แค่ค้นหา)

วิธีใช้:
    python compare_ranking.py "อาการของหัดเยอรมัน"
"""
import sys
from improved_search import ImprovedSearchRAG

sys.stdout.reconfigure(encoding="utf-8")


def title(src):
    """บรรทัดแรกของ chunk (หัวข้อ) ไว้ดูว่าเป็น chunk เรื่องอะไร"""
    return src.get("content", "").strip().split("\n")[0][:55]


def show(name, results, score_name):
    print(f"\n--- {name} ---")
    if not results:
        print("  (ไม่เจอเลย)")
    for rank, (doc_id, score, src) in enumerate(results, start=1):
        print(f"  {rank:>2}. {doc_id:<7} {score_name} {score:8.4f}  {title(src)}")


query = sys.argv[1] if len(sys.argv) > 1 else "อาการของหัดเยอรมัน"
rag = ImprovedSearchRAG()
print(f"\nคำถาม: {query}")

# ระบบเดิม (1): standard BM25 + vector -> RRF
rag.configure(use_thai_bm25=False, use_reranker=False)
print("\n================ ระบบเดิม (1) ================")
show("BM25 (standard)", rag.sparse_search(query, k=10), "BM25")
show("Vector (bge-m3)", rag.dense_search(query, k=10), "sim ")
show("RRF -> 5 chunks ที่ส่งให้ LLM", rag.hybrid_search(query, k=5), "RRF ")

# ระบบใหม่ (2): thai BM25 + vector -> RRF 10 ตัว -> reranker เลือก 5
rag.configure(use_thai_bm25=True, use_reranker=False)
print("\n================ ระบบใหม่ (2) ================")
sparse = rag.sparse_search(query, k=rag.num_candidates)
dense = rag.dense_search(query, k=rag.num_candidates)
show("BM25 (thai)", sparse, "BM25")
show("Vector (bge-m3) — ตู้เดียวกับระบบเดิม ผลจึงเหมือนกัน", dense, "sim ")
show("RRF -> ผู้สมัคร 10 ตัว", rag.rrf_fusion([sparse, dense])[:rag.num_candidates], "RRF ")
rag.configure(use_thai_bm25=True, use_reranker=True)
show("Reranker -> 5 chunks ที่ส่งให้ LLM", rag.hybrid_search(query, k=5), "rerank")
