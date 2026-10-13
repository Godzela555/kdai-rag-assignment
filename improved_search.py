"""
ข้อ (2): ระบบค้นหาที่ปรับปรุงแล้ว (ต่อยอดจาก onlysearchAuthenticRAG.py โดยไม่แก้ไฟล์เดิม)

ปรับปรุง 2 จุด (ไม่เปลี่ยน LLM ที่ใช้สร้างคำตอบสุดท้าย):

  A. Search algorithm — BM25 ที่ตัดคำภาษาไทยได้
     ใช้ index "improved-bm25-index-thai" (สร้างด้วย improved_ingest.py)
     แทน "anthropic-bm25-index" ที่ใช้ standard analyzer

  B. Reranking — Cross-encoder reranker (BAAI/bge-reranker-v2-m3)
     1) ดึงผู้สมัคร 10 chunks จาก BM25 + Vector แล้วรวมด้วย RRF (เหมือนเดิม แต่กว้างขึ้น)
     2) แบ่งแต่ละ chunk เป็นหน้าต่างสั้น ๆ (~500 ตัวอักษร) เพราะ reranker อ่านได้ครั้งละ ~512 tokens
     3) ให้ reranker อ่าน "คำถาม + หน้าต่าง" คู่กันทีละคู่ แล้วให้คะแนนความเกี่ยวข้อง 0-1
        คะแนนของ chunk = คะแนนของหน้าต่างที่ดีที่สุด (MaxP)
     4) เลือก 5 อันดับแรกตามคะแนน reranker ส่งให้ LLM

     ทำไมดีกว่า RRF อย่างเดียว: RRF ดูแค่ "ลำดับ" จาก 2 ระบบ ไม่ได้อ่านเนื้อหาจริง
     ส่วน bi-encoder (bge-m3) แปลงคำถามและเอกสารเป็น vector แยกกัน
     แต่ cross-encoder อ่านคำถามและเอกสารพร้อมกัน จึงตัดสินความเกี่ยวข้องได้แม่นกว่า
     (แนวคิดเดียวกับ Jev reranker ในบทความที่อาจารย์แนะนำ — เลือกใช้ bge-reranker-v2-m3
      เพราะรันในเครื่องได้ฟรี รองรับภาษาไทย และเป็นตระกูลเดียวกับ bge-m3 ที่ใช้ทำ embedding)
"""
import os
import time
from sentence_transformers import CrossEncoder
from onlysearchAuthenticRAG import AuthenticSearchRAG

IMPROVED_BM25_INDEX = "improved-bm25-index-thai"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
WINDOW_CHARS = 500     # ความยาวหน้าต่างสำหรับ reranker (ตัวอักษร)
WINDOW_STRIDE = 400    # เลื่อนทีละ 400 -> ซ้อนกัน 100 ตัวอักษร กันคำตอบถูกตัดกลางประโยค


class ImprovedSearchRAG(AuthenticSearchRAG):
    def __init__(self, opensearch_host="localhost", opensearch_port=9200,
                 use_thai_bm25=True, use_reranker=True, rerank_mode="maxp", num_candidates=10):
        super().__init__(opensearch_host, opensearch_port)
        self.baseline_bm25_index = self.bm25_index_name
        self.num_candidates = num_candidates
        self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        self.configure(use_thai_bm25, use_reranker, rerank_mode)

    def configure(self, use_thai_bm25=True, use_reranker=True, rerank_mode="maxp"):
        """เปิด/ปิดแต่ละการปรับปรุง (ใช้ตอนวัดผลแยกทีละส่วน)"""
        self.use_thai_bm25 = use_thai_bm25
        self.use_reranker = use_reranker
        self.rerank_mode = rerank_mode
        self.bm25_index_name = IMPROVED_BM25_INDEX if use_thai_bm25 else self.baseline_bm25_index

    def call_qwen_api(self, prompt, max_tokens=500, temperature=0.2, retries=3):
        """เรียก LLM ตัวเดิม แต่ถ้าชนโควตา Groq ฟรี (429: 8,000 tokens/นาที) ให้รอแล้วลองใหม่"""
        for attempt in range(retries + 1):
            answer = super().call_qwen_api(prompt, max_tokens=max_tokens, temperature=temperature)
            if "Error code: 429" not in answer or attempt == retries:
                return answer
            print("Groq rate limit — รอ 60 วินาทีแล้วลองใหม่...")
            time.sleep(60)

    def _fill_missing_context(self, results):
        """ผลจาก vector index ไม่มี contextualized_content — ดึงมาจาก BM25 index ให้ครบ"""
        missing = [doc_id for doc_id, _, src in results if "contextualized_content" not in src]
        if missing:
            found = self.opensearch_client.mget(index=self.bm25_index_name, body={"ids": missing})
            extra = {d["_id"]: d["_source"] for d in found["docs"] if d.get("found")}
            for doc_id, _, src in results:
                if doc_id in extra:
                    src["contextualized_content"] = extra[doc_id].get("contextualized_content", "")
        return results

    @staticmethod
    def split_windows(text, size=WINDOW_CHARS, stride=WINDOW_STRIDE):
        """แบ่ง chunk ยาว ๆ เป็นหน้าต่างสั้น ๆ ที่ซ้อนกันเล็กน้อย"""
        if len(text) <= size:
            return [text]
        return [text[i:i + size] for i in range(0, len(text) - size + stride, stride)]

    def rerank(self, query, results):
        """ให้ cross-encoder ให้คะแนน (คำถาม, chunk) แล้วเรียงใหม่

        rerank_mode = "maxp" (ค่าเริ่มต้น):
            reranker อ่านได้ครั้งละ ~512 tokens แต่ chunk ของเราบางอันยาวถึง 8,000 ตัวอักษร
            ถ้าส่งทั้ง chunk ไป ส่วนท้ายจะถูกตัดทิ้งและ reranker มองไม่เห็นคำตอบที่อยู่ท้าย chunk
            จึงแบ่ง chunk เป็นหน้าต่างละ ~500 ตัวอักษร ให้คะแนนทุกหน้าต่าง
            แล้วใช้ "คะแนนสูงสุด" เป็นคะแนนของ chunk (MaxP: Dai & Callan, 2019)
        rerank_mode = "first512": ส่งทั้ง chunk แล้วปล่อยให้ถูกตัดที่ 512 tokens (ใช้เทียบในการวัดผล)
        """
        if self.rerank_mode == "first512":
            pairs = [(query, f"{src.get('content', '')}\n\n{src.get('contextualized_content', '')}")
                     for _, _, src in results]
            scores = self.reranker.predict(pairs, batch_size=8)
        else:
            pairs, owner = [], []
            for idx, (_, _, src) in enumerate(results):
                context = src.get("contextualized_content", "")
                for window in self.split_windows(src.get("content", "")):
                    pairs.append((query, f"{window}\n\n{context}"))
                    owner.append(idx)
            window_scores = self.reranker.predict(pairs, batch_size=16)
            scores = [float("-inf")] * len(results)
            for idx, s in zip(owner, window_scores):
                scores[idx] = max(scores[idx], float(s))
        reranked = [(doc_id, float(score), src) for (doc_id, _, src), score in zip(results, scores)]
        return sorted(reranked, key=lambda x: x[1], reverse=True)

    def hybrid_search(self, query, k=5, rrf_k=60):
        if not self.use_reranker:
            return super().hybrid_search(query, k=k, rrf_k=rrf_k)

        # 1) ดึงผู้สมัครให้กว้างขึ้น (10 จากแต่ละระบบ) แล้วรวมด้วย RRF
        sparse_results = self.sparse_search(query, k=self.num_candidates)
        dense_results = self.dense_search(query, k=self.num_candidates)
        print(f"\nSparse search found {len(sparse_results)} results")
        print(f"Dense search found {len(dense_results)} results")
        candidates = self.rrf_fusion([sparse_results, dense_results], k=rrf_k)[:self.num_candidates]
        candidates = self._fill_missing_context(candidates)

        # 2) Rerank ด้วย cross-encoder แล้วเลือก top-k
        reranked = self.rerank(query, candidates)
        print(f"Reranked {len(candidates)} candidates -> top {k}")
        return reranked[:k]


def main():
    if "GROQ_API_KEY" not in os.environ:
        print("Error: GROQ_API_KEY environment variable is not set")
        return

    rag = ImprovedSearchRAG()
    questions = [
        'โรคหัดและโรคหัดเยอรมันแตกต่างกันอย่างไร?',
        'อธิบายสาเหตุของโรคหัดเยอรมันและการป้องกัน',
        'ทำไมโรคหัดเยอรมันจึงมีอันตรายกับหญิงตั้งครรภ์?',
        'ถ้าคนที่ฉีดวัคซีนป้องกันโรคหัดเยอรมันแล้ว จะมีโอกาสติดเชื้อหรือไม่?',
        'โรคหัดเยอรมันมีผลกระทบอย่างไรต่อระบบสาธารณสุขและเศรษฐกิจของประเทศ?'
    ]
    results = rag.search_multiple_questions(questions, k=5)
    rag.export_results_to_json(results, "improved_rag_search_results.json")


if __name__ == "__main__":
    main()
