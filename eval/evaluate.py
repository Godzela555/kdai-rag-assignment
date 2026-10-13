"""
ข้อ (3): วัดผลเปรียบเทียบระบบเดิม (1) กับระบบที่ปรับปรุง (2)

วัดเฉพาะส่วน "ค้นหา" (retrieval) เพราะ LLM ตัวเดียวกันจะตอบได้ดีก็ต่อเมื่อได้ chunk ที่ถูกต้อง
ใช้ชุดคำถาม 2 ชุด ซึ่งแต่ละข้อระบุ chunk ที่มีคำตอบจริง ("relevant") ไว้ก่อนรันการค้นหา:
  - eval_questions.json      : 24 ข้อ ถามตามหัวข้อบทความ (กำหนดคำตอบจากหัวข้อของ chunk)
  - eval_questions_hard.json : 20 ข้อ ถามข้อเท็จจริงเฉพาะ (คำตอบ = ทุก chunk ที่มีวลี "anchor" อยู่จริง
                               ใช้โปรแกรมหาอัตโนมัติ จึงไม่มีการเลือกเข้าข้างระบบใด)

ตัวชี้วัด (ใช้ 5 chunks ที่ส่งให้ LLM เหมือนระบบจริง):
  Hit@1    : chunk อันดับ 1 เป็น chunk ที่ถูกต้องหรือไม่
  Hit@5    : มี chunk ที่ถูกต้องอยู่ใน 5 อันดับแรกหรือไม่
  Recall@5 : เจอ chunk ที่ถูกต้องครบกี่ % (บางคำถามมีคำตอบอยู่ 2 chunks)
  MRR@5    : ค่าเฉลี่ยของ 1/อันดับ ของ chunk ถูกต้องตัวแรก (ยิ่งใกล้ 1 ยิ่งดี)
  BM25=0   : จำนวนคำถามที่ BM25 หาไม่เจอเลยแม้แต่ chunk เดียว

วิธีรัน (จากโฟลเดอร์หลักของโปรเจกต์):
  python eval/evaluate.py
"""
import contextlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improved_search import ImprovedSearchRAG  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
K = 5

CONFIGS = [
    ("(1) ระบบเดิม: Standard BM25 + Vector + RRF", dict(use_thai_bm25=False, use_reranker=False)),
    ("(2A) + Thai BM25", dict(use_thai_bm25=True, use_reranker=False)),
    ("(2B) + Thai BM25 + Reranker (อ่านแค่ 512 tokens แรกของ chunk)", dict(use_thai_bm25=True, use_reranker=True, rerank_mode="first512")),
    ("(2) ระบบใหม่: Thai BM25 + Reranker แบบ MaxP", dict(use_thai_bm25=True, use_reranker=True, rerank_mode="maxp")),
]


def recall_at(retrieved, relevant, k):
    """Recall@k: ใน k อันดับแรก เจอ chunk ที่ถูกต้องกี่ส่วนจากทั้งหมด (บางคำถามมีคำตอบ 2-4 chunks)"""
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def score(retrieved, relevant):
    first = next((i + 1 for i, d in enumerate(retrieved) if d in relevant), None)
    return {
        "hit1": 1.0 if first == 1 else 0.0,
        "hit5": 1.0 if first else 0.0,
        "recall1": recall_at(retrieved, relevant, 1),
        "recall3": recall_at(retrieved, relevant, 3),
        "recall5": recall_at(retrieved, relevant, 5),
        "mrr5": 1.0 / first if first else 0.0,
        "first_rank": first,
    }


def summarize(name, rows, bm25_zero, sec):
    n = len(rows)
    return {
        "config": name,
        "Hit@1": sum(r["hit1"] for r in rows) / n,
        "Hit@5": sum(r["hit5"] for r in rows) / n,
        "Recall@1": sum(r["recall1"] for r in rows) / n,
        "Recall@3": sum(r["recall3"] for r in rows) / n,
        "Recall@5": sum(r["recall5"] for r in rows) / n,
        "MRR@5": sum(r["mrr5"] for r in rows) / n,
        "BM25=0": bm25_zero,
        "sec/query": sec,
    }


QUESTION_SETS = [
    ("ชุดคำถามทั่วไป (ถามตามหัวข้อของบทความ)", "eval_questions.json"),
    ("ชุดคำถามเจาะลึก (ข้อเท็จจริงเฉพาะที่ซ่อนอยู่กลาง chunk ยาว ๆ)", "eval_questions_hard.json"),
]


def run_set(rag, questions):
    summary, details = [], {}
    for name, cfg in CONFIGS:
        rag.configure(**cfg)
        rows, bm25_zero, t0 = [], 0, time.time()
        for q in questions:
            with contextlib.redirect_stdout(io.StringIO()):
                if not rag.sparse_search(q["question"], k=K * 2):
                    bm25_zero += 1
                retrieved = [doc_id for doc_id, _, _ in rag.hybrid_search(q["question"], k=K)]
            rows.append({"id": q["id"], "question": q["question"], "relevant": q["relevant"],
                         "retrieved": retrieved, **score(retrieved, q["relevant"])})
        n = len(rows)
        summary.append(summarize(name, rows, f"{bm25_zero}/{n}", (time.time() - t0) / n))
        details[name] = rows
        print(f"done: {name}", file=sys.stderr)
    return summary, details


def to_markdown(title, questions, summary, details):
    lines = [f"## {title} — {len(questions)} คำถาม, top-{K}", "",
             "| ระบบ | Hit@1 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | BM25 หาไม่เจอ | วินาที/คำถาม |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['config']} | {s['Hit@1']:.2f} | {s['Hit@5']:.2f} | {s['Recall@1']:.3f} | "
                     f"{s['Recall@3']:.3f} | {s['Recall@5']:.3f} | {s['MRR@5']:.3f} | {s['BM25=0']} | "
                     f"{s['sec/query']:.1f} |")

    base, final = CONFIGS[0][0], CONFIGS[-1][0]
    improved = sum(1 for a, b in zip(details[base], details[final]) if (b["first_rank"] or 99) < (a["first_rank"] or 99))
    worse = sum(1 for a, b in zip(details[base], details[final]) if (b["first_rank"] or 99) > (a["first_rank"] or 99))
    lines += ["", f"เทียบ (1) กับ (2): จัดอันดับ chunk ที่ถูกต้องได้ **ดีขึ้น {improved} ข้อ**, แย่ลง {worse} ข้อ, "
                  f"เท่าเดิม {len(questions) - improved - worse} ข้อ",
              "", "<details><summary>รายคำถาม: อันดับของ chunk ที่ถูกต้องตัวแรก (— = ไม่อยู่ใน top-5)</summary>", "",
              "| # | คำถาม | chunk ที่ถูกต้อง | " + " | ".join(n.split(" ")[0] for n, _ in CONFIGS) + " |",
              "|---|---|---|" + "---|" * len(CONFIGS)]
    for i, q in enumerate(questions):
        ranks = [details[name][i]["first_rank"] for name, _ in CONFIGS]
        lines.append(f"| {q['id']} | {q['question']} | {', '.join(q['relevant'])} | " +
                     " | ".join(str(r) if r else "—" for r in ranks) + " |")
    lines += ["", "</details>", ""]
    return lines


def rescore_set(old):
    """คิดคะแนนใหม่จากอันดับที่บันทึกไว้ใน eval_results.json (ไม่ค้นหาใหม่ ผลค้นหาเหมือนเดิมทุกตัว)"""
    summary, details = [], {}
    for s in old["summary"]:
        name = s["config"]
        rows = [{**r, **score(r["retrieved"], r["relevant"])} for r in old["details"][name]]
        summary.append(summarize(name, rows, s["BM25=0"], s["sec/query"]))
        details[name] = rows
    return summary, details


def main():
    # --rescore: เพิ่มตัวชี้วัดใหม่จากผลเดิม โดยไม่ต้องรันค้นหาทั้งหมดซ้ำ (ประมาณ 15 นาทีบน CPU)
    rescore = "--rescore" in sys.argv
    if rescore:
        old_results = json.load(open(os.path.join(HERE, "eval_results.json"), encoding="utf-8"))
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            rag = ImprovedSearchRAG()

    lines, all_results = ["# ผลการวัด Retrieval: ระบบเดิม (1) vs ระบบที่ปรับปรุง (2)", ""], {}
    for title, filename in QUESTION_SETS:
        questions = json.load(open(os.path.join(HERE, filename), encoding="utf-8"))
        summary, details = rescore_set(old_results[filename]) if rescore else run_set(rag, questions)
        all_results[filename] = {"summary": summary, "details": details}
        lines += to_markdown(title, questions, summary, details)

    json.dump(all_results, open(os.path.join(HERE, "eval_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    md = "\n".join(lines) + "\n"
    open(os.path.join(HERE, "eval_results.md"), "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
