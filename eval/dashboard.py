"""
สร้าง Dashboard ผลการวัดใน OpenSearch Dashboards (http://localhost:5601)

1) อ่าน eval/eval_results.json แล้วเก็บลง index "rag-eval-results" (1 document = 1 คำถาม x 1 ระบบ)
2) สร้าง index pattern, กราฟ และ dashboard "RAG: ระบบเดิม (1) vs ระบบใหม่ (2)"

รันซ้ำได้ (ทับของเดิม) และไม่แตะ index ของระบบ RAG

วิธีใช้:
    python eval/dashboard.py
"""
import json
import os
import urllib.request

from opensearchpy import OpenSearch, helpers

OPENSEARCH = {"host": "localhost", "port": 9200}
DASHBOARDS_URL = "http://localhost:5601"
INDEX = "rag-eval-results"
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "eval_results.json")

# ชื่อสั้นของแต่ละระบบ ขึ้นต้นด้วยเลขลำดับเพื่อให้กราฟเรียงถูก
SYSTEM_LABELS = {
    "(1)": "1 · ระบบเดิม",
    "(2A)": "2 · +Thai BM25",
    "(2B)": "3 · +Reranker 512",
    "(2)": "4 · ระบบใหม่ (MaxP)",
}
SET_LABELS = {
    "eval_questions.json": "ชุดทั่วไป (24 ข้อ)",
    "eval_questions_hard.json": "ชุดเจาะลึก (20 ข้อ)",
}
NOT_IN_TOP5 = 6  # ใช้แทน "ไม่อยู่ใน 5 อันดับ" ในแผนภาพสี


def system_label(config):
    return SYSTEM_LABELS[config.split()[0]]


def first_rank(retrieved, relevant):
    for rank, doc_id in enumerate(retrieved[:5], start=1):
        if doc_id in relevant:
            return rank
    return NOT_IN_TOP5


def build_docs(results):
    for set_file, data in results.items():
        sec = {s["config"]: s["sec/query"] for s in data["summary"]}
        for config, rows in data["details"].items():
            for row in rows:
                yield {
                    "_index": INDEX,
                    "_id": f"{row['id']}|{config.split()[0]}",
                    "set": SET_LABELS[set_file],
                    "system": system_label(config),
                    "qid": row["id"],
                    "question": row["question"],
                    "relevant": row["relevant"],
                    "retrieved": row["retrieved"],
                    "first_rank": first_rank(row["retrieved"], row["relevant"]),
                    "hit1": row["hit1"],
                    "hit5": row["hit5"],
                    "recall1": row["recall1"],
                    "recall3": row["recall3"],
                    "recall5": row["recall5"],
                    "mrr5": row["mrr5"],
                    "sec_per_query": sec[config],
                }


def load_index():
    client = OpenSearch(hosts=[OPENSEARCH], use_ssl=False, verify_certs=False)
    if client.indices.exists(index=INDEX):
        client.indices.delete(index=INDEX)
    client.indices.create(index=INDEX, body={
        "settings": {"number_of_replicas": 0},
        "mappings": {"properties": {
            "set": {"type": "keyword"},
            "system": {"type": "keyword"},
            "qid": {"type": "keyword"},
            "question": {"type": "text", "analyzer": "thai",
                         "fields": {"keyword": {"type": "keyword"}}},
            "relevant": {"type": "keyword"},
            "retrieved": {"type": "keyword"},
            "first_rank": {"type": "integer"},
            "hit1": {"type": "float"},
            "hit5": {"type": "float"},
            "recall1": {"type": "float"},
            "recall3": {"type": "float"},
            "recall5": {"type": "float"},
            "mrr5": {"type": "float"},
            "sec_per_query": {"type": "float"},
        }},
    })
    with open(RESULTS_FILE, encoding="utf-8") as f:
        ok, _ = helpers.bulk(client, build_docs(json.load(f)), refresh=True)
    print(f"เก็บผลการวัด {ok} รายการลง index {INDEX}")


# ---------- OpenSearch Dashboards saved objects ----------

def save_object(obj_type, obj_id, attributes, references=()):
    body = json.dumps({"attributes": attributes, "references": list(references)}).encode("utf-8")
    req = urllib.request.Request(
        f"{DASHBOARDS_URL}/api/saved_objects/{obj_type}/{obj_id}?overwrite=true",
        data=body, method="POST",
        headers={"Content-Type": "application/json", "osd-xsrf": "true"})
    urllib.request.urlopen(req).read()


def search_source(with_index=True):
    src = {"query": {"query": "", "language": "kuery"}, "filter": []}
    if with_index:
        src["indexRefName"] = "kibanaSavedObjectMeta.searchSourceJSON.index"
    return {"searchSourceJSON": json.dumps(src)}


INDEX_REF = [{"name": "kibanaSavedObjectMeta.searchSourceJSON.index",
              "type": "index-pattern", "id": INDEX}]


def bar_chart(vis_id, title, field, label, y_max=None):
    """กราฟแท่ง: ค่าเฉลี่ยของ field แยกตามระบบ (แกน x) และชุดคำถาม (สี)"""
    scale = {"type": "linear", "mode": "normal", "defaultYExtents": False}
    if y_max is not None:
        scale.update({"setYExtents": True, "min": 0, "max": y_max})
    vis_state = {
        "title": title, "type": "histogram",
        "params": {
            "type": "histogram", "grid": {"categoryLines": False},
            "categoryAxes": [{"id": "CategoryAxis-1", "type": "category", "position": "bottom",
                              "show": True, "style": {}, "scale": {"type": "linear"},
                              "labels": {"show": True, "filter": False, "truncate": 100, "rotate": 0},
                              "title": {}}],
            "valueAxes": [{"id": "ValueAxis-1", "name": "LeftAxis-1", "type": "value",
                           "position": "left", "show": True, "style": {}, "scale": scale,
                           "labels": {"show": True, "rotate": 0, "filter": False, "truncate": 100},
                           "title": {"text": label}}],
            "seriesParams": [{"show": True, "type": "histogram", "mode": "normal",
                              "data": {"label": label, "id": "1"}, "valueAxis": "ValueAxis-1",
                              "drawLinesBetweenPoints": True, "lineWidth": 2, "showCircles": True}],
            "addTooltip": True, "addLegend": True, "legendPosition": "top",
            "times": [], "addTimeMarker": False, "labels": {"show": True},
            "thresholdLine": {"show": False, "value": 1, "width": 1, "style": "full", "color": "#E7664C"},
        },
        "aggs": [
            {"id": "1", "enabled": True, "type": "avg", "schema": "metric",
             "params": {"field": field, "customLabel": label}},
            {"id": "2", "enabled": True, "type": "terms", "schema": "segment",
             "params": {"field": "system", "orderBy": "_key", "order": "asc", "size": 10,
                        "otherBucket": False, "missingBucket": False, "customLabel": "ระบบ"}},
            {"id": "3", "enabled": True, "type": "terms", "schema": "group",
             "params": {"field": "set", "orderBy": "_key", "order": "asc", "size": 5,
                        "otherBucket": False, "missingBucket": False, "customLabel": "ชุดคำถาม"}},
        ],
    }
    save_object("visualization", vis_id, {
        "title": title, "visState": json.dumps(vis_state), "uiStateJSON": "{}",
        "description": "", "version": 1, "kibanaSavedObjectMeta": search_source(),
    }, INDEX_REF)


def recall_chart(vis_id, title, query):
    """กราฟแท่ง: Recall@1 / @3 / @5 ของแต่ละระบบ ในชุดคำถามเดียว"""
    metrics = [("1", "recall1", "Recall@1"), ("2", "recall3", "Recall@3"), ("3", "recall5", "Recall@5")]
    vis_state = {
        "title": title, "type": "histogram",
        "params": {
            "type": "histogram", "grid": {"categoryLines": False},
            "categoryAxes": [{"id": "CategoryAxis-1", "type": "category", "position": "bottom",
                              "show": True, "style": {}, "scale": {"type": "linear"},
                              "labels": {"show": True, "filter": False, "truncate": 100, "rotate": 0},
                              "title": {}}],
            "valueAxes": [{"id": "ValueAxis-1", "name": "LeftAxis-1", "type": "value",
                           "position": "left", "show": True, "style": {},
                           "scale": {"type": "linear", "mode": "normal", "defaultYExtents": False,
                                     "setYExtents": True, "min": 0, "max": 1},
                           "labels": {"show": True, "rotate": 0, "filter": False, "truncate": 100},
                           "title": {"text": "Recall"}}],
            "seriesParams": [{"show": True, "type": "histogram", "mode": "normal",
                              "data": {"label": label, "id": mid}, "valueAxis": "ValueAxis-1",
                              "drawLinesBetweenPoints": True, "lineWidth": 2, "showCircles": True}
                             for mid, _, label in metrics],
            "addTooltip": True, "addLegend": True, "legendPosition": "top",
            "times": [], "addTimeMarker": False, "labels": {"show": True},
            "thresholdLine": {"show": False, "value": 1, "width": 1, "style": "full", "color": "#E7664C"},
        },
        "aggs": [{"id": mid, "enabled": True, "type": "avg", "schema": "metric",
                  "params": {"field": field, "customLabel": label}} for mid, field, label in metrics] + [
            {"id": "4", "enabled": True, "type": "terms", "schema": "segment",
             "params": {"field": "system", "orderBy": "_key", "order": "asc", "size": 10,
                        "otherBucket": False, "missingBucket": False, "customLabel": "ระบบ"}},
        ],
    }
    src = {"query": {"query": query, "language": "kuery"}, "filter": [],
           "indexRefName": "kibanaSavedObjectMeta.searchSourceJSON.index"}
    save_object("visualization", vis_id, {
        "title": title, "visState": json.dumps(vis_state), "uiStateJSON": "{}",
        "description": "", "version": 1,
        "kibanaSavedObjectMeta": {"searchSourceJSON": json.dumps(src)},
    }, INDEX_REF)


def rank_heatmap(vis_id, title, query):
    """แผนภาพสี: อันดับของ chunk ที่ถูกต้อง ทุกคำถาม x ทุกระบบ (6 = ไม่อยู่ใน 5 อันดับ)"""
    vis_state = {
        "title": title, "type": "heatmap",
        "params": {
            "type": "heatmap", "addTooltip": True, "addLegend": True, "enableHover": False,
            "legendPosition": "right", "times": [], "colorsNumber": 4, "colorSchema": "Reds",
            "setColorRange": True,
            "colorsRange": [{"from": 1, "to": 1.5}, {"from": 1.5, "to": 3.5},
                            {"from": 3.5, "to": 5.5}, {"from": 5.5, "to": 6.5}],
            "invertColors": False, "percentageMode": False,
            "valueAxes": [{"show": False, "id": "ValueAxis-1", "type": "value",
                           "scale": {"type": "linear", "defaultYExtents": False},
                           "labels": {"show": True, "rotate": 0, "overwriteColor": False,
                                      "color": "black"}}],
        },
        "aggs": [
            {"id": "1", "enabled": True, "type": "max", "schema": "metric",
             "params": {"field": "first_rank", "customLabel": "อันดับของ chunk ที่ถูก"}},
            {"id": "2", "enabled": True, "type": "terms", "schema": "segment",
             "params": {"field": "system", "orderBy": "_key", "order": "asc", "size": 10,
                        "otherBucket": False, "missingBucket": False, "customLabel": "ระบบ"}},
            {"id": "3", "enabled": True, "type": "terms", "schema": "group",
             "params": {"field": "qid", "orderBy": "_key", "order": "asc", "size": 50,
                        "otherBucket": False, "missingBucket": False, "customLabel": "คำถาม"}},
        ],
    }
    src = {"query": {"query": query, "language": "kuery"}, "filter": [],
           "indexRefName": "kibanaSavedObjectMeta.searchSourceJSON.index"}
    save_object("visualization", vis_id, {
        "title": title, "visState": json.dumps(vis_state), "uiStateJSON": "{}",
        "description": "", "version": 1,
        "kibanaSavedObjectMeta": {"searchSourceJSON": json.dumps(src)},
    }, INDEX_REF)


EXPLAIN_MD = """### อ่านกราฟยังไง

ทุกตัวเลขมาจาก `eval/eval_results.json` (ชุดคำถาม 44 ข้อ กำหนดเฉลยไว้ก่อนรัน) วัดเฉพาะการค้นหา: 5 chunks ที่ส่งให้ LLM มี chunk ที่ถูกอยู่อันดับไหน

* **Hit@1** อันดับ 1 ถูกไหม (1.0 = ถูกทุกข้อ)
* **MRR@5** เฉลี่ยของ 1 ÷ อันดับของตัวที่ถูก (อันดับ 2 ได้ 0.5)
* **Recall@k** ใน k อันดับแรก เจอ chunk ที่ถูกครบกี่ส่วน (บางข้อมีคำตอบหลาย chunk Recall@1 จึงเต็ม 1.0 ไม่ได้ทุกข้อ)
* **แผนภาพสี** เลขในช่อง = อันดับของ chunk ที่ถูก ยิ่งแดงยิ่งแย่ (6 = ไม่อยู่ใน 5 อันดับ)

**ระบบ:** 1 = ของอาจารย์ (standard BM25 + vector + RRF) · 2 = +ตัดคำไทย · 3 = +reranker อ่าน 512 tokens แรก · 4 = +reranker แบบ MaxP (ระบบใหม่)
"""


def build_dashboard():
    save_object("index-pattern", INDEX, {"title": INDEX})

    bar_chart("rag-eval-hit1", "Hit@1: อันดับ 1 ถูกไหม", "hit1", "Hit@1", y_max=1)
    bar_chart("rag-eval-mrr", "MRR@5: ตัวที่ถูกอยู่อันดับไหน", "mrr5", "MRR@5", y_max=1)
    bar_chart("rag-eval-sec", "เวลาที่ใช้ต่อคำถาม (วินาที)", "sec_per_query", "วินาที")
    recall_chart("rag-eval-recall-hard", "Recall@k: ชุดเจาะลึก", 'set:"ชุดเจาะลึก (20 ข้อ)"')
    recall_chart("rag-eval-recall-general", "Recall@k: ชุดทั่วไป", 'set:"ชุดทั่วไป (24 ข้อ)"')
    rank_heatmap("rag-eval-rank-hard", "อันดับรายข้อ: ชุดเจาะลึก", 'set:"ชุดเจาะลึก (20 ข้อ)"')
    rank_heatmap("rag-eval-rank-general", "อันดับรายข้อ: ชุดทั่วไป", 'set:"ชุดทั่วไป (24 ข้อ)"')
    save_object("visualization", "rag-eval-explain", {
        "title": "อ่านกราฟยังไง",
        "visState": json.dumps({"title": "อ่านกราฟยังไง", "type": "markdown", "aggs": [],
                                "params": {"markdown": EXPLAIN_MD, "fontSize": 12,
                                           "openLinksInNewTab": False}}),
        "uiStateJSON": "{}", "description": "", "version": 1,
        "kibanaSavedObjectMeta": search_source(with_index=False),
    })

    # ตำแหน่งบนจอ (กว้างเต็ม = 48 ช่อง)
    layout = [
        ("rag-eval-explain", 0, 0, 48, 9),
        ("rag-eval-hit1", 0, 9, 24, 14),
        ("rag-eval-mrr", 24, 9, 24, 14),
        ("rag-eval-recall-hard", 0, 23, 24, 14),
        ("rag-eval-recall-general", 24, 23, 24, 14),
        ("rag-eval-rank-hard", 0, 37, 24, 26),
        ("rag-eval-rank-general", 24, 37, 24, 26),
        ("rag-eval-sec", 0, 63, 48, 12),
    ]
    panels, refs = [], []
    for i, (vis_id, x, y, w, h) in enumerate(layout):
        panels.append({"version": "2.19.1", "panelIndex": str(i + 1), "embeddableConfig": {},
                       "gridData": {"x": x, "y": y, "w": w, "h": h, "i": str(i + 1)},
                       "panelRefName": f"panel_{i}"})
        refs.append({"name": f"panel_{i}", "type": "visualization", "id": vis_id})
    save_object("dashboard", "rag-eval-dashboard", {
        "title": "RAG: ระบบเดิม (1) vs ระบบใหม่ (2)",
        "description": "ผลการวัด retrieval จาก eval/eval_results.json",
        "panelsJSON": json.dumps(panels), "optionsJSON": json.dumps({"useMargins": True, "hidePanelTitles": False}),
        "version": 1, "timeRestore": False, "kibanaSavedObjectMeta": search_source(with_index=False),
    }, refs)
    print(f"สร้าง dashboard แล้ว: {DASHBOARDS_URL}/app/dashboards#/view/rag-eval-dashboard")


if __name__ == "__main__":
    load_index()
    build_dashboard()
