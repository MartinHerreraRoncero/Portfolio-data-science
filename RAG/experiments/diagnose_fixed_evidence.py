"""Control manual posterior: aislar redactor y verificador del selector de pasajes."""

import hashlib
import json
from pathlib import Path
import time

from experiments.diagnose_total_component import _hash, validate_case
from rag_pensiones.answer import parse_chat_response, validate_chat_stats
from rag_pensiones.grounding import (GENERATION_JSON_SCHEMA, OllamaGrounder, build_generation_messages,
                                     format_to_public_claims, validate_generation_payload)
from rag_pensiones.ingest import PROJECT_ROOT
from rag_pensiones.search import load_index, validate_index


def main():
    path = PROJECT_ROOT / "evaluation/total_component_diagnostic_case.json"
    case = json.loads(path.read_text(encoding="utf-8"))
    index = load_index(PROJECT_ROOT / "data/processed/airef_2025_embeddings.json")
    chunks = {c.chunk_id: c for c, _ in validate_index(index)}
    validate_case(case, chunks)
    # Referencia manual fijada en el fixture; no transmitir su etiqueta ni su afirmación al modelo.
    reference = next(c for c in case["verifier_controls"] if c["id"] == "correct_total")
    if reference["chunk_id"] != case["oracle_chunk_id"]:
        raise ValueError("Manual evidence must belong to the oracle source")
    chunk = chunks[reference["chunk_id"]]
    evidence = [{"evidence_id": "E1", "source_id": "S1", "doc_id": chunk.doc_id,
                 "source_file": chunk.source_file, "page_number": chunk.page_number, "quote": reference["quote"]}]
    messages = build_generation_messages(case["query"], evidence)
    grounder = OllamaGrounder()
    digest = grounder.generator.get_digest()
    started = time.perf_counter()
    raw = grounder.generator.chat(messages, GENERATION_JSON_SCHEMA)
    if grounder.generator.get_digest() != digest:
        raise ValueError("Generator digest changed during manual control")
    parsed = parse_chat_response(raw, grounder.model)
    stats = validate_chat_stats(parsed["stats"])
    generated = validate_generation_payload(parsed["payload"], evidence)
    verified = (grounder.verify(generated["claims"], evidence, expected_digest=digest)
                if generated["status"] == "answered" else None)
    report = {"schema_version": "fixed-evidence-diagnostic-v1", "query": case["query"],
              "mode": "manual_quote_without_retrieval_or_selection", "model": grounder.model,
              "model_digest": digest, "case_sha256": _hash(case), "index_sha256": _hash(index),
              "evidence": evidence, "generation": generated, "verification": verified,
              "claims_with_citations": format_to_public_claims(generated["claims"], evidence),
              "generation_prompt_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest(),
              "generation_stats": stats, "chats": 2 if verified else 1,
              "duration_seconds": time.perf_counter() - started}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
