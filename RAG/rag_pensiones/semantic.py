"""Flujo RAG semántico local con planificación estructurada y recuperación por necesidad."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

from rag_pensiones.answer import (
    ALLOWED_MODELS,
    DEFAULT_GENERATION_MODEL,
    OllamaGenerator,
    _generate_and_validate,
    validate_chat_stats,
)
from rag_pensiones.embeddings import OllamaEmbedder
from rag_pensiones.qwen_embeddings import QwenEmbedder, MODEL as QWEN_EMBEDDING_MODEL
from rag_pensiones.grounding import (
    SCHEMA_VERSION as GROUNDING_SCHEMA_VERSION,
    OllamaGrounder,
    validate_grounded_result,
)
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.planning import (
    DEFAULT_SCOPE_YEARS,
    OllamaPlanner,
    extract_years,
    validate_plan,
    validate_resolved_question,
)
from rag_pensiones.search import (
    DEFAULT_INDEX_PATH,
    SearchHit,
    load_index,
    search,
    validate_index,
)

SCHEMA_VERSION = "rag-semantic-v1"
GROUNDED_SCHEMA_VERSION = "rag-grounded-v1"
SHA256_HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_sha256_hex(hash_val: str, name: str = "hash") -> str:
    """Valida estrictamente que el valor sea un hash SHA-256 hexadecimal de 64 caracteres no vacío."""
    if isinstance(hash_val, bool) or not isinstance(hash_val, str):
        raise ValueError(f"{name} must be a string, got {type(hash_val).__name__}")
    if not hash_val.strip():
        raise ValueError(f"{name} cannot be empty or contain only whitespace")
    if len(hash_val) != 64 or not SHA256_HEX_PATTERN.fullmatch(hash_val):
        raise ValueError(
            f"{name} must be a valid 64-character hexadecimal SHA-256 string, got {hash_val!r}"
        )
    return hash_val


def _validate_parameters(params: dict, name: str = "parameters") -> dict:
    """Valida estrictamente que los parámetros sean un dict serializable en JSON sin NaN ni Inf."""
    if isinstance(params, bool) or not isinstance(params, dict):
        raise ValueError(f"{name} must be a dict, got {type(params).__name__}")
    try:
        json.dumps(params, ensure_ascii=False, allow_nan=False)
    except Exception as exc:
        raise ValueError(f"{name} must be strictly JSON-serializable without NaN/Inf: {exc}") from exc
    return copy.deepcopy(params)


def answer_semantic(
    index: dict,
    query: str,
    top_k: int = 4,
    model: str = DEFAULT_GENERATION_MODEL,
    embedder: OllamaEmbedder | None = None,
    generator: OllamaGenerator | None = None,
    planner: OllamaPlanner | None = None,
    grounder: OllamaGrounder | None = None,
    query_builder=None,
) -> dict:
    """Ejecuta el flujo RAG semántico local completo: planifica, resuelve dependencias, busca por necesidad y responde."""
    # 1. Validaciones previas exhaustivas antes de contactar con la red
    validate_index(index)

    if isinstance(query, bool) or not isinstance(query, str):
        raise ValueError(f"query must be a string, got {type(query).__name__}")
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query cannot be empty or contain only whitespace")
    if len(cleaned_query) > 1000:
        raise ValueError(f"query length exceeds 1000 characters ({len(cleaned_query)})")
    if query_builder is not None and not callable(query_builder):
        raise ValueError("query_builder must be callable or None")

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer (not bool), got {top_k!r}")

    if model not in ALLOWED_MODELS:
        raise ValueError(f"model {model!r} is not allowed. Choose from {sorted(ALLOWED_MODELS)}")

    # Comprobación de consistencia de generador, planificador y grounder inyectados
    if generator is not None:
        gen_model = getattr(generator, "model", None)
        if gen_model != model:
            raise ValueError(
                f"Injected generator model {gen_model!r} does not match requested model {model!r}"
            )

    if planner is not None:
        p_model = getattr(planner, "model", None)
        if p_model is not None and p_model != model:
            raise ValueError(
                f"Injected planner model {p_model!r} does not match requested model {model!r}"
            )
        p_gen = getattr(planner, "generator", None)
        if p_gen is not None:
            p_gen_model = getattr(p_gen, "model", None)
            if p_gen_model is not None and p_gen_model != model:
                raise ValueError(
                    f"Injected planner generator model {p_gen_model!r} does not match requested model {model!r}"
                )
            if generator is not None and p_gen is not generator:
                raise ValueError(
                    "Injected generator and injected planner's generator must be the same instance"
                )

    if grounder is not None:
        g_model = getattr(grounder, "model", None)
        if g_model is not None and g_model != model:
            raise ValueError(
                f"Injected grounder model {g_model!r} does not match requested model {model!r}"
            )
        g_gen = getattr(grounder, "generator", None)
        if g_gen is not None:
            g_gen_model = getattr(g_gen, "model", None)
            if g_gen_model is not None and g_gen_model != model:
                raise ValueError(
                    f"Injected grounder generator model {g_gen_model!r} does not match requested model {model!r}"
                )
            if generator is not None and g_gen is not generator:
                raise ValueError(
                    "Injected generator and injected grounder's generator must be the same instance"
                )
            if planner is not None:
                p_gen = getattr(planner, "generator", None)
                if p_gen is not None and g_gen is not p_gen:
                    raise ValueError(
                        "Injected planner's generator and injected grounder's generator must be the same instance"
                    )

    # 2. Inicialización compartida del generador, planificador y grounder (mismo cliente y configuración)
    if generator is None:
        if grounder is not None and getattr(grounder, "generator", None) is not None:
            generator = grounder.generator
        elif planner is not None and getattr(planner, "generator", None) is not None:
            generator = planner.generator
        else:
            generator = OllamaGenerator(model=model)

    if planner is None:
        planner = OllamaPlanner(generator=generator, model=model)
    elif getattr(planner, "generator", None) is not None and planner.generator is not generator:
        raise ValueError("Injected planner generator does not match resolved generator")

    if grounder is not None and getattr(grounder, "generator", None) is not None and grounder.generator is not generator:
        raise ValueError("Injected grounder generator does not match resolved generator")

    # 3. Ámbito temporal (scope_years) extraído de las fechas de publicación del índice
    index_scope_years: set[str] = set()
    for c in index.get("chunks", []):
        if isinstance(c, dict):
            pub = c.get("published_at", "")
            if isinstance(pub, str):
                index_scope_years.update(extract_years(pub))
    if not index_scope_years:
        index_scope_years = set(DEFAULT_SCOPE_YEARS)

    # 4. Generación del plan semántico estructurado
    plan_report = planner.plan(cleaned_query, scope_years=index_scope_years)
    if not isinstance(plan_report, dict) or isinstance(plan_report, bool):
        raise ValueError(f"Planner output must be a dict, got {type(plan_report).__name__}")
    for req_key in ("plan", "model", "model_digest", "prompt_sha256", "stats", "parameters"):
        if req_key not in plan_report:
            raise ValueError(f"Planner output missing required key: {req_key!r}")

    if plan_report["model"] != model:
        raise ValueError(
            f"Plan model {plan_report['model']!r} does not match requested model {model!r}"
        )

    pin_digest = plan_report["model_digest"]
    if not isinstance(pin_digest, str) or not pin_digest.strip():
        raise ValueError("Plan model_digest must be a non-empty string")

    # Revalidar hash, estadísticas y parámetros del plan ANTES de inicializar o usar el embedder
    plan_prompt_sha256 = _validate_sha256_hex(plan_report["prompt_sha256"], "Plan prompt_sha256")
    validated_plan_stats = validate_chat_stats(plan_report["stats"])
    validated_plan_params = _validate_parameters(plan_report.get("parameters"), "Plan parameters")

    # Validación exhaustiva del plan generado contra la consulta y el scope_years
    validated_plan = validate_plan(plan_report["plan"], query=cleaned_query, scope_years=index_scope_years)

    # 5. Inicializar cliente de embedding si no fue inyectado
    if embedder is None:
        embedder = QwenEmbedder() if index["model"] == QWEN_EMBEDDING_MODEL else OllamaEmbedder()

    # 6. Ejecución topológica secuencial de cada nodo
    subquestions = validated_plan["subquestions"]
    parts: list[dict] = []
    parts_by_id: dict[str, dict] = {}
    discovered_chunks: dict[str, dict] = {}
    discovered_chunk_records: dict = {}
    chunk_id_to_source_id: dict[str, str] = {}
    next_source_idx = 1

    num_searches = 0
    num_resolutions = 0
    num_answers = 0
    executed_resolution_stats: list[dict] = []
    executed_answer_stats: list[dict] = []

    for node in subquestions:
        node_id = node["id"]
        depends_on = node["depends_on"]

        # Comprobar si algún nodo padre no fue respondido (status != "answered")
        blocking_parents = [pid for pid in depends_on if parts_by_id[pid]["status"] != "answered"]
        if blocking_parents:
            execution_status = "blocked_dependency"
            status = "insufficient_evidence"
            reasons_list = [
                f"{pid} ({parts_by_id[pid]['status']}: {parts_by_id[pid]['reason']})"
                for pid in blocking_parents
            ]
            reason = f"Dependencia bloqueada por nodo(s) no respondido(s): {', '.join(reasons_list)}"
            part_dict = {
                "id": node_id,
                "question": node["question"],
                "required_evidence": node["required_evidence"],
                "depends_on": node["depends_on"],
                "resolved_question": None,
                "execution_status": execution_status,
                "status": status,
                "claims": [],
                "reason": reason,
                "sources_ids": [],
                "retrievals": [],
                "answer_prompt_sha256": None,
                "stats": None,
            }
            if grounder is not None:
                part_dict["answer_prompt_sha256_kind"] = "aggregate"
                part_dict["grounding"] = None
            parts.append(part_dict)
            parts_by_id[node_id] = part_dict
            continue

        # Nodo ejecutable: resolver dependencias si depends_on no está vacío
        resolution_metadata = None
        if depends_on:
            deps_for_resolve = [
                {
                    "id": pid,
                    "question": parts_by_id[pid]["question"],
                    "status": parts_by_id[pid]["status"],
                    "claims": parts_by_id[pid]["claims"],
                }
                for pid in depends_on
            ]
            resolve_res = planner.resolve(
                question=node["question"],
                original_query=cleaned_query,
                dependencies=deps_for_resolve,
                expected_digest=pin_digest,
                scope_years=index_scope_years,
            )

            if not isinstance(resolve_res, dict) or isinstance(resolve_res, bool):
                raise ValueError(f"Resolve output must be a dict, got {type(resolve_res).__name__}")
            for k in ("question", "model", "model_digest", "prompt_sha256", "stats", "parameters"):
                if k not in resolve_res:
                    raise ValueError(f"Resolve output missing required key: {k!r}")
            if resolve_res["model"] != model:
                raise ValueError(
                    f"Resolve model {resolve_res['model']!r} does not match requested model {model!r}"
                )
            if resolve_res["model_digest"] != pin_digest:
                raise RuntimeError(
                    f"Resolve model_digest mismatch: expected {pin_digest!r}, got {resolve_res['model_digest']!r}"
                )

            # Revalidar hash, stats, parameters y question resuelta ANTES de buscar o generar para el nodo hijo
            resolve_prompt_sha256 = _validate_sha256_hex(
                resolve_res["prompt_sha256"], "Resolve prompt_sha256"
            )
            validated_resolve_stats = validate_chat_stats(resolve_res["stats"])
            validated_resolve_params = _validate_parameters(
                resolve_res.get("parameters"), "Resolve parameters"
            )
            resolved_question = validate_resolved_question(
                resolved_question=resolve_res["question"],
                question=node["question"],
                original_query=cleaned_query,
                dependencies=deps_for_resolve,
                scope_years=index_scope_years,
            )

            num_resolutions += 1
            resolution_metadata = {
                "model": resolve_res["model"],
                "model_digest": resolve_res["model_digest"],
                "prompt_sha256": resolve_prompt_sha256,
                "stats": validated_resolve_stats,
                "parameters": validated_resolve_params,
            }
            executed_resolution_stats.append(validated_resolve_stats)
            search_query = resolved_question
        else:
            resolved_question = None
            search_query = node["question"]

        if query_builder is not None:
            search_query = query_builder(search_query, node["required_evidence"])
            if not isinstance(search_query, str) or not search_query.strip() or len(search_query) > 1000:
                raise ValueError("Execution query must be a nonempty string of at most 1000 characters")

        # Búsqueda semántica UNA VEZ para este nodo con su pregunta autónoma/resuelta
        hits = search(
            index=index,
            query=search_query,
            top_k=top_k,
            client=embedder,
        )
        num_searches += 1

        # Deduplicación global por chunk_id asignando S1..SN estables en orden de descubrimiento
        node_search_hits: list[SearchHit] = []
        node_sources_ids: list[str] = []
        node_retrievals: list[dict] = []

        for hit in hits:
            cid = hit.chunk.chunk_id
            if cid not in discovered_chunks:
                src_id = f"S{next_source_idx}"
                next_source_idx += 1
                chunk_id_to_source_id[cid] = src_id
                discovered_chunks[cid] = {
                    "source_id": src_id,
                    "doc_id": hit.chunk.doc_id,
                    "chunk_id": cid,
                    "source_file": hit.chunk.source_file,
                    "published_at": hit.chunk.published_at,
                    "page_number": hit.chunk.page_number,
                    "text": hit.chunk.text,
                    "retrievals": [],
                }
                discovered_chunk_records[cid] = hit.chunk

            src_id = chunk_id_to_source_id[cid]
            retrieval_entry = {
                "subquestion_id": node_id,
                "rank": hit.rank,
                "score": hit.score,
            }
            discovered_chunks[cid]["retrievals"].append(retrieval_entry)

            # Construir SearchHit con rank=IDnumglobal sin mutar hits originales
            global_rank = int(src_id[1:])
            node_search_hits.append(SearchHit(rank=global_rank, score=hit.score, chunk=hit.chunk))
            node_sources_ids.append(src_id)
            node_retrievals.append({
                "source_id": src_id,
                "rank": hit.rank,
                "score": hit.score,
            })

        # Optional, source-validated expansion before grounding. Raw retrievals
        # remain separate; derived passages receive globally stable source IDs.
        prepare = getattr(grounder, "prepare_evidence", None) if grounder is not None else None
        if callable(prepare):
            expanded = prepare(search_query, hits)
            if not isinstance(expanded, list) or not expanded:
                raise ValueError("Evidence preparation must return nonempty SearchHit list")
            prepared_hits, prepared_ids = [], []
            seed_ids = list(node_sources_ids)
            for hit in expanded:
                if not isinstance(hit, SearchHit):
                    raise ValueError("Invalid prepared evidence hit")
                cid = hit.chunk.chunk_id
                if cid in discovered_chunk_records and discovered_chunk_records[cid] != hit.chunk:
                    raise ValueError("Prepared chunk ID conflicts with an existing source")
                if cid not in discovered_chunks:
                    src_id = f"S{next_source_idx}"
                    next_source_idx += 1
                    chunk_id_to_source_id[cid] = src_id
                    discovered_chunk_records[cid] = hit.chunk
                    discovered_chunks[cid] = {
                        "source_id": src_id, "doc_id": hit.chunk.doc_id, "chunk_id": cid,
                        "source_file": hit.chunk.source_file, "published_at": hit.chunk.published_at,
                        "page_number": hit.chunk.page_number, "text": hit.chunk.text, "retrievals": [],
                    }
                src_id = chunk_id_to_source_id[cid]
                if src_id in prepared_ids:
                    raise ValueError("Duplicate prepared evidence source")
                discovered_chunks[cid].setdefault("expansions", []).append({
                    "subquestion_id": node_id, "seed_source_ids": seed_ids,
                })
                prepared_ids.append(src_id)
                prepared_hits.append(SearchHit(rank=int(src_id[1:]), score=hit.score, chunk=hit.chunk))
            node_search_hits, node_sources_ids = prepared_hits, prepared_ids

        # Generación estructurada validada con citas literales SOLO contra fuentes de este nodo
        if grounder is None:
            ans_res = _generate_and_validate(
                query=search_query,
                hits=node_search_hits,
                model=model,
                generator=generator,
                original_query=cleaned_query,
                expected_digest=pin_digest,
            )
            num_answers += 1
            executed_answer_stats.append(ans_res["stats"])

            part_dict = {
                "id": node_id,
                "question": node["question"],
                "required_evidence": node["required_evidence"],
                "depends_on": node["depends_on"],
                "resolved_question": resolved_question,
                "execution_status": "executed",
                "status": ans_res["status"],
                "claims": ans_res["claims"],
                "reason": ans_res["reason"],
                "sources_ids": node_sources_ids,
                "retrievals": node_retrievals,
                "answer_prompt_sha256": ans_res["prompt_sha256"],
                "stats": ans_res["stats"],
            }
        else:
            grounded_res = grounder.answer(
                query=search_query,
                hits=node_search_hits,
                original_query=cleaned_query,
                expected_digest=pin_digest,
            )
            validated_grounded = validate_grounded_result(
                grounded_res, node_search_hits, pin_digest, expected_model=model
            )
            num_answers += 1
            executed_answer_stats.append(validated_grounded["stats"])

            part_dict = {
                "id": node_id,
                "question": node["question"],
                "required_evidence": node["required_evidence"],
                "depends_on": node["depends_on"],
                "resolved_question": resolved_question,
                "execution_status": "executed",
                "status": validated_grounded["status"],
                "claims": validated_grounded["claims"],
                "reason": validated_grounded["reason"],
                "sources_ids": node_sources_ids,
                "retrievals": node_retrievals,
                "answer_prompt_sha256": validated_grounded["prompt_sha256"],
                "answer_prompt_sha256_kind": "aggregate",
                "stats": validated_grounded["stats"],
                "grounding": validated_grounded["grounding"],
            }

        if query_builder is not None:
            part_dict["execution_query"] = search_query
        if resolution_metadata is not None:
            part_dict["resolution"] = resolution_metadata

        parts.append(part_dict)
        parts_by_id[node_id] = part_dict

    # 7. Determinación del estado global y consolidación de afirmaciones
    all_answered = all(p["status"] == "answered" for p in parts)
    if all_answered:
        global_status = "answered"
        global_claims: list[dict] = []
        for p in parts:
            global_claims.extend(p["claims"])
        global_reason = ""
    else:
        global_status = "insufficient_evidence"
        global_claims = []
        unanswered_parts = [p for p in parts if p["status"] != "answered"]
        details = "; ".join(f"{p['id']}: {p['reason']}" for p in unanswered_parts)
        global_reason = (
            f"Evidencia insuficiente o dependencia bloqueada en {len(unanswered_parts)} de {len(parts)} partes. "
            f"Detalle: {details}"
        )
        if len(global_reason) > 1200:
            global_reason = global_reason[:1197] + "..."

    # 8. Estadísticas separadas por fase (planning / resolutions / answers / total)
    planning_stats = copy.deepcopy(validated_plan_stats)

    resolution_stats = {
        "prompt_eval_count": sum(s["prompt_eval_count"] for s in executed_resolution_stats),
        "eval_count": sum(s["eval_count"] for s in executed_resolution_stats),
        "eval_duration": sum(s["eval_duration"] for s in executed_resolution_stats),
        "total_duration": sum(s["total_duration"] for s in executed_resolution_stats),
    }
    if any("load_duration" in s for s in executed_resolution_stats):
        resolution_stats["load_duration"] = sum(
            s.get("load_duration", 0) for s in executed_resolution_stats
        )
    if any("prompt_eval_duration" in s for s in executed_resolution_stats):
        resolution_stats["prompt_eval_duration"] = sum(
            s.get("prompt_eval_duration", 0) for s in executed_resolution_stats
        )

    answers_stats = {
        "prompt_eval_count": sum(s["prompt_eval_count"] for s in executed_answer_stats),
        "eval_count": sum(s["eval_count"] for s in executed_answer_stats),
        "eval_duration": sum(s["eval_duration"] for s in executed_answer_stats),
        "total_duration": sum(s["total_duration"] for s in executed_answer_stats),
    }
    if any("load_duration" in s for s in executed_answer_stats):
        answers_stats["load_duration"] = sum(
            s.get("load_duration", 0) for s in executed_answer_stats
        )
    if any("prompt_eval_duration" in s for s in executed_answer_stats):
        answers_stats["prompt_eval_duration"] = sum(
            s.get("prompt_eval_duration", 0) for s in executed_answer_stats
        )

    total_stats = {
        "prompt_eval_count": (
            planning_stats["prompt_eval_count"]
            + resolution_stats["prompt_eval_count"]
            + answers_stats["prompt_eval_count"]
        ),
        "eval_count": (
            planning_stats["eval_count"]
            + resolution_stats["eval_count"]
            + answers_stats["eval_count"]
        ),
        "eval_duration": (
            planning_stats["eval_duration"]
            + resolution_stats["eval_duration"]
            + answers_stats["eval_duration"]
        ),
        "total_duration": (
            planning_stats["total_duration"]
            + resolution_stats["total_duration"]
            + answers_stats["total_duration"]
        ),
    }
    if any("load_duration" in s for s in (planning_stats, resolution_stats, answers_stats)):
        total_stats["load_duration"] = (
            planning_stats.get("load_duration", 0)
            + resolution_stats.get("load_duration", 0)
            + answers_stats.get("load_duration", 0)
        )
    if any("prompt_eval_duration" in s for s in (planning_stats, resolution_stats, answers_stats)):
        total_stats["prompt_eval_duration"] = (
            planning_stats.get("prompt_eval_duration", 0)
            + resolution_stats.get("prompt_eval_duration", 0)
            + answers_stats.get("prompt_eval_duration", 0)
        )

    if grounder is None:
        api_calls = {
            "plan": 1,
            "resolutions": num_resolutions,
            "answers": num_answers,
            "searches": num_searches,
            "total": 1 + num_resolutions + num_answers + num_searches,
        }
        report_schema_version = SCHEMA_VERSION
        report_stats = {
            "planning": planning_stats,
            "resolutions": resolution_stats,
            "answers": answers_stats,
            "total": total_stats,
            "api_calls": api_calls,
        }
        validation_info = {
            "citation_quotes_checked": True,
            "semantic_faithfulness_evaluated": False,
            "plan_semantics_verified": False,
        }
    else:
        report_schema_version = GROUNDED_SCHEMA_VERSION
        grounded_counts = {
            "selection": sum(
                1
                for p in parts
                if p.get("grounding")
                for t in p["grounding"].get("trace", [])
                if t.get("stage") == "selection"
            ),
            "generation": sum(
                1
                for p in parts
                if p.get("grounding")
                for t in p["grounding"].get("trace", [])
                if t.get("stage") == "generation"
            ),
            "verification": sum(
                1
                for p in parts
                if p.get("grounding")
                for t in p["grounding"].get("trace", [])
                if t.get("stage") == "verification"
            ),
            "repair_selection": sum(
                1
                for p in parts
                if p.get("grounding")
                for t in p["grounding"].get("trace", [])
                if t.get("stage") == "repair_selection"
            ),
            "repair_generation": sum(
                1
                for p in parts
                if p.get("grounding")
                for t in p["grounding"].get("trace", [])
                if t.get("stage") == "repair_generation"
            ),
            "reverification": sum(
                1
                for p in parts
                if p.get("grounding")
                for t in p["grounding"].get("trace", [])
                if t.get("stage") == "reverification"
            ),
        }
        api_calls = {
            "plan": 1,
            "resolutions": num_resolutions,
            "selection": grounded_counts["selection"],
            "generation": grounded_counts["generation"],
            "verification": grounded_counts["verification"],
            "repair_selection": grounded_counts["repair_selection"],
            "repair_generation": grounded_counts["repair_generation"],
            "reverification": grounded_counts["reverification"],
            "searches": num_searches,
            "total": (
                1
                + num_resolutions
                + sum(grounded_counts.values())
                + num_searches
            ),
        }

        grounding_stages = {}
        for stg in (
            "selection",
            "generation",
            "verification",
            "repair_selection",
            "repair_generation",
            "reverification",
        ):
            stg_traces = [
                t
                for p in parts
                if p.get("grounding")
                for t in p["grounding"].get("trace", [])
                if t.get("stage") == stg
            ]
            stg_dict = {
                "prompt_eval_count": sum(t["stats"]["prompt_eval_count"] for t in stg_traces),
                "eval_count": sum(t["stats"]["eval_count"] for t in stg_traces),
                "eval_duration": sum(t["stats"]["eval_duration"] for t in stg_traces),
                "total_duration": sum(t["stats"]["total_duration"] for t in stg_traces),
            }
            if any("load_duration" in t["stats"] for t in stg_traces):
                stg_dict["load_duration"] = sum(
                    t["stats"].get("load_duration", 0) for t in stg_traces
                )
            if any("prompt_eval_duration" in t["stats"] for t in stg_traces):
                stg_dict["prompt_eval_duration"] = sum(
                    t["stats"].get("prompt_eval_duration", 0) for t in stg_traces
                )
            grounding_stages[stg] = stg_dict

        report_stats = {
            "planning": planning_stats,
            "resolutions": resolution_stats,
            "answers": answers_stats,
            "grounding_stages": grounding_stages,
            "total": total_stats,
            "api_calls": api_calls,
        }

        any_verification = any(
            p.get("grounding")
            and any(
                t.get("stage") in ("verification", "reverification")
                for t in p["grounding"].get("trace", [])
            )
            for p in parts
        )
        validation_info = {
            "citation_quotes_checked": True,
            "plan_semantics_verified": False,
            "semantic_support_checked_by_model": any_verification,
            "semantic_faithfulness_certified": False,
        }

    # 9. Hash canónico de hashes de prompts individuales en orden (null explícito para bloqueados)
    ordered_prompt_hashes: list[str | None] = [plan_prompt_sha256]
    for p in parts:
        if p["depends_on"]:
            if p["execution_status"] == "blocked_dependency":
                ordered_prompt_hashes.append(None)
            else:
                res_meta = p.get("resolution")
                ordered_prompt_hashes.append(res_meta["prompt_sha256"] if res_meta else None)

    for p in parts:
        if p["execution_status"] == "blocked_dependency":
            ordered_prompt_hashes.append(None)
        else:
            ordered_prompt_hashes.append(p.get("answer_prompt_sha256"))

    canonical_hashes_json = json.dumps(ordered_prompt_hashes, separators=(",", ":"))
    global_prompt_sha256 = hashlib.sha256(canonical_hashes_json.encode("utf-8")).hexdigest()

    # 10. Estructuración del reporte canónico final
    parameters = {
        "temperature": 0,
        "seed": 42,
        "num_ctx": 8192,
        "num_predict": 1024,
        "top_k": top_k,
    }

    plan_metadata = {
        "subquestions": validated_plan["subquestions"],
        "model": plan_report["model"],
        "model_digest": plan_report["model_digest"],
        "prompt_sha256": plan_prompt_sha256,
        "stats": validated_plan_stats,
        "parameters": validated_plan_params,
    }

    report = {
        "schema_version": report_schema_version,
        "query": cleaned_query,
        "status": global_status,
        "claims": global_claims,
        "reason": global_reason,
        "plan": plan_metadata,
        "parts": parts,
        "sources": list(discovered_chunks.values()),
        "top_k": top_k,
        "top_k_requested": top_k,
        "corpus_size": len(index["chunks"]),
        "retrieval_model": index["model"],
        "retrieval_model_digest": index["model_digest"],
        "index_input_sha256": index.get("input_sha256"),
        "generation_model": model,
        "generation_model_digest": pin_digest,
        "parameters": parameters,
        "prompt_sha256": global_prompt_sha256,
        "prompt_hashes": ordered_prompt_hashes,
        "stats": report_stats,
        "api_calls": api_calls,
        "validation": validation_info,
    }
    if grounder is not None:
        report["node_answers"] = num_answers

    return report


def format_text_semantic(report: dict) -> str:
    """Genera una salida legible en texto estructurando plan, dependencias, respuestas por parte y citas."""
    is_grounded = report.get("schema_version") == GROUNDED_SCHEMA_VERSION

    if not is_grounded:
        lines = [
            "=" * 72,
            "RESPUESTA RAG SEMÁNTICA — PENSIONES AIREF 2025",
            "=" * 72,
            f"Consulta: {report['query']}",
            f"Modelo generador / planificador: {report['generation_model']}",
            f"Estado global: {'Respondida con citas comprobadas' if report['status'] == 'answered' else 'Abstención / Respuesta parcial por evidencia insuficiente'}",
            f"Nodos del plan evaluados: {len(report['parts'])}",
            "",
            "PLAN SEMÁNTICO:",
        ]
        for node in report["plan"]["subquestions"]:
            deps_str = ", ".join(node["depends_on"]) if node["depends_on"] else "ninguna"
            lines.append(f"  [{node['id']}] \"{node['question']}\"")
            lines.append(f"       Evidencia requerida: {node['required_evidence']}")
            lines.append(f"       Dependencias: {deps_str}")
            part_match = next((p for p in report["parts"] if p["id"] == node["id"]), None)
            if part_match and part_match.get("resolved_question"):
                lines.append(f"       Consulta resuelta: \"{part_match['resolved_question']}\"")
        lines.append("")

        lines.append("DESGLOSE POR NODO:")
        for part in report["parts"]:
            lines.append(f"  [{part['id']}] Subpregunta: {part['question']}")
            if part["execution_status"] == "blocked_dependency":
                lines.append("      Estado de ejecución: Dependencia bloqueada")
                lines.append(f"      Estado: Abstención ({part['status']})")
                lines.append(f"      Motivo de bloqueo: {part['reason']}")
            elif part["status"] == "answered":
                lines.append("      Estado: Respondida")
                lines.append("      Afirmaciones:")
                for c_idx, claim in enumerate(part["claims"], start=1):
                    srcs = ", ".join(sorted({ev["source_id"] for ev in claim["evidence"]}))
                    lines.append(f"        {c_idx}. [{srcs}] {claim['text']}")
                    for ev in claim["evidence"]:
                        src_meta = next(
                            (s for s in report["sources"] if s["source_id"] == ev["source_id"]), None
                        )
                        if src_meta:
                            lines.append(
                                f"           - Fuente {ev['source_id']} ({src_meta['source_file']}, "
                                f"pág. {src_meta['page_number']}, {src_meta['published_at']}):"
                            )
                        else:
                            lines.append(f"           - Fuente {ev['source_id']}:")
                        lines.append(f'             "{ev["quote"]}"')
            else:
                lines.append("      Estado: Abstención por evidencia insuficiente")
                lines.append(f"      Motivo de abstención: {part['reason']}")
            lines.append("")

        if report["status"] != "answered":
            lines.append("MOTIVO GLOBAL DE LA ABSTENCIÓN / RESPUESTA PARCIAL:")
            lines.append(f"  {report['reason']}")
            lines.append("")

        lines.append("FUENTES RECUPERADAS (DEDUPLICADAS):")
        for src in report["sources"]:
            retrievals_str = ", ".join(
                f"{r['subquestion_id']} (rango {r['rank']}, score {r['score']:.4f})"
                for r in src["retrievals"]
            )
            lines.append(
                f"  [{src['source_id']}] {src['source_file']} (pág. {src['page_number']}, "
                f"publicado: {src['published_at']})"
            )
            lines.append(f"       Recuperado en: {retrievals_str}")
        lines.append("")

        lines.append("NOTA METODOLÓGICA:")
        lines.append(
            "  Corpus limitado al informe de la AIReF de marzo de 2025 (páginas 95-99). Las cifras futuras "
            "son proyecciones macroeconómicas de un escenario base, no datos observados. La validación "
            "comprueba la presencia literal exacta de las citas en los fragmentos recuperados para cada subpregunta, "
            "pero no evalúa la fidelidad semántica global ni garantiza que las afirmaciones se deduzcan lógicamente "
            "de las fuentes. La recuperación se realiza por similitud coseno top-k independiente por nodo "
            "(no es cross-encoder ni fusión de scores)."
        )
        return "\n".join(lines)

    # Modo Grounded
    lines = [
        "=" * 72,
        "RESPUESTA RAG SEMÁNTICA FUNDAMENTADA — PENSIONES AIREF 2025",
        "=" * 72,
        f"Consulta: {report['query']}",
        f"Modelo generador / planificador: {report['generation_model']}",
        f"Estado global: {'Respondida con apoyo evaluado por modelo' if report['status'] == 'answered' else 'Abstención / Respuesta parcial por evidencia insuficiente'}",
        f"Nodos del plan evaluados: {len(report['parts'])}",
        "",
        "PLAN SEMÁNTICO:",
    ]
    for node in report["plan"]["subquestions"]:
        deps_str = ", ".join(node["depends_on"]) if node["depends_on"] else "ninguna"
        lines.append(f"  [{node['id']}] \"{node['question']}\"")
        lines.append(f"       Evidencia requerida: {node['required_evidence']}")
        lines.append(f"       Dependencias: {deps_str}")
        part_match = next((p for p in report["parts"] if p["id"] == node["id"]), None)
        if part_match and part_match.get("resolved_question"):
            lines.append(f"       Consulta resuelta: \"{part_match['resolved_question']}\"")
    lines.append("")

    lines.append("DESGLOSE POR NODO:")
    for part in report["parts"]:
        lines.append(f"  [{part['id']}] Subpregunta: {part['question']}")
        if part["execution_status"] == "blocked_dependency":
            lines.append("      Estado de ejecución: Dependencia bloqueada")
            lines.append(f"      Estado: Abstención ({part['status']})")
            lines.append(f"      Motivo de bloqueo: {part['reason']}")
        else:
            g_info = part.get("grounding")
            if g_info:
                rep_att = g_info.get("repair_attempts", 0)
                acc_cnt = len(g_info.get("accepted_claims", []))
                rej_cnt = len(g_info.get("rejected_claims", []))
                lines.append(f"      Intentos de reparación: {rep_att}")
                lines.append(f"      Afirmaciones aceptadas: {acc_cnt}, rechazadas: {rej_cnt}")

            if part["status"] == "answered":
                lines.append("      Estado: Respondida (apoyo evaluado por modelo)")
                lines.append("      Afirmaciones:")
                for c_idx, claim in enumerate(part["claims"], start=1):
                    srcs = ", ".join(sorted({ev["source_id"] for ev in claim["evidence"]}))
                    lines.append(f"        {c_idx}. [{srcs}] {claim['text']}")
                    for ev in claim["evidence"]:
                        src_meta = next(
                            (s for s in report["sources"] if s["source_id"] == ev["source_id"]), None
                        )
                        if src_meta:
                            lines.append(
                                f"           - Fuente {ev['source_id']} ({src_meta['source_file']}, "
                                f"pág. {src_meta['page_number']}, {src_meta['published_at']}):"
                            )
                        else:
                            lines.append(f"           - Fuente {ev['source_id']}:")
                        lines.append(f'             "{ev["quote"]}"')
            else:
                lines.append("      Estado: Abstención por evidencia insuficiente")
                lines.append(f"      Motivo de abstención: {part['reason']}")
                if g_info and g_info.get("accepted_claims"):
                    lines.append("      Afirmaciones parciales aceptadas (no autosuficientes):")
                    for c_idx, pclaim in enumerate(g_info["accepted_claims"], start=1):
                        lines.append(f"        {c_idx}. {pclaim['text']}")
        lines.append("")

    if report["status"] != "answered":
        lines.append("MOTIVO GLOBAL DE LA ABSTENCIÓN / RESPUESTA PARCIAL:")
        lines.append(f"  {report['reason']}")
        lines.append("")

    lines.append("FUENTES RECUPERADAS (DEDUPLICADAS):")
    for src in report["sources"]:
        retrievals_str = ", ".join(
            f"{r['subquestion_id']} (rango {r['rank']}, score {r['score']:.4f})"
            for r in src["retrievals"]
        )
        lines.append(
            f"  [{src['source_id']}] {src['source_file']} (pág. {src['page_number']}, "
            f"publicado: {src['published_at']})"
        )
        lines.append(f"       Recuperado en: {retrievals_str}")
    lines.append("")

    lines.append("NOTA METODOLÓGICA:")
    lines.append(
        "  Corpus limitado al informe de la AIReF de marzo de 2025 (páginas 95-99). Las cifras futuras "
        "son proyecciones macroeconómicas de un escenario base, no datos observados. La validación "
        "comprueba la presencia literal exacta de las citas en los fragmentos recuperados para cada subpregunta "
        "y el apoyo semántico evaluado por el modelo local (Qwen). Este soporte no certifica verdad formal "
        "ni fidelidad absoluta, pues el evaluador comparte sesgos con el generador. La presencia de afirmaciones "
        "parciales aceptadas en nodos no respondidos no autoriza la resolución de dependencias ni constituye una respuesta completa."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la CLI de generación semántica de respuestas fundamentadas."""
    parser = argparse.ArgumentParser(
        description="Flujo RAG semántico local con planificación estructurada y recuperación por necesidad."
    )
    parser.add_argument("--query", required=True, type=str, help="Texto de la consulta.")
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=4,
        help="Número de fragmentos a recuperar por cada nodo (defecto 4).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_GENERATION_MODEL,
        choices=ALLOWED_MODELS,
        help="Modelo generador local Ollama: Qwen 3.5 o K2 Horizon 7B.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Ruta al archivo JSON con el índice de embeddings.",
    )
    parser.add_argument(
        "--grounded",
        action="store_true",
        default=False,
        help="Activar fundamentación con selección, verificación y reparación.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emitir reporte completo en formato JSON UTF-8 a stdout.",
    )

    try:
        args = parser.parse_args(argv)

        if not isinstance(args.query, str) or not args.query.strip():
            sys.stderr.write("Error: query cannot be empty or contain only whitespace\n")
            return 1

        if len(args.query.strip()) > 1000:
            sys.stderr.write(
                f"Error: query length exceeds 1000 characters ({len(args.query.strip())})\n"
            )
            return 1

        if isinstance(args.top_k, bool) or not isinstance(args.top_k, int) or args.top_k <= 0:
            sys.stderr.write(f"Error: top-k must be a positive integer, got {args.top_k!r}\n")
            return 1

        if args.model not in ALLOWED_MODELS:
            sys.stderr.write(
                f"Error: model {args.model!r} is not allowed. Choose from {sorted(ALLOWED_MODELS)}\n"
            )
            return 1

        index_path = validate_project_path(args.index)
        index_data = load_index(index_path)

        grounder = OllamaGrounder(model=args.model) if args.grounded else None

        result = answer_semantic(
            index=index_data,
            query=args.query,
            top_k=args.top_k,
            model=args.model,
            grounder=grounder,
        )

        if args.json:
            sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        else:
            sys.stdout.write(format_text_semantic(result) + "\n")

        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
