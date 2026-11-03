"""Carry existing plan fields as query data, without rewriting or another model."""
import json


def compose_execution_query(question: str, required_evidence: str) -> str:
    for value, limit, name in [(question, 600, 'question'), (required_evidence, 250, 'required_evidence')]:
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError(f'{name} must be a nonempty string of at most {limit} characters')
    result = json.dumps(dict(pregunta=question, evidencia_requerida=required_evidence),
                        ensure_ascii=False, separators=(',', ':'))
    if len(result) > 1000:
        raise ValueError('Execution query exceeds 1000 characters; no truncation')
    return result
