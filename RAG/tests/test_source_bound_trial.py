"""Offline invariants for source binding and the separate specialist adapter."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from experiments.source_bound_trial import (
    MINICHECK, MiniCheckClient, RecordedGenerator, SourceBoundGrounder,
    build_catalog, checkpoint, restore_selection, selection_schema, validate_controls, run_verifiers,
)


def hit(text, rank=1):
    return SimpleNamespace(rank=rank, chunk=SimpleNamespace(
        text=text, chunk_id=f"chunk-{rank}", page_number=95, start_char=0))


class SourceBindingTests(unittest.TestCase):
    def test_schema_excludes_incoherent_selected_and_abstention_states(self):
        schema = selection_schema([{"id": "R1"}])
        selected, abstained = [b["properties"] for b in schema["anyOf"]]
        self.assertEqual(selected["reason"], {"const": ""})
        self.assertEqual(selected["ids"]["minItems"], 1)
        self.assertEqual(abstained["ids"]["maxItems"], 0)
        self.assertEqual(abstained["reason"]["minLength"], 1)

    def test_offsets_preserve_original_whitespace_and_literal_quotes(self):
        text = "  El gasto total es del 16,1%  del PIB en 2050.\nLa unidad siguiente tiene otro sujeto. "
        catalog = build_catalog([hit(text)])
        self.assertEqual(len(catalog["records"]), 2)
        for r in catalog["records"]:
            raw = text[r["start_in_chunk"]:r["end_in_chunk"]]
            self.assertEqual(" ".join(raw.split()), r["quote"])
            self.assertEqual(r["chunk_id"], "chunk-1")
        self.assertEqual(catalog["records"][0]["years"], ["2050"])
        self.assertIn("16,1%", catalog["records"][0]["numbers"])

    def test_oversize_unit_is_logged_not_truncated(self):
        text = "Texto " + "largo " * 140 + ". Otra oración completa con un dato de 2023."
        catalog = build_catalog([hit(text)])
        self.assertEqual(len(catalog["omitted"]), 1)
        self.assertGreater(len(catalog["omitted"][0]["quote"]), 700)
        self.assertEqual(catalog["records"][0]["quote"], "Otra oración completa con un dato de 2023.")

    def test_two_sources_cannot_be_reassigned(self):
        records = build_catalog([hit("La primera fuente aporta el total."), hit("La segunda fuente describe una parte.", 2)])["records"]
        p = restore_selection({"status": "selected", "ids": ["R2"], "reason": ""}, records)
        self.assertEqual(p["evidence"], [{"source_id": "S2", "quote": "La segunda fuente describe una parte."}])

    def test_chunk_boundary_fragments_are_logged_not_selected(self):
        h = hit("mitad de una frase anterior. Esta oración está completa. Y termina a mitad")
        h.chunk.start_char = 100
        c = build_catalog([h])
        self.assertEqual([r["quote"] for r in c["records"]], ["Esta oración está completa."])
        self.assertEqual([r["omission"] for r in c["omitted"]], ["boundary", "boundary"])

    def test_unknown_duplicate_or_too_many_ids_rejected(self):
        records = build_catalog([hit("Una oración suficientemente extensa.")])["records"]
        for ids in (["S2"], ["R1", "R1"], [None], ["R1"] * 5):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                restore_selection({"status": "selected", "ids": ids, "reason": ""}, records)

    def test_abstention_cannot_keep_evidence(self):
        records = build_catalog([hit("Una oración suficientemente extensa.")])["records"]
        with self.assertRaises(ValueError):
            restore_selection({"status": "insufficient_evidence", "ids": ["R1"], "reason": "Falta"}, records)

    def test_total_utf8_citation_budget_enforced(self):
        records = [{"id": f"R{i}", "source_id": f"S{i}", "quote": "á" * 650} for i in range(2)]
        with self.assertRaisesRegex(ValueError, "byte budget"):
            restore_selection({"status": "selected", "ids": ["R0", "R1"], "reason": ""}, records)

    def test_controls_cannot_include_gold(self):
        data = {"schema_version": "relational-controls-v1", "cases": [
            {"id": "c1", "claim": "Dato", "evidence": ["Cita suficientemente larga."], "expected": True}]}
        with self.assertRaises(ValueError):
            validate_controls(data)

    def test_atomic_checkpoint_is_valid_json(self):
        with tempfile.TemporaryDirectory(dir="tmp") as directory:
            path = Path(directory) / "result.json"
            checkpoint({"value": "á"}, path)
            checkpoint({"value": "otra"}, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": "otra"})
            self.assertFalse(path.with_suffix(".json.pending").exists())


class SpecialistTests(unittest.TestCase):
    def client(self, label="Yes", **updates):
        client = MiniCheckClient()
        client.get_digest = MagicMock(return_value="digest")
        raw = dict(model=MINICHECK, done=True, done_reason="stop", response=label,
                   prompt_eval_count=20, eval_count=1, eval_duration=100, total_duration=200)
        raw.update(updates)
        client.transport._open_request = MagicMock(return_value=json.dumps(raw).encode())
        return client

    def test_document_claim_format_unconstrained_label_and_exact_model(self):
        c = self.client()
        result = c.verify("La cifra es 2.", ["El documento afirma una cifra de 2."], "digest")
        self.assertEqual(result["verdict"], "supported")
        req = c.transport._open_request.call_args[0][0]
        body = json.loads(req.data)
        self.assertEqual(body["model"], MINICHECK)
        self.assertEqual(body["prompt"], "Document: El documento afirma una cifra de 2.\nClaim: La cifra es 2.\n")
        self.assertNotIn("format", body)
        self.assertNotIn("grammar", body)

    def test_truncated_unknown_and_wrong_model_keep_raw_trace(self):
        for changes in ({"done_reason": "length"}, {"response": "Probably yes"}, {"model": "other"}):
            c = self.client(**changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                c.verify("Una afirmación.", ["Una evidencia suficientemente larga."], "digest")
            self.assertIn("raw_response", c.calls[0])
            self.assertIn("error", c.calls[0])

    def test_digest_drift_rejected_and_trace_retained(self):
        c = self.client()
        c.get_digest.side_effect = ["digest", "changed"]
        with self.assertRaisesRegex(RuntimeError, "during"):
            c.verify("Afirmación", ["Evidencia suficientemente larga."], "digest")
        self.assertIn("raw_response", c.calls[0])

    def test_preflight_budget_does_not_call_network(self):
        c = self.client()
        with self.assertRaisesRegex(ValueError, "6500"):
            c.verify("Afirmación", ["á" * 4000], "digest")
        c.transport._open_request.assert_not_called()

    def test_no_is_negative_not_a_technical_failure(self):
        self.assertEqual(self.client("No").verify("Dato", ["Cita"], "digest")["verdict"], "not_supported")

    def test_failed_generation_preserves_call(self):
        c = RecordedGenerator()
        with patch("rag_pensiones.answer.OllamaGenerator.chat", side_effect=RuntimeError("connection failed")):
            with self.assertRaises(RuntimeError):
                c.chat([{"role": "user", "content": "Pregunta"}], {"type": "object"})
        self.assertIn("connection failed", c.calls[0]["error"])


if __name__ == "__main__":
    unittest.main()
