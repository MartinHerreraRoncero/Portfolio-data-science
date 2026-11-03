"""Tests exhaustivos y deterministas offline para el buscador semántico."""

import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.embeddings import (
    DIMENSION,
    DOCUMENT_PREFIX,
    MODEL_NAME,
    QUERY_PREFIX,
    SCHEMA_VERSION,
)
from rag_pensiones.ingest import PROJECT_ROOT
from rag_pensiones.search import (
    DEFAULT_INDEX_PATH,
    LEGACY_INDEX_PATH,
    SearchHit,
    load_index,
    main,
    search,
    validate_index,
)


class MockOllamaClient:
    """Cliente simulado offline para pruebas de búsqueda sin contactar red."""

    def __init__(
        self,
        digest: str = "fake_test_digest_sha256",
        vector: list[float] | None = None,
        digest_after: str | None = None,
    ) -> None:
        self.digest = digest
        self.digest_after = digest_after if digest_after is not None else digest
        self.vector = vector if vector is not None else ([1.0] + [0.0] * (DIMENSION - 1))
        self.calls: list[str] = []

    def get_digest(self) -> str:
        count = self.calls.count("get_digest")
        self.calls.append("get_digest")
        if count == 0:
            return self.digest
        return self.digest_after

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append("embed_texts")
        self.last_texts = texts
        return [list(self.vector)]


def _make_valid_chunk(
    idx: int,
    vec: list[float] | None = None,
    doc_id: str = "doc_test_1",
    source_file: str = "test_doc.pdf",
    published_at: str = "2025-03-31",
    page_number: int = 1,
) -> dict:
    if vec is None:
        vec = [0.0] * DIMENSION
        vec[0] = 1.0
    return {
        "chunk_id": f"chunk_id_{idx}",
        "doc_id": doc_id,
        "source_file": source_file,
        "published_at": published_at,
        "page_number": page_number,
        "start_char": 0,
        "end_char": 100,
        "text": f"Texto del fragmento {idx}",
        "strategy": "page-local-v1:max_chars=1000:overlap_chars=150",
        "embedding": list(vec),
    }


def _make_valid_index(chunks: list[dict] | None = None, digest: str = "fake_test_digest_sha256") -> dict:
    if chunks is None:
        chunks = [_make_valid_chunk(1), _make_valid_chunk(2)]
    return {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "model_digest": digest,
        "dimension": DIMENSION,
        "document_prefix": DOCUMENT_PREFIX,
        "query_prefix": QUERY_PREFIX,
        "input_sha256": "4cac2ee74838d6c5caa37efa0f27336689cd1ff12b6cbf9aa1948e1a36314cdc",
        "chunks": chunks,
    }


class TestCosineAndRankingOffline(unittest.TestCase):
    """Pruebas de similitud coseno, ranking, valores conocidos y pureza sin mutación."""

    def test_known_value_vectors_ranking(self):
        """Verifica ranking con vectores ortogonales, opuestos, idénticos y en ángulo de 45 grados."""
        v_ident = [1.0] + [0.0] * (DIMENSION - 1)
        v_ortho = [0.0, 1.0] + [0.0] * (DIMENSION - 2)
        v_opp = [-1.0] + [0.0] * (DIMENSION - 1)
        v_45 = [1.0, 1.0] + [0.0] * (DIMENSION - 2)

        chunks = [
            _make_valid_chunk(1, v_ortho),
            _make_valid_chunk(2, v_ident),
            _make_valid_chunk(3, v_opp),
            _make_valid_chunk(4, v_45),
        ]
        index = _make_valid_index(chunks)
        client = MockOllamaClient(vector=v_ident)

        hits = search(index, "consulta sobre pensiones", top_k=4, client=client)

        self.assertEqual(len(hits), 4)
        # Rank 1: v_ident (score 1.0)
        self.assertEqual(hits[0].rank, 1)
        self.assertEqual(hits[0].chunk.chunk_id, "chunk_id_2")
        self.assertAlmostEqual(hits[0].score, 1.0, places=6)

        # Rank 2: v_45 (score 1/sqrt(2) approx 0.70710678)
        self.assertEqual(hits[1].rank, 2)
        self.assertEqual(hits[1].chunk.chunk_id, "chunk_id_4")
        self.assertAlmostEqual(hits[1].score, 1.0 / math.sqrt(2.0), places=6)

        # Rank 3: v_ortho (score 0.0)
        self.assertEqual(hits[2].rank, 3)
        self.assertEqual(hits[2].chunk.chunk_id, "chunk_id_1")
        self.assertAlmostEqual(hits[2].score, 0.0, places=6)

        # Rank 4: v_opp (score -1.0)
        self.assertEqual(hits[3].rank, 4)
        self.assertEqual(hits[3].chunk.chunk_id, "chunk_id_3")
        self.assertAlmostEqual(hits[3].score, -1.0, places=6)

    def test_stable_tie_breaking(self):
        """Verifica desempate estable por posición original cuando las puntuaciones son idénticas."""
        v = [0.0, 1.0] + [0.0] * (DIMENSION - 2)
        chunks = [
            _make_valid_chunk(10, v),
            _make_valid_chunk(20, v),
            _make_valid_chunk(30, v),
        ]
        index = _make_valid_index(chunks)
        client = MockOllamaClient(vector=v)

        hits = search(index, "pregunta", top_k=3, client=client)

        self.assertEqual(len(hits), 3)
        self.assertEqual(hits[0].chunk.chunk_id, "chunk_id_10")
        self.assertEqual(hits[1].chunk.chunk_id, "chunk_id_20")
        self.assertEqual(hits[2].chunk.chunk_id, "chunk_id_30")
        for h in hits:
            self.assertAlmostEqual(h.score, 1.0, places=6)

    def test_top_k_one_and_top_k_larger_than_corpus(self):
        """Verifica comportamiento cuando top_k=1 y cuando top_k supera el tamaño del corpus."""
        chunks = [_make_valid_chunk(1), _make_valid_chunk(2)]
        index = _make_valid_index(chunks)
        client = MockOllamaClient()

        # top_k = 1
        hits1 = search(index, "consulta", top_k=1, client=client)
        self.assertEqual(len(hits1), 1)
        self.assertEqual(hits1[0].rank, 1)

        # top_k = 100 > corpus_size (2)
        hits_all = search(index, "consulta", top_k=100, client=client)
        self.assertEqual(len(hits_all), 2)
        self.assertEqual(hits_all[0].rank, 1)
        self.assertEqual(hits_all[1].rank, 2)

    def test_query_and_top_k_invalid(self):
        """Verifica rechazo de queries vacías/blancos y top_k no enteros o no positivos."""
        index = _make_valid_index()
        client = MockOllamaClient()

        # Query no string o vacía tras strip
        with self.assertRaises(ValueError):
            search(index, "", client=client)
        with self.assertRaises(ValueError):
            search(index, "   \n\t  ", client=client)
        with self.assertRaises(ValueError):
            search(index, None, client=client)  # type: ignore

        # top_k booleanos, <= 0 o tipos incorrectos
        with self.assertRaises(ValueError):
            search(index, "consulta", top_k=True, client=client)
        with self.assertRaises(ValueError):
            search(index, "consulta", top_k=False, client=client)
        with self.assertRaises(ValueError):
            search(index, "consulta", top_k=0, client=client)
        with self.assertRaises(ValueError):
            search(index, "consulta", top_k=-3, client=client)
        with self.assertRaises(ValueError):
            search(index, "consulta", top_k="2", client=client)  # type: ignore

    def test_index_immutability(self):
        """Verifica que search no muta en absoluto el diccionario ni los vectores del índice original."""
        chunks = [_make_valid_chunk(1), _make_valid_chunk(2)]
        index = _make_valid_index(chunks)
        index_snapshot = copy.deepcopy(index)
        client = MockOllamaClient()

        search(index, "consulta", top_k=2, client=client)

        self.assertEqual(index, index_snapshot)


class TestIndexCorruptionAndValidation(unittest.TestCase):
    """Pruebas de corrupción del índice y comprobación de que no contacta con el cliente ante fallo."""

    def test_validation_fails_does_not_call_client(self):
        """Verifica que si la validación falla, jamás se contacta al cliente de embeddings."""
        class FailClient:
            def get_digest(self):
                raise AssertionError("get_digest called despite invalid index!")

            def embed_texts(self, texts):
                raise AssertionError("embed_texts called despite invalid index!")

        corrupted = _make_valid_index()
        corrupted["schema_version"] = "invalid_version"

        with self.assertRaises(ValueError):
            search(corrupted, "consulta", client=FailClient())

    def test_corrupted_metadata_and_prefixes(self):
        """Verifica rechazo de versión, modelo, prefijos y dimensión erróneos."""
        for field, bad_val in [
            ("schema_version", "embeddings-v2"),
            ("model", "qwen2.5:7b"),
            ("dimension", 512),
            ("dimension", True),
            ("document_prefix", "bad: "),
            ("query_prefix", "query: "),
            ("model_digest", ""),
            ("model_digest", "   "),
            ("input_sha256", ""),
            ("input_sha256", "   "),
            ("chunks", []),
        ]:
            idx = _make_valid_index()
            idx[field] = bad_val
            with self.assertRaises(ValueError, msg=f"Field {field}={bad_val!r} should fail"):
                validate_index(idx)

    def test_duplicate_chunk_ids_rejected(self):
        """Verifica rechazo de chunk_id duplicados."""
        c1 = _make_valid_chunk(1)
        c2 = _make_valid_chunk(1)  # mismo chunk_id
        idx = _make_valid_index([c1, c2])
        with self.assertRaises(ValueError):
            validate_index(idx)

    def test_vector_corruptions_nan_inf_bool_zero_norm(self):
        """Verifica rechazo de vectores con NaN, Inf, booleanos, norma cero o dimensión incorrecta."""
        # NaN
        vec_nan = [0.0] * DIMENSION
        vec_nan[5] = float("nan")
        with self.assertRaises(ValueError):
            validate_index(_make_valid_index([_make_valid_chunk(1, vec_nan)]))

        # Inf
        vec_inf = [0.0] * DIMENSION
        vec_inf[10] = float("inf")
        with self.assertRaises(ValueError):
            validate_index(_make_valid_index([_make_valid_chunk(1, vec_inf)]))

        # bool
        vec_bool = [0.0] * DIMENSION
        vec_bool[0] = True  # type: ignore
        with self.assertRaises(ValueError):
            validate_index(_make_valid_index([_make_valid_chunk(1, vec_bool)]))

        # zero norm
        vec_zero = [0.0] * DIMENSION
        with self.assertRaises(ValueError):
            validate_index(_make_valid_index([_make_valid_chunk(1, vec_zero)]))

        # wrong dimension
        vec_short = [1.0] * (DIMENSION - 1)
        with self.assertRaises(ValueError):
            validate_index(_make_valid_index([_make_valid_chunk(1, vec_short)]))

    def test_chunk_field_corruptions_and_inconsistent_doc_metadata(self):
        """Verifica rechazo de campos inválidos y metadatos incompatibles para un mismo doc_id."""
        # Campo faltante
        c = _make_valid_chunk(1)
        del c["text"]
        with self.assertRaises(ValueError):
            validate_index(_make_valid_index([c]))

        # Inconsistencia en source_file para el mismo doc_id
        c1 = _make_valid_chunk(1, doc_id="doc_A", source_file="file1.pdf")
        c2 = _make_valid_chunk(2, doc_id="doc_A", source_file="file2.pdf")
        with self.assertRaises(ValueError):
            validate_index(_make_valid_index([c1, c2]))

        # Inconsistencia en published_at para el mismo doc_id
        c3 = _make_valid_chunk(3, doc_id="doc_B", published_at="2025-01-01")
        c4 = _make_valid_chunk(4, doc_id="doc_B", published_at="2025-03-31")
        with self.assertRaises(ValueError):
            validate_index(_make_valid_index([c3, c4]))

    def test_load_index_json_and_path_boundaries(self):
        """Verifica load_index con archivos inexistentes, NaN en JSON y rutas fuera del proyecto."""
        # Ruta externa al proyecto
        with self.assertRaises(ValueError):
            load_index(Path("C:/outside_project_index.json"))

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            tmppath = Path(tmpdir)

            # Archivo inexistente
            with self.assertRaises(FileNotFoundError):
                load_index(tmppath / "no_existe.json")

            # Archivo vacío
            empty_file = tmppath / "empty.json"
            empty_file.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_index(empty_file)

            # JSON con literal NaN (prohibido)
            nan_file = tmppath / "nan.json"
            nan_file.write_text('{"schema_version": NaN}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_index(nan_file)

            # JSON malformado sintácticamente
            bad_json = tmppath / "bad.json"
            bad_json.write_text('{"unclosed": ', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_index(bad_json)


class TestDigestAndClientErrors(unittest.TestCase):
    """Pruebas de cambio o discordancia de digest y fallos del cliente Ollama."""

    def test_digest_mismatch_before_embed(self):
        """Verifica aborto si el digest del cliente no coincide con el del índice antes de embed."""
        index = _make_valid_index(digest="digest_AAA")
        client = MockOllamaClient(digest="digest_BBB")

        with self.assertRaises(RuntimeError) as ctx:
            search(index, "consulta", client=client)
        self.assertIn("digest mismatch", str(ctx.exception))
        self.assertNotIn("embed_texts", client.calls)

    def test_digest_changed_after_embed(self):
        """Verifica aborto si el digest del cliente cambia durante la llamada de embedding."""
        index = _make_valid_index(digest="digest_AAA")
        client = MockOllamaClient(digest="digest_AAA", digest_after="digest_CHANGED")

        with self.assertRaises(RuntimeError) as ctx:
            search(index, "consulta", client=client)
        self.assertIn("digest changed", str(ctx.exception))

    def test_invalid_query_vector_returned_by_client(self):
        """Verifica rechazo si el cliente devuelve vectores de dimensión o valores incorrectos."""
        index = _make_valid_index()

        # Vector con dimensión errónea
        client_bad_dim = MockOllamaClient(vector=[1.0] * 10)
        with self.assertRaises(ValueError):
            search(index, "consulta", client=client_bad_dim)

        # Vector con norma cero
        client_zero_norm = MockOllamaClient(vector=[0.0] * DIMENSION)
        with self.assertRaises(ValueError):
            search(index, "consulta", client=client_zero_norm)

        # Vector con NaN
        bad_vec = [1.0] * DIMENSION
        bad_vec[0] = float("nan")
        client_nan = MockOllamaClient(vector=bad_vec)
        with self.assertRaises(ValueError):
            search(index, "consulta", client=client_nan)

    def test_ollama_client_exception_propagates(self):
        """Verifica que errores de conexión de Ollama no se capturan silenciosamente."""
        class ErrorClient:
            def get_digest(self):
                return "fake_test_digest_sha256"

            def embed_texts(self, texts):
                raise ConnectionError("No se puede contactar con Ollama")

        index = _make_valid_index()
        with self.assertRaises(ConnectionError):
            search(index, "consulta", client=ErrorClient())


class TestCLIIntegrationOffline(unittest.TestCase):
    """Pruebas de la CLI en modo texto, JSON y manejo de errores sin llamadas reales a Ollama."""

    def test_cli_text_output_spanish_and_disclaimer(self):
        """Verifica salida estándar en español con metadatos y aviso metodológico explícito."""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            idx_file = Path(tmpdir) / "test_idx.json"
            data = _make_valid_index([_make_valid_chunk(1), _make_valid_chunk(2)])
            idx_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            fake_client = MockOllamaClient()
            with patch("rag_pensiones.search.OllamaEmbedder", return_value=fake_client):
                out_buf = io.StringIO()
                err_buf = io.StringIO()
                with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
                    code = main(["--query", "¿Cuál es el gasto?", "--top-k", "2", "--index", str(idx_file)])

                self.assertEqual(code, 0)
                self.assertEqual(err_buf.getvalue(), "")
                out = out_buf.getvalue()
                self.assertIn("Consulta: ¿Cuál es el gasto?", out)
                self.assertIn("Modelo: embeddinggemma:300m", out)
                self.assertIn("Resultados recuperados: 2", out)
                self.assertIn("La similitud coseno mide proximidad semántica", out)
                self.assertIn("[Resultado 1]", out)
                self.assertIn("Puntuación (similitud coseno):", out)
                self.assertIn("Documento: test_doc.pdf", out)
                self.assertIn("ID de fragmento: chunk_id_1", out)
                self.assertIn("Texto del fragmento 1", out)

    def test_cli_json_output(self):
        """Verifica salida JSON UTF-8 con todas las claves y campos requeridos."""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            idx_file = Path(tmpdir) / "test_idx.json"
            data = _make_valid_index([_make_valid_chunk(1)])
            idx_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            fake_client = MockOllamaClient()
            with patch("rag_pensiones.search.OllamaEmbedder", return_value=fake_client):
                out_buf = io.StringIO()
                err_buf = io.StringIO()
                with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
                    code = main(["--query", "consulta test", "--top-k", "5", "--index", str(idx_file), "--json"])

                self.assertEqual(code, 0)
                self.assertEqual(err_buf.getvalue(), "")
                parsed = json.loads(out_buf.getvalue())
                self.assertEqual(parsed["query"], "consulta test")
                self.assertEqual(parsed["model"], MODEL_NAME)
                self.assertEqual(parsed["corpus_size"], 1)
                self.assertEqual(parsed["top_k_requested"], 5)
                self.assertEqual(len(parsed["hits"]), 1)
                self.assertEqual(parsed["hits"][0]["rank"], 1)
                self.assertEqual(parsed["hits"][0]["chunk"]["chunk_id"], "chunk_id_1")

    def test_cli_errors_to_stderr_and_non_zero_exit_no_traceback(self):
        """Verifica que errores de argumentos o validación emiten mensaje a stderr sin traceback."""
        # Query vacía
        err_buf = io.StringIO()
        out_buf = io.StringIO()
        with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
            code = main(["--query", "   "])
        self.assertNotEqual(code, 0)
        self.assertIn("Error:", err_buf.getvalue())
        self.assertNotIn("Traceback", err_buf.getvalue())

        # top_k inválido
        err_buf = io.StringIO()
        with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
            code = main(["--query", "valida", "--top-k", "0"])
        self.assertNotEqual(code, 0)
        self.assertIn("Error:", err_buf.getvalue())

        # Ruta de índice fuera del proyecto
        err_buf = io.StringIO()
        with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
            code = main(["--query", "valida", "--index", "C:/outside/index.json"])
        self.assertNotEqual(code, 0)
        self.assertIn("Error:", err_buf.getvalue())

    def test_real_13_vector_index_offline_search(self):
        """Verifica búsqueda sobre el índice real local data/processed/airef_2025_embeddings.json."""
        index_data = load_index(LEGACY_INDEX_PATH)
        self.assertEqual(len(index_data["chunks"]), 13)

        fake_client = MockOllamaClient(digest=index_data["model_digest"])
        hits = search(index_data, "sostenibilidad y pensiones", top_k=3, client=fake_client)

        self.assertEqual(len(hits), 3)
        self.assertEqual(hits[0].rank, 1)
        self.assertEqual(hits[1].rank, 2)
        self.assertEqual(hits[2].rank, 3)
        for h in hits:
            self.assertIsInstance(h, SearchHit)
            self.assertIsInstance(h.chunk, ChunkRecord)
            self.assertFalse(hasattr(h.chunk, "embedding"))
            self.assertTrue(math.isfinite(h.score))


if __name__ == "__main__":
    unittest.main()
