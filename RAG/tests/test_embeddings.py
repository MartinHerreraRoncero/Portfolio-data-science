"""Pruebas unitarias e integración offline para el módulo de embeddings."""

from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.embeddings import (
    DIMENSION,
    DOCUMENT_PREFIX,
    MODEL_NAME,
    QUERY_PREFIX,
    SCHEMA_VERSION,
    OllamaEmbedder,
    build_embeddings,
    main,
    validate_chunk_record,
)
from rag_pensiones.ingest import PROJECT_ROOT


class MockHTTPResponse:
    """Simulador de respuesta HTTP de urllib."""

    def __init__(self, data: bytes, code: int = 200):
        self._data = data
        self.code = code

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def _make_dummy_vector(dim: int = DIMENSION, base_val: float = 1.0) -> list[float]:
    """Genera un vector sintético de dimensión dada."""
    return [base_val] * dim


def _make_dummy_chunk(chunk_idx: int = 1, text: str = "Texto de prueba") -> ChunkRecord:
    """Crea un ChunkRecord válido para pruebas."""
    return ChunkRecord(
        chunk_id=f"chunk_{chunk_idx:04d}",
        doc_id="doc_test_123",
        source_file="test_doc.pdf",
        published_at="2025-03-31",
        page_number=95,
        start_char=(chunk_idx - 1) * 100,
        end_char=chunk_idx * 100,
        text=text,
        strategy="page-local-v1:max_chars=1000:overlap_chars=150",
    )


class TestEmbeddingsOffline(unittest.TestCase):
    """Pruebas de cliente OllamaEmbedder y función build_embeddings con transporte mockeado."""

    def test_ollama_embedder_payload_prefixes_and_truncate(self):
        """Verifica payload exacto, prefijos de documento y parámetro truncate=False."""
        captured_requests = []

        def fake_open(req, timeout):
            captured_requests.append((req.full_url, json.loads(req.data.decode("utf-8"))))
            # Responder con vector válido
            resp_payload = {
                "model": MODEL_NAME,
                "embeddings": [_make_dummy_vector(DIMENSION, 2.0)],
            }
            return MockHTTPResponse(json.dumps(resp_payload).encode("utf-8"))

        mock_opener = MagicMock()
        mock_opener.open.side_effect = fake_open

        client = OllamaEmbedder(opener=mock_opener)
        texts = [f"{DOCUMENT_PREFIX}Contenido del fragmento"]
        vectors = client.embed_texts(texts)

        self.assertEqual(len(captured_requests), 1)
        url, body = captured_requests[0]
        self.assertTrue(url.endswith("/api/embed"))
        self.assertEqual(body["model"], MODEL_NAME)
        self.assertEqual(body["input"], texts)
        self.assertIs(body["truncate"], False)
        self.assertEqual(body["dimensions"], DIMENSION)
        self.assertEqual(body["keep_alive"], "5m")

        # Verificar normalización L2 del vector devuelto
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), DIMENSION)
        norm = math.hypot(*vectors[0])
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_unicode_preservation(self):
        """Comprueba que textos con caracteres Unicode complejos se envían y conservan fielmente."""
        unicode_text = "GRÁFICO 44: Δ 2050-2029 = 6,2% — «Sostenibilidad» € ñ á ü"
        chunk = _make_dummy_chunk(1, text=unicode_text)

        captured_input = []

        def fake_open(req, timeout):
            if req.full_url.endswith("/api/tags"):
                payload = {"models": [{"name": MODEL_NAME, "digest": "sha256:fixed_digest"}]}
                return MockHTTPResponse(json.dumps(payload).encode("utf-8"))
            if req.full_url.endswith("/api/embed"):
                body = json.loads(req.data.decode("utf-8"))
                captured_input.extend(body["input"])
                payload = {"model": MODEL_NAME, "embeddings": [_make_dummy_vector(DIMENSION, 1.0)]}
                return MockHTTPResponse(json.dumps(payload).encode("utf-8"))
            raise NotImplementedError(req.full_url)

        mock_opener = MagicMock()
        mock_opener.open.side_effect = fake_open
        client = OllamaEmbedder(opener=mock_opener)

        result = build_embeddings([chunk], client)
        self.assertEqual(len(captured_input), 1)
        self.assertEqual(captured_input[0], f"{DOCUMENT_PREFIX}{unicode_text}")
        self.assertEqual(result["chunks"][0]["text"], unicode_text)

    def test_vector_dimension_and_count_mismatch(self):
        """Verifica rechazo de dimensiones distintas a 768 y discordancia en número de vectores."""
        mock_opener = MagicMock()
        client = OllamaEmbedder(opener=mock_opener)

        # Caso A: dimensión 767 en vez de 768
        mock_opener.open.return_value = MockHTTPResponse(
            json.dumps({"model": MODEL_NAME, "embeddings": [_make_dummy_vector(767, 1.0)]}).encode("utf-8")
        )
        with self.assertRaises(ValueError) as ctx:
            client.embed_texts(["Texto"])
        self.assertIn("dimension 767, expected 768", str(ctx.exception))

        # Caso B: discordancia de número de vectores (se envían 2, devuelve 1)
        mock_opener.open.return_value = MockHTTPResponse(
            json.dumps({"model": MODEL_NAME, "embeddings": [_make_dummy_vector(DIMENSION, 1.0)]}).encode("utf-8")
        )
        with self.assertRaises(ValueError) as ctx:
            client.embed_texts(["Texto 1", "Texto 2"])
        self.assertIn("Expected 2 embeddings, received 1", str(ctx.exception))

    def test_vector_nan_inf_bool_and_zero_norm_rejected(self):
        """Verifica rechazo estricto de NaN, Inf, bool y vectores de norma cero."""
        mock_opener = MagicMock()
        client = OllamaEmbedder(opener=mock_opener)

        # NaN
        vec_nan = _make_dummy_vector(DIMENSION, 1.0)
        vec_nan[5] = float("nan")
        mock_opener.open.return_value = MockHTTPResponse(
            json.dumps({"model": MODEL_NAME, "embeddings": [vec_nan]}).encode("utf-8")
        )
        with self.assertRaises(ValueError):
            client.embed_texts(["Texto"])

        # Inf
        vec_inf = _make_dummy_vector(DIMENSION, 1.0)
        vec_inf[10] = float("inf")
        mock_opener.open.return_value = MockHTTPResponse(
            json.dumps({"model": MODEL_NAME, "embeddings": [vec_inf]}).encode("utf-8")
        )
        with self.assertRaises(ValueError):
            client.embed_texts(["Texto"])

        # Bool en el vector (True no es un flotante válido)
        vec_bool = _make_dummy_vector(DIMENSION, 1.0)
        vec_bool[0] = True  # type: ignore
        mock_opener.open.return_value = MockHTTPResponse(
            json.dumps({"model": MODEL_NAME, "embeddings": [vec_bool]}).encode("utf-8")
        )
        with self.assertRaises(ValueError):
            client.embed_texts(["Texto"])

        # Norma cero
        vec_zero = [0.0] * DIMENSION
        mock_opener.open.return_value = MockHTTPResponse(
            json.dumps({"model": MODEL_NAME, "embeddings": [vec_zero]}).encode("utf-8")
        )
        with self.assertRaises(ValueError) as ctx:
            client.embed_texts(["Texto"])
        self.assertIn("invalid norm", str(ctx.exception))

    def test_http_connection_and_json_errors(self):
        """Comprueba manejo correcto de HTTPError, URLError, Timeout y JSON malformado."""
        mock_opener = MagicMock()
        client = OllamaEmbedder(opener=mock_opener)

        # HTTPError 500
        mock_opener.open.side_effect = urllib.error.HTTPError(
            "http://127.0.0.1:11434/api/tags", 500, "Internal Server Error", {}, io.BytesIO(b"error fatal")
        )
        with self.assertRaises(RuntimeError) as ctx:
            client.get_digest()
        self.assertIn("HTTP error 500", str(ctx.exception))

        # URLError (Conexión rechazada)
        mock_opener.open.side_effect = urllib.error.URLError("Connection refused")
        with self.assertRaises(ConnectionError):
            client.get_digest()

        # TimeoutError
        mock_opener.open.side_effect = TimeoutError("Timed out")
        with self.assertRaises(TimeoutError):
            client.get_digest()

        # JSON malformado
        mock_opener.open.side_effect = None
        mock_opener.open.return_value = MockHTTPResponse(b"NOT A JSON")
        with self.assertRaises(ValueError):
            client.get_digest()

    def test_rejects_remote_base_urls(self):
        for url in ("https://external.invalid", "http://127.0.0.1:11434@external.invalid", "http://localhost:11434"):
            with self.assertRaises(ValueError):
                OllamaEmbedder(base_url=url)

    def test_tags_validation_and_model_mismatch(self):
        """Verifica que el modelo y digest se validen contra /api/tags y /api/embed."""
        mock_opener = MagicMock()
        client = OllamaEmbedder(opener=mock_opener)

        # Modelo no presente en tags
        mock_opener.open.return_value = MockHTTPResponse(
            json.dumps({"models": [{"name": "otro_modelo:latest", "digest": "sha256:123"}]}).encode("utf-8")
        )
        with self.assertRaises(ValueError) as ctx:
            client.get_digest()
        self.assertIn("not found in Ollama /api/tags", str(ctx.exception))

        # Modelo presente pero digest vacío
        mock_opener.open.return_value = MockHTTPResponse(
            json.dumps({"models": [{"name": MODEL_NAME, "digest": "   "}]}).encode("utf-8")
        )
        with self.assertRaises(ValueError) as ctx:
            client.get_digest()
        self.assertIn("empty or invalid digest", str(ctx.exception))

        # Discordancia de modelo en respuesta de embed
        mock_opener.open.return_value = MockHTTPResponse(
            json.dumps({"model": "otro_modelo:300m", "embeddings": [_make_dummy_vector(DIMENSION, 1.0)]}).encode("utf-8")
        )
        with self.assertRaises(ValueError) as ctx:
            client.embed_texts(["Texto"])
        self.assertIn("does not match expected", str(ctx.exception))

    def test_digest_change_aborts_without_publishing(self):
        """Verifica que si el digest del modelo cambia durante la inferencia, se aborte la operación."""
        digest_calls = 0

        def fake_open(req, timeout):
            nonlocal digest_calls
            if req.full_url.endswith("/api/tags"):
                digest_calls += 1
                current_digest = "sha256:first_digest" if digest_calls == 1 else "sha256:changed_digest"
                return MockHTTPResponse(
                    json.dumps({"models": [{"name": MODEL_NAME, "digest": current_digest}]}).encode("utf-8")
                )
            if req.full_url.endswith("/api/embed"):
                return MockHTTPResponse(
                    json.dumps({"model": MODEL_NAME, "embeddings": [_make_dummy_vector(DIMENSION, 1.0)]}).encode("utf-8")
                )
            raise NotImplementedError(req.full_url)

        mock_opener = MagicMock()
        mock_opener.open.side_effect = fake_open
        client = OllamaEmbedder(opener=mock_opener)

        chunk = _make_dummy_chunk(1)
        with self.assertRaises(RuntimeError) as ctx:
            build_embeddings([chunk], client)
        self.assertIn("Model digest changed during embedding", str(ctx.exception))

    def test_chunk_validation_invariants(self):
        """Valida que entradas vacías, duplicados o campos corruptos se rechacen."""
        mock_opener = MagicMock()
        client = OllamaEmbedder(opener=mock_opener)

        # Chunks vacíos
        with self.assertRaises(ValueError):
            build_embeddings([], client)

        # Duplicado de chunk_id
        c1 = _make_dummy_chunk(1)
        c2 = _make_dummy_chunk(1)
        with self.assertRaises(ValueError) as ctx:
            build_embeddings([c1, c2], client)
        self.assertIn("Duplicate chunk_id", str(ctx.exception))

        # Texto vacío
        c_empty_text = ChunkRecord(
            chunk_id="c_empty",
            doc_id="d1",
            source_file="f1.pdf",
            published_at="2025-03-31",
            page_number=1,
            start_char=0,
            end_char=10,
            text="   ",
            strategy="strat",
        )
        with self.assertRaises(ValueError) as ctx:
            validate_chunk_record(c_empty_text)
        self.assertIn("text must be a non-empty string", str(ctx.exception))

        # Offsets incoherentes (start_char >= end_char)
        c_bad_offsets = ChunkRecord(
            chunk_id="c_bad",
            doc_id="d1",
            source_file="f1.pdf",
            published_at="2025-03-31",
            page_number=1,
            start_char=50,
            end_char=50,
            text="Texto",
            strategy="strat",
        )
        with self.assertRaises(ValueError):
            validate_chunk_record(c_bad_offsets)


class TestCLIIntegrationOffline(unittest.TestCase):
    """Pruebas de la CLI con mocks de transporte y 13 filas simuladas."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_fake_jsonl(self, count: int = 13) -> Path:
        """Crea un archivo JSONL con `count` registros de chunk válidos."""
        jsonl_path = self.temp_path / "fake_chunks.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i in range(1, count + 1):
                rec = _make_dummy_chunk(i, text=f"Texto del fragmento número {i} para RAG.")
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        return jsonl_path

    def test_cli_metadata_order_and_13_fake_rows(self):
        """Ejecuta la CLI con 13 fragmentos simulados verificando metadatos, orden y hashes."""
        input_path = self._create_fake_jsonl(13)
        output_path = self.temp_path / "fake_embeddings.json"

        expected_digest = "sha256:777888999aaa"

        def fake_open(req, timeout):
            if req.full_url.endswith("/api/tags"):
                return MockHTTPResponse(
                    json.dumps({"models": [{"name": MODEL_NAME, "digest": expected_digest}]}).encode("utf-8")
                )
            if req.full_url.endswith("/api/embed"):
                body = json.loads(req.data.decode("utf-8"))
                num_texts = len(body["input"])
                embeddings = [_make_dummy_vector(DIMENSION, 1.0 + (idx * 0.01)) for idx in range(num_texts)]
                return MockHTTPResponse(json.dumps({"model": MODEL_NAME, "embeddings": embeddings}).encode("utf-8"))
            raise NotImplementedError(req.full_url)

        with patch.object(urllib.request.OpenerDirector, "open", side_effect=fake_open):
            exit_code = main(["--input", str(input_path), "--output", str(output_path)])
            self.assertEqual(exit_code, 0)

        self.assertTrue(output_path.exists())
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validar contrato del esquema JSON de salida
        self.assertEqual(data["schema_version"], SCHEMA_VERSION)
        self.assertEqual(data["model"], MODEL_NAME)
        self.assertEqual(data["model_digest"], expected_digest)
        self.assertEqual(data["dimension"], DIMENSION)
        self.assertEqual(data["document_prefix"], DOCUMENT_PREFIX)
        self.assertEqual(data["query_prefix"], QUERY_PREFIX)

        # Validar hash SHA-256 de entrada calculado una sola vez
        expected_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
        self.assertEqual(data["input_sha256"], expected_sha256)

        # Validar los 13 fragmentos
        chunks = data["chunks"]
        self.assertEqual(len(chunks), 13)
        for i, chunk in enumerate(chunks, start=1):
            self.assertEqual(chunk["chunk_id"], f"chunk_{i:04d}")
            self.assertEqual(chunk["text"], f"Texto del fragmento número {i} para RAG.")
            self.assertIn("embedding", chunk)
            self.assertEqual(len(chunk["embedding"]), DIMENSION)
            norm = math.hypot(*chunk["embedding"])
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_cli_malformed_jsonl_and_midflight_failure_preserve_previous_output(self):
        """Verifica que errores en JSONL o durante la inferencia conservan intacta la salida anterior."""
        output_path = self.temp_path / "existing_embeddings.json"
        sentinel_content = '{"preservar": "contenido_previo_intacto"}'
        output_path.write_text(sentinel_content, encoding="utf-8")

        # Escenario 1: JSONL con línea malformada
        bad_jsonl = self.temp_path / "bad.jsonl"
        bad_jsonl.write_text(
            '{"chunk_id": "c1", "doc_id": "d1", "source_file": "s", "published_at": "2025-03-31", '
            '"page_number": 1, "start_char": 0, "end_char": 10, "text": "t", "strategy": "strat"}\n'
            'ESTA LINEA NO ES JSON\n',
            encoding="utf-8",
        )

        with self.assertRaises(ValueError):
            main(["--input", str(bad_jsonl), "--output", str(output_path)])
        self.assertEqual(output_path.read_text(encoding="utf-8"), sentinel_content)

        # Escenario 2: Fallo a mitad de inferencia en Ollama
        good_jsonl = self._create_fake_jsonl(2)

        def failing_open(req, timeout):
            if req.full_url.endswith("/api/tags"):
                return MockHTTPResponse(
                    json.dumps({"models": [{"name": MODEL_NAME, "digest": "sha256:abc"}]}).encode("utf-8")
                )
            if req.full_url.endswith("/api/embed"):
                raise urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", {}, io.BytesIO(b"down"))
            raise NotImplementedError(req.full_url)

        with patch.object(urllib.request.OpenerDirector, "open", side_effect=failing_open):
            with self.assertRaises(RuntimeError):
                main(["--input", str(good_jsonl), "--output", str(output_path)])

        self.assertEqual(output_path.read_text(encoding="utf-8"), sentinel_content)

    def test_cli_rejects_external_paths_and_same_input_output(self):
        """Verifica validación de rutas fuera del proyecto y coincidencia de input/output."""
        fake_input = self._create_fake_jsonl(1)
        external_path = Path("C:/ruta_externa_prohibida/salida.json")

        with self.assertRaises(ValueError) as ctx:
            main(["--input", str(fake_input), "--output", str(external_path)])
        self.assertIn("outside project root", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            main(["--input", str(fake_input), "--output", str(fake_input)])
        self.assertIn("Output path cannot be the same as input path", str(ctx.exception))

    def test_cli_cleanup_only_own_temp_files(self):
        """Verifica que solo se borran temporales propios y no archivos ajenos con prefijo similar."""
        unrelated_temp = self.temp_path / "custom_output.json.tmp_otro_proceso"
        unrelated_temp.write_text("no me toques", encoding="utf-8")

        target_output = self.temp_path / "custom_output.json"
        bad_jsonl = self.temp_path / "corrupt.jsonl"
        bad_jsonl.write_text("invalido", encoding="utf-8")

        with self.assertRaises(ValueError):
            main(["--input", str(bad_jsonl), "--output", str(target_output)])

        # El archivo temporal no relacionado debe seguir intacto
        self.assertTrue(unrelated_temp.exists())
        self.assertEqual(unrelated_temp.read_text(encoding="utf-8"), "no me toques")
        # El archivo de salida nunca llegó a crearse
        self.assertFalse(target_output.exists())


if __name__ == "__main__":
    unittest.main()
