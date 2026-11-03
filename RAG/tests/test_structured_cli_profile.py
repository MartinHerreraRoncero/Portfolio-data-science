"""CLI routing only; actual correction records are covered by replay integration tests."""
import contextlib
import inspect
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import continuous_quality as quality
from experiments import query_coverage as covered
from experiments import query_preserved as query
from rag_pensiones.ingest import PROJECT_ROOT


class StructuredCliProfileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=PROJECT_ROOT / 'tmp')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.index = self.root / 'index.json'
        self.units = self.root / 'units.json'
        self.pages = self.root / 'pages.jsonl'
        self.index.write_text('{}', encoding='utf-8')
        self.units.write_text('{}', encoding='utf-8')
        self.pages.write_text('', encoding='utf-8')
        self.args = ['--query', 'Pregunta de control', '--index', str(self.index),
                     '--units', str(self.units), '--pages', str(self.pages),
                     '--output', str(self.root / 'answer.json')]

    def query_cli(self, flags):
        report = {'status': 'insufficient_evidence', 'outcome': {'status': 'insufficient_evidence'}}
        with patch.object(query, 'load_index', return_value={}), \
             patch.object(query, 'Q3VerifierClient') as verifier, \
             patch.object(query, 'answer_preserved', return_value=report) as historical, \
             patch.object(query, 'checkpoint'), \
             patch.object(query, 'format_answer', return_value='Respuesta histórica'), \
             patch.object(covered, 'answer_covered', return_value=report) as hybrid, \
             patch.object(covered, 'format_covered_answer', return_value='Respuesta híbrida'), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(query.legacy_main(self.args + flags), 0)
        return hybrid, historical, verifier

    def test_covered_cli_default_optin_and_optout_pass_strict_bools(self):
        for flags, expected in [([], True), (['--coverage-correction'], True),
                                (['--no-coverage-correction'], False)]:
            with self.subTest(flags=flags):
                hybrid, historical, verifier = self.query_cli(flags)
                self.assertIs(hybrid.call_args.kwargs['coverage_correction'], expected)
                historical.assert_not_called()
                verifier.assert_called_once()

    def test_historical_query_modes_keep_their_path_without_automatic_correction(self):
        for mode in ['--factual-only', '--legacy-verifier']:
            for flags in [[], ['--no-coverage-correction']]:
                with self.subTest(mode=mode, flags=flags):
                    hybrid, historical, verifier = self.query_cli([mode] + flags)
                    hybrid.assert_not_called()
                    historical.assert_called_once()
                    self.assertNotIn('coverage_correction', historical.call_args.kwargs)
                    if mode == '--legacy-verifier':
                        self.assertIsNone(historical.call_args.kwargs['verifier'])
                        verifier.assert_not_called()
                    else:
                        verifier.assert_called_once()

    def test_explicit_positive_conflict_fails_before_paths_or_clients(self):
        for mode in ['--factual-only', '--legacy-verifier']:
            with self.subTest(mode=mode), \
                 patch.object(query, 'output_path') as destination, \
                 patch.object(query, 'load_index') as loader, \
                 patch.object(query, 'Q3VerifierClient') as verifier, \
                 patch.object(query, 'answer_preserved') as historical, \
                 patch.object(covered, 'answer_covered') as hybrid, \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as rejected:
                    query.legacy_main(['--query', 'Control', mode, '--coverage-correction'])
                self.assertEqual(rejected.exception.code, 2)
                for mock in [destination, loader, verifier, historical, hybrid]:
                    mock.assert_not_called()

    def test_quality_run_selects_correct_actual_correction_and_source_modes(self):
        cases = [([], True, True), (['--coverage-correction'], True, True),
                 (['--no-coverage-correction'], False, True),
                 (['--legacy-semantic'], False, False),
                 (['--legacy-semantic', '--coverage-correction'], True, False),
                 (['--legacy-semantic', '--no-coverage-correction'], False, False)]
        for flags, correction, structured in cases:
            with self.subTest(flags=flags), patch.object(quality, 'run', return_value={'overall_pass': True}) as run:
                self.assertEqual(quality.legacy_main(['run', '--output-dir', str(self.root / 'run')] + flags), 0)
                self.assertIs(run.call_args.kwargs['coverage_correction'], correction)
                self.assertIs(run.call_args.kwargs['structured_retrieval'], structured)

    def test_prepare_cli_and_python_api_defaults_remain_explicit_compatibility(self):
        with patch.object(quality, 'prepare') as prepare:
            self.assertEqual(quality.legacy_main(['prepare', '--output-dir', str(self.root / 'protocol')]), 0)
            self.assertIs(prepare.call_args.kwargs.get('coverage_correction', False), False)
        for function in [quality.run, quality.prepare, covered.answer_covered]:
            with self.subTest(function=function.__name__):
                self.assertIs(inspect.signature(function).parameters['coverage_correction'].default, False)


if __name__ == '__main__':
    unittest.main()
