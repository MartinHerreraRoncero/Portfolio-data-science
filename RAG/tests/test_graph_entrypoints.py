"""Public entrypoints route to the active 9B graph without model execution."""
import contextlib
import inspect
import io
import unittest
from unittest.mock import patch

from experiments import continuous_quality,graph_quality,query_graph,query_preserved
from rag_pensiones.graph_types import CHAT_MODEL


class GraphEntrypointTests(unittest.TestCase):
    def test_public_mains_forward_exact_arguments_and_return_values(self):
        for public,target in [(query_preserved,query_graph),(continuous_quality,graph_quality)]:
            arguments=['unchanged','arguments']
            with self.subTest(module=public.__name__),patch.object(target,'main',return_value=37)as forward, \
                 patch.object(public,'legacy_main',side_effect=AssertionError('Archived route called')):
                self.assertEqual(public.main(arguments),37)
                forward.assert_called_once_with(arguments)

    def test_query_cli_uses_graph_without_initializing_historical_27b_client(self):
        services=object();report={'status':'answered'}
        with patch.object(query_preserved,'Q3VerifierClient',side_effect=AssertionError('27B initialization')), \
             patch.object(query_graph,'load_default_services',return_value=services), \
             patch.object(query_graph,'answer_graph',return_value=report)as answer, \
             patch.object(query_graph,'format_graph_answer',return_value='Graph9B'), \
             contextlib.redirect_stdout(io.StringIO())as output:
            self.assertEqual(query_preserved.main(['--query','Gasto en pensiones']),0)
        self.assertEqual(CHAT_MODEL,'qwen3.5:9b')
        self.assertIs(answer.call_args.kwargs['services'],services)
        self.assertEqual(answer.call_args.args,('Gasto en pensiones',))
        self.assertIn('Graph9B',output.getvalue())

    def test_public_help_is_graph_help_and_has_no_inference(self):
        for public,expected in [(query_preserved,'--thread-id'),(continuous_quality,'LangGraph')]:
            with self.subTest(module=public.__name__),contextlib.redirect_stdout(io.StringIO())as output, \
                 patch.object(query_graph,'load_default_services',side_effect=AssertionError('Help loaded models')), \
                 patch.object(graph_quality,'run',side_effect=AssertionError('Help evaluated models')):
                with self.assertRaises(SystemExit)as exit_:public.main(['--help'])
                self.assertEqual(exit_.exception.code,0)
                self.assertIn(expected,output.getvalue())
        with contextlib.redirect_stdout(io.StringIO())as output:
            with self.assertRaises(SystemExit)as exit_:continuous_quality.main(['run','--help'])
        self.assertEqual(exit_.exception.code,0)
        self.assertIn('--suite',output.getvalue());self.assertIn('--asset-dir',output.getvalue())

    def test_archived_main_signatures_remain_explicitly_available(self):
        for module in [query_preserved,continuous_quality]:
            self.assertEqual(list(inspect.signature(module.legacy_main).parameters),['argv'])
            self.assertIsNone(inspect.signature(module.legacy_main).parameters['argv'].default)
            self.assertIsNot(module.main,module.legacy_main)


if __name__=='__main__':unittest.main()
