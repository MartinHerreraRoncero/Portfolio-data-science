"""Evaluator integrity tests; no model calls or dependence on expected LLM wording."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import graph_quality as q


class GraphQualityTests(unittest.TestCase):
    def test_actual_embedding_lifecycle_matches_frozen_protocol_without_changing_role_or_vectors(self):
        call=dict(stage='embedding',endpoint='/api/embed',model=q.EMBED_MODEL,attempted=True,error=None,
            digest_before=q.EMBED_DIGEST,digest_after=q.EMBED_DIGEST,
            request=dict(model=q.EMBED_MODEL,input=['Synthetic embedding input'],dimensions=q.EMBED_DIMENSION,
                         truncate=False,keep_alive=q.EMBED_KEEP_ALIVE,options=q.model_lifecycle_policy()['embedding_options']),
            response=dict(model=q.EMBED_MODEL,embeddings=[[1.]+[0.]*(q.EMBED_DIMENSION-1)]))
        before=copy.deepcopy(call);q._validate_calls(dict(stage_calls=[call]));self.assertEqual(call,before)
        for value in ['5m',None,0,False]:
            changed=copy.deepcopy(call);changed['request']['keep_alive']=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'lifecycle'):
                q._validate_calls(dict(stage_calls=[changed]))
        changed=copy.deepcopy(call);changed['request']['options']['num_gpu']=1
        with self.assertRaisesRegex(ValueError,'placement'):q._validate_calls(dict(stage_calls=[changed]))
        out,protocol,_=self.fixture()
        self.assertEqual(protocol['model_lifecycle_policy'],q.model_lifecycle_policy())
        changed=copy.deepcopy(protocol);changed['model_lifecycle_policy']['embedding_keep_alive']='5m'
        with self.assertRaisesRegex(ValueError,'identity mismatch'):q._validate_protocol(changed,self.services)
        settings=q.stage_chat_settings('intent_audit')
        extract=dict(stage='intent_audit',endpoint='/api/chat',model=q.CHAT_MODEL,attempted=True,error=None,
            digest_before=q.CHAT_DIGEST,digest_after=q.CHAT_DIGEST,
            request=dict(model=q.CHAT_MODEL,stream=False,keep_alive=q.chat_keep_alive('intent_audit'),**settings),
            response=dict(model=q.CHAT_MODEL,done=True,done_reason='stop',prompt_eval_count=64,eval_count=4))
        q._validate_calls(dict(stage_calls=[extract]))
        extract['request']['keep_alive']='5m'
        with self.assertRaisesRegex(ValueError,'lifecycle'):q._validate_calls(dict(stage_calls=[extract]))

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=q.PROJECT_ROOT/'tmp')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source=self.root/'bound.py';self.source.write_text('# fixed source\n',encoding='utf-8')
        self.services=SimpleNamespace(scope='legacy_2025_reference',documents={'pages':[]})
        self.manifest={'reference_text_only':True,'exposed_card_ids':[],'scope':self.services.scope}
        self.addCleanup(patch.stopall)
        patch.object(q,'_code_files',side_effect=lambda:{q._rel(self.source):q._sha(self.source)}).start()
        patch.object(q,'_dependencies',return_value={'test-dependency':'1'}).start()
        patch.object(q,'_runtime_manifest',side_effect=lambda s:copy.deepcopy(self.manifest)).start()
        self.replay=patch.object(q,'_runtime_replay',return_value={'validated':True}).start()
        patch.object(q,'_format',side_effect=lambda report:json.dumps(report['publication'],sort_keys=True)).start()
        self.rubric=patch.object(q,'evaluate_case_reference',side_effect=lambda *args:
            dict(whole_correct=True,main_direct_correct=True,safe=True,direct=True,warnings=[],failures=[])).start()

    def fixture(self,suite='legacy_reference'):
        out=self.root/'run'
        if suite!='legacy_reference':
            self.services.scope=q.SCOPES[suite]
            self.manifest.update(reference_text_only=False,scope=self.services.scope)
        protocol=q.prepare(out,services=self.services,suite=suite)
        entries=[]
        for case in protocol['cases']:
            report=dict(schema_version='langgraph-rag-report-v1',profile=q.PROFILE,
                query=case['query'],context=case['context'],status='answered',
                publication={'status':'answered','claims':[],'missing_parts':[],'reason':''},
                numeric_facts=[],stage_calls=[],manifest=copy.deepcopy(self.manifest))
            entries.append(dict(case,attempted=True,error=None,answer=report,
                formatted_answer=q._format(report),wall_seconds=0.1))
        results=dict(schema_version=q.RESULTS,profile=q.PROFILE,run_status='completed',
            protocol_sha256=q._sha(out/'protocol.json'),cases=entries)
        path=out/'results.json';q._write_new(path,results)
        return out,protocol,results

    def rewrite(self,path,value):
        path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')

    def test_prepare_exposes_only_query_context_not_oracle_and_pins_baseline(self):
        out,p,_=self.fixture()
        self.assertEqual(set(p['cases'][0]),{'id','query','context'})
        self.assertEqual(p['chat_model'],q.CHAT_MODEL)
        self.assertEqual(p['baseline_sha256'],q._sha(q.DEFAULT_BASELINE))
        self.assertTrue(p['reference_text_only'])
        self.assertEqual(p['runtime_manifest']['exposed_card_ids'],[])
        with self.assertRaises(FileExistsError):q.prepare(out,services=self.services)

    def test_check_calls_lineage_replay_before_pure_rubric_and_keeps_metrics(self):
        out,_,_=self.fixture()
        score=q.check(out/'results.json',services=self.services)
        self.assertTrue(score['overall_pass'])
        self.assertEqual(score['execution_kind'],'injected_test_services')
        self.assertFalse(score['adoption_eligible'])
        self.assertEqual(score['summary']['whole_correct_count'],11)
        self.assertEqual(self.replay.call_count,11)
        self.assertEqual(self.rubric.call_count,11)
        self.assertIn('text-only',score['warning'])

    def test_query_context_and_format_tampering_are_rejected(self):
        out,_,original=self.fixture()
        for target in ('query','context','formatted'):
            data=copy.deepcopy(original)
            if target=='query':data['cases'][0]['answer']['query']='Otra pregunta'
            elif target=='context':data['cases'][0]['answer']['context']={'report_year':'2026'}
            else:data['cases'][0]['formatted_answer']='Cifra inventada'
            self.rewrite(out/'results.json',data)
            with self.subTest(target=target),self.assertRaises(ValueError):q.check(out/'results.json',services=self.services)

    def test_runtime_lineage_failure_is_not_a_quality_pass(self):
        out,_,_=self.fixture();self.replay.side_effect=ValueError('Tampered source cell')
        with self.assertRaisesRegex(ValueError,'Tampered source'):q.check(out/'results.json',services=self.services)

    def test_technical_failure_retains_denominator_and_never_becomes_abstention(self):
        out,_,data=self.fixture();data['cases'][0]['error']='TimeoutError: local model';data['cases'][0]['answer']=None
        self.rewrite(out/'results.json',data)
        score=q.check(out/'results.json',services=self.services)
        self.assertFalse(score['overall_pass']);self.assertEqual(score['status'],'technical_error')
        self.assertEqual(score['summary']['total_cases'],11);self.assertEqual(len(score['case_results']),11)
        self.assertFalse(score['case_results'][0]['safe'])
        self.assertEqual(score['technical_errors'],['b01'])

    def test_all_natural_technical_failures_keep_all_24_frozen_source_groups(self):
        out,_,data=self.fixture(suite='natural_dev')
        for entry in data['cases']:
            entry.update(error='TimeoutError: local model',answer=None)
        self.rewrite(out/'results.json',data)
        score=q.check(out/'results.json',services=self.services)
        self.assertFalse(score['overall_pass']);self.assertEqual(score['status'],'technical_error')
        self.assertEqual(score['summary']['total_cases'],32)
        self.assertEqual(score['summary']['required_source_group_count'],24)
        self.assertEqual(score['summary']['passed_count'],0)
        self.assertEqual(score['summary']['safe_count'],0)
        self.assertEqual(len(score['technical_errors']),32)
        for key in ('retrieval_source_recall','selection_source_recall','exact_fact_recovery'):
            self.assertEqual(score['summary'][key],0.0)
        for row in score['case_results']:
            self.assertFalse(row['passed']);self.assertFalse(row['safe'])
            for key in ('retrieved_source_group_count','selected_source_group_count','matched_required_fact_count'):
                self.assertEqual(row[key],0)
        byid={r['case_id']:r for r in score['case_results']}
        self.assertEqual(byid['d30']['required_source_group_count'],2)
        self.assertEqual(byid['d22']['required_source_group_count'],0)
        self.replay.assert_not_called();self.rubric.assert_not_called()

    def test_one_natural_technical_failure_cannot_shrink_recall_denominator(self):
        out,_,data=self.fixture(suite='natural_dev')
        data['cases'][0].update(error='TimeoutError: local model',answer=None)
        # A later case retrieved its known source. Its retrieval credit must
        # still be measured against all24 source groups, including the timeout.
        data['cases'][1]['answer']['retrieval']=dict(catalog=[dict(
            card_id='fixture:2026:pensions',observation_ids=['airef:2026:388'])])
        self.rewrite(out/'results.json',data)
        score=q.check(out/'results.json',services=self.services)
        self.assertEqual(score['technical_errors'],['d01'])
        self.assertEqual(score['summary']['total_cases'],32)
        self.assertEqual(score['summary']['required_source_group_count'],24)
        self.assertAlmostEqual(score['summary']['retrieval_source_recall'],1/24)
        self.assertEqual(score['case_results'][0]['required_source_group_count'],1)
        self.assertEqual(score['case_results'][0]['retrieved_source_group_count'],0)
        self.assertEqual(self.replay.call_count,31)
        self.assertFalse(score['overall_pass']);self.assertFalse(score['adoption_eligible'])

    def test_models_and_options_are_not_relabelled(self):
        out,_,original=self.fixture()
        call=dict(endpoint='/api/chat',model=q.CHAT_MODEL,attempted=True,error=None,
            digest_before=q.CHAT_DIGEST,digest_after=q.CHAT_DIGEST,
            request=dict(model=q.CHAT_MODEL,options=q.CHAT_OPTIONS,stream=False,think=False,keep_alive=q.chat_keep_alive('default')),
            response=dict(model=q.CHAT_MODEL))
        for mutate in ('model','response','options','digest'):
            data=copy.deepcopy(original);c=copy.deepcopy(call)
            if mutate=='model':c['model']='hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M'
            elif mutate=='response':c['response']['model']='other'
            elif mutate=='options':c['request']['options']['num_ctx']=4096
            else:c['digest_after']='0'*64
            data['cases'][0]['answer']['stage_calls']=[c];self.rewrite(out/'results.json',data)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):q.check(out/'results.json',services=self.services)

    def test_changed_sources_or_manifest_invalidate_evaluation(self):
        out,_,data=self.fixture()
        self.source.write_text('# changed\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'drift'):q.check(out/'results.json',services=self.services)

    def test_uniform_stage_settings_and_actual_token_bounds_are_enforced(self):
        settings=q.stage_chat_settings('plan_check_batch')
        call=dict(stage='plan_check_batch',endpoint='/api/chat',model=q.CHAT_MODEL,
            attempted=True,error=None,digest_before=q.CHAT_DIGEST,digest_after=q.CHAT_DIGEST,
            request=dict(model=q.CHAT_MODEL,stream=False,keep_alive=q.chat_keep_alive('default'),**settings),
            response=dict(model=q.CHAT_MODEL,done=True,done_reason='stop',
                prompt_eval_count=1000,eval_count=1024))
        q._validate_calls(dict(stage_calls=[call]))
        self.assertEqual(settings,q.stage_chat_settings('generate'))
        for mutation in ('think','options','truncated','tokens','missing_tokens'):
            changed=copy.deepcopy(call)
            if mutation=='think':changed['request']['think']=True
            elif mutation=='options':changed['request']['options']['num_predict']=4096
            elif mutation=='truncated':changed['response']['done_reason']='length'
            elif mutation=='tokens':changed['response']['eval_count']=1025
            else:changed['response'].pop('prompt_eval_count')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                q._validate_calls(dict(stage_calls=[changed]))

    def test_default_stage_rejects_expanded_output_or_context_overflow(self):
        settings=q.stage_chat_settings('generate')
        call=dict(stage='generate',endpoint='/api/chat',model=q.CHAT_MODEL,
            attempted=True,error=None,digest_before=q.CHAT_DIGEST,digest_after=q.CHAT_DIGEST,
            request=dict(model=q.CHAT_MODEL,stream=False,keep_alive=q.chat_keep_alive('default'),**settings),
            response=dict(model=q.CHAT_MODEL,done=True,done_reason='stop',
                prompt_eval_count=1000,eval_count=1024))
        q._validate_calls(dict(stage_calls=[call]))
        for key,value in [('eval_count',1025),('prompt_eval_count',8192)]:
            changed=copy.deepcopy(call);changed['response'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                q._validate_calls(dict(stage_calls=[changed]))

    def test_missing_cases_unknown_protocol_fields_and_overwrite_rejected(self):
        out,p,data=self.fixture();data['cases'].pop();self.rewrite(out/'results.json',data)
        with self.assertRaises(ValueError):q.check(out/'results.json',services=self.services)
        p['unexpected_override']=True
        self.rewrite(out/'protocol.json',p)
        with self.assertRaises(ValueError):q.check(out/'results.json',services=self.services)
        with self.assertRaises(FileExistsError):q._write_new(out/'results.json',{})

    def test_true_baseline_metrics_may_not_regress(self):
        baseline=q._read(q.DEFAULT_BASELINE)
        rows=[{k:c[k]for k in ('case_id','whole_correct','main_direct_correct','safe')}for c in baseline['case_results']]
        self.assertEqual(q._baseline_regressions(rows,baseline,baseline['reference_sha256']),[])
        rows[0]['whole_correct']=False
        self.assertIn('b01:whole_correct',q._baseline_regressions(rows,baseline,baseline['reference_sha256']))
        rows[0]['safe']=1
        with self.assertRaises(ValueError):q._baseline_regressions(rows,baseline,baseline['reference_sha256'])

    def test_legacy_query_cannot_change_while_reusing_original_gold(self):
        cases,_,_=q.load_cases('legacy_reference');cases[0]['query']='A different easier question'
        with self.assertRaisesRegex(ValueError,'Legacy question'):
            q._validate_reference_inputs(cases,'legacy_reference',q.DEFAULT_BASELINE,q.DEFAULT_REFERENCE_REL)

    def test_reserved_loader_denies_implicit_release_without_reading_file(self):
        with patch.object(q,'_read')as read:
            with self.assertRaises(PermissionError):q.load_cases('natural_heldout')
            read.assert_not_called()

    def test_natural_oracles_do_not_reach_public_case_inputs(self):
        cases,labels,_=q.load_cases('natural_dev')
        self.assertEqual(len(cases),32)
        self.assertTrue(labels['d01']['required_facts'])
        self.assertTrue(all(set(c)=={'id','query','context'}for c in cases))
        self.assertTrue(all('cuadro'not in c['query'].lower()and'gráfico'not in c['query'].lower()for c in cases))

    def test_natural_fact_exactness_warnings_and_unsafe_substitution(self):
        case=dict(id='synthetic',query='Dato',context=[])
        fact=dict(observation_id='f:1',value_exact='1.2500',unit='percent_gdp')
        expected=dict(behavior='answer',required_facts=[fact])
        report=dict(status='answered',numeric_facts=[dict(observation_id='f:1',value='1.25',unit='percent_gdp')],
            publication={'claims':[{'text':'Dato fuente','evidence':[{'observation_id':'f:1'}]}]})
        self.assertTrue(q._natural_grade(case,expected,report)['passed'])
        report['numeric_facts'][0]['value']='1.251'
        self.assertFalse(q._natural_grade(case,expected,report)['passed'])
        negative=dict(behavior='population_fidelity',forbidden_substitute_facts=['f:1'],allowed_statuses=['insufficient_evidence'])
        self.assertFalse(q._natural_grade(case,negative,report)['safe'])
        report.update(status='insufficient_evidence',numeric_facts=[],publication={'claims':[]})
        self.assertTrue(q._natural_grade(case,negative,report)['passed'])

    def test_private_fact_without_public_citation_does_not_answer_question(self):
        f=dict(observation_id='f:1',value_exact='12',unit='percent_gdp')
        report=dict(status='answered',numeric_facts=[dict(observation_id='f:1',value='12',unit='percent_gdp')],publication={'claims':[]})
        self.assertFalse(q._natural_grade(dict(id='x'),dict(behavior='answer',required_facts=[f]),report)['passed'])

    def test_natural_extra_source_requires_review_not_automatic_success(self):
        case=dict(id='x',query='Otra unidad',context=[])
        expected=dict(behavior='unit_fidelity',forbidden_substitute_facts=[],allowed_statuses=['insufficient_evidence'])
        report=dict(status='answered',numeric_facts=[],publication={})
        result=q._natural_grade(case,expected,report)
        self.assertFalse(result['passed']);self.assertTrue(result['review_required'])

    def test_equivalent_source_label_requires_independent_vintage_review(self):
        wanted=dict(observation_id='oracle:1',value_exact='16.125',unit='percent_gdp',
            book='2025',metric='Pensiones',series='Opinión 2025',year=2050)
        got=dict(wanted,observation_id='other:1',series='AIReF 2025',value='16.125',period='2050')
        expected=dict(behavior='answer',required_facts=[wanted])
        report=dict(status='answered',numeric_facts=[got],publication={'claims':[
            {'text':'Dato de la fuente alternativa','evidence':[{'observation_id':'other:1'}]}]})
        case=dict(id='x',query='Según AIReF en 2025',context=[])
        row=q._natural_grade(case,expected,report)
        self.assertFalse(row['passed']);self.assertEqual(row['failures'],[])
        self.assertTrue(row['review_required']);self.assertTrue(row['equivalence_candidates'])
        for key,value in [('value','16.126'),('unit','eur'),('book','2026'),('period','2070')]:
            with self.subTest(key=key):
                bad=copy.deepcopy(report);bad['numeric_facts'][0][key]=value
                self.assertTrue(q._natural_grade(case,expected,bad)['failures'])
        expected['allow_verified_equivalent_source']=False
        row=q._natural_grade(case,expected,report)
        self.assertEqual(row['failures'],[]);self.assertTrue(row['review_required']);self.assertFalse(row['passed'])

    def test_alternative_metric_label_without_gold_flag_requires_independent_review(self):
        wanted=dict(observation_id='oracle:1',value_exact='16.125',unit='percent_gdp',
            book='2025',metric='Pensiones',series='Opinión 2025',year=2050)
        got=dict(wanted,observation_id='other:1',metric='Evolución del gasto en pensiones (%PIB)',
            series='AIReF 2025',value='16.125',period='2050')
        expected=dict(behavior='answer',required_facts=[wanted])
        report=dict(status='answered',numeric_facts=[got],publication={'claims':[
            {'text':'Dato de fuente alternativa','evidence':[{'observation_id':'other:1'}]}]})
        case=dict(id='metric_labels',query='Previsión de gasto en pensiones',context=[])
        for label in ('Evolución del gasto en pensiones (%PIB)','Intereses'):
            report['numeric_facts'][0]['metric']=label
            row=q._natural_grade(case,expected,report)
            self.assertFalse(row['passed']);self.assertEqual(row['failures'],[])
            self.assertTrue(row['review_required'])
            requirement=row['equivalence_candidates'][0]
            self.assertEqual(requirement['expected_metric'],'Pensiones')
            self.assertEqual(requirement['candidates'][0]['metric'],label)
        for key,value in [('value','16.126'),('unit','eur'),('book','2026'),('period','2070'),('derived',True)]:
            with self.subTest(key=key):
                bad=copy.deepcopy(report);bad['numeric_facts'][0][key]=value
                self.assertTrue(q._natural_grade(case,expected,bad)['failures'])
        report['publication']['claims']=[]
        self.assertTrue(q._natural_grade(case,expected,report)['failures'])
        report['publication']['claims']=[{'text':'Published','evidence':[{'observation_id':'other:1'}]}]
        expected['allow_verified_equivalent_source']=False
        row=q._natural_grade(case,expected,report)
        self.assertEqual(row['failures'],[]);self.assertTrue(row['review_required']);self.assertFalse(row['passed'])

    def test_derived_or_supporting_fact_cannot_recover_even_the_exact_required_id(self):
        wanted=dict(observation_id='oracle:1',value_exact='16.125',unit='percent_gdp',
            book='2025',metric='Pensiones',series='Opinión 2025',year=2050)
        expected=dict(behavior='answer',required_facts=[wanted])
        for oid in ('oracle:1','other:1'):
            for flag in ('derived','supporting_only'):
                with self.subTest(observation_id=oid,flag=flag):
                    fact=dict(wanted,observation_id=oid,value='16.125',period='2050',**{flag:True})
                    report=dict(status='answered',numeric_facts=[fact],publication={'claims':[
                        dict(text='Published value',evidence=[dict(observation_id=oid)])]})
                    row=q._natural_grade(dict(id='non_primary'),expected,report)
                    self.assertTrue(row['failures']);self.assertEqual(row['equivalence_candidates'],[])
                    self.assertEqual(row['matched_required_fact_count'],0);self.assertFalse(row['passed'])

    def test_alternative_source_cannot_bypass_required_public_warning(self):
        warning='Total publicado no reconciliado con sus componentes.'
        wanted=dict(observation_id='oracle:warn',value_exact='16.125',unit='percent_gdp',
            book='2025',metric='Pensiones',series='Opinión 2025',year=2050,
            numeric_reconciliation_failed=True)
        got=dict(wanted,observation_id='other:warn',metric='Evolución del gasto en pensiones (%PIB)',
            series='AIReF 2025',value='16.125',period='2050',warning=warning,calculation_safe=False)
        citation=dict(observation_id='other:warn',source_kind='structured_cell',source_id='DB:other:warn',
            source_path='source.xlsx',source_sheet='Source',source_cell='F8',source_sha256='a'*64)
        report=dict(status='answered',numeric_facts=[got],publication=dict(status='answered',missing_parts=[],claims=[
            dict(text='Dato de fuente alternativa. Aviso: '+warning,evidence=[citation])]))
        expected=dict(behavior='answer',required_facts=[wanted])
        case=dict(id='mandatory_warning',query='Total publicado',context=[])
        row=q._natural_grade(case,expected,report)
        self.assertEqual(row['failures'],[]);self.assertTrue(row['review_required'])
        for field,value in [('warning',None),('calculation_safe',True),('calculation_safe',None)]:
            with self.subTest(field=field,value=value):
                bad=copy.deepcopy(report);bad['numeric_facts'][0][field]=value
                row=q._natural_grade(case,expected,bad)
                self.assertTrue(row['failures']);self.assertEqual(row['equivalence_candidates'],[])
        report['publication']['claims'][0]['text']='Dato alternativo sin aviso visible.'
        row=q._natural_grade(case,expected,report)
        self.assertTrue(row['failures']);self.assertEqual(row['equivalence_candidates'],[])

    def test_supporting_source_does_not_recover_the_required_main_fact(self):
        wanted=dict(observation_id='oracle:main',value_exact='16.125',unit='percent_gdp',
            book='2025',metric='Pensiones',series='Opinión 2025',year=2050)
        main=dict(observation_id='other:main',value='4.5',unit='percent_gdp',
            book='2025',metric='Intereses',series='Opinión 2025',period='2050')
        support=dict(wanted,value='16.125',period='2050',supporting_only=True)
        expected=dict(behavior='answer',required_facts=[wanted])
        report=dict(status='answered',numeric_facts=[main,support],publication={'claims':[
            dict(text='Intereses: 4.5. Nota de fuente: Pensiones 16.125.',evidence=[
                dict(observation_id='other:main'),dict(observation_id='oracle:main',supporting_only=True)])]})
        original=copy.deepcopy(report)
        row=q._natural_grade(dict(id='support_is_not_main'),expected,report)
        self.assertFalse(row['passed']);self.assertIn('missing_exact_fact:oracle:main',row['failures'])
        self.assertEqual(row['matched_required_fact_count'],0)
        self.assertEqual(report,original) # Provenance/replay still receive every supporting value and citation.
        genuine=copy.deepcopy(report);genuine['numeric_facts'][0]=dict(support,supporting_only=False)
        genuine['publication']['claims'][0]['evidence'][0]=dict(observation_id='oracle:main')
        row=q._natural_grade(dict(id='genuine_main'),expected,genuine)
        self.assertTrue(row['passed']);self.assertEqual(row['matched_required_fact_count'],1)

    def test_supporting_citation_or_claim_does_not_establish_primary_publication(self):
        wanted=dict(observation_id='oracle:main',value_exact='16.125',unit='percent_gdp',
            book='2025',metric='Pensiones',series='Opinión 2025',year=2050)
        fact=dict(wanted,value='16.125',period='2050')
        expected=dict(behavior='answer',required_facts=[wanted])
        for supporting_claim in (False,True):
            with self.subTest(supporting_claim=supporting_claim):
                claim=dict(text='Nota secundaria: Pensiones 16.125.',evidence=[
                    dict(observation_id='oracle:main',supporting_only=not supporting_claim)])
                if supporting_claim:claim['supporting_only']=True
                report=dict(status='answered',numeric_facts=[fact],publication={'claims':[claim]})
                row=q._natural_grade(dict(id='supporting_publication'),expected,report)
                self.assertFalse(row['passed']);self.assertEqual(row['matched_required_fact_count'],0)
                self.assertTrue(row['failures'])

    def test_equivalence_adjudication_requires_explicit_semantic_attestation(self):
        out,_,data=self.fixture();entry=data['cases'][0]
        entry['query']='Según el Cuadro 12, ¿qué gasto en pensiones publica la serie AIReF 2026?'
        citation=dict(source_id='DB:other:1',observation_id='other:1',source_cell='F8',source_sheet='Source')
        entry['answer']['publication']['claims']=[dict(text='Source fact',evidence=[citation])]
        self.rewrite(out/'results.json',data)
        row=dict(case_id=entry['id'],passed=False,safe=True,failures=[],review_required=['Equivalent source'],
            equivalence_candidates=[dict(expected_observation_id='oracle:1',candidates=[dict(observation_id='other:1')])])
        decision=dict(case_id=entry['id'],approved=True,reviewer='Independent QA',whole_answer_reviewed=True,
            rationale='Verified original cell, definition, scenario, scope and publication vintage.',evidence=[citation],
            case_input_sha256=__import__('hashlib').sha256(q._canonical({k:entry[k]for k in ('id','query','context')}).encode()).hexdigest())
        doc=dict(schema_version='langgraph-adjudications-v1',policy=q.REVIEW_POLICY,
            results_sha256=q._sha(out/'results.json'),protocol_sha256=q._sha(out/'protocol.json'),decisions=[decision])
        path=out/'equivalence.json';self.rewrite(path,doc)
        with self.assertRaisesRegex(ValueError,'Equivalent source'):
            q._apply_adjudications([copy.deepcopy(row)],data['cases'],path,out/'results.json',out/'protocol.json')
        decision['equivalence_checks']=[dict(expected_observation_id='oracle:1',actual_observation_id='other:1',
            vintage_scenario_verified=True,definition_verified=True,scope_verified=True,
            requested_series_compatible=True,primary_source_verified=True)]
        decision['equivalence_checks'][0]['definition_verified']=False
        self.rewrite(path,doc)
        with self.assertRaisesRegex(ValueError,'Equivalent source'):
            q._apply_adjudications([copy.deepcopy(row)],data['cases'],path,out/'results.json',out/'protocol.json')
        decision['equivalence_checks'][0]['definition_verified']=True
        for constraint in (None,False):
            with self.subTest(requested_source_constraint_verified=constraint):
                decision['equivalence_checks'][0]['requested_source_constraint_verified']=constraint
                self.rewrite(path,doc)
                with self.assertRaisesRegex(ValueError,'requested-source-constraint'):
                    q._apply_adjudications([copy.deepcopy(row)],data['cases'],path,out/'results.json',out/'protocol.json')
        decision['equivalence_checks'][0]['requested_source_constraint_verified']=True
        decision['whole_answer_reviewed']=False;self.rewrite(path,doc)
        with self.assertRaisesRegex(ValueError,'whole-answer review'):
            q._apply_adjudications([copy.deepcopy(row)],data['cases'],path,out/'results.json',out/'protocol.json')
        decision['whole_answer_reviewed']=True
        self.rewrite(path,doc);rows=[copy.deepcopy(row)]
        q._apply_adjudications(rows,data['cases'],path,out/'results.json',out/'protocol.json')
        self.assertTrue(rows[0]['passed'])
        decision.update(approved=False,rationale='Independent source inspection found a different economic definition; equal values do not establish equivalence.')
        self.rewrite(path,doc);rows=[copy.deepcopy(row)]
        q._apply_adjudications(rows,data['cases'],path,out/'results.json',out/'protocol.json')
        self.assertFalse(rows[0]['passed']);self.assertFalse(rows[0]['safe'])
        self.assertIn('independent_review_rejected',rows[0]['failures'])

    def test_previous_review_policy_cannot_relabel_a_new_protocol(self):
        out,_,data=self.fixture();entry=data['cases'][0]
        doc=dict(schema_version='langgraph-adjudications-v1',policy='independent-pending-only-results-bound-v1',
            results_sha256=q._sha(out/'results.json'),protocol_sha256=q._sha(out/'protocol.json'),decisions=[])
        path=out/'old_policy.json';self.rewrite(path,doc)
        with self.assertRaisesRegex(ValueError,'frozen execution'):
            q._apply_adjudications([],data['cases'],path,out/'results.json',out/'protocol.json')

    def test_adjudications_bound_to_results_and_citations_only_resolve_pending(self):
        out,_,data=self.fixture();entry=data['cases'][0]
        evidence=dict(source_id='S1',doc_id='source',page_number=1,quote='Literal source evidence')
        entry['answer']['publication']['claims']=[dict(text='Verified claim',evidence=[evidence])]
        self.rewrite(out/'results.json',data)
        row=dict(case_id=entry['id'],passed=False,safe=True,failures=[],review_required=['Equivalent source definition/scope requires independent review'])
        decision=dict(case_id=entry['id'],approved=True,reviewer='Independent QA',
            rationale='Independently checked exact source, definition and scope.',evidence=[evidence],
            case_input_sha256=__import__('hashlib').sha256(q._canonical({k:entry[k]for k in ('id','query','context')}).encode()).hexdigest())
        doc=dict(schema_version='langgraph-adjudications-v1',policy=q.REVIEW_POLICY,
            results_sha256=q._sha(out/'results.json'),protocol_sha256=q._sha(out/'protocol.json'),decisions=[decision])
        path=out/'adjudications.json';self.rewrite(path,doc)
        rows=[copy.deepcopy(row)]
        q._apply_adjudications(rows,data['cases'],path,out/'results.json',out/'protocol.json')
        self.assertTrue(rows[0]['passed']);self.assertFalse(rows[0]['automatic_passed'])
        self.assertFalse(rows[0]['review_required'])
        for alteration in ('results','evidence','failure'):
            changed=copy.deepcopy(doc);rows=[copy.deepcopy(row)]
            if alteration=='results':changed['results_sha256']='0'*64
            elif alteration=='evidence':changed['decisions'][0]['evidence'][0]['quote']='Invented quotation'
            else:rows[0]['failures']=['wrong_exact_value']
            self.rewrite(path,changed)
            with self.subTest(alteration=alteration),self.assertRaises(ValueError):
                q._apply_adjudications(rows,data['cases'],path,out/'results.json',out/'protocol.json')

    def test_material_definition_clarification_is_pending_not_wrong_number(self):
        f=dict(observation_id='f:1',unit='percent_gdp',value_exact='1')
        expected=dict(behavior='answer_or_definition_clarification',required_fact_groups=[[f]])
        report=dict(status='clarification',numeric_facts=[],publication={'claims':[]})
        result=q._natural_grade(dict(id='x'),expected,report)
        self.assertFalse(result['passed']);self.assertEqual(result['failures'],[])
        self.assertTrue(result['review_required'])

    def test_ambiguous_query_requires_a_material_axis_not_a_table_number(self):
        expected=dict(behavior='labelled_candidates_or_material_clarification',material_axes=['report_edition','unit'])
        report=dict(status='clarification',numeric_facts=[],publication={},plan={'clarification_field':'edition'})
        self.assertTrue(q._natural_grade(dict(id='x'),expected,report)['passed'])
        report['plan']['clarification_field']='period'
        self.assertFalse(q._natural_grade(dict(id='x'),expected,report)['passed'])

    def test_ambiguous_answer_requires_published_exact_relevant_candidates(self):
        wanted=[dict(observation_id='p:'+book,book=book,metric='Pensiones',series='Forecast',
            period='2050',unit='percent_gdp',value_exact=value)for book,value in [('2025','16'),('2026','17')]]
        expected=dict(behavior='labelled_candidates_or_material_clarification',acceptable_known_facts=wanted)
        def report(facts):
            return dict(status='answered',numeric_facts=facts,publication={'claims':[
                dict(text='Published candidates',evidence=[dict(observation_id=f['observation_id'])for f in facts])]})
        facts=[dict(w,value=w['value_exact'])for w in wanted]
        self.assertTrue(q._natural_grade(dict(id='x'),expected,report(facts))['passed'])
        wrong=[dict(f,metric='Intereses')for f in facts]
        row=q._natural_grade(dict(id='x'),expected,report(wrong))
        self.assertFalse(row['passed']);self.assertTrue(row['failures'])
        unknown=[dict(f,observation_id='other:'+f['book'])for f in facts]
        row=q._natural_grade(dict(id='x'),expected,report(unknown))
        self.assertFalse(row['passed']);self.assertEqual(row['failures'],[]);self.assertTrue(row['review_required'])
        hidden=report(facts);hidden['publication']['claims']=[]
        self.assertFalse(q._natural_grade(dict(id='x'),expected,hidden)['passed'])

    def test_partial_causal_explanation_requires_review_even_with_exact_number(self):
        fact=dict(observation_id='p:2026',value_exact='17',unit='percent_gdp')
        expected=dict(behavior='answer_numeric_and_ground_explanation_or_explicit_partial',required_facts=[fact])
        report=dict(status='partial',numeric_facts=[dict(fact,value='17')],publication={'claims':[
            dict(text='Exact numeric part',evidence=[dict(observation_id='p:2026')])]})
        case=dict(id='x')
        self.assertTrue(q._natural_grade(case,expected,report)['passed'])
        report['publication']['claims'].append(dict(text='A generic explanation',evidence=[
            dict(source_kind='pdf_text',doc_id='2025-source',quote='Generic literal explanation from 2025')]))
        row=q._natural_grade(case,expected,report)
        self.assertFalse(row['passed']);self.assertEqual(row['failures'],[])
        self.assertTrue(any('2026 causal'in p for p in row['review_required']))

    def test_partial_positive_negative_case_is_reviewed_and_substitutions_fail(self):
        expected=dict(behavior='population_fidelity',forbidden_substitute_facts=['aggregate'],
            allowed_statuses=['partial','insufficient_evidence'])
        report=dict(status='partial',numeric_facts=[dict(observation_id='unknown',book='2026')],
            publication={'claims':[dict(text='A new source',evidence=[dict(observation_id='unknown')])]})
        row=q._natural_grade(dict(id='x'),expected,report)
        self.assertFalse(row['passed']);self.assertTrue(row['review_required']);self.assertEqual(row['failures'],[])
        report['numeric_facts'][0]['observation_id']='aggregate'
        report['publication']['claims'][0]['evidence'][0]['observation_id']='aggregate'
        row=q._natural_grade(dict(id='x'),expected,report)
        self.assertFalse(row['safe']);self.assertIn('forbidden_aggregate_or_unit_substitution',row['failures'])


if __name__=='__main__':unittest.main()
