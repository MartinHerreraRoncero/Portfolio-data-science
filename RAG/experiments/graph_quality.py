"""Versioned, source-bound evaluation for the local 9B LangGraph profile.

Expected answers are read only by scoring. Inference receives query/context only.
The historical eleven-case comparison is an explicit text-only reference scope.
"""
from __future__ import annotations
import argparse
import copy
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

from experiments.continuous_quality import (DEFAULT_QUERIES_REL, DEFAULT_REFERENCE_REL,
    EXPECTED_CASE_IDS, evaluate_case_reference, strict_load_json, validate_reference_ledger)
from rag_pensiones.ingest import PROJECT_ROOT, validate_project_path
from rag_pensiones.graph_types import (CHAT_MODEL, CHAT_DIGEST, CHAT_OPTIONS,
    EMBED_MODEL, EMBED_DIGEST, EMBED_DIMENSION, EMBED_KEEP_ALIVE, PROFILE, stage_chat_settings,chat_keep_alive,model_lifecycle_policy)

PROTOCOL = 'langgraph-quality-protocol-v1'
RESULTS = 'langgraph-quality-results-v1'
SCORE = 'langgraph-quality-score-v1'
REVIEW_POLICY = 'independent-pending-only-results-bound-v2'
DEFAULT_BASELINE = 'evaluation/continuous_quality_2026-10-27/accepted_baseline_v2.json'
DEFAULT_BASELINE_SHA256 = 'f05be601ce1c49fd2a58175d73a3ed50e9538d7fda98e84e585ce9c2fe734430'
BENCH_DIR = 'evaluation/langgraph_rag_2026-11-01/design_audit'
SUITES = ('legacy_reference', 'natural_dev', 'natural_heldout')
SCOPES = {'legacy_reference':'legacy_2025_reference', 'natural_dev':'production_full',
          'natural_heldout':'production_full'}
# Independent primary198 oracle: same semantic identity and exact value in C12/G20.
# Other aliases are never accepted automatically from a matching name or number.
KNOWN_EQUIVALENT_FACT_IDS = (frozenset(('airef:2026:376','airef:2026:4174')),
    frozenset(('airef:2026:377','airef:2026:4179')),
    frozenset(('airef:2026:378','airef:2026:4199')))


def _path(path):
    return validate_project_path(Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path)


def _read(path):
    return strict_load_json(_path(path).read_bytes())


def _sha(path):
    return hashlib.sha256(_path(path).read_bytes()).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _same(left, right):
    return _canonical(left) == _canonical(right)


def _rel(path):
    return _path(path).relative_to(PROJECT_ROOT).as_posix()


def _write_new(path, value):
    path = _path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _checkpoint(path, value):
    path = _path(path)
    temporary = path.with_suffix(path.suffix + '.checkpoint.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def _code_files():
    paths = sorted(PROJECT_ROOT.glob('rag_pensiones/*.py')) + sorted(PROJECT_ROOT.glob('experiments/*.py'))
    paths += sorted(PROJECT_ROOT.glob('requirements*.txt'))
    return {_rel(p): _sha(p) for p in paths}


def _dependencies():
    return {name:importlib.metadata.version(name) for name in
            ('langgraph','langgraph-checkpoint-sqlite','duckdb','pypdf')}


def _runtime_manifest(services):
    from rag_pensiones.graph_validation import runtime_manifest, verify_manifest
    manifest = runtime_manifest(services)
    verify_manifest(manifest, services=services)
    return manifest


def _services(suite, services=None, asset_dir=None):
    if services is None:
        from rag_pensiones.graph_pipeline import load_default_services
        services = load_default_services(asset_dir=asset_dir, scope=SCOPES[suite])
    if getattr(services, 'scope', None) != SCOPES[suite]:
        raise ValueError('Evaluation corpus/service scope mismatch')
    return services


def _execution_kind(services):
    from rag_pensiones.graph_models import LocalOllamaTransport
    transport=getattr(getattr(services,'client',None),'transport',None)
    return 'local_live'if type(transport)is LocalOllamaTransport else'injected_test_services'


def load_cases(suite, *, reserved_release=False):
    if suite not in SUITES:
        raise ValueError('Unknown evaluation suite')
    if suite == 'natural_heldout' and reserved_release is not True:
        raise PermissionError('Reserved questions require explicit QA/root release after implementation freeze')
    path = DEFAULT_QUERIES_REL if suite == 'legacy_reference' else (
        BENCH_DIR + '/natural_' + ('dev' if suite == 'natural_dev' else 'heldout') + '_v1.json')
    payload = _read(path)
    if suite!='legacy_reference':
        frozen=_read(BENCH_DIR+'/benchmark_freeze_v1.json')
        if frozen.get('files',{}).get(Path(path).name)!=_sha(path):
            raise ValueError('Frozen natural benchmark changed')
        if payload.get('oracle_sha256')!=_sha(payload['oracle_path']):
            raise ValueError('Primary benchmark oracle changed')
    rows = payload.get('cases')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Empty or malformed case roster')
    public = []
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get('id'), str)
                or not isinstance(row.get('query'), str) or not row['query'].strip()):
            raise ValueError('Invalid case identity/query')
        context = {'report_year':'2025'} if suite == 'legacy_reference' else row.get('context', [])
        if not isinstance(context, (dict,list)):
            raise ValueError('Invalid conversational context')
        public.append(dict(id=row['id'], query=row['query'], context=copy.deepcopy(context)))
    if len({c['id'] for c in public}) != len(public):
        raise ValueError('Duplicate case identities')
    if suite == 'legacy_reference' and tuple(c['id'] for c in public) != EXPECTED_CASE_IDS:
        raise ValueError('Historical reference roster/order mismatch')
    if suite != 'legacy_reference' and len(public) != payload.get('case_count'):
        raise ValueError('Natural case count mismatch')
    return public, {c['id']:copy.deepcopy(c.get('expected', {})) for c in rows}, _rel(path)


def _validate_reference_inputs(cases,suite,baseline_path,reference_path):
    reference=_read(reference_path);validate_reference_ledger(reference)
    baseline=_read(baseline_path)
    if _path(baseline_path)==_path(DEFAULT_BASELINE) and _sha(baseline_path)!=DEFAULT_BASELINE_SHA256:
        raise ValueError('Accepted baseline was modified')
    if baseline.get('overall_pass')is not True:raise ValueError('Baseline is not an accepted passing report')
    _baseline_regressions(baseline.get('case_results',[]),baseline,_sha(reference_path))
    if suite=='legacy_reference' and any(c['query']!=reference['cases'][c['id']]['query']for c in cases):
        raise ValueError('Legacy question differs from frozen reference')
    return reference


def prepare(output_dir, *, suite='legacy_reference', services=None, asset_dir=None,
            baseline_path=DEFAULT_BASELINE, reference_path=DEFAULT_REFERENCE_REL,
            reserved_release=False):
    out = _path(output_dir)
    if any((out/name).exists() for name in ('protocol.json','results.json','quality_report.json')):
        raise FileExistsError('Evaluation destination already contains artifacts')
    cases, _, case_path = load_cases(suite, reserved_release=reserved_release)
    services = _services(suite, services, asset_dir)
    manifest = _runtime_manifest(services)
    _validate_reference_inputs(cases,suite,baseline_path,reference_path)
    protocol = dict(schema_version=PROTOCOL, profile=PROFILE, suite=suite,
        source_scope=SCOPES[suite], reference_text_only=(suite=='legacy_reference'),
        scope_note=('Compatibility control: 13 accepted 2025 text chunks, contextual report2025 and zero exposed numeric cards; not production coverage.'
                    if suite=='legacy_reference' else 'Production full certified numeric catalog and both full report corpora.'),
        chat_model=CHAT_MODEL,chat_digest=CHAT_DIGEST,chat_options=copy.deepcopy(CHAT_OPTIONS),
        chat_stage_settings={stage:stage_chat_settings(stage)for stage in ('default','plan_check_batch')},
        embedding_model=EMBED_MODEL,embedding_digest=EMBED_DIGEST,embedding_dimension=EMBED_DIMENSION,
        model_lifecycle_policy=model_lifecycle_policy(),
        cases=cases, case_file=case_path,case_file_sha256=_sha(case_path),
        reference_path=_rel(reference_path),reference_sha256=_sha(reference_path),
        baseline_path=_rel(baseline_path),baseline_sha256=_sha(baseline_path),
        source_files=_code_files(),dependencies=_dependencies(),runtime_manifest=manifest,
        reserved_release=reserved_release is True,manual_review_policy=REVIEW_POLICY,
        execution_kind=_execution_kind(services))
    _validate_protocol(protocol, services)
    _write_new(out/'protocol.json',protocol)
    return protocol


def _validate_protocol(protocol, services):
    keys={'schema_version','profile','suite','source_scope','reference_text_only','scope_note',
        'chat_model','chat_digest','chat_options','chat_stage_settings','embedding_model','embedding_digest','embedding_dimension','model_lifecycle_policy',
        'cases','case_file','case_file_sha256','reference_path','reference_sha256','baseline_path',
        'baseline_sha256','source_files','dependencies','runtime_manifest','reserved_release','manual_review_policy','execution_kind'}
    if not isinstance(protocol,dict) or set(protocol)!=keys:
        raise ValueError('Graph protocol keys mismatch')
    if protocol.get('schema_version') != PROTOCOL or protocol.get('profile') != PROFILE:
        raise ValueError('Unknown graph evaluation protocol/profile')
    if protocol.get('manual_review_policy')!=REVIEW_POLICY:raise ValueError('Unknown manual adjudication policy')
    if protocol.get('execution_kind')!=_execution_kind(services):raise ValueError('Producer live/test transport identity mismatch')
    suite = protocol.get('suite')
    if suite not in SUITES or protocol.get('source_scope') != SCOPES[suite]:
        raise ValueError('Invalid corpus scope')
    if (type(protocol['reference_text_only'])is not bool or protocol['reference_text_only']!=(suite=='legacy_reference')
            or type(protocol['reserved_release'])is not bool):raise ValueError('Invalid scope/release flags')
    expected = dict(chat_model=CHAT_MODEL,chat_digest=CHAT_DIGEST,chat_options=CHAT_OPTIONS,
        chat_stage_settings={stage:stage_chat_settings(stage)for stage in ('default','plan_check_batch')},
        embedding_model=EMBED_MODEL,embedding_digest=EMBED_DIGEST,embedding_dimension=EMBED_DIMENSION,
        model_lifecycle_policy=model_lifecycle_policy())
    if any(not _same(protocol.get(k),v) for k,v in expected.items()):
        raise ValueError('Graph role/model/options identity mismatch')
    if not _same(protocol.get('source_files'),_code_files()) or not _same(protocol.get('dependencies'),_dependencies()):
        raise ValueError('Frozen graph code/dependency drift')
    for key in ('case_file','reference_path','baseline_path'):
        hashkey = {'case_file':'case_file_sha256','reference_path':'reference_sha256','baseline_path':'baseline_sha256'}[key]
        if _sha(protocol[key]) != protocol.get(hashkey):
            raise ValueError('Frozen evaluator input drift: '+key)
    cases, _, path = load_cases(suite,reserved_release=protocol.get('reserved_release') is True)
    if path != protocol['case_file'] or not _same(cases,protocol.get('cases')):
        raise ValueError('Protocol question/context bindings altered')
    _validate_reference_inputs(cases,suite,protocol['baseline_path'],protocol['reference_path'])
    if not _same(_runtime_manifest(services),protocol.get('runtime_manifest')):
        raise ValueError('Runtime asset/prompt manifest drift')
    if suite == 'legacy_reference':
        manifest=protocol['runtime_manifest']
        if manifest.get('reference_text_only') is not True or manifest.get('exposed_card_ids') != []:
            raise ValueError('Legacy reference must explicitly expose no numeric cards')


def _validate_calls(report):
    calls=report.get('stage_calls')
    if not isinstance(calls,list):
        raise ValueError('Missing truthful stage-call records')
    for c in calls:
        req=c.get('request',{});endpoint=c.get('endpoint')
        model=CHAT_MODEL if endpoint=='/api/chat' else EMBED_MODEL if endpoint=='/api/embed' else None
        digest=CHAT_DIGEST if endpoint=='/api/chat' else EMBED_DIGEST
        if model is None or c.get('model')!=model or req.get('model')!=model:
            raise ValueError('Disallowed model or operation in actual trace')
        if c.get('error') or c.get('attempted') is not True:
            raise ValueError('Technical stage error is not an abstention')
        if c.get('digest_before')!=digest or c.get('digest_after')!=digest:
            raise ValueError('Stage model identity drift')
        if not isinstance(c.get('response'),dict) or c['response'].get('model')!=model:
            raise ValueError('Returned stage model mismatch')
        if req.get('keep_alive')!=(chat_keep_alive(c.get('stage'))if endpoint=='/api/chat'else EMBED_KEEP_ALIVE):
            raise ValueError('Actual model lifecycle differs from profile')
        if endpoint=='/api/chat':
            if not isinstance(c.get('stage'),str)or not c['stage'].strip():
                raise ValueError('Missing truthful graph stage')
            settings=stage_chat_settings(c['stage'],req.get('messages'));response=c['response']
            if (not _same(req.get('options'),settings['options'])or req.get('stream')is not False
                    or req.get('think')is not settings['think']):
                raise ValueError('Unfaithful graph chat stage options')
            if response.get('done')is not True or response.get('done_reason')!='stop':
                raise ValueError('Incomplete technical graph chat is not an abstention')
            if any(type(response.get(k))is not int or response[k]<0 for k in ('prompt_eval_count','eval_count')):
                raise ValueError('Invalid truthful graph token statistics')
            if (response['eval_count']>settings['options']['num_predict']
                    or response['prompt_eval_count']+response['eval_count']>settings['options']['num_ctx']):
                raise ValueError('Graph stage token budget exceeded')
        if endpoint=='/api/embed' and (type(req.get('dimensions'))is not int or req.get('dimensions')!=EMBED_DIMENSION
                or req.get('truncate')is not False):
            raise ValueError('Embedding dimension mismatch')
        if endpoint=='/api/embed' and req.get('options')!=model_lifecycle_policy()['embedding_options']:
            raise ValueError('Embedding placement options mismatch')


def _runtime_replay(report, services):
    from rag_pensiones.graph_validation import replay_graph_report
    return replay_graph_report(report,services=services)


def _format(report):
    from rag_pensiones.graph_pipeline import format_graph_answer
    return format_graph_answer(report)


def _pages(services):
    pages=services.documents.get('pages',[])
    if isinstance(pages,dict):pages=list(pages.values())
    return pages


def _provenance(report,services):
    pages=_pages(services)
    canonical={(p.get('doc_id'),p.get('page_number')):p for p in pages}
    provenance={}
    for claim in report['publication'].get('claims',[]):
        for e in claim.get('evidence',[]):
            sid=e.get('source_id');kind=e.get('source_kind')
            if kind=='structured_cell':
                provenance[sid]=dict(e,structured_verified=True,source_file=e.get('source_path'))
            else:
                page=canonical.get((e.get('doc_id'),e.get('page_number')))
                if page is None or e.get('quote','') not in page.get('text',''):
                    raise ValueError('Text citation not bound to canonical document/page')
                provenance[sid]=dict(doc_id=page['doc_id'],page_number=page['page_number'],source_file=page['source_file'])
    # The pure historical rubric is page-keyed; production natural grading never uses it.
    bynum={p['page_number']:p for p in pages}
    if len({p['doc_id'] for p in pages})>1:
        raise ValueError('Historical rubric requires the explicit single-document reference scope')
    return provenance,bynum


def _decimal(value):
    try:result=Decimal(str(value))
    except (InvalidOperation,TypeError,ValueError):raise ValueError('Invalid fact number')
    if not result.is_finite():raise ValueError('Nonfinite fact number')
    return result


def _fact_value(f):
    return f.get('value',f.get('value_exact'))


def _period(f):
    if f.get('period') is not None:return str(f['period'])
    if f.get('category_literal') is not None:return str(f['category_literal'])
    if f.get('year') is not None:return str(f['year'])
    return str(f.get('start_year'))+'-'+str(f.get('end_year'))


def _natural_grade(case, expected, report):
    """Deterministic known-source grading; unknown equivalent sources require review."""
    status=report['status'];pub=report['publication']
    facts=[f for f in report.get('numeric_facts',[])if f.get('supporting_only')is not True]
    behavior=expected.get('behavior');failures=[];pending=[];equivalence_candidates=[]
    published_ids=set()
    for claim in pub.get('claims',[]):
        if claim.get('supporting_only')is True:continue
        for ev in claim.get('evidence',[]):
            if ev.get('supporting_only')is True:continue
            oid=ev.get('observation_id')
            if not oid and str(ev.get('source_id','')).startswith('DB:'):
                oid=ev['source_id'][3:]
            if oid:published_ids.add(oid)
    def matched(wanted):
        for got in facts:
            if got.get('derived'):continue
            if got.get('observation_id')not in published_ids:continue
            same_id=got.get('observation_id')==wanted['observation_id']
            alias=any({got.get('observation_id'),wanted['observation_id']}<=group for group in KNOWN_EQUIVALENT_FACT_IDS)
            if same_id or alias:
                if _decimal(_fact_value(got))!=_decimal(wanted['value_exact']):return False
                if got.get('unit')!=wanted['unit']:return False
                if any(got.get(k)!=wanted.get(k,wanted.get(k+'_literal'))for k in ('book','metric','series')):return False
                if _period(got)!=_period(wanted):return False
                if wanted.get('numeric_reconciliation_failed') is True:
                    if (not got.get('warning') or got.get('calculation_safe') is not False
                            or str(got['warning']) not in _format(report)):return False
                return True
        return False
    required=expected.get('required_facts',[])
    def missing_fact(wanted):
        if matched(wanted):return False
        candidates=[g for g in facts if g.get('observation_id')in published_ids and not g.get('derived')
                and g.get('observation_id')!=wanted['observation_id']
                and g.get('unit')==wanted['unit'] and _decimal(_fact_value(g))==_decimal(wanted['value_exact'])
                and _period(g)==_period(wanted)
                and g.get('book')==wanted.get('book',wanted.get('book_literal'))
                and (wanted.get('numeric_reconciliation_failed')is not True
                     or (g.get('warning') and g.get('calculation_safe')is False
                         and str(g['warning'])in _format(report)))]
        if candidates:
            pending.append('Equivalent source definition/scope/vintage/series requires independent review:'+wanted['observation_id'])
            equivalence_candidates.append(dict(expected_observation_id=wanted['observation_id'],
                expected_metric=wanted.get('metric',wanted.get('metric_literal')),
                expected_series=wanted.get('series',wanted.get('series_literal')),
                candidates=[dict(observation_id=g['observation_id'],metric=g.get('metric'),
                                 series=g.get('series'))for g in candidates]))
            return False
        return True
    for f in required:
        if missing_fact(f):failures.append('missing_exact_fact:'+f['observation_id'])
    for group in expected.get('required_fact_groups',[]):
        if behavior=='answer_or_definition_clarification'and status=='clarification':continue
        if not any(matched(f)for f in group) and all(missing_fact(f)for f in group):
            failures.append('missing_edition_fact_group')
    if behavior=='answer' and status!='answered':failures.append('expected_answer')
    if behavior=='answer_or_definition_clarification' and status=='clarification':
        pending.append('Definition conflict must be independently justified; not an automatic pass')
    if behavior=='labelled_candidates_or_material_clarification':
        if status=='clarification':
            text=_format(report).lower()
            if any(w in text for w in ('numero de cuadro','número de cuadro','numero de grafico','número de gráfico',
                    'número de tabla','numero de tabla','qué cuadro','qué gráfico','qué tabla','cual cuadro','cuál cuadro')):
                failures.append('asks_for_figure_number')
            mapping={'report_edition':{'edition'},'unit':{'unit'},'scope':{'population','metric'},
                     'period':{'period'},'scenario':{'scenario'}}
            allowed=set().union(*(mapping.get(axis,set())for axis in expected.get('material_axes',[])))
            if (report.get('plan')or{}).get('clarification_field')not in allowed:
                failures.append('clarification_axis_not_material')
        elif status=='answered':
            known=expected.get('acceptable_known_facts',[])
            answered=[f for f in facts if f.get('observation_id')in published_ids]
            if known and all(matched(f)for f in known):pass
            else:
                # New sources remain open-world, but two editions of an unrelated
                # measure cannot satisfy a question about the requested metric.
                keys=('book','metric','unit')
                plausible=[g for g in answered if any(_period(g)==_period(w)
                    and all(g.get(k)==w.get(k,w.get(k+'_literal'))for k in keys)for w in known)]
                if known and answered and not plausible:failures.append('ambiguity_candidates_wrong_metric_scope_or_period')
                else:pending.append('Unspecified or alternative edition/unit candidates require independent source/scope review')
        else:failures.append('ambiguity_not_resolved_or_explained')
    negative=behavior in {'unit_fidelity','growth_not_level','population_fidelity','territory_fidelity',
        'individual_fidelity','horizon_fidelity','incompatible_comparison'}
    if negative:
        forbidden=set(expected.get('forbidden_substitute_facts',[]))
        if any(f.get('observation_id')in forbidden and f.get('observation_id')in published_ids for f in facts):failures.append('forbidden_aggregate_or_unit_substitution')
        if status=='answered'or pub.get('claims'):
            pending.append('Open-world positive requires independent source/scope validation')
        elif status not in expected.get('allowed_statuses',[]):failures.append('invalid_safe_status')
    if behavior=='answer_numeric_and_ground_explanation_or_explicit_partial':
        has_text=any(e.get('source_kind')=='pdf_text'for c in pub.get('claims',[])for e in c.get('evidence',[]))
        if status=='answered'or has_text:pending.append('2026 causal explanation requires independent narrative-source review')
        if status not in ('answered','partial'):failures.append('mixed_numeric_result_not_preserved')
    if status=='error':failures.append('technical_error')
    if expected.get('comparability_must_be_proven_before_arithmetic') and any(f.get('derived')for f in facts):
        pending.append('Derived cross-edition comparability requires independent definition evidence')
    groups=[[f]for f in required]+expected.get('required_fact_groups',[])
    def collect(value):
        found={}
        if isinstance(value,dict):
            if isinstance(value.get('card_id'),str) and isinstance(value.get('observation_ids'),list):
                found[value['card_id']]=set(value['observation_ids'])
            for child in value.values():found.update(collect(child))
        elif isinstance(value,list):
            for child in value:found.update(collect(child))
        return found
    cards=collect(report.get('retrieval',{}))
    available=set().union(*cards.values())if cards else set()
    selected_ids={cid for s in (report.get('plan')or{}).get('selections',[])for cid in s.get('card_ids',[])}
    selected=set().union(*(cards[cid]for cid in selected_ids if cid in cards))if selected_ids else set()
    def source_hit(group,ids):
        expected_ids={f['observation_id']for f in group}
        for aliases in KNOWN_EQUIVALENT_FACT_IDS:
            if expected_ids&aliases:expected_ids|=aliases
        return bool(ids&expected_ids)
    return dict(case_id=case['id'],status=status,passed=not failures and not pending,
        safe=not failures,failures=failures,review_required=pending,equivalence_candidates=equivalence_candidates,
        required_fact_count=len(groups),matched_required_fact_count=sum(any(matched(f)for f in group)for group in groups),
        required_source_group_count=len(groups),retrieved_source_group_count=sum(source_hit(g,available)for g in groups),
        selected_source_group_count=sum(source_hit(g,selected)for g in groups))


def _apply_adjudications(rows,entries,path,results_path,protocol_path):
    """Record independent semantic review; never bypass numerical/lineage failures."""
    document=_read(path)
    if (document.get('schema_version')!='langgraph-adjudications-v1'
            or document.get('policy')!=REVIEW_POLICY
            or document.get('results_sha256')!=_sha(results_path)
            or document.get('protocol_sha256')!=_sha(protocol_path)):
        raise ValueError('Adjudications not bound to this frozen execution')
    decisions=document.get('decisions')
    if not isinstance(decisions,list):raise ValueError('Invalid adjudication decisions')
    byid={r['case_id']:r for r in rows};inputs={e['id']:e for e in entries};seen=set()
    for decision in decisions:
        cid=decision.get('case_id')
        if cid in seen or cid not in byid:raise ValueError('Duplicate/unknown adjudication case')
        seen.add(cid);row=byid[cid];entry=inputs[cid]
        if row.get('failures') or not row.get('review_required'):
            raise ValueError('Adjudication cannot override a failure or an already automatic decision')
        if (type(decision.get('approved'))is not bool or not isinstance(decision.get('reviewer'),str)
                or not decision['reviewer'].strip() or not isinstance(decision.get('rationale'),str)
                or len(decision['rationale'].strip())<30):raise ValueError('Unattributed or unreasoned independent review')
        citations=decision.get('evidence')
        if not isinstance(citations,list) or not citations:raise ValueError('Independent review requires cited evidence')
        actual=[e for c in entry['answer']['publication'].get('claims',[])for e in c.get('evidence',[])]
        for cited in citations:
            if not any(_same(cited,e)for e in actual):raise ValueError('Review evidence is not a replay-verified published citation')
        if row.get('equivalence_candidates'):
            if decision['approved'] and decision.get('whole_answer_reviewed')is not True:
                raise ValueError('Equivalent source requires explicit whole-answer review')
            checks=decision.get('equivalence_checks',[])
            if not isinstance(checks,list):raise ValueError('Invalid equivalence attestations')
            for requirement in row['equivalence_candidates']:
                choices={c['observation_id']for c in requirement['candidates']}
                accepted=[c for c in checks if c.get('expected_observation_id')==requirement['expected_observation_id']
                    and c.get('actual_observation_id')in choices]
                if decision['approved']:
                    if len(accepted)!=1 or any(accepted[0].get(k)is not True for k in
                        ('vintage_scenario_verified','definition_verified','scope_verified',
                         'requested_series_compatible','primary_source_verified',
                         'requested_source_constraint_verified')):
                        raise ValueError('Equivalent source requires independent vintage, scenario, definition, scope, primary-source and requested-source-constraint verification')
                    if not any(e.get('observation_id')==accepted[0]['actual_observation_id']for e in citations):
                        raise ValueError('Equivalent fact attestation lacks its published source citation')
        if decision.get('case_input_sha256')!=hashlib.sha256(_canonical({k:entry[k]for k in ('id','query','context')}).encode()).hexdigest():
            raise ValueError('Adjudication case identity/context mismatch')
        if any('2026 causal' in p for p in row['review_required']):
            # Page + document identity was already validated by replay; review must cite 2026.
            from rag_pensiones.graph_documents import source_registry
            ids={s['source_sha256']for s in source_registry() if s['book']=='2026'}
            if not any(e.get('doc_id')in ids and e.get('source_kind')=='pdf_text'for e in citations):
                raise ValueError('Causal2026 adjudication lacks literal 2026 documentary evidence')
        row['automatic_passed']=row['passed'];row['adjudication']=copy.deepcopy(decision)
        row['pending_before_review']=row['review_required'];row['review_required']=[]
        row['passed']=decision['approved']
        if not decision['approved']:row['failures'].append('independent_review_rejected');row['safe']=False
    return dict(path=_rel(path),sha256=_sha(path),policy=REVIEW_POLICY,
        reviewed_cases=sorted(seen),assurance='Attributed independent review recorded; not automatic semantic proof')


def _baseline_regressions(rows,baseline,reference_sha):
    if (baseline.get('schema_version')!='continuous-quality-report-v1'
            or baseline.get('reference_sha256')!=reference_sha):raise ValueError('Baseline identity mismatch')
    old=baseline.get('case_results')
    if not isinstance(old,list) or tuple(c.get('case_id')for c in old)!=EXPECTED_CASE_IDS:
        raise ValueError('Baseline roster mismatch')
    regressions=[]
    for previous,current in zip(old,rows):
        if previous['case_id']!=current['case_id']:raise ValueError('Comparison case mismatch')
        for name in ('whole_correct','main_direct_correct','safe'):
            if type(previous.get(name))is not bool or type(current.get(name))is not bool:
                raise ValueError('Baseline metric must be a strict bool')
            if previous[name] and not current[name]:regressions.append(current['case_id']+':'+name)
    return regressions


def check(results_path, *, protocol_path=None, services=None, asset_dir=None, output_path=None,adjudications_path=None):
    results_path=_path(results_path);protocol_path=_path(protocol_path or results_path.parent/'protocol.json')
    protocol=_read(protocol_path);services=_services(protocol['suite'],services,asset_dir)
    _validate_protocol(protocol,services)
    results=_read(results_path)
    if (results.get('schema_version')!=RESULTS or results.get('protocol_sha256')!=_sha(protocol_path)
            or results.get('run_status')!='completed'):raise ValueError('Incomplete or mismatched graph run')
    cases=results.get('cases')
    if not isinstance(cases,list) or len(cases)!=len(protocol['cases']):raise ValueError('Incomplete case roster')
    _,expectations,_=load_cases(protocol['suite'],reserved_release=protocol['reserved_release'])
    reference=_read(protocol['reference_path']);rows=[];technical=[]
    for expected_case,entry in zip(protocol['cases'],cases):
        if any(not _same(entry.get(k),expected_case[k])for k in ('id','query','context')):
            raise ValueError('Run question/context binding mismatch')
        report=entry.get('answer')
        if isinstance(report,dict):
            for call in report.get('stage_calls',[]):
                if call.get('model')not in (CHAT_MODEL,EMBED_MODEL):
                    raise ValueError('Disallowed model in graph trace, including technical failures')
        if entry.get('error') or entry.get('attempted')is not True or isinstance(report,dict)and report.get('status')=='error':
            technical.append(entry['id'])
            row=dict(case_id=entry['id'],query=entry['query'],status='error',failures=['technical_error'])
            if protocol['suite']=='legacy_reference':
                row.update(whole_correct=False,main_direct_correct=False,safe=False,direct=False,warnings=[])
            else:
                expected=expectations[entry['id']]
                required_groups=len(expected.get('required_facts',[]))+len(expected.get('required_fact_groups',[]))
                row.update(passed=False,safe=False,review_required=[],required_fact_count=required_groups,
                    matched_required_fact_count=0,required_source_group_count=required_groups,
                    retrieved_source_group_count=0,selected_source_group_count=0)
            rows.append(row);continue
        if (not isinstance(report,dict) or report.get('query')!=entry['query']
                or not _same(report.get('context'),entry['context']) or report.get('profile')!=PROFILE
                or report.get('schema_version')!='langgraph-rag-report-v1'):
            raise ValueError('Graph answer identity mismatch')
        if not _same(report.get('manifest'),protocol['runtime_manifest']):raise ValueError('Answer asset manifest mismatch')
        _validate_calls(report)
        _runtime_replay(report,services)
        if _format(report)!=entry.get('formatted_answer'):raise ValueError('Formatted answer differs from replayed publication')
        if protocol['suite']=='legacy_reference':
            provenance,pages=_provenance(report,services)
            row=evaluate_case_reference(entry['id'],report['publication'],provenance,pages,reference['cases'][entry['id']])
            row.update(case_id=entry['id'],query=entry['query'])
        else:row=_natural_grade(expected_case,expectations[entry['id']],report)
        rows.append(row)
    adjudications=None
    if adjudications_path is not None:
        if protocol['suite']=='legacy_reference':raise ValueError('Historical rubric cannot be overridden by adjudications')
        adjudications=_apply_adjudications(rows,cases,adjudications_path,results_path,protocol_path)
    regressions=[]
    if protocol['suite']=='legacy_reference' and not technical:
        regressions=_baseline_regressions(rows,_read(protocol['baseline_path']),protocol['reference_sha256'])
    if protocol['suite']=='legacy_reference':
        summary=dict(total_cases=len(cases),whole_correct_count=sum(x['whole_correct']for x in rows),
            main_direct_count=sum(x['main_direct_correct']for x in rows),safe_count=sum(x['safe']for x in rows),
            warnings=[w for x in rows for w in x['warnings']],regressions=regressions)
        passed=not technical and summary['whole_correct_count']==len(cases) and summary['safe_count']==len(cases) and not regressions
    else:
        summary=dict(total_cases=len(cases),passed_count=sum(x['passed']for x in rows),
            safe_count=sum(x['safe']for x in rows),review_required_count=sum(bool(x['review_required'])for x in rows))
        denominator=sum(x.get('required_source_group_count',0)for x in rows)
        summary['required_source_group_count']=denominator
        for name,key in [('retrieval_source_recall','retrieved_source_group_count'),('selection_source_recall','selected_source_group_count'),('exact_fact_recovery','matched_required_fact_count')]:
            summary[name]=sum(x.get(key,0)for x in rows)/denominator if denominator else None
        passed=not technical and summary['passed_count']==len(cases)
    walls=[e.get('wall_seconds')for e in cases]
    if any(type(v)not in (int,float)or not math.isfinite(v)or v<0 for v in walls):
        raise ValueError('Invalid measured case latency')
    call_counts={'chat':0,'embedding':0}
    for entry in cases:
        for call in (entry.get('answer')or{}).get('stage_calls',[]):
            if call.get('attempted')is True:
                kind='chat'if call.get('endpoint')=='/api/chat'else'embedding'
                call_counts[kind]+=1
    output=dict(schema_version=SCORE,profile=PROFILE,suite=protocol['suite'],source_scope=protocol['source_scope'],
        execution_kind=protocol['execution_kind'],adoption_eligible=passed and protocol['execution_kind']=='local_live',
        scope_note=protocol['scope_note'],overall_pass=passed,status='technical_error'if technical else 'pass'if passed else
            'review_required'if any(x.get('review_required')for x in rows)and not any(x.get('failures')for x in rows)else'quality_fail',
        results_path=_rel(results_path),results_sha256=_sha(results_path),protocol_sha256=_sha(protocol_path),
        baseline_sha256=protocol['baseline_sha256'],reference_sha256=protocol['reference_sha256'],
        summary=summary,technical_errors=technical,case_results=rows,adjudications=adjudications,
        measured_cost=dict(case_wall_seconds_total=sum(walls),case_wall_seconds_mean=sum(walls)/len(walls),
            stage_call_counts=call_counts,note='Observed wall time/call counts only; no memory peak inferred'),
        warning='Known queries only; text-only historical compatibility is not production coverage; reserved new wordings are not unseen domains.')
    _validate_protocol(protocol,services)
    if output_path is not None:_write_new(output_path,output)
    return output


def run(output_dir, *, suite='legacy_reference',services=None,asset_dir=None,
        baseline_path=DEFAULT_BASELINE,reference_path=DEFAULT_REFERENCE_REL,reserved_release=False):
    services=_services(suite,services,asset_dir)
    protocol=prepare(output_dir,suite=suite,services=services,baseline_path=baseline_path,
        reference_path=reference_path,reserved_release=reserved_release)
    out=_path(output_dir);destination=out/'results.json'
    result=dict(schema_version=RESULTS,profile=PROFILE,run_status='running',protocol_sha256=_sha(out/'protocol.json'),cases=[])
    _write_new(destination,result)
    from rag_pensiones.graph_pipeline import answer_graph
    for case in protocol['cases']:
        _validate_protocol(protocol,services)
        entry=dict(copy.deepcopy(case),attempted=False,error=None,answer=None,formatted_answer=None)
        started=time.perf_counter()
        try:
            entry['attempted']=True
            entry['answer']=answer_graph(case['query'],context=copy.deepcopy(case['context']),services=services)
            entry['formatted_answer']=_format(entry['answer'])
            if entry['answer'].get('status')=='error':entry['error']='Graph returned a technical error'
        except Exception as exc:entry['error']=type(exc).__name__+': '+str(exc)
        entry['wall_seconds']=time.perf_counter()-started
        result['cases'].append(entry)
        try:_validate_protocol(protocol,services)
        except Exception as exc:
            result.update(run_status='invalidated',error='Post-attempt drift: '+str(exc))
            _checkpoint(destination,result)
            raise
        _checkpoint(destination,result)
    result['run_status']='completed';_checkpoint(destination,result)
    return check(destination,services=services,output_path=out/'quality_report.json')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','run'):
        p=commands.add_parser(name);p.add_argument('--output-dir',required=True)
        p.add_argument('--suite',choices=SUITES,default='legacy_reference');p.add_argument('--asset-dir')
        p.add_argument('--baseline',default=DEFAULT_BASELINE);p.add_argument('--reference',default=DEFAULT_REFERENCE_REL)
        p.add_argument('--reserved-release',action='store_true')
    p=commands.add_parser('check');p.add_argument('--results',required=True)
    p.add_argument('--protocol');p.add_argument('--output',required=True);p.add_argument('--asset-dir')
    p.add_argument('--adjudications')
    args=parser.parse_args(argv)
    try:
        if args.command=='check':
            value=check(args.results,protocol_path=args.protocol,asset_dir=args.asset_dir,output_path=args.output,
                adjudications_path=args.adjudications)
        else:
            function=prepare if args.command=='prepare'else run
            value=function(args.output_dir,suite=args.suite,asset_dir=args.asset_dir,
                baseline_path=args.baseline,reference_path=args.reference,reserved_release=args.reserved_release)
        print(json.dumps(value.get('summary',{'prepared':True}),ensure_ascii=False))
        return 0 if value.get('overall_pass',True) else 1
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
