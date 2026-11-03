"""Verified source metadata and source-subtotal scope, without model inference."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from rag_pensiones import graph_context as m
from rag_pensiones.graph_validation import execute_plan,InvalidPlan
from rag_pensiones.graph_planning import compile_capabilities


class GraphContextTests(unittest.TestCase):
    def test_query_view_preserves_verified_published_alias_independent_of_wording(self):
        card=dict(card_id='synthetic',metric='Etiqueta Ábrevíada',series='Versión literal',book='2037',
                  unit='euros',scope='ámbito original',block='Objeto',quantity_role='level',title='Contenedor')
        semantic=dict(semantic_label='Magnitud desarrollada',series_role='forecast_vintage',
                      source_scope='total',scope_qualifier='sin partida temporal',forecast_role='report_base')
        before=copy.deepcopy(card);default=m.source_measure_view(card,semantic)
        canonical=m.source_measure_view(card,semantic,query='¿Magnitud desarrollada?')
        self.assertEqual(canonical['published_measure_name'],card['metric'])
        expected=copy.deepcopy(default)
        self.assertEqual(canonical,expected)
        for query,history in [('¿ETIQUETA, ABREVIADA?',None),
                              ('¿Y ahora?',dict(history=[dict(role='user',content='Etiqueta abreviada')]))]:
            self.assertEqual(m.source_measure_view(card,semantic,query=query,context=history),default)
        assistant=dict(history=[dict(role='assistant',content='Etiqueta abreviada')])
        self.assertEqual(m.source_measure_view(card,semantic,query='¿Y ahora?',context=assistant),canonical)
        self.assertEqual(m.source_measure_view(card,semantic,context=assistant),default)
        uncurated=m.source_measure_view(card,None,query='Otra pregunta')
        self.assertEqual(uncurated['measure'],card['metric'])
        self.assertEqual(uncurated['published_measure_name'],card['metric'])
        self.assertEqual(card,before)
        cap=dict(capability_id='fixed-capability',selection=dict(card_ids=[card['card_id']],periods=['2039'],operation='lookup'))
        with patch.object(m,'compact_semantic_context',return_value=semantic):
            original=m.capability_options([cap],[card],{})
            scoped=m.capability_options([cap],[card],{},query='Magnitud desarrollada')
        self.assertEqual(scoped[0]['capability_id'],cap['capability_id'])
        self.assertEqual((scoped[0]['operation'],scoped[0]['periods']),('lookup',['2039']))
        self.assertEqual(scoped[0]['source_metadata'][0],expected)
        self.assertEqual(original[0]['source_metadata'][0],default)

    def test_grounding_mixed_literal_alias_views_have_uniform_lossless_wire_fields(self):
        from rag_pensiones.graph_planning import grounding_inputs
        from rag_pensiones.graph_pipeline import pack_candidate_table,unpack_candidate_table,_wire_size,_model_messages
        rows=[]
        for cid,alias,measure in [('first','Etiqueta corta','Magnitud desarrollada'),('second','Otra etiqueta','Segunda magnitud')]:
            rows.append(dict(card_id=cid,metric=alias,series='Versión literal',book='2037',unit='euros',scope='original',
                block='Objeto',quantity_role='level',title='Contenedor',period_kind='year',periods=['2039'],
                semantic_context=dict(semantic_label=measure,series_role='forecast_vintage',source_scope='total',
                                      other_verified_qualifier={'condition':'Original scope'})))
        query='¿Etiqueta corta en 2039?'
        n=dict(need_id='n',kind='data',book='2037',unit='euros',quantity_request='level',periods=['2039'],series=None)
        before=copy.deepcopy(rows);inputs=grounding_inputs(dict(needs=[n]),rows,query)
        sources=inputs['n']['sources']
        self.assertEqual(sources[0]['published_measure_name'],'Etiqueta corta')
        self.assertEqual(sources[1]['published_measure_name'],rows[1]['metric'])
        self.assertEqual(set(sources[0]),set(sources[1]))
        self.assertEqual(sources[0]['source_scope'],'total')
        self.assertEqual(sources[0]['semantic_context'],{'other_verified_qualifier':{'condition':'Original scope'}})
        self.assertEqual(unpack_candidate_table(pack_candidate_table(sources)),sources)
        data=dict(query=query,candidates=sources);schema=dict(type='object',properties={})
        self.assertLessEqual(_wire_size('grounding',data,schema),6400)
        self.assertEqual(json.loads(_model_messages('grounding',data,schema)[1]['content'])['query'],query)
        self.assertEqual(rows,before)

    def test_forecast_default_only_compares_verified_roles_with_compatible_requested_horizon(self):
        def card(cid,**changes):
            row=dict(card_id=cid,metric='Published measure',series=cid,book='2026',unit='percent_gdp',
                     quantity_role='level',period_kind='year',periods=['2030','2050'],scope='aggregate',block='Object')
            row.update(changes);return row
        base=card('base');old=card('old',periods=['2025','2030','2050'])
        external=card('external');alternative=card('alternative');other=card('other');unknown=card('unknown')
        rows=[base,old,external,alternative,other,unknown]
        roles=['report_base','prior_report','external_updated','demographic_alternative','prior_report',None]
        entries={row['card_id']:dict(raw=copy.deepcopy(row),proposed=dict(semantic_label='Family B'if row is other else'Family A',
                  forecast_role=role))for row,role in zip(rows,roles)}
        n=dict(need_id='n',kind='data',book='2026',series=None,source_block=None,unit='percent_gdp',
               quantity_request='level',periods=['2050'])
        before=copy.deepcopy(rows)
        with patch.object(m,'_entries',return_value=entries):
            self.assertEqual([r['card_id']for r in m.prefer_current_forecast(n,rows,{})],['base','other','unknown'])
            for changes in [dict(series='old'),dict(source_block='Object')]:
                self.assertEqual(m.prefer_current_forecast(dict(n,**changes),rows,{}),rows)
            self.assertEqual([r['card_id']for r in m.prefer_current_forecast(dict(n,book=None),rows,{})],['base','other','unknown'])
            # With no requested horizon, differing grids are left unresolved.
            self.assertIn(old,m.prefer_current_forecast(dict(n,periods=[]),rows,{}))
            for changes in [dict(unit='euros'),dict(quantity_role='change'),dict(period_kind='interval'),dict(periods=['2070'])]:
                changed=dict(base,**changes);entries['base']['raw']=copy.deepcopy(changed)
                self.assertIn(old,m.prefer_current_forecast(n,[changed,old],{}),changes)
            entries['base']['raw']=copy.deepcopy(base)
            no_horizon=dict(old,periods=['2070']);entries['old']['raw']=copy.deepcopy(no_horizon)
            self.assertIn(no_horizon,m.prefer_current_forecast(n,[base,no_horizon],{}))
        self.assertEqual(rows,before)

    def test_verified_pension_roles_default_to_report_base_after_grounding_without_overriding_explicit_sources(self):
        from rag_pensiones.graph_planning import prepare_need_candidates
        raw='Evolución del gasto en pensiones (% PIB)'
        series=['AIReF 2026','Opinión 2025','AR 2024 actualizado','AIReF + Población EUROPOP2025']
        rows=[copy.deepcopy(self.row(raw,'Gráfico 4','2026',series=label))for label in series]
        services=SimpleNamespace(catalog=self.catalog,source_policy=dict(groups=[]),semantic_context=self.registry)
        n=self.need(metric='Gasto público en pensiones',book='2026',series=None,periods=['2050'])
        before=copy.deepcopy(rows)
        prepared=prepare_need_candidates(n,rows,services)
        self.assertEqual([row['series']for row in prepared],['AIReF 2026'])
        self.assertEqual([cap['selection']['card_ids']for cap in compile_capabilities(n,rows,services)],[[rows[0]['card_id']]])
        for label in series[1:]:
            explicit=dict(n,series=label)
            self.assertEqual(prepare_need_candidates(explicit,rows,services),rows)
            caps=compile_capabilities(explicit,rows,services)
            self.assertEqual([self.cards[cap['source_card_id']]['series']for cap in caps],[label])
        self.assertEqual(prepare_need_candidates(dict(n,source_block='Gráfico 4'),rows,services),rows)
        self.assertEqual(prepare_need_candidates(dict(n,book=None),rows,services),[rows[0]])
        # A default role cannot supply an endpoint absent from both figures.
        short=copy.deepcopy(self.row('Gasto bruto público en pensiones (% PIB)','Gráfico 1','2026',series='AIReF'))
        self.assertEqual(prepare_need_candidates(dict(n,periods=['2070']),[short,rows[1]],services),[])
        for row in rows[2:]:
            entry=self.entries[row['card_id']]
            self.assertNotIn('context_forecast_vintage',entry['proposed'])
            compact=m.compact_semantic_context(row,self.registry)
            self.assertIn('forecast_role',compact);self.assertNotIn('forecast_role_scope',compact)
            self.assertTrue(entry['proposed']['forecast_role_scope'])
            self.assertNotIn('forecast_vintage',compact)
        for row in self.cards.values():
            if row['book']=='2026'and row['block']in {'Gráfico 1','Gráfico 4'}and row['series']=='Histórico':
                self.assertNotIn(row['card_id'],self.entries)
        self.assertEqual(rows,before)

    @classmethod
    def setUpClass(cls):
        cls.catalog=json.loads((m.ROOT/'data/processed/langgraph_9b_assets/catalog.json').read_text(encoding='utf-8'))
        cls.registry=m.load_semantic_context(cls.catalog)
        cls.cards={card['card_id']:card for card in cls.catalog['cards']}
        cls.entries={cid:e for e in cls.registry['entries']for cid in e['source_card_ids']}
        (m.ROOT/'tmp/unit_graph').mkdir(parents=True,exist_ok=True)

    def row(self,metric,block,book='2025',unit='percent_gdp',series=None):
        return next(card for card in self.cards.values()if card['metric']==metric and card['block'].strip()==block
                    and card['book']==book and card['unit']==unit and(series is None or card['series']==series))

    def need(self,**changes):
        need=dict(need_id='n',kind='data',metric='educación',book='2026',series='AIReF',unit='percent_gdp',
                  periods=['2030'],quantity_request='level',scope_terms=[],population='aggregate',source_block=None)
        need.update(changes);return need

    def test_promoted_registry_is_exact_source_proposal_and_fixed_bytes(self):
        proposal=json.loads((m.ROOT/'evaluation/langgraph_rag_2026-11-01/design_audit/source_context_enrichment_proposal_v1.json').read_text(encoding='utf-8'))
        extension=json.loads((m.ROOT/'evaluation/langgraph_rag_2026-11-01/design_audit/source_context_enrichment_extension_proposal_v2.json').read_text(encoding='utf-8'))
        forecast=json.loads((m.ROOT/'evaluation/langgraph_rag_2026-11-01/design_audit/pension_forecast_context_extension_v3.json').read_text(encoding='utf-8'))
        expected=copy.deepcopy(proposal['entries']+extension['entries'])
        for annotation in forecast['existing_card_annotations']:
            for entry in expected:
                if entry['source_card_ids']==annotation['source_card_ids']:
                    self.assertEqual(entry['raw'],annotation['raw'])
                    self.assertEqual(set(annotation['proposed_additions']),{'forecast_role','forecast_role_scope'})
                    entry['proposed'].update(annotation['proposed_additions'])
        self.assertEqual(self.registry['entries'],expected+forecast['entries'])
        self.assertEqual(self.registry['primary_evidence'],{**proposal['primary_evidence'],**extension['primary_evidence'],**forecast['primary_evidence']})
        self.assertEqual(len(self.entries),71)
        self.assertEqual(len(self.registry['primary_evidence']),44)
        original=m.ROOT/'data/processed/graph_semantic_context_v1.json'
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),'c9210545c35fccf62d2fe0793e37b894f10d49b898904bfdb3a3ec280a874d38')
        previous=m.ROOT/'data/processed/graph_semantic_context_v2.json'
        self.assertEqual(hashlib.sha256(previous.read_bytes()).hexdigest(),'a838cca7f433d880a0e348315eb5a21940ffa8c28496dab8eb0e9c83e7bcd4b6')
        self.assertEqual(hashlib.sha256(m.CONTEXT_PATH.read_bytes()).hexdigest(),m.CONTEXT_SHA256)
        changed=copy.deepcopy(self.registry);changed['entries'][0]['proposed']['semantic_label']='Forged meaning'
        with self.assertRaisesRegex(ValueError,'In-memory'):m.verify_semantic_context(changed,self.catalog)

    def test_registry_file_and_raw_catalog_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp/unit_graph')as directory:
            path=Path(directory)/'registry.json';path.write_bytes(m.CONTEXT_PATH.read_bytes()+b' ')
            with patch.object(m,'CONTEXT_PATH',path),self.assertRaisesRegex(ValueError,'bytes changed'):
                m.load_semantic_context(self.catalog)
        changed=copy.deepcopy(self.catalog);changed['cards'][0]['title']='Altered'
        with self.assertRaises(ValueError):m.verify_semantic_context(self.registry,changed)

    def test_compaction_preserves_raw_labels_and_only_omits_duplicate_vintage(self):
        annual=self.row('Educación','Gráfico 25','2026',series='AIReF')
        before=copy.deepcopy(annual);context=m.compact_semantic_context(annual,self.registry)
        self.assertEqual(context['forecast_vintage'],'2026');self.assertEqual(annual,before)
        summary=self.row('Educación','Cuadro 12','2026',series='AIReF 2026')
        self.assertNotIn('forecast_vintage',m.compact_semantic_context(summary,self.registry))
        changed=dict(summary,series='Different')
        with self.assertRaisesRegex(ValueError,'Candidate differs'):m.compact_semantic_context(changed,self.registry)

    def test_related_education_is_discovered_without_mutating_source_or_copies(self):
        annual=copy.deepcopy(self.row('Educación','Gráfico 25','2026',series='AIReF'))
        annual['matched_need_ids']=['n'];before=copy.deepcopy(annual)
        summary=self.row('Educación','Cuadro 12','2026',series='AIReF 2026')
        result=m.expand_context_candidates(self.catalog,[annual],[self.need()],self.registry)
        self.assertIn(summary['card_id'],{card['card_id']for card in result})
        self.assertEqual(annual,before);self.assertNotIn('semantic_context',self.cards[annual['card_id']])
        self.assertEqual(next(card for card in result if card['card_id']==summary['card_id'])['series'],'AIReF 2026')

    def test_related_discovery_never_changes_vintage_period_book_unit_or_explicit_source(self):
        annual=copy.deepcopy(self.row('Educación','Gráfico 25','2026',series='AIReF'))
        annual['matched_need_ids']=['n']
        summary=self.row('Educación','Cuadro 12','2026',series='AIReF 2026')
        for changes in [dict(series='Opinión 2025'),dict(periods=['2199']),dict(book='2025'),
                        dict(unit='euros'),dict(unit='percentage_points_gdp'),dict(source_block='Gráfico 25')]:
            with self.subTest(changes=changes):
                result=m.expand_context_candidates(self.catalog,[annual],[self.need(**changes)],self.registry)
                self.assertNotIn(summary['card_id'],{card['card_id']for card in result})

    def test_discovery_defers_tentative_years_but_never_compiles_unavailable_horizon(self):
        annual=copy.deepcopy(self.row('Educación','Gráfico 25','2026',series='AIReF'));annual['matched_need_ids']=['n']
        summary=self.row('Educación','Cuadro 12','2026',series='AIReF 2026')
        result=m.expand_context_candidates(self.catalog,[annual],[self.need(periods=['2026','2050'])],self.registry,metadata_discovery=True)
        self.assertIn(summary['card_id'],{card['card_id']for card in result})
        services=SimpleNamespace(catalog=self.catalog,source_policy=dict(groups=[]))
        self.assertEqual(compile_capabilities(self.need(periods=['2199']),result,services),[])

    def test_production_authentication_cannot_disable_registry_in_memory(self):
        from rag_pensiones import graph_pipeline as pipeline
        service=SimpleNamespace(catalog=self.catalog,documents={},source_policy={},semantic_context=None,scope='production_full')
        with patch.object(pipeline,'validate_catalog'),patch.object(pipeline,'validate_document_corpus'),patch.object(pipeline,'verify_policy_asset'):
            with self.assertRaisesRegex(ValueError,'registry is required'):pipeline.GraphServices.authenticate(service)

    def test_subtotals_remain_visible_independent_of_query_and_history(self):
        for name in ['Modelizados','No modelizados']:
            row=self.row(name,'Cuadro 9');before=copy.deepcopy(row)
            for query in ['Gasto total','Otra redaccion',None]:
                self.assertTrue(m.subtotal_scope_allowed(row,self.registry,query))
                m.enforce_context_scope(dict(selections=[dict(card_ids=[row['card_id']])]),[row],self.registry,query)
            view=m.source_measure_view(row,m.compact_semantic_context(row,self.registry))
            self.assertEqual(view['semantic_context']['source_scope'],'subtotal')
            self.assertEqual(view['published_measure_name'],row['metric'])
            self.assertEqual(row,before)

    def test_manual_execution_revalidates_subtotal_and_authentic_card_before_sql(self):
        row=self.row('Modelizados','Cuadro 9');need=self.need(book='2025',metric='Modelizados',series=row['series'],periods=['2050'])
        plan=dict(status='ready',selections=[dict(need_id='n',card_ids=[row['card_id']],periods=['2050'],operation='lookup')])
        services=SimpleNamespace(catalog=self.catalog,semantic_context=self.registry)
        with patch('rag_pensiones.graph_validation.policy.query_observation')as sql:
            m.enforce_context_scope(plan,[row],self.registry,query='Gasto total')
            sql.assert_not_called()
            forged=dict(row,card_id='forged');plan['selections'][0]['card_ids']=['forged']
            with self.assertRaisesRegex(ValueError,'authenticated catalog'):execute_plan(plan,dict(needs=[need]),[forged],services,query='Modelizados')
            sql.assert_not_called()
            disabled=SimpleNamespace(catalog=self.catalog,semantic_context=None,scope='production_full')
            with self.assertRaisesRegex(ValueError,'registry is required before SQL'):
                execute_plan(plan,dict(needs=[need]),[forged],disabled,query='Modelizados')
            sql.assert_not_called()

    def test_cap_options_are_self_contained_and_bind_actual_capability(self):
        row=copy.deepcopy(self.row('Educación','Cuadro 12','2026'));row['matched_need_ids']=['n']
        services=SimpleNamespace(catalog=self.catalog,source_policy=dict(groups=[]))
        need=self.need(series=row['series']);caps=compile_capabilities(need,[row],services)
        options=m.capability_options(caps,[row],self.registry)
        self.assertEqual([o['capability_id']for o in options],[c['capability_id']for c in caps])
        source=options[0]['source_metadata'][0]
        self.assertEqual(next(iter(source)),'measure')
        self.assertEqual(source['measure'],self.entries[row['card_id']]['proposed']['semantic_label'])
        self.assertEqual(source['published_measure_name'],row['metric']);self.assertEqual(source['series'],row['series'])
        self.assertEqual(source['series_role'],self.entries[row['card_id']]['proposed']['series_role'])
        self.assertIn('semantic_context',source);self.assertNotIn('title',source);self.assertNotIn('value',source)
        parents=self.entries[row['card_id']]['proposed'].get('parent_card_ids',[])
        self.assertEqual(source['semantic_context'].get('parent_card_ids',[]),parents)
        self.assertEqual(m.compact_semantic_context(row,self.registry).get('parent_card_ids',[]),parents)
        from rag_pensiones.graph_pipeline import pack_candidate_table,unpack_candidate_table
        restored=unpack_candidate_table(pack_candidate_table([source]))[0]
        self.assertEqual(restored['semantic_context'].get('parent_card_ids',[]),parents)

    def test_expanded_family_and_grounding_use_same_verified_measure_and_literal_series(self):
        from rag_pensiones.graph_planning import grounding_inputs
        from rag_pensiones.graph_pipeline import pack_candidate_table,unpack_candidate_table
        row=copy.deepcopy(self.row('Cuidados larga duración','Cuadro 9'))
        n=self.need(book='2025',series=None,periods=['2050'])
        enriched=m.expand_context_candidates(self.catalog,[dict(row,matched_need_ids=['n'])],[n],self.registry)
        context=m.compact_semantic_context(row,self.registry)
        self.assertEqual(context['semantic_label'],'Gasto público en cuidados de larga duración')
        unreviewed=next(copy.deepcopy(c)for c in self.cards.values()if c['card_id']not in self.entries
            and c['book']=='2025'and c['period_kind']=='year'and c['unit']=='percent_gdp'and'2050'in c['periods'])
        unreviewed['matched_need_ids']=['n']
        sources=grounding_inputs(dict(needs=[n]),[*enriched,unreviewed],'Dato en 2050')['n']['sources']
        actual=next(s for s in sources if s['card_id']==row['card_id'])
        self.assertEqual(actual['measure'],context['semantic_label'])
        self.assertEqual(actual['published_measure_name'],row['metric'])
        self.assertEqual(actual['series'],row['series'])
        self.assertEqual(actual['series_role'],context['series_role'])
        other=next(s for s in sources if s['card_id']==unreviewed['card_id'])
        self.assertIsNone(other['semantic_context']);self.assertIsNone(other['series_role'])
        self.assertEqual(unpack_candidate_table(pack_candidate_table(sources)),sources)




class ForecastDefaultWithoutBookTests(unittest.TestCase):
    def test_default_family_identity_and_horizon_remain_strict_without_requested_book(self):
        base=dict(card_id='base',metric='Published source',series='Base',book='2090',unit='euros',
                  quantity_role='level',period_kind='year',periods=['2100','2110'],scope='aggregate',block='Table')
        old=dict(base,card_id='old',series='Previous')
        n=dict(need_id='n',kind='data',book=None,series=None,source_block=None,unit='euros',quantity_request='level',periods=['2110'])
        def evaluate(rows,labels=None,roles=None,need=None,registry={}):
            entries={r['card_id']:dict(raw=copy.deepcopy(r),proposed=dict(semantic_label=(labels or {}).get(r['card_id'],'Family'),
                forecast_role=(roles or {}).get(r['card_id'],'report_base'if r['card_id']=='base'else'prior_report')))for r in rows}
            before=copy.deepcopy((rows,entries))
            with patch.object(m,'_entries',return_value=entries):result=m.prefer_current_forecast(need or n,rows,registry)
            self.assertEqual((rows,entries),before)
            return {r['card_id']for r in result}
        self.assertEqual(evaluate([base,old]),{'base'})
        self.assertEqual(evaluate([old]),{'old'})
        for changes in [dict(book='2080'),dict(unit='persons'),dict(quantity_role='change'),dict(period_kind='interval',periods=['2100-2110'])]:
            different=dict(old,**changes)
            self.assertIn('old',evaluate([base,different]),changes)
        self.assertIn('old',evaluate([base,old],labels={'old':'Other metric'}))
        self.assertIn('old',evaluate([base,old],roles={'old':None}))
        self.assertIn('old',evaluate([dict(base,periods=['2100']),old]))
        self.assertIn('old',evaluate([base,old],need=dict(n,periods=['2199'])))
        for changes in [dict(series='Previous'),dict(source_block='Table'),dict(book='2080')]:
            self.assertEqual(evaluate([base,old],need=dict(n,**changes)),{'base','old'})
        self.assertEqual(evaluate([base,old],registry=None),{'base','old'})
        self.assertEqual(evaluate([base,old],need=dict(n,periods=[])),{'base'})
        self.assertIn('old',evaluate([base,dict(old,periods=['2110'])],need=dict(n,periods=[])))
        otherbase=dict(base,card_id='otherbase',book='2080')
        otherold=dict(old,card_id='otherold',book='2080')
        self.assertEqual(evaluate([base,old,otherold]),{'base','otherold'})
        self.assertEqual(evaluate([base,old,otherbase,otherold],roles={'otherbase':'report_base'}),{'base','otherbase'})


if __name__=='__main__':unittest.main()
