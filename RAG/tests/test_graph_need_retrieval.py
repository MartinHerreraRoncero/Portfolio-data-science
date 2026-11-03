"""Metadata filtering and fair candidate exposure, without any model inference."""
import copy
import json
import unittest

from rag_pensiones import graph_catalog as m
from rag_pensiones import graph_pipeline as pipeline
from rag_pensiones.graph_types import EMBED_DIMENSION
from rag_pensiones.graph_models import GraphStageClient
from rag_pensiones.graph_validation import validate_plan,InvalidPlan


def catalog(cards):
    observations = {}
    for card in cards:
        card.update(block='synthetic', metric=card.get('metric', 'indicator'), series='reference',
                    period_kind=card.get('period_kind', 'year'), scope='aggregate', quantity_role='level',
                    title='Synthetic source', period_warnings=[], observation_ids=[])
        for period in card['periods']:
            oid = card['card_id'] + ':' + period
            card['observation_ids'].append(oid)
            observations[oid] = dict(observation_id=oid, period=period, value='1', **{
                k: card[k] for k in ['book', 'block', 'metric', 'series', 'unit',
                                     'period_kind', 'scope', 'quantity_role']})
    result = dict(schema_version=m.SCHEMA, database_sha256='synthetic', sources={},
                  cards=cards, observations=observations)
    result['catalog_hash'] = m.canonical_hash(m._content(result))
    return result


def card(cid, book='2026', unit='percent_gdp', periods=None, description='indicator'):
    return dict(card_id=cid, book=book, unit=unit,period_kind='year', periods=periods or ['2050'], description=description)


def need(nid='N1', book='2026', unit='percent_gdp', periods=None, query='indicator'):
    return dict(need_id=nid, kind='data', book=book, unit=unit, periods=periods or ['2050'], query_span=query)


class GraphNeedRetrievalTests(unittest.TestCase):
    def test_discovery_retains_a_possible_safe_pair_until_grounding_resolves_endpoints(self):
        from types import SimpleNamespace
        from rag_pensiones.graph_planning import grounding_inputs,apply_grounding,compile_capabilities
        c=catalog([card('amount',book='2025',unit='euros',periods=['2050','2070'],description='importe')])
        for observation in c['observations'].values():observation['calculation_safe']=True
        c['catalog_hash']=m.canonical_hash(m._content(c));row=c['cards'][0]
        service=SimpleNamespace(catalog=c,source_policy=dict(groups=[]))
        q='Informe 2025: compara el importe de 2050 y 2070.'
        for periods in [[],['2025','2050','2070']]:
            with self.subTest(periods=periods):
                n=dict(need(book='2025',unit='euros'),metric='importe',quantity_request='change',periods=periods,
                       population='aggregate',scope_terms=[],series=None,source_block=None)
                before=copy.deepcopy(n)
                self.assertFalse(m.derivation_operands_eligible(row,n,c))
                self.assertTrue(m.derivation_operands_eligible(row,n,c,metadata_discovery=True))
                self.assertEqual(m.retrieve_catalog_for_needs(c,q,None,[n]),[])
                found=m.retrieve_catalog_for_needs(c,q,None,[n],metadata_discovery=True)
                self.assertEqual([source['card_id']for source in found],['amount'])
                inputs=grounding_inputs(dict(needs=[n]),found,q)
                self.assertIn('N1',inputs)
                self.assertEqual(compile_capabilities(n,found,service),[])
                self.assertEqual(n,before)
                if periods:
                    intent,_=apply_grounding(dict(needs=[n]),inputs,dict(N1=dict(scenario=None,target_years=['2050','2070'])))
                    caps=compile_capabilities(intent['needs'][0],found,service)
                    self.assertEqual([cap['selection']['operation']for cap in caps],['difference'])
                    self.assertEqual(caps[0]['selection']['periods'],['2050','2070'])
                missing=dict(n,periods=['2199'])
                self.assertEqual(compile_capabilities(missing,found,service),[])
        for periods in [[],['2025','2050','2070']]:
            n=dict(need(book='2025',unit='euros'),quantity_request='change',periods=periods)
            for edit in [dict(calculation_safe=False),dict(global_calculation_prohibition=True),dict(value='NaN'),
                         dict(value='Infinity'),dict(series='incompatible')]:
                changed=copy.deepcopy(c);changed['observations'][row['observation_ids'][1]].update(edit)
                self.assertFalse(m.derivation_operands_eligible(row,n,changed,metadata_discovery=True),edit)
        n=dict(n,periods=['2025','2199','2200'])
        self.assertFalse(m.derivation_operands_eligible(row,n,c,metadata_discovery=True))
        changed=copy.deepcopy(c)
        for observation in changed['observations'].values():observation['series']='same but not source identity'
        self.assertFalse(m.derivation_operands_eligible(row,dict(n,periods=[]),changed,metadata_discovery=True))

    def test_interval_discovery_defers_incomplete_horizons_but_never_accepts_one_year(self):
        mean=dict(card('mean',unit='euros',periods=['2050-2070']),period_kind='interval')
        change=dict(mean,unit='percentage_points_gdp',quantity_role='published_change')
        for source,quantity,unit in [(mean,'mean','euros'),(change,'change','percentage_points_gdp')]:
            for periods in [[],['2025','2050','2070']]:
                n=dict(need(unit=unit),quantity_request=quantity,periods=periods)
                self.assertTrue(m.candidate_matches_need(source,n,metadata_discovery=True))
            n=dict(n,periods=['2050'])
            self.assertFalse(m.candidate_matches_need(source,n,metadata_discovery=True))

    def test_one_year_growth_excludes_levels_and_intervals_before_grounding(self):
        from rag_pensiones.graph_planning import grounding_inputs
        c=catalog([card('gdp-level',unit='percent',periods=['2024','2025'],description='growth'),
            card('total-growth',unit='percent_growth',periods=['2024'],description='growth'),
            card('component-growth',unit='percent_growth',periods=['2024'],description='growth'),
            dict(card('interval-mean',unit='percent',periods=['2020-2024'],description='growth'),period_kind='interval')])
        c['cards'][0]['series']='Opinión 2025'
        c['cards'][1]['series']='Total';c['cards'][2]['series']='Componente'
        c['cards'][1]['quantity_role']=c['cards'][2]['quantity_role']='annual_growth'
        for row in c['cards']:
            for oid in row['observation_ids']:
                c['observations'][oid].update(series=row['series'],quantity_role=row['quantity_role'])
        for obs in c['observations'].values():obs['calculation_safe']=True
        c['catalog_hash']=m.canonical_hash(m._content(c))
        n=dict(need(unit='unspecified',periods=['2024']),quantity_request='change',series='2025',metric='growth')
        for discovery in [False,True]:
            found=m.retrieve_catalog_for_needs(c,'growth',None,[n],metadata_discovery=discovery)
            self.assertEqual({row['card_id']for row in found},{'total-growth','component-growth'})
            self.assertFalse(m.candidate_matches_need(c['cards'][0],n,metadata_discovery=discovery))
            self.assertFalse(m.candidate_matches_need(c['cards'][3],n,metadata_discovery=discovery))
        sources=grounding_inputs(dict(needs=[n]),c['cards'],'Growth en 2024')['N1']['sources']
        self.assertEqual({row['card_id']for row in sources},{'total-growth','component-growth'})
        self.assertFalse(m.derivation_operands_eligible(c['cards'][0],n,c))
        full=dict(n,periods=['2024','2025'])
        self.assertTrue(m.derivation_operands_eligible(c['cards'][0],full,c))
        c['observations'][c['cards'][0]['observation_ids'][1]]['calculation_safe']=False
        self.assertFalse(m.derivation_operands_eligible(c['cards'][0],full,c))
        annual=dict(c['cards'][1],periods=['2024','2025'])
        self.assertFalse(m.candidate_matches_need(annual,full,metadata_discovery=True))

    def test_published_change_recall_is_not_consumed_by_unsafe_level_operands(self):
        levels=[card('level'+str(i),periods=['2050','2070'],description='indicator '*20)for i in range(20)]
        published=dict(card('published',unit='percentage_points_gdp',periods=['2050-2070'],description='indicator'),period_kind='interval')
        c=catalog([*levels,published]);n=dict(need(unit='percentage_points_gdp',periods=['2050-2070']),quantity_request='change')
        before=copy.deepcopy(c)
        found=m.retrieve_catalog_for_needs(c,'indicator',None,[n],top_k_per_need=1)
        self.assertEqual([row['card_id']for row in found],['published'])
        self.assertEqual(c,before)
        level=c['cards'][0]
        for oid in level['observation_ids']:c['observations'][oid]['calculation_safe']=True
        self.assertTrue(m.derivation_operands_eligible(level,n,c))
        c['catalog_hash']=m.canonical_hash(m._content(c))
        c['observations'][level['observation_ids'][1]]['series']='different'
        self.assertFalse(m.derivation_operands_eligible(level,n,c))
        c['observations'][level['observation_ids'][1]]['series']='reference'
        c['observations'][level['observation_ids'][0]]['global_calculation_prohibition']=True
        self.assertFalse(m.derivation_operands_eligible(level,n,c))
        self.assertFalse(m.derivation_operands_eligible(level,dict(n,periods=['2050']),c))
        # The same eligibility contract applies to ordinary percentage points.
        plain=catalog([card('unsafe',unit='percent',periods=['2050','2070']),
            dict(card('change',unit='percentage_points',periods=['2050-2070']),period_kind='interval')])
        self.assertEqual([r['card_id']for r in m.retrieve_catalog_for_needs(plain,'indicator',None,
            [dict(n,unit='percentage_points')],top_k_per_need=1)],['change'])

    def test_unrelated_published_intervals_cannot_displace_a_safe_relevant_level(self):
        relevant=dict(card('pension',periods=['2050','2070'],description='pensiones gasto'),metric='pensiones')
        unrelated=[dict(card('health'+str(i),unit='percentage_points_gdp',periods=['2050-2070'],description='sanidad gasto'),
                        metric='sanidad',period_kind='interval')for i in range(12)]
        c=catalog([relevant,*unrelated])
        for oid in c['cards'][0]['observation_ids']:c['observations'][oid]['calculation_safe']=True
        c['catalog_hash']=m.canonical_hash(m._content(c))
        n=dict(need(unit='percentage_points_gdp',periods=['2050-2070']),quantity_request='change',metric='pensiones gasto')
        self.assertEqual(m.retrieve_catalog_for_needs(c,'pensiones gasto',None,[n],top_k_per_need=1)[0]['card_id'],'pension')
        vector=[1.]+[0.]*(EMBED_DIMENSION-1);other=[0.,1.]+[0.]*(EMBED_DIMENSION-2)
        hybrid=m.attach_embeddings(c,[vector,*[other]*len(unrelated)])
        self.assertEqual(m.retrieve_catalog_for_needs(hybrid,'pensiones gasto',vector,[n],top_k_per_need=1)[0]['card_id'],'pension')

    def test_source_context_is_preserved_and_duplicate_need_selections_rejected(self):
        c=catalog([card('source')]); n=need()
        compact=pipeline._compact_cards(dict(candidates=c['cards'],interpretation=dict(needs=[n])))
        self.assertEqual(compact[0]['block'],c['cards'][0]['block'])
        self.assertEqual(compact[0]['title'],c['cards'][0]['title'])
        selection=dict(need_id='N1',card_ids=['source'],periods=['2050'],operation='lookup')
        with self.assertRaisesRegex(InvalidPlan,'every interpreted need'):
            validate_plan(dict(status='ready',selections=[selection,copy.deepcopy(selection)]),
                          dict(needs=[n]),c['cards'])

    def test_pool_is_filtered_before_topk_not_after_it(self):
        c = catalog([card('wrong', book='2025', description='target ' * 20),
                     card('right', description='target ' + 'unrelated ' * 100)])
        self.assertEqual(m.retrieve_catalog(c, 'target', top_k=1)[0]['card_id'], 'wrong')
        found = m.retrieve_catalog_for_needs(c, 'target', None, [need(query='target')], top_k_per_need=1)
        self.assertEqual([x['card_id'] for x in found], ['right'])
        self.assertEqual(found[0]['retrieval']['rank'], 1)
        with self.assertRaises(ValueError):
            m.retrieve_catalog(c, 'target', allowed_card_ids={'missing'})

    def test_units_remain_strict_and_levels_only_offer_possible_difference_operands(self):
        self.assertFalse(m.candidate_matches_need(card('pp', unit='percentage_points_gdp'), need()))
        self.assertTrue(m.candidate_matches_need(card('level'), need(unit='percentage_points_gdp')))
        self.assertFalse(m.candidate_matches_need(card('level'), need(unit='euros')))
        self.assertTrue(m.candidate_matches_need(card('money', unit='euros_per_person'), need(unit='euros')))
        self.assertFalse(m.candidate_matches_need(card('share'), need(unit='pensions_count')))
        self.assertTrue(m.candidate_matches_need(card('count', unit='pensions_count'), need(unit='pensions_count')))
        self.assertTrue(m.candidate_matches_need(card('people', unit='persons'), need(unit='persons')))
        self.assertFalse(m.candidate_matches_need(card('count', unit='pensions_count'), need(unit='persons')))
        self.assertFalse(m.candidate_matches_need(card('level'), dict(need(), unit='invented')))

    def test_intervals_offer_only_published_labels_or_their_actual_endpoints(self):
        interval = dict(card('interval', periods=['2023-2070']),period_kind='interval')
        self.assertFalse(m.candidate_matches_need(interval, need(periods=['2023', '2070'])))
        self.assertTrue(m.candidate_matches_need(interval, need(periods=['2023-2070'])))
        self.assertFalse(m.candidate_matches_need(interval, need(periods=['2050'])))
        levels = dict(card('levels', periods=['2023', '2070']),period_kind='year')
        self.assertFalse(m.candidate_matches_need(levels, need(periods=['2023-2070'])))
        self.assertTrue(m.candidate_matches_need(levels, need(unit='percentage_points_gdp',periods=['2023-2070'])))
        self.assertFalse(m.candidate_matches_need(levels, need(periods=['2023', '2040', '2070'])))

    def test_point_level_never_matches_a_mean_interval_even_through_a_manual_plan(self):
        c=catalog([dict(card('mean',periods=['2022-2050']),period_kind='interval')])
        n=dict(need(periods=['2050']),population='aggregate',scope_terms=[],series=None)
        self.assertFalse(m.candidate_matches_need(c['cards'][0],n))
        for operation in ['lookup','published_change','difference']:
            selection=dict(need_id='N1',card_ids=['mean'],periods=['2022-2050'],operation=operation)
            with self.subTest(operation=operation),self.assertRaisesRegex(InvalidPlan,'temporal shape'):
                validate_plan(dict(status='ready',selections=[selection]),dict(needs=[n]),c['cards'])
        average=dict(n,periods=['2022-2050'])
        self.assertTrue(m.candidate_matches_need(c['cards'][0],average))
        selected=dict(need_id='N1',card_ids=['mean'],periods=['2022-2050'],operation='lookup')
        validate_plan(dict(status='ready',selections=[selected]),dict(needs=[average]),c['cards'])

    def test_published_change_requires_one_real_full_interval_not_two_adjacent_ones(self):
        n=dict(need(unit='percentage_points_gdp',periods=['2023','2070']),
               population='aggregate',scope_terms=[],series=None)
        full=catalog([dict(card('full',unit='percentage_points_gdp',periods=['2023-2070']),period_kind='interval')])
        self.assertTrue(m.candidate_matches_need(full['cards'][0],n))
        selected=dict(need_id='N1',card_ids=['full'],periods=['2023-2070'],operation='published_change')
        validate_plan(dict(status='ready',selections=[selected]),dict(needs=[n]),full['cards'])
        for requested in [['2023'],['2050']]:
            self.assertFalse(m.candidate_matches_need(full['cards'][0],dict(n,periods=requested)))
        adjacent=catalog([dict(card('adjacent',unit='percentage_points_gdp',periods=['2023-2050','2050-2070']),period_kind='interval')])
        self.assertFalse(m.candidate_matches_need(adjacent['cards'][0],n))
        selected=dict(selected,card_ids=['adjacent'],periods=['2023-2050','2050-2070'])
        with self.assertRaisesRegex(InvalidPlan,'temporal shape'):
            validate_plan(dict(status='ready',selections=[selected]),dict(needs=[n]),adjacent['cards'])
        self.assertFalse(m.candidate_matches_need(dict(adjacent['cards'][0],unit='percent_gdp'),n))

    def test_unspecified_edition_keeps_both_books_even_with_small_budget(self):
        c = catalog([card('a', book='2025'), card('b', book='2026')])
        found = m.retrieve_catalog_for_needs(c, 'indicator', None, [need(book=None)], top_k_per_need=1)
        self.assertEqual({x['book'] for x in found}, {'2025', '2026'})
        explicit = m.retrieve_catalog_for_needs(c, 'indicator', None, [need(book='2025')])
        self.assertEqual({x['book'] for x in explicit}, {'2025'})

    def test_needs_are_interleaved_and_shared_cards_have_all_need_provenance(self):
        c = catalog([card('a1', book='2025'), card('a2', book='2025'), card('b', book='2026')])
        needs = [need('old', book='2025'), need('new'), need('shared', book='2025')]
        found = m.retrieve_catalog_for_needs(c, 'indicator', None, needs)
        self.assertEqual(found[0]['matched_need_ids'], ['old', 'shared'])
        self.assertEqual(found[1]['matched_need_ids'], ['new'])
        self.assertEqual(len(found), len({x['card_id'] for x in found}))
        self.assertEqual({x['need_id'] for x in found[0]['need_retrieval']}, {'old', 'shared'})
        self.assertEqual(m.retrieve_catalog_for_needs(c, 'indicator', None, [dict(need(), kind='text')]), [])

    def test_hybrid_indices_and_full_catalog_stay_unchanged(self):
        c = catalog([card('a', book='2025'), card('b')])
        v = [1.] + [0.] * (EMBED_DIMENSION - 1)
        c = m.attach_embeddings(c, [v, v]); before = copy.deepcopy(c)
        found = m.retrieve_catalog_for_needs(c, 'indicator', v, [need()])
        self.assertEqual([x['card_id'] for x in found], ['b'])
        self.assertEqual(found[0]['retrieval']['mode'], 'hybrid_rrf')
        self.assertEqual(c, before)
        found[0]['periods'].append('2099')
        self.assertEqual(c, before)

    def test_followup_search_keeps_the_resolved_context_and_interpreted_metric(self):
        c = catalog([card('p', description='pensions'), card('i', description='interest')])
        n = need(query='and in that year?'); n['metric'] = 'pensions'
        found = m.retrieve_catalog_for_needs(c, 'pensions forecast', None, [n], top_k_per_need=1)
        self.assertEqual(found[0]['card_id'], 'p')

    def test_distinct_concepts_in_one_query_get_distinct_lexical_rankings(self):
        c = catalog([card('p', description='pensions'), card('h', description='health')])
        pension = dict(need('pension', query='pensions and health'), metric='pensions')
        health = dict(need('health', query='pensions and health'), metric='health')
        found = m.retrieve_catalog_for_needs(c, 'pensions and health', None,
                                             [pension, health], top_k_per_need=1)
        self.assertEqual([(r['card_id'], r['matched_need_ids']) for r in found],
                         [('p', ['pension']), ('h', ['health'])])

    def test_packing_keeps_a_candidate_for_each_independent_need(self):
        class Client:
            calls = []
            def chat(self, messages, schema, stage):
                self.payload = json.loads(messages[1]['content'])
                return {}
        class Services:
            client = Client()
        services = Services()
        rows = [dict(card_id=f'{nid}-{i}', matched_need_ids=[nid], metric='m' * 900+f'{nid}-{i}')
                for i in range(3) for nid in ['N1', 'N2', 'N3']]
        pipeline._chat(services, 'plan', dict(candidates=rows), pipeline.obj({}))
        kept = pipeline.unpack_candidate_table(services.client.payload['candidates'])
        self.assertGreater(services.client.payload['omitted_candidate_count'], 0)
        self.assertEqual({nid for c in kept for nid in c['matched_need_ids']}, {'N1', 'N2', 'N3'})

    def test_packing_preserves_each_edition_for_each_ambiguous_need(self):
        class Client:
            calls = []
            def chat(self, messages, schema, stage):
                self.payload = json.loads(messages[1]['content'])
                return {}
        class Services:
            client = Client()
        services = Services()
        rows = [dict(card_id=f'{nid}-{book}-{i}', book=book, matched_need_ids=[nid], metric='m' * 900+f'{nid}-{book}-{i}')
                for i in range(2) for nid in ['N1', 'N2'] for book in ['2025', '2026']]
        pipeline._chat(services, 'plan', dict(candidates=rows), pipeline.obj({}))
        kept = pipeline.unpack_candidate_table(services.client.payload['candidates'])
        self.assertGreater(services.client.payload['omitted_candidate_count'], 0)
        self.assertEqual({(nid,c['book']) for c in kept for nid in c['matched_need_ids']},
                         {('N1','2025'),('N1','2026'),('N2','2025'),('N2','2026')})

    def test_candidate_table_roundtrip_preserves_every_field_and_identifier(self):
        rows=[dict(card_id=f'card:actual-{i}',book='2025',block=f'Objeto {i}',
                   metric='magnitud '+str(i),title='magnitud '+str(i)if i%2 else'otro título',
                   series='serie publicada',unit='percent_gdp',scope='',role='level',
                   period_kind='year',periods=['2023','2050'],matched_need_ids=['N1','N2'],
                   unsafe_periods=[] if i%2 else['2023'],available_period_count=48)
              for i in range(12)]
        original=copy.deepcopy(rows);packed=pipeline.pack_candidate_table(rows)
        self.assertEqual(pipeline.unpack_candidate_table(packed),original)
        self.assertEqual(rows,original)
        self.assertNotIn('card_id',packed['common'])
        self.assertLess(len(json.dumps(packed)),len(json.dumps(rows)))
        decoded=pipeline.unpack_candidate_table(packed);decoded[0]['periods'].append('9999')
        self.assertEqual(rows,original)
        self.assertEqual(pipeline.unpack_candidate_table(pipeline.pack_candidate_table([])),[])

    def test_malformed_candidate_tables_fail_without_losing_identity(self):
        packed=pipeline.pack_candidate_table([dict(card_id='a',metric='x',title='x')])
        bad=copy.deepcopy(packed);bad['columns'].append('card_id')
        with self.assertRaises(ValueError):pipeline.unpack_candidate_table(bad)
        bad=copy.deepcopy(packed);bad['rows'][0]=[]
        with self.assertRaises(ValueError):pipeline.unpack_candidate_table(bad)
        bad=copy.deepcopy(packed);bad['rows']*=2
        with self.assertRaises(ValueError):pipeline.unpack_candidate_table(bad)
        with self.assertRaises(ValueError):pipeline.pack_candidate_table([dict(card_id='a',title=None)])
        with self.assertRaises(ValueError):pipeline.pack_candidate_table([dict(card_id='a'),dict(card_id='b',metric='x')])

    def test_policy_candidate_cannot_displace_another_concepts_best_original_source(self):
        class Client:
            calls=[]
            def chat(self,messages,schema,stage):
                self.payload=json.loads(messages[1]['content']);return {}
        class Services:client=Client()
        rows=[dict(card_id='policy',book='2025',matched_need_ids=['N1','N2'],need_ranks={'N1':9,'N2':1},
                   source_policy=[dict(role='preferred_summary')],metric='p'*900)]
        rows += [dict(card_id=f'original-{i}',book='2025',matched_need_ids=['N1'],need_ranks={'N1':i+1},
                      source_policy=[],metric='x'*900+str(i))for i in range(7)]
        services=Services();pipeline._chat(services,'plan',dict(candidates=rows),pipeline.obj({}))
        kept=pipeline.unpack_candidate_table(services.client.payload['candidates'])
        self.assertGreater(services.client.payload['omitted_candidate_count'],0)
        self.assertTrue({'policy','original-0'}<={r['card_id']for r in kept})

    def test_oversized_protected_minimum_fails_before_any_model_request(self):
        class NoNetwork:
            def __init__(self):self.calls=[]
            def request(self,*args):self.calls.append(args);raise AssertionError('Network forbidden')
        transport=NoNetwork()
        class Services:client=GraphStageClient(transport=transport)
        rows=[dict(card_id=str(i),book='2025',matched_need_ids=[f'N{i}'],metric='x'*5000+str(i))for i in range(2)]
        with self.assertRaisesRegex(ValueError,'byte budget'):
            pipeline._chat(Services(),'plan',dict(candidates=rows),pipeline.obj({}))
        self.assertEqual(transport.calls,[])


if __name__ == '__main__':
    unittest.main()
