import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
from rag_pensiones import graph_pipeline

class TopologyTests(unittest.TestCase):
    def test_ui_nodes_and_routes_match_actual_graph_builder(self):
        class Capture:
            def __init__(self,*args):self.nodes=[];self.edges=[];self.conditions=[]
            def add_node(self,name,fn):self.nodes.append(name)
            def add_edge(self,source,target):
                self.edges.extend((name,target) for name in (source if isinstance(source,list) else [source]))
            def add_conditional_edges(self,source,fn):self.conditions.append((source,fn))
            def compile(self,**kwargs):return self
        with patch.object(graph_pipeline,'StateGraph',Capture):graph=graph_pipeline.build_graph(None)
        output=subprocess.run(['node','-e',"const g=require('./experiments/prototype_chat.js');process.stdout.write(JSON.stringify({nodes:g.GRAPH_NODES.map(n=>n[0]),edges:g.GRAPH_EDGES}));"],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,check=True)
        ui=json.loads(output.stdout)
        self.assertEqual(set(ui['nodes']),set(graph.nodes))
        states=[]
        for error in [[],['synthetic']]:
            for ambiguous in [False,True]:
                for valid in [False,True]:
                    for status in ['ready','clarification']:
                        for repairs in [0,1,2]:
                            for missing in [False,True]:
                                states.append(dict(errors=error,conversation_resolution={'status':'ambiguous' if ambiguous else 'resolved'},plan_valid=valid,plan={'status':status},repairs=repairs,text_missing=missing))
        expected=set(graph.edges)
        for source,condition in graph.conditions:
            for state in states:
                targets=condition(state)
                expected.update((source,target)for target in (targets if isinstance(targets,list)else [targets]))
        self.assertEqual({(a,b)for a,b,_ in ui['edges']},expected)

if __name__=='__main__':unittest.main()
