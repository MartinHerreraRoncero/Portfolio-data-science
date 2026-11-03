"""User-facing local semantic graph, exclusively9B for text and4B for embeddings."""
import os
os.environ['LANGSMITH_TRACING']='false'
os.environ['LANGCHAIN_TRACING_V2']='false'
import argparse
import json
from pathlib import Path
from rag_pensiones.graph_pipeline import answer_graph,format_graph_answer,load_default_services
from rag_pensiones.structured_retrieval import ROOT
from rag_pensiones.qwen_embeddings import _json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query',required=True)
    parser.add_argument('--context',type=Path,help='JSON conversation history or explicit context')
    parser.add_argument('--assets',type=Path)
    parser.add_argument('--thread-id')
    parser.add_argument('--resume',help='Answer a pending material clarification')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/langgraph_rag/checkpoints.sqlite3')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--json',action='store_true')
    parser.add_argument('--progress',action='store_true',help='Emit content-free local node events on stdout')
    args=parser.parse_args(argv)
    def safe(path):
        path=Path(path).resolve();path.relative_to(ROOT);return path
    destination=safe(args.output)if args.output else None
    if destination and destination.exists():raise FileExistsError('Query output exists; use a new path')
    context=_json(safe(args.context).read_bytes())if args.context else None
    from contextlib import nullcontext
    if args.progress:
        from experiments.prototype_progress import observe_nodes,stdout_event
        stdout_event(dict(node='load',state='start'))
    services=load_default_services(asset_dir=safe(args.assets)if args.assets else None)
    if args.progress:stdout_event(dict(node='load',state='end'))
    with observe_nodes(stdout_event) if args.progress else nullcontext():
        report=answer_graph(args.query,context=context,thread_id=args.thread_id,resume=args.resume,
                            checkpoint_path=safe(args.checkpoint),services=services)
    if destination:
        if args.progress:stdout_event(dict(node='report',state='start'))
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
        if args.progress:stdout_event(dict(node='report',state='end'))
    # Single-line JSON keeps model-supplied text from impersonating telemetry lines.
    print(json.dumps(report if args.json else {'status':report['status']},ensure_ascii=False,separators=(',',':'))if args.progress else
          json.dumps(report,ensure_ascii=False,indent=2)if args.json else format_graph_answer(report))
    return int(report['status']=='error')


if __name__=='__main__':raise SystemExit(main())
