"""JSON contracts for the source-bound LangGraph profile."""
from dataclasses import asdict, dataclass, field
import copy
from typing import Any

PROFILE = 'langgraph-9b-v8'
CHAT_MODEL = 'qwen3.5:9b-q8_0'
CHAT_DIGEST = 'a8ee95bc3592ebaa624014ca8b429d25e0ef5f45030dc35786da7ff7e9e1b60a'
EMBED_MODEL = 'qwen3-embedding:4b'
EMBED_DIGEST = 'df5bd2e3c74cd8d069d21dc038f1b359fcdc9458fce1c99bd43c9eb1518ff907'
EMBED_DIMENSION = 2560
EMBED_KEEP_ALIVE = -1
EMBED_OPTIONS = {'num_gpu': 0, 'num_ctx': 8192, 'num_batch': 128}
CHAT_OPTIONS = {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0.0, 'repeat_penalty': 1.0, 'seed': 42, 'num_ctx': 32768, 'num_predict': 8192, 'presence_penalty': 1.5}
CHAT_TEMPLATE_TOKEN_RESERVE = 128
MAX_THINKING_OUTPUT_TOKENS = 24576


def chat_input_bound(messages):
    return (sum(len(message['content'].encode('utf-8'))for message in messages)
            +CHAT_TEMPLATE_TOKEN_RESERVE+16*max(0,len(messages)-2))


def stage_chat_settings(stage,messages=None):
    options=copy.deepcopy(CHAT_OPTIONS)
    if messages is not None:
        remaining=options['num_ctx']-chat_input_bound(messages)
        options['num_predict']=max(options['num_predict'],min(MAX_THINKING_OUTPUT_TOKENS,remaining))
    return dict(think=False,options=options)


def chat_keep_alive(stage):
    return -1


def model_lifecycle_policy():
    return dict(chat_keep_alive=dict(default=chat_keep_alive('default'),interpret=chat_keep_alive('interpret'),intent_audit=chat_keep_alive('intent_audit')),
                embedding_keep_alive=EMBED_KEEP_ALIVE,embedding_options=copy.deepcopy(EMBED_OPTIONS),
                thinking_output_policy=dict(enabled=False,minimum_reserved=CHAT_OPTIONS['num_predict'],maximum=MAX_THINKING_OUTPUT_TOKENS,context=CHAT_OPTIONS['num_ctx']))


@dataclass(frozen=True)
class CatalogCard:
    card_id: str
    book: str | None
    block: str
    metric: str | None
    series: str | None
    unit: str
    period_kind: str
    quantity_role: str
    scope: str
    title: str
    periods: list[str]
    observation_ids: list[str]
    source_path: str
    source_sha256: str
    description: str
    period_warnings: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphQueryState:
    query: str
    context: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict] = field(default_factory=list)
    plan: dict[str, Any] | None = None
    data_results: list[dict] = field(default_factory=list)
    text_results: list[dict] = field(default_factory=list)
    missing_parts: list[dict] = field(default_factory=list)
    retries: int = 0
