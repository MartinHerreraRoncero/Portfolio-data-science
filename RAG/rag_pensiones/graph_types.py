"""JSON contracts for the source-bound LangGraph profile."""
from dataclasses import asdict, dataclass, field
import copy
from typing import Any

PROFILE = 'langgraph-9b-v6'
CHAT_MODEL = 'qwen3.5:9b'
CHAT_DIGEST = '56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90'
EMBED_MODEL = 'qwen3-embedding:4b'
EMBED_DIGEST = 'df5bd2e3c74cd8d069d21dc038f1b359fcdc9458fce1c99bd43c9eb1518ff907'
EMBED_DIMENSION = 2560
EMBED_KEEP_ALIVE = '0s'
CHAT_OPTIONS = {'temperature': 0, 'seed': 42, 'num_ctx': 8192, 'num_predict': 1024, 'presence_penalty': 0}


def stage_chat_settings(stage):
    options=copy.deepcopy(CHAT_OPTIONS)
    return dict(think=False,options=options)


def chat_keep_alive(stage):
    return '0s'if stage=='intent_audit'else'5m'


def model_lifecycle_policy():
    return dict(chat_keep_alive=dict(default=chat_keep_alive('default'),intent_audit=chat_keep_alive('intent_audit')),
                embedding_keep_alive=EMBED_KEEP_ALIVE)


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
