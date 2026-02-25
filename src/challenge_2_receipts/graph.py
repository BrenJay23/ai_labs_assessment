import json
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from .ocr import run_ocr
from .pipelines import improved_ocr, extract_entities, unified_pipeline


class PipelineState(TypedDict):
    image_path: str
    tier: str
    raw_text: Optional[str]
    improved_text: Optional[str]
    entities: Optional[dict]
    context: Optional[str]


# --- Nodes ---


def ocr_node(state: PipelineState) -> PipelineState:
    state["raw_text"] = run_ocr(Path(state["image_path"]))["raw_text"]
    return state


def improved_ocr_node(state: PipelineState) -> PipelineState:
    state["improved_text"] = improved_ocr(Path(state["image_path"]))
    return state


def entity_extraction_node(state: PipelineState) -> PipelineState:
    text = state["improved_text"] if state.get("improved_text") else state["raw_text"]
    state["entities"] = extract_entities(text).model_dump()
    return state


def unified_node(state: PipelineState) -> PipelineState:
    state["entities"] = unified_pipeline(Path(state["image_path"])).model_dump()
    return state


def context_node(state: PipelineState) -> PipelineState:
    if state.get("entities"):
        state["context"] = json.dumps(state["entities"], indent=2)
    elif state.get("improved_text"):
        state["context"] = state["improved_text"]
    else:
        state["context"] = state["raw_text"]
    return state


# --- Routing ---


def route_entry(state: PipelineState) -> str:
    tier = state["tier"]
    if tier == "Tier 1":
        return "ocr"
    elif tier == "Tier 2":
        return "improved_ocr"
    elif tier == "Tier 3":
        return "ocr"
    elif tier == "Tier 4":
        return "improved_ocr"
    elif tier == "Tier 5":
        return "unified"
    return "unified"


def route_after_ocr(state: PipelineState) -> str:
    return "entity_extraction" if state["tier"] == "Tier 3" else "context"


def route_after_improved_ocr(state: PipelineState) -> str:
    return "entity_extraction" if state["tier"] == "Tier 4" else "context"


# --- Graph ---


def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("ocr", ocr_node)
    graph.add_node("improved_ocr", improved_ocr_node)
    graph.add_node("entity_extraction", entity_extraction_node)
    graph.add_node("unified", unified_node)
    graph.add_node("context", context_node)

    graph.set_conditional_entry_point(
        route_entry,
        {
            "ocr": "ocr",
            "improved_ocr": "improved_ocr",
            "unified": "unified",
        },
    )

    graph.add_conditional_edges(
        "ocr",
        route_after_ocr,
        {"entity_extraction": "entity_extraction", "context": "context"},
    )

    graph.add_conditional_edges(
        "improved_ocr",
        route_after_improved_ocr,
        {"entity_extraction": "entity_extraction", "context": "context"},
    )

    graph.add_edge("entity_extraction", "context")
    graph.add_edge("unified", "context")
    graph.add_edge("context", END)

    return graph.compile()


pipeline_graph = build_graph()


def run_graph(image_path: Path, tier: str) -> dict:
    result = pipeline_graph.invoke(
        {
            "image_path": str(image_path),
            "tier": tier,
            "raw_text": None,
            "improved_text": None,
            "entities": None,
            "context": None,
        }
    )
    return {"context": result["context"], "display": result["context"]}
