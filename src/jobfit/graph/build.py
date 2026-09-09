"""Assembling the graph.

The shape:

    START -> intake -> supervisor -+-> fit_screener -+
                                   +-> brief_writer -+-> supervisor
                                   +-> finalize --------> END

Two things make LangGraph worth the dependency here rather than an ``if``
statement in a while loop, which is a fair question to ask of any framework:

* **conditional edges plus reducers.** Routing is data on the state, and the
  append-only fields (trace, handoffs, assessments) merge through reducers that
  are declared once instead of managed by hand at every return;
* **``interrupt_before``.** Human approval before the writer runs is a one-line
  change here and is milestone M6's main deliverable. It is exposed as
  ``approve_briefs`` already, so the claim is checkable rather than promised.

Termination is decided on the edge, not by the model. Even if the supervisor
insisted on looping forever, the step ceiling routes to finalize.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from jobfit.graph.nodes import (
    GraphDeps,
    make_finalize,
    make_intake,
    make_screener,
    make_supervisor,
    make_writer,
)
from jobfit.graph.state import PipelineState

NODE_INTAKE = "intake"
NODE_SUPERVISOR = "supervisor"
NODE_SCREENER = "fit_screener"
NODE_WRITER = "brief_writer"
NODE_FINALIZE = "finalize"


def route_from_supervisor(state: PipelineState) -> str:
    """Where to go after the supervisor.

    The ceiling is enforced here as well as inside the supervisor node. Belt and
    braces on purpose: this is the one function that makes termination a
    property of the graph rather than of the model's cooperation.
    """
    if state.step_count > _ceiling(state):
        return NODE_FINALIZE
    if state.next_agent == NODE_SCREENER:
        return NODE_SCREENER
    if state.next_agent == NODE_WRITER:
        return NODE_WRITER
    return NODE_FINALIZE


def _ceiling(state: PipelineState) -> int:
    """Hard bound on graph steps, independent of settings being sane."""
    return max(2, min(state.request.limit * 6 + 6, 60))


def build_graph(deps: GraphDeps, *, approve_briefs: bool = False) -> Any:
    """Compile the pipeline.

    ``approve_briefs=True`` pauses the run before the writer, so a human can
    inspect the assessment first. That is the human-in-the-loop primitive
    milestone M6 builds on; it works today.
    """
    builder = StateGraph(PipelineState)

    builder.add_node(NODE_INTAKE, make_intake(deps))
    builder.add_node(NODE_SUPERVISOR, make_supervisor(deps))
    builder.add_node(NODE_SCREENER, make_screener(deps))
    builder.add_node(NODE_WRITER, make_writer(deps))
    builder.add_node(NODE_FINALIZE, make_finalize(deps))

    builder.add_edge(START, NODE_INTAKE)
    builder.add_edge(NODE_INTAKE, NODE_SUPERVISOR)

    builder.add_conditional_edges(
        NODE_SUPERVISOR,
        route_from_supervisor,
        {
            NODE_SCREENER: NODE_SCREENER,
            NODE_WRITER: NODE_WRITER,
            NODE_FINALIZE: NODE_FINALIZE,
        },
    )

    # Both specialists report back to the supervisor, which is what makes this a
    # supervised team rather than a fixed pipeline with extra steps.
    builder.add_edge(NODE_SCREENER, NODE_SUPERVISOR)
    builder.add_edge(NODE_WRITER, NODE_SUPERVISOR)
    builder.add_edge(NODE_FINALIZE, END)

    return builder.compile(interrupt_before=[NODE_WRITER] if approve_briefs else None)


def graph_mermaid(deps: GraphDeps) -> str:
    """The architecture diagram, generated from the graph that actually runs.

    Used for the README, so the picture cannot drift away from the code.
    """
    return build_graph(deps).get_graph().draw_mermaid()
