from __future__ import annotations

import math
from typing import List, Tuple

import networkx as nx
import plotly.graph_objects as go

from node import Node
from custom_types import NodeInfo


def shorten_edge(x0: float, y0: float, x1: float, y1: float, r: float) -> tuple[float, float, float, float]:
    """
    Shorten a line so it starts/ends at the circle boundary.
    r = node radius (same for all nodes)
    """
    dx = x1 - x0
    dy = y1 - y0
    dist = math.sqrt(dx * dx + dy * dy)

    if dist == 0:
        return x0, y0, x1, y1

    ux = dx / dist
    uy = dy / dist

    return (
        x0 + ux * r,
        y0 + uy * r,
        x1 - ux * r,
        y1 - uy * r,
    )


def create_graph_window(
    nodes: List[Node],
    connections: List[Tuple[Node, Node]],
    state_by_id: dict[int, NodeInfo] | None = None,
    title: str = "Supply Chain Network",
) -> None:
    """
    Interactive supply chain graph using NetworkX + Plotly.
    Nodes = Node objects, dynamic values come from state_by_id.
    """
    node_radius = 0.06

    graph = nx.DiGraph()

    # Add nodes with metadata
    for node in nodes:
        st = state_by_id.get(node.id) if state_by_id else None
        graph.add_node(
            node.id,
            label=node.name,
            inventory=st.inventory if st else 0,
            backorders=st.backorders if st else 0,
            remaining_time=st.remaining_time if st else 0,
        )

    # Add directed edges
    for src, tgt in connections:
        graph.add_edge(src.id, tgt.id)

    # Layout
    pos = nx.spring_layout(graph, seed=42)

    # Build edge traces
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []

    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        xs, ys, xe, ye = shorten_edge(x0, y0, x1, y1, node_radius)
        edge_x += [xs, xe, None]
        edge_y += [ys, ye, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2, color="black"),
        hoverinfo="none",
    )

    # Build node traces (on top)
    node_x: list[float] = []
    node_y: list[float] = []
    hover_text: list[str] = []
    labels: list[str] = []

    for node_id in graph.nodes():
        x, y = pos[node_id]
        data = graph.nodes[node_id]

        node_x.append(x)
        node_y.append(y)
        labels.append(data["label"])
        hover_text.append(
            f"<b>{data['label']}</b><br>"
            f"Inventory: {data['inventory']}<br>"
            f"Backorders: {data['backorders']}<br>"
            f"Remaining time: {data['remaining_time']}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="bottom center",
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(
            size=40,
            color="dodgerblue",
            line=dict(width=2, color="black"),
        ),
    )

    # Arrow heads (annotations)
    annotations = []
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        xs, ys, xe, ye = shorten_edge(x0, y0, x1, y1, node_radius)

        annotations.append(
            dict(
                ax=xs,
                ay=ys,
                x=xe,
                y=ye,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor="black",
            )
        )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            annotations=annotations,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    fig.show()
