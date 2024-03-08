from graphviz import Digraph
from typing import Tuple, Set

from lib.value import Value


def trace(root: Value) -> Tuple[Set[Value], Set[Tuple[Value, Value]]]:
    """
    Builds a depth-first set of all nodes and edges in an expression graph.
    """
    nodes, edges = set(), set()

    def build(v: Value):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def render_expression(root: Value):
    diagram = Digraph(
        format="svg", graph_attr={"rankdir": "LR"}
    )  # LR means to draw from left to right

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))

        # For any value in the graph, create a rectangular 'record' node for it
        diagram.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }"
            % (
                n._name,
                n.data,
                n.grad,
            ),
            shape="record",
        )

        if n._op:
            # If this Value was created by an operation, create an op node
            diagram.node(name=uid + n._op, label=n._op)

            # Connect the node to the operation
            diagram.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # Connect n1 to the op node of n2
        diagram.edge(str(id(n1)), str(id(n2)) + n2._op)

    return diagram
