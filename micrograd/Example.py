import os

# Option 1: Escape backslashes
os.environ["PATH"] += os.pathsep + 'C:\\Users\\user\\PycharmProjects\\pythonProject4\\Graphviz\\bin'

# Option 2: Use a raw string
os.environ["PATH"] += os.pathsep + r'C:\Users\user\PycharmProjects\pythonProject4\Graphviz\bin'

from micrograd.engine import Value
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return edges, nodes

def draw_dot(root, format='svg', rankdir='LR'):
    assert rankdir in ('LR', 'TB', 'BT', 'LR', 'RL')
    edges, nodes = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    for node in nodes:
        dot.node(name=str(id(node)), label="{ data %.4f | grad %.4f }" % (node.data, node.grad), shape='record')
        if node._op:
            dot.node(name=str(id(node)) + node._op, label=node._op)
            dot.edge(str(id(node)) + node._op, str(id(node)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

x = Value(1.0)
y = (x * 2 + 1).relu()
y.backward()
dot = draw_dot(y)
dot.render('computation_graph', view=True)
