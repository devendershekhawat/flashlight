try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, Circle
    import networkx as nx
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .Tensor import Tensor


def trace(root: Tensor):
    nodes, edges = set(), set()
    
    def build(v: Tensor):
        if v not in nodes:
            nodes.add(v)
            for child in v._children:
                edges.add((child, v))
                build(child)
    
    build(root)
    return nodes, edges


def draw_dot(root: Tensor, format='svg', rankdir='TB', use_graphviz=None):
    nodes, edges = trace(root)
    
    # Auto-detect if not specified
    if use_graphviz is None:
        use_graphviz = HAS_GRAPHVIZ
    
    if use_graphviz:
        if not HAS_GRAPHVIZ:
            raise ImportError("graphviz not available. Install with: pip install graphviz")
        return _draw_with_graphviz(nodes, edges, format, rankdir)
    else:
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib/networkx not available. Install with: pip install matplotlib networkx")
        return _draw_with_matplotlib(nodes, edges, rankdir)


def _draw_with_graphviz(nodes, edges, format, rankdir):
    if not HAS_GRAPHVIZ:
        raise ImportError("graphviz not available")
    assert rankdir in ['LR', 'TB']
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})  # type: ignore[name-defined]
    
    # Track operations separately
    operations = {}
    
    for n in nodes:
        tensor_id = str(id(n))
        shape_str = str(n.shape).replace(' ', '')
        label = f"{shape_str}"
        if n._label:
            label = f"{n._label}"
        
        # Square node for tensor
        dot.node(name=tensor_id, label=label, shape='box', style='filled',
                fillcolor='lightyellow', fontsize='10')
        
        # If tensor has an operation, create circular operation node
        if n._op:
            op_id = tensor_id + '_' + n._op
            dot.node(name=op_id, label=n._op, shape='circle', style='filled',
                    fillcolor='lightblue', fontsize='10')
            operations[n] = op_id
    
    # Add edges: child tensors -> operation -> result tensor
    for n1, n2 in edges:
        if n2 in operations:
            # Connect child tensor to operation
            dot.edge(str(id(n1)), operations[n2])
            # Connect operation to result tensor
            dot.edge(operations[n2], str(id(n2)))
        else:
            # Direct tensor to tensor (shouldn't happen with proper structure)
            dot.edge(str(id(n1)), str(id(n2)))
    
    return dot


def _draw_with_matplotlib(nodes, edges, rankdir):
    """Pure Python matplotlib-based rendering (no system executables needed)."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib/networkx not available")
    assert rankdir in ['LR', 'TB']
    
    # Create directed graph
    G = nx.DiGraph()  # type: ignore[name-defined]
    
    # Track operations and positions
    operations = {}
    tensor_positions = {}
    op_positions = {}
    
    # Add nodes and track operations
    for n in nodes:
        tensor_id = id(n)
        G.add_node(tensor_id, type='tensor', tensor=n)  # type: ignore[name-defined]
        
        if n._op:
            op_id = f"{tensor_id}_{n._op}"
            G.add_node(op_id, type='operation', op=n._op)  # type: ignore[name-defined]
            operations[n] = op_id
    
    # Add edges
    for n1, n2 in edges:
        if n2 in operations:
            # Child tensor -> operation -> result tensor
            G.add_edge(id(n1), operations[n2])  # type: ignore[name-defined]
            G.add_edge(operations[n2], id(n2))  # type: ignore[name-defined]
        else:
            G.add_edge(id(n1), id(n2))  # type: ignore[name-defined]
    
    # Create layout
    if rankdir == 'LR':
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)  # type: ignore[name-defined]
    else:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)  # type: ignore[name-defined]
        # Rotate for top-bottom
        pos = {node: (y, -x) for node, (x, y) in pos.items()}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))  # type: ignore[name-defined]
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw edges first
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=20,  # type: ignore[name-defined]
                          edge_color='gray', width=1.5, alpha=0.6,
                          connectionstyle='arc3,rad=0.1')
    
    # Draw tensor nodes (square)
    tensor_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'tensor']  # type: ignore[name-defined]
    for node_id in tensor_nodes:
        tensor = G.nodes[node_id]['tensor']  # type: ignore[name-defined]
        x, y = pos[node_id]
        shape_str = str(tensor.shape).replace(' ', '')
        label = f"{shape_str}"
        if tensor._label:
            label = f"{tensor._label}\n{shape_str}"
        
        # Draw square
        square = FancyBboxPatch((x-0.15, y-0.1), 0.3, 0.2,  # type: ignore[name-defined]
                               boxstyle="round,pad=0.02", 
                               facecolor='lightyellow',
                               edgecolor='black', linewidth=1.5)
        ax.add_patch(square)
        ax.text(x, y, label, ha='center', va='center', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    # Draw operation nodes (circle)
    op_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'operation']  # type: ignore[name-defined]
    for node_id in op_nodes:
        op_name = G.nodes[node_id]['op']  # type: ignore[name-defined]
        x, y = pos[node_id]
        
        # Draw circle
        circle = Circle((x, y), 0.12, facecolor='lightblue',  # type: ignore[name-defined]
                       edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y, op_name, ha='center', va='center', fontsize=9, weight='bold')
    
    plt.tight_layout()  # type: ignore[name-defined]
    return fig