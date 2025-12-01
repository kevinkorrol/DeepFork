"""
Utilities for visualizing the MCTS search tree with Graphviz.

Provides helpers to convert a tree of MCTSNode objects into Graphviz data and
render a styled image containing an HTML board for each node.
"""

import graphviz


def visualize_mcts_graph(root_node, filename: str = "mcts_tree"):
    """
    Generate and render a Graphviz visualization of the MCTS tree rooted at root_node.

    :param root_node: Root MCTSNode of the search tree
    :param filename: Output filename without extension
    :return: None. Renders a PNG file via graphviz.Digraph.render
    """
    graph_data, _ = to_graph_data(root_node)
    dot = graphviz.Digraph(
        comment='MCTS Search Tree',
        graph_attr={'rankdir': 'TB', 'splines': 'true', 'ranksep': '0.5'},
        node_attr={'fontname': 'Arial', 'fontsize': '10', 'labelloc': 't'},
        edge_attr={'fontname': 'Arial', 'fontsize': '8'}
    )

    # Add Nodes
    for node in graph_data['nodes']:
        dot.node(
            str(node['id']),
            label=node['label'],
            fillcolor=node['fillcolor'],
            style=node['style'],
            shape=node['shape']
        )

    # Add Edges
    for edge in graph_data['edges']:
        dot.edge(
            str(edge['source']),
            str(edge['target']),
            label=edge['label'],
            penwidth=str(edge['penwidth'])
        )

    # Render the graph to a file (e.g., PDF, PNG)
    dot.render(filename, view=True, format='png')


def get_html_board_label(node) -> str:
    """
    Build an HTML table label representing the node's chessboard and stats.

    The label combines an 8x8 board with Unicode chess glyphs and a stats row
    containing move, visit count, Q value, and prior P.

    :param node: MCTSNode with a .board attribute and stats (move, visit_count, prior_est)
    :return: HTML label string for Graphviz
    """
    # Unicode characters for chess pieces (Standard FEN mapping)
    PIECE_UNICODE = {
        'p': '&#9823;', 'n': '&#9822;', 'b': '&#9821;', 'r': '&#9820;', 'q': '&#9819;', 'k': '&#9818;',
        'P': '&#9817;', 'N': '&#9816;', 'B': '&#9815;', 'R': '&#9814;', 'Q': '&#9813;', 'K': '&#9812;',
    }

    # Determine the color of the square based on rank/file
    def get_square_color(rank_idx, file_idx):
        if (rank_idx + file_idx) % 2 == 0:
            return "#EAD8C1"  # Light square
        else:
            return "#B58863"  # Dark square

    fen_parts = node.board.fen().split(' ')
    piece_placement = fen_parts[0]

    html_rows = []
    rank_idx = 0
    file_idx = 0
    current_row_html = ''

    for char in piece_placement:
        if char == '/':
            # End of rank: finalize the current row and reset for the next
            html_rows.append(current_row_html)
            current_row_html = ''
            rank_idx += 1
            file_idx = 0
            continue

        if char.isdigit():
            # Empty squares (Fix: using &nbsp; from previous suggestion)
            num_empty = int(char)
            for _ in range(num_empty):
                bg_color = get_square_color(rank_idx, file_idx)
                current_row_html += f'<TD BGCOLOR="{bg_color}" WIDTH="20" HEIGHT="20">&nbsp;</TD>'
                file_idx += 1
        else:
            # Piece
            piece_char = PIECE_UNICODE.get(char, '')
            bg_color = get_square_color(rank_idx, file_idx)
            # Ensure piece is vertically centered
            current_row_html += f'<TD BGCOLOR="{bg_color}" WIDTH="20" HEIGHT="20" VALIGN="MIDDLE"><FONT POINT-SIZE="18">{piece_char}</FONT></TD>'
            file_idx += 1

    # Add the last row after the loop finishes
    html_rows.append(current_row_html)

    # 1. Build the Board Table
    html_table = f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
    for row_html in html_rows:
        html_table += f'<TR>{row_html}</TR>'  # Each row is wrapped cleanly

    # 2. Add the Stats Row (same as before)
    q_value = node.Q()
    fillcolor = 'green' if q_value > 0.1 else ('red' if q_value < -0.1 else 'yellow')

    stats_label = (
        f"<FONT POINT-SIZE='10' COLOR='BLACK'>"
        f"Move: {node.move.uci() if node.move else 'ROOT'}<BR/>"
        f"N: {node.visit_count} | Q: {q_value:.3f} | P: {node.prior_est:.3f}"
        f"</FONT>"
    )

    html_table += f'<TR><TD COLSPAN="8" BGCOLOR="{fillcolor}">{stats_label}</TD></TR>'
    html_table += '</TABLE>>'
    return html_table


def to_graph_data(node, node_id: int = 0, graph_data: dict = None) -> tuple[dict, int]:
    """
    Recursively collect nodes and edges to represent the MCTS tree in Graphviz.

    :param node: Current MCTSNode being serialized
    :param node_id: Integer id to assign to this node
    :param graph_data: Accumulator dict with 'nodes' and 'edges' lists
    :return: Tuple (graph_data, next_available_id)
    """
    if graph_data is None:
        graph_data = {'nodes': [], 'edges': []}

    # Label is an HTML string
    html_label = get_html_board_label(node)

    graph_data['nodes'].append({
        'id': node_id,
        'label': html_label,
        'fillcolor': 'white', # Background
        'style': 'filled',
        'shape': 'box',  # Must be 'box' or 'plain' for HTML labels
    })

    next_node_id = node_id + 1

    # Edge Data
    for move, (child, est) in node.children.items():
        if child is not None:
            child_id = next_node_id

            child_visits = child.visit_count
            puct = node.Q() + node.U(1.0, est, child_visits)
            edge_label = f"{move.uci()}\nPUCT: {puct:.3f}"

            graph_data['edges'].append({
                'source': node_id,
                'target': child_id,
                'label': edge_label,
                'penwidth': 1 + child_visits / (node.visit_count + 1) * 3,
            })

            # Recurse to children
            graph_data, next_node_id = to_graph_data(child, child_id, graph_data)

    return graph_data, next_node_id


if __name__ == "__main__":
    pass