import networkx as nx
import matplotlib.pyplot as plt
"""
_code_<x>.py: graphs that might be good codes!
x = `dipyramid`: given n, generate a n-gonic dipyramid

By reflection symmetry, the distance is always 1 if the input is placed on the pyramid heads.
So, to have a chance at a good code, you need to put at least 1 qubit in the base.

EXPORT FUNCTIONS:
`adj_dipyramid`
"""

def adj_dipyramid(n):
    """
    Adjacency matrix of graph of dipyramid with `n`-gonic base.
    """
    return nx.adjacency_matrix(create_dipyramid(n)).toarray()

def create_dipyramid(base_polygon):
    """
    Create a dipyramid graph based on the number of sides `base_polygon` in the base polygon.
    """
    G = nx.Graph()
    
    # Add base polygon vertices
    for i in range(base_polygon):
        G.add_node(f"B{i}")
    
    # Add top and bottom vertices
    G.add_node("T")
    G.add_node("B")
    
    # Connect base polygon vertices
    for i in range(base_polygon):
        G.add_edge(f"B{i}", f"B{(i+1)%base_polygon}")
        
    # Connect top and bottom vertices to all base vertices
    for i in range(base_polygon):
        G.add_edge("T", f"B{i}")
        G.add_edge("B", f"B{i}")
    
    return G

def visualize_dipyramid(graph):
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(graph, k=0.5, iterations=50)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=12, font_weight='bold')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Create dipyramids
    triangular_dipyramid = create_dipyramid(3)
    square_dipyramid = create_dipyramid(4)
    pentagonal_dipyramid = create_dipyramid(5)
    hexagonal_dipyramid = create_dipyramid(6)

    # # Visualize dipyramids
    # visualize_dipyramid(triangular_dipyramid)
    # visualize_dipyramid(square_dipyramid)
    # visualize_dipyramid(pentagonal_dipyramid)
    # visualize_dipyramid(hexagonal_dipyramid)