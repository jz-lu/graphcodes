import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
"""
_code_<x>.py: graphs that might be good codes!
x = `convex`: A generic convex hull based framework that makes any convex shape in D dimensions.
"""

def coords2graph(coords):
    """
    Takes a set of coordinates and builds a convex hull from them.
    Then the hull is mapped to a graph in adjacency matrix representation
    which is returned.

    Input:
        * `coords`: a N x D numpy array representing N points in D dimensions.
    
    Returns:
        * `adj_mat`: adjacency matrix representing graph of convex hull of `coords`.
    """

    hull = ConvexHull(coords)
    
    # Extract indices of the vertices forming the convex hull
    hull_vertices = hull.vertices

    # Number of vertices in the convex hull
    n = len(hull_vertices)

    # Initialize the adjacency matrix with zeros
    adjacency_matrix = np.zeros((n, n), dtype=int)

    # Create a mapping from original point indices to hull vertex indices
    index_map = {original_idx: new_idx for new_idx, original_idx in enumerate(hull_vertices)}

    # Iterate over the simplices (edges) in the convex hull
    for simplex in hull.simplices:
        # Map the original indices to the new indices for the convex hull
        i, j = simplex
        i_mapped = index_map[i]
        j_mapped = index_map[j]
        # Mark the vertices as connected in the adjacency matrix
        adjacency_matrix[i_mapped, j_mapped] = 1
        adjacency_matrix[j_mapped, i_mapped] = 1

    print("Adjacency Matrix of the Convex Hull:")
    print(adjacency_matrix)

def plot3d(points):
    # Step 3: Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Step 4: Plot the convex hull as a solid
    ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=hull.simplices, cmap='viridis', alpha=0.8)

    # Optional: Plot the points
    ax.scatter(points[:,0], points[:,1], points[:,2], c='r', marker='o')

    # Step 5: Customize the plot
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Convex Hull')

    # Step 6: Show the plot
    plt.show()

def plotwire(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(points[:,0], points[:,1], points[:,2], 'o')

    # Plot the surface triangles
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])  # Here we cycle back to the first coordinate
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'r-')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Convex Hull')

    plt.show()

points= np.array([
    [0,0,0],
    [4,0,0],
    [4,4,0],
    [0,4,0],
    [0,0,4],
    [4,0,4],
    [4,4,4],
    [0,4,4]
])

xi = np.sqrt(2) + 1
basic = np.array((xi, xi, 1))
from sympy.utilities.iterables import multiset_permutations
perms = np.array([[xi, xi, 1], [1, xi, xi], [xi, 1, xi]]) / 2
n = len(list(perms)) * 8
coords = []
for perm in perms:
    for i in range(2):
        for j in range(2):
            for k in range(2):
                coords.append([(-1)**i * perm[0], (-1)**j * perm[1], (-1)**k * perm[2]])
points = np.array(coords)
hull = ConvexHull(points)

plotwire(points)
