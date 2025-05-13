"""Generate the adjacency matrices for circle graphs"""

def type_nearest_neighbor(n, M):
    """
    Build the adjacency matrix for an undirected graph with nodes 0..n.
    
    - Nodes 0..n-1 are arranged on a circle of length n.
      Two distinct nodes i, j in [0..n-1] are connected if
      their circular distance ≤ M.
    - Node n is connected to every node 0..n-1.
    
    Returns:
        adj: list of lists, size (n+1) x (n+1), with 0/1 entries.
    """
    # Initialize all-zero (n+1)x(n+1) matrix
    adj = [[0] * (n + 1) for _ in range(n + 1)]
    
    # Helper to compute circular distance on a ring of size n
    def circ_dist(i, j):
        d = abs(i - j)
        return min(d, n - d)
    
    # Connect nodes 0..n-1 according to M
    for i in range(n):
        for j in range(i + 1, n):
            if circ_dist(i, j) <= M and i != j:
                adj[i][j] = 1
                adj[j][i] = 1
    
    # Connect node n to every node 0..n-1
    for i in range(n):
        adj[i][n] = 1
        adj[n][i] = 1
    
    return adj
