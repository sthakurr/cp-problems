
** Disjoint Set/Union Find Algorithm **: It allows us to keep track of non-overlapping subsets of a set and perform find and union operations! Used to detect cycles in a graph. This method assumes that the graph doesn’t contain any self-loops as the implementation assumes that all of the vertices have self-loops (i.e. they are their own parents).
Whenever we find that both the vertices of any edge correspond to the same subset, then there is a CYCLE!

Refer to [this](https://www.geeksforgeeks.org/union-find/) for more details!

** Spannning Tree **: A subset of a connected graph that connects all the vertices of a graph with the minimum possible number of edges. There are multiple spanning trees possible for a connected graph.
Minimum Spanning Tree is the one which has the minimum weight as it is defined on a weighted graph. If we have n vertices, the minimum spanning tree will have (n-1) edges.

- Every undirected and connected graph has a minimum of one spanning tree!
- For any complete graph, we have a total of n(n-2) spanning trees
- Spanning Tree is always minimally connected and maximally acyclic
- In a complete graph, we can create a spanning tree by removing a maximum of E-N+1 edges.

For more: https://techvidvan.com/tutorials/spanning-tree/

There are 2 algorithms to find the MST of a graph and they both follow Greedy approach:
1. ** Prim's Algorithm **: Maintain two sets of vertices; one with the ones that are added in the MST and the other with the one not added yet
    - Find the cut (connecting edges) between the two sets of vertices
    - Choose the one with the minimum weight and add that edge and the corresponding vertex to the MST
    - Repeat this unil (V-1) edges are there
2. ** Kruskal's Algorithm **:
    - Sort all the edges according to their weights
    - Take the smallest edge and check if it forms a cycle with the spannign tree formed so far. If no cycle, include that edge otherwise discard it
    - Repeat this until there are (V-1) edges

    Prim iterates over the vertices whereas Kruskal iterates over the edges hence there's no need to check for cycles in Prim's but Kruskal's has this step of checking for cycles in between!

