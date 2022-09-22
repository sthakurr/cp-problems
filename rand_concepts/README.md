**Time Complexities**

For Recursive calls:
Refer to [this](https://stackoverflow.com/questions/13467674/determining-complexity-for-recursive-functions-big-o-notation) for examples!

- If the function is being called recursively n times before reaching the base case, it is O(n), often called linear.
- If the function is called n-5 for each time, so we deduct five from n before calling the function, but n-5 is also O(n). (Actually called order of n/5 times. And, O(n/5) = O(n) ).
- This function is log(n) base 5, for every time we divide by 5 before calling the function so its O(log(n))(base 5), often called logarithmic and most often Big O notation and complexity analysis uses base 2.
- If each function call calls itself twice unless it has been recursed n times, it is O(2^n), or exponential.
- (For loop + Recursive call): the for loop takes n/2 since we're increasing by 2, and the recursion takes n/5 and since the for loop is called recursively, therefore, the time complexity is in (n/5) * (n/2) = n^2/10

For loops:


**Disjoint Set/Union Find Algorithm**: It allows us to keep track of non-overlapping subsets of a set and perform find and union operations! Used to detect cycles in a graph. This method assumes that the graph doesn’t contain any self-loops as the implementation assumes that all of the vertices have self-loops (i.e. they are their own parents).
Whenever we find that both the vertices of any edge correspond to the same subset, then there is a CYCLE!

Refer to [this](https://www.geeksforgeeks.org/union-find/) for more details!

**Spannning Tree**: A subset of a connected graph that connects all the vertices of a graph with the minimum possible number of edges. There are multiple spanning trees possible for a connected graph.
Minimum Spanning Tree is the one which has the minimum weight as it is defined on a weighted graph. If we have n vertices, the minimum spanning tree will have (n-1) edges.

- Every undirected and connected graph has a minimum of one spanning tree!
- For any complete graph, we have a total of n(n-2) spanning trees
- Spanning Tree is always minimally connected and maximally acyclic
- In a complete graph, we can create a spanning tree by removing a maximum of E-N+1 edges.

For more: https://techvidvan.com/tutorials/spanning-tree/

There are 2 algorithms to find the MST of a graph and they both follow Greedy approach:
1. **Prim's Algorithm**: Maintain two sets of vertices; one with the ones that are added in the MST and the other with the one not added yet
    - Find the cut (connecting edges) between the two sets of vertices
    - Choose the one with the minimum weight and add that edge and the corresponding vertex to the MST
    - Repeat this unil (V-1) edges are there
2. **Kruskal's Algorithm**:
    - Sort all the edges according to their weights
    - Take the smallest edge and check if it forms a cycle with the spannign tree formed so far. If no cycle, include that edge otherwise discard it
    - Repeat this until there are (V-1) edges

**NOTE**: Prim iterates over the vertices whereas Kruskal iterates over the edges hence there's no need to check for cycles in Prim's but Kruskal's has this step of checking for cycles in between!


**Dijkstra's algorithm**: 

Find the shortest path in O(E*log(v)) time! It is an SSSP (Single Sourced Shortest Path) algorithm!
It is a Greedy Algorithm and starts by assigning infinity to each of the distances from source node to every other node.

**Topological Sorting** (Kahn's Algorithm):
Given a Directed Acyclic Graph (DAG), we want to sort the nodes such that for any directed edge u --> v, u comes before v in the ordering!
