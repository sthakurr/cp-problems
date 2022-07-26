# cp-problems

A repository containing the hints and (my) solutions to some of the Leetcode problems. Can contain some common tips and strategies too that I may encounter during my journey with Competitive Programming.


## Concepts and some intuitive references:

### Recursion

[5 Simple Steps for Solving Any Recursion Problem](https://www.youtube.com/watch?v=ngCos392W4w)

### Depth-First Search:

In Depth First Search, we recursively expand potential candidate until the defined goal is achieved. After that, we backtrack to explore the next potential candidate. Viewing this as a graph, we explore all the possibilities by going till the last node in that branch (exhausting the depth) and then jumping to the adjacent node to explore the other possibilities. This is known as Backtracking.

Given a graph adjacency dictionary containing the neighbors of each node, the dfs can be implemented as:

visited = set() # Set to keep track of visited nodes of graph.

def dfs(visited, graph, node):  #function for dfs 
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

Breadth-First Search:


### Trie (Prefix Tree)
[Implementation](https://leetcode.com/problems/implement-trie-prefix-tree/)

### Sweep Line Algo:
https://leetcode.com/problems/maximum-sum-obtained-of-any-permutation/discuss/854206/JavaC%2B%2BPython-Sweep-Line
sweep line algo: leetcode 253, 1589, 1109
** NOTE that for all intervals inputs, this method should be the first intuition you come up with. **

Reservoir Sampling:
A family of randomized algorithms for sampling from a population of unknown size.

## building-heap-from-array
[Building Heap from Array](https://www.geeksforgeeks.org/building-heap-from-array/):
