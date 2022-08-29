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

### Digit Dynamic Programming:

Refer to [this](https://youtu.be/sCwokuCoQB8) for an intuition!
Whenever we want to solve problems involving patterns in digits of numbers within a given range, digit dynamic programming helps!
Sequentially move forward by removing one digit at a time and checking for the conditions. Time complexity is O(log(n)^2)

### Bisect module

--> Works with sorted arrays, uses the concept of binary search
Returns the position where a new number should be inserted in a sorted array in O(logn). efficient because it reduces the overhead time to sort the list
insort() returns the sorted array after inserting the element in O(n)

### Bitmasking and DP

Use bitmasking when keeping a track of subsets included for a given setting. For example, if there are 5 persons and 1st and 4th of them are wearing a cap, then it can be represented with 10010. Masks can be used to 

### Trie (Prefix Tree)
[Implementation](https://leetcode.com/problems/implement-trie-prefix-tree/)

Reservoir Sampling:
A family of randomized algorithms for sampling from a population of unknown size.

## building-heap-from-array
[Building Heap from Array](https://www.geeksforgeeks.org/building-heap-from-array/):
use heapify()
most of the operations in a heap are performed in log(n)
