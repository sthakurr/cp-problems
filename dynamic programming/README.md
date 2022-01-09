# Dynamic Programming

Simple Intuition: While solving for the Fibonacci series, instead of computing some values again and again, we can store them when they are first computed and can be used further. This is known as memoization.

Naive Recursive solutions can be optimised using Dynamic Programming.

5 steps to solving DP problems:
- Visualise examples: usually using a directed acyclic graph
- Find an appropriate subproblem
- Find relationships among subproblems
- Generalize the relationship
- Implement by solving subproblems in order

Longest Increasing Subsequence:
For a sequence a1, a2, ... , an, find the length of the longest increasing subsequence ai1, ai2,..., aik.


Cherry Pickup:

[Link](https://leetcode.com/problems/cherry-pickup-ii/solution/) to the problem.

Approaches: Depth-First Search with memoization Top Down or Bottom up

