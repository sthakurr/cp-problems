# Dynamic Programming

DP: Finding relations between various subproblems! (Remember Your Past!)

**NOTE**: Use DP when the output is expecting an integer value and not the exact answers (permutations or sublists or arrays etc etc..)

Quick motivation (from interviewbit.com):
A *writes down "1+1+1+1+1+1+1+1 =" on a sheet of paper*

A : "What's that equal to?"
B : *counting* "Eight!"

A *writes down another "1+" on the left*
A : "What about that?"
B : *quickly* "Nine!"
A : "How'd you know it was nine so fast?"
A : "You just added one more"
A : "So you didn't need to recount because you remembered there were eight! Dynamic Programming is just a fancy way to say 'remembering stuff to save time later'"

Another Intuition: While solving for the Fibonacci series, instead of computing some values again and again, we can store them when they are first computed and can be used further. This is known as memoization.

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

