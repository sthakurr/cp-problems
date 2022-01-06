# cp-problems

A repository containing the hints and (my) solutions to some of the Leetcode problems. Can contain some common tips and strategies too that I may encounter during my journey with Competitive Programming.

## Some common tricks to remember:

To check for a palindrome:

' return s == s[::-1] '

## Some common concepts to remember:

Depth-First Search:

In Depth First Search, we recursively expand potential candidate until the defined goal is achieved. After that, we backtrack to explore the next potential candidate. Viewing this as a graph, we explore all the possibilities by going till the last node in that branch (exhausting the depth) and then jumping to the adjacent node to explore the other possibilities. This is known as Backtracking.

Sweep Line Algo:
https://leetcode.com/problems/maximum-sum-obtained-of-any-permutation/discuss/854206/JavaC%2B%2BPython-Sweep-Line
sweep line algo: leetcode 253, 1589, 1109
** NOTE that for all intervals inputs, this method should be the first intuition you come up with. **
