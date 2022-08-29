This includes the important data structures in python, useful in solving some complex problems efficiently.

## Binary Tree
A data structure with the following properties:
- Each left subnode should have smaller value than its parent node
- Each right subnode should have greater value than its parent node
- Each left and right subtree must also be a binary search tree

[Link](https://leetcode.com/problems/all-elements-in-two-binary-search-trees/) to the problem: All elements in two binary search trees
Main Concept: InOrderly Traversal and then using two stacks to store the values while traversing. 

Inorderly traversal: First traversing the leftmost branch of both the BSTs (while storing the values in a stack) until last node is reached. Compare both the stacks and add the smaller value to the final result array. Include the base cases like when one stack is empty then don't compare but append the entries of the non-empty array. After appending, move the root pointer to the right value. Here the base case checking the lengths of both the stacks will come into play. If the root.right value is None, it should now point to the top value of that stack and then its right pointer. This way, all the right pointers are also included. Refer to [this](https://leetcode.com/problems/all-elements-in-two-binary-search-trees/discuss/1720210/JavaC%2B%2BPython-A-very-very-detailed-EXPLANATION) for a more detailed explanation.

```python
class Solution(object):
    def getAllElements(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: List[int]
        """
        all_list = []
        st1 = []
        st2 = []
        while root1 != None or root2 != None or len(st1)>0 or len(st2)>0:
            while root1 != None:
                st1.append(root1)
                root1 = root1.left
            while root2 != None:
                st2.append(root2)
                root2 = root2.left
            if len(st2) == 0 or (len(st1) > 0 and st1[-1].val < st2[-1].val):
                root1 = st1.pop()
                all_list.append(root1.val)
                root1 = root1.right
            else:
                root2 = st2.pop()
                all_list.append(root2.val)
                root2 = root2.right
        return all_list
```

[Link](https://leetcode.com/problems/maximum-depth-of-binary-tree/) to the problem: Finding maximum depth of a binary tree (Recursive Solution)

```python
class Solution(object):    
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0
        else:
            return max(self.maxDepth(root.left), self.maxDepth(root.right))+1
```

## Linked List
