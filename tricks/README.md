# Eliminating the base cases 

[Link](https://leetcode.com/problems/can-place-flowers/) to the problem: Can Place Flowers

The base cases here can be easily eliminated by adding zeros at the start and end of the original flowerbed array.

Best solution:

class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        newflower = [0] + flowerbed + [0]
        for i in range(1, len(newflower) - 1):
            if newflower[i - 1] == 0 and  newflower[i] == 0 and  newflower[i + 1] == 0:
                newflower[i] = 1
                n -= 1
        return n <= 0

# Inorder Traversal Utility Function

def inorder(root, arr = []):
    if root:
        inorder(root.left, arr)
        arr.append(root.val)
        inorder(root.right, arr)