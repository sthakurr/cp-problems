# Eliminating the base cases 

[Link](https://leetcode.com/problems/can-place-flowers/) to the problem: Can Place Flowers

The base cases here can be easily eliminated by adding zeros at the start and end of the original flowerbed array.

Best solution:

```python
class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        newflower = [0] + flowerbed + [0]
        for i in range(1, len(newflower) - 1):
            if newflower[i - 1] == 0 and  newflower[i] == 0 and  newflower[i + 1] == 0:
                newflower[i] = 1
                n -= 1
        return n <= 0
```

# Inorder Traversal Utility Function

```python
def inorder(root, arr = []):
    if root:
        inorder(root.left, arr)
        arr.append(root.val)
        inorder(root.right, arr)
```

# Bit Manipulation for keeping track of new numbers
[Link](https://leetcode.com/problems/single-number/) to a sample problem: Single Number

Here, XOR operation can be used to add bitwise numbers when new numbers are seen and destroy the bits of numbers that are already seen before. This way, the number that was present only once, its bits survive in the end and hence the result.

```python
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # o1 oN 
        
        # cant sort 
        # cant use map 
        
        # can use pointers or swap 
        res = 0 
        for n in nums: 
            res = n ^ res  # XOR operation
        return res
```