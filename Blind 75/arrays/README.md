Kadane's Algorithm:

Problem statement: [Link](https://leetcode.com/problems/maximum-subarray/) to the problem

Trick:
Using the max(local_maximum(idx-1)+arr[idx], arr[idx]) ----> Kadane's Algorithm (relation between the current and previous subproblems)
if somehow arr[idx] is greater than the local_maximum(idx-1)+arr[idx], then it'll start the new subarray search from index=idx!!!
Covering all possible subarrays in O(n)

My Solution:
class Solution(object):
    def locMax(self, nums, index):
        if index == 0:
            return nums[0], nums[0]
        else:
            res, prevLocSum = self.locMax(nums, index-1)
            locSum = max(prevLocSum+nums[index], nums[index])
            if locSum > res:
                res = locSum
            return res, locSum        
        
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res, locSum = self.locMax(nums, len(nums)-1)
        return res
