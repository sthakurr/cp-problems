This contains the tricks used in the questions from the Blind 75 list. The list of those 75 LeetCode questions can be found [here](https://www.teamblind.com/post/New-Year-Gift---Curated-List-of-Top-75-LeetCode-Questions-to-Save-Your-Time-OaM1orEU). The [spreadsheet](https://docs.google.com/spreadsheets/d/1ePf1uB47mhQZR-xQZevAed6TvezhO6hJzEoeyRCnHaE/edit?usp=sharing) contains the one liners to these solutions. 

[Two Sums](https://leetcode.com/problems/two-sum/)

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hashmap = {}
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in hashmap:
                return [hashmap[complement], i]
            else:
                hashmap[nums[i]] = i