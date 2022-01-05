# Pairs of Songs with Total Duration divisible by 60 (Difficulty: Medium)

[Link](https://leetcode.com/problems/pairs-of-songs-with-total-durations-divisible-by-60/) to the problem.

Idea: Efficient storage of remainders (when divided by 60) instead of using a nested for loop to find the right pairs. Then make use of that array of remainders to perform further operations to find the total number of songs. This way, we only need to consider each song length modulo 60 and we can count the number of songs with (length % 60) equal to r, and store that in an array of size 60.

My Implementation:

class Solution(object):
    def cumSum(self, num):
        sum_ = 0
        for i in range(1, num+1):
            sum_ += i
        return sum_
    
    def numPairsDivisibleBy60(self, time):
        """
        :type time: List[int]
        :rtype: int
        """
        numPairs = 0
        divisibleBy = 60
        modulo = [0]*divisibleBy
        for i in range(len(time)):
            rem = time[i]%60 
            modulo[rem] += 1
        
        numPairs += self.cumSum(modulo[0] - 1)
        for i in range(1, 31):
            if i == 30:
                numPairs += self.cumSum(modulo[30]-1)
            else:
                total = modulo[i] * modulo[60-i]
                numPairs += total
        
        return numPairs