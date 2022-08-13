### Sorting with a key

Given an array of subarrays, we can sort it according to the first element and in case of a tie we can sort it wrt the second element too.

sorted(arr, key=(lambda i: (i[0], -i[0])))

If we want to sort using a user-defined comparison:
we need to define a comparator function as follows:
```python
def comp(A,B):
    # condition when A is prioritised
    # return -ve
    # when B is sorted first
    # return +ve
    # else return 0 
```

For example: If I want to sort a list of integers according to the number formed by string-joining any two of them:
```python
def comp(A,B):
    A, B = str(A), str(B)
    if A+B > B+A:
        return -1
    elif B+A < A+B:
        return 1
    else:
        return 0
```

Then use:
```python
from functools import key_to_comp
sorted(list_, key_to_comp(comp))
```

### Always copy the lists using copy.deepcopy()

Copying the lists using:
 
```python
temp = out
temp = out[:]
temp = out.copy()
```
will only create a new variable referenced to the original variable. So, changing the new variable will also change the original variable.

Instead do:

```python
temp = copy.deepcopy(out)
```

### To check for a palindrome:

' return s == s[::-1] '