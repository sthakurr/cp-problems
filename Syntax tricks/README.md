### Sorting with a key

Given an array of subarrays, we can sort it according to the first element and in case of a tie we can sort it wrt the second element too.

sorted(arr, key=(lambda i: (i[0], -i[0])))

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