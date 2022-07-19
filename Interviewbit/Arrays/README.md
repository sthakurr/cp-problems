## Pointers in C and C++

Pointers are special variables, used to store the addresses of other variables. They are declared using the special character (*) and can be initialised to address of any variable.

```bash
int c;
int *p;
c = 5
p = &c; // stores the address of c
```
 
We can access the value any pointer is storing at its address that it is pointing to using another (*). So, *p will be equal to 5.
Also, we can change the value of any location using its pointer variable like:

```bash
*p = 8 // now c = 8!!
```

## Arrays and Pointers

Arrays are assigned contiguous blocks of memory. 

```bash
int A[5];
```

> The array names point to the base address of the first element! 

So,
A will point to the base address and (A+1) will then point to the starting address of (A+1) where 1 would be called the offset.
So, if A[0] starts from 101, A[1] will start from 105 so:

```bash
printf(A) // will print A[0]
printf(A+1) // will print A[1]
printf(*(A+1)) // will print the value at 105; value of A[1]
```

> The offset in the pointer variables are set according to the datatype: int will give 4 bytes, char will have 1 byte and so on..

### Pointers: different for single variables and arrays

If x is a single variable stored at 300 and int A[3] is an array stored at 400, then:

```bash
int *p = &x
printf(*(A+1)) // will print the value of A[1]
printf(*(p+1)) // will print garbage
```

The neighbors of single variables do not have connected access!

```bash
A++ // will give compilation error
// whereas
int *p = A
p++ // will execute fine
``` 

### Pointers and multidimensional arrays

```bash
int A[2][3];
int *p = A // will give a compilation error since the array name here is not an integer pointer but is pointing to an array
int *p[3] = A // works fine
```

Array Pointer Arithmetics:

```bash
print B     // an array pointer
print *B    // value of this array pointer (an integer pointer: B[0])
print B[0]      // an integer pointer
print &B[0]     // address of an integer pointer
print &B[0][0]      // was a value hence & used to obtain the address
// all the above three statements will return the base address of the 2D array

print (B+1) // offset is of 12 bytes here
print &B[1]     // base address of B[1]
print *(B+1)
print B[1]
print &B[1][0]
// all the above statements will return the base address of the second array in B 

// A General Formula:
B[i][j] = *(B[i] + j) = *( *(B+i) + j)
```

Think the above in terms of array and integer pointer. * (any array pointer) returns an integer pointer! (or any other datatype for that matter)

*( *(B+1))   // equals the value of B[1][0]
*( *B + 1) // equals the value of B[0][1]

## Sorting Algorithms

![Sorting Algorithms](/assets/sorting.png)

### Insertion Sort
Check every element with its preceding values and if they are greater than the key, then shift the values rightwards and insert the key at their location. This can be made more efficient by checking for greater values continuously since the preceding values are getting sorted accordingly, as shown below:

```bash
def insertionSort(arr):
    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):
        key = arr[i]
        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i-1
        while j >=0 and key < arr[j] :
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key 

```

### Merge Sort
Merge sort repeatedly breaks down a list into several sublists until each sublist consists of a single element and merging those sublists in a manner that results into a sorted list.

Top-Down recursive approach:
```bash
# and merge them in sorted order
def merge(Arr, start, mid, end) :

	# create a temp array
	temp[] = [0] * (end - start + 1)

	# crawlers for both intervals and for temp
	i, j, k = start, mid+1, 0

	# traverse both lists and in each iteration add smaller of both elements in temp 
	while(i <= mid and j <= end) :
		if(Arr[i] <= Arr[j]) :
			temp[k] = Arr[i]
			k += 1; i += 1
		else :
			temp[k] = Arr[j]
			k += 1; j += 1

	# add elements left in the first interval 
	while(i <= mid) :
		temp[k] = Arr[i]
		k += 1; i += 1

	# add elements left in the second interval 
	while(j <= end) :
		temp[k] = Arr[j]
		k += 1; j += 1

	# copy temp to original interval
	for i in range (start, end+1) :
		Arr[i] = temp[i - start]
```

Bottom Up iterative approach:
