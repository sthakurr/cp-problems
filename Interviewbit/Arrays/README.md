## Pointers in C and C++

Pointers are special variables, used to store the addresses of other variables. They are declared using the special character (*) and can be initialised to address of any variable.

```
int c;
int *p;
c = 5
p = &c; // stores the address of c
```
 
We can access the value any pointer is storing at its address that it is pointing to using another (*). So, *p will be equal to 5.
Also, we can change the value of any location using its pointer variable like:

```
*p = 8 // now c = 8!!
```

## Arrays and Pointers

Arrays are assigned contiguous blocks of memory. 

```
int A[5];
```

> The array names point to the base address of the first element! 

So,
A will point to the base address and (A+1) will then point to the starting address of (A+1) where 1 would be called the offset.
So, if A[0] starts from 101, A[1] will start from 105 so:

```
printf(A) // will print A[0]
printf(A+1) // will print A[1]
printf(*(A+1)) // will print the value at 105; value of A[1]
```

> The offset in the pointer variables are set according to the datatype: int will give 4 bytes, char will have 1 byte and so on..

### Pointers: different for single variables and arrays

If x is a single variable stored at 300 and int A[3] is an array stored at 400, then:

```
int *p = &x
printf(*(A+1)) // will print the value of A[1]
printf(*(p+1)) // will print garbage
```

The neighbors of single variables do not have connected access!

```
A++ // will give compilation error
// whereas
int *p = A
p++ // will execute fine
``` 

### Pointers and multidimensional arrays

```
int A[2][3];
int *p = A // will give a compilation error since the array name here is not an integer pointer but is pointing to an array
int *p[3] = A // works fine
```

Array Pointer Arithmetics:

```
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