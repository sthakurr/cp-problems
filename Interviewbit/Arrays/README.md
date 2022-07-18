## Pointers in C and C++

Pointers are special variables, used to store the addresses of other variables. They are declared using the special character (*) and can be initialised to address of any variable.

int c;
int *p;
c = 5
p = &c; // stores the address of c
 
We can access the value any pointer is storing at its address that it is pointing to using another (*). So, *p will be equal to 5.
Also, we can change the value of any location using its pointer variable like:

*p = 8 // now c = 8!!

## Arrays and Memory

Arrays are assigned contiguous blocks of memory. 

int A[5];
A will point to the base address and (A+1) will then point to the starting address of (A+1) where 1 would be called the offset.
So, if A[0] starts from 101, A[1] will start from 105 so:

printf(A) // will print A[0]
printf(A+1) // will print A[1]
printf(*(A+1)) // will print the value at 105; value of A[1]

The offset in the pointer variables are set according to the datatype: int will give 4 bytes, char will have 1 byte and so on..