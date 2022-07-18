## Pointers in C and C++

Pointers are special variables, used to store the addresses of other variables. They are declared using the special character (*) and can be initialised to address of any variable.

int c;
int *p;
c = 5
p = &c; // stores the address of c
 
We can access the value any pointer is storing at its address that it is pointing to using another (*). So, *p will be equal to 5.
Also, we can change the value of any location using its pointer variable like:

*p = 8 // now c = 8!!