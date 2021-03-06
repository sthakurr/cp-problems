## Bit Manipulation

The base of a number system is defined by the number of digits in that system.

1 byte = 8 bits
integer = 4 bytes = 32 bits
character = 1 byte = 8 bits
long long = 8 bytes = 64 bits

Range of integer values:
-   Signed bit:     -2^(31) to (2^31 - 1)
-   Unsigned bit:   0 to (2^32 - 1)

> If the result of a computation requires more bits than allocatted then only the first n bits (the number allocated to that datatype) will be converted to integer and will come out as the output!

### Storing negative values in binary

We can use the leftmost bit as the sign bit (1 would mean +ve and 0 would mean -ve). But this does not leas to the right arithmetic results. So, another encoding mechanism for the negative values is used known as **2's complement**.

### 1's and 2's complement of a binary number

if n = 1001
n's 1's complement is obtained by inverting all the bits: 0110
it's 2's complement is obtained by adding 1 to it: 0110 + 0001 = 0111

Using the 2's complement encoding, 001 + 111 = [1]000 where [1] is overflow ans is not stored. So, 2's complement allows us to perform normal arithmetic computations.

### Operators

-   AND: A&B
-   OR: A|B
-   XOR: A^B
-   NOT: ~A
Remember, if the leftmost bit according to the datatype of any of the above results is 1, then it'll be a negative integer value.

-   Leftshift operators:    A << x shifts the bits of A to left by x positions 
-   rightshift operators:   A >> x shifts the bits of A to the right by x positions and the last x bits are lost this way.

> A << x = multiplication by pow(2, x)
> A >> x = division by pow(2, x)
> 1 << x = pow(2, x)

### Bit Tricks

-   x & (x-1) will clear the lowest set bit of x
-   x & ~(x-1) extracts the lowest set bit of x (all others are clear). Pretty patterns when applied to a linear sequence.
-   x & (x + (1 << n)) = x, with the run of set bits (possibly length 0) starting at bit n cleared.
-   x & ~(x + (1 << n)) = the run of set bits (possibly length 0) in x, starting at bit n.
-   x | (x + 1) = x with the lowest cleared bit set.
-   x | ~(x + 1) = extracts the lowest cleared bit of x (all others are set).
-   x | (x - (1 << n)) = x, with the run of cleared bits (possibly length 0) starting at bit n set.
-   x | ~(x - (1 << n)) = the lowest run of cleared bits (possibly length 0) in x, starting at bit n are the only clear bits.

> Remember that bitwise operators can work with integers too but operation is applied bitwise. So & and && are very different.