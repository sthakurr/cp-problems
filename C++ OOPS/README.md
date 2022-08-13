## [Object-Oriented Programming in C++](https://www.geeksforgeeks.org/object-oriented-programming-in-cpp/?ref=lbp)

NOTE: Also refer to Interviewbit/Arrays for Pointers in C++

Major Concepts:

1. **Class** - a user-defined data type to bind together some data members and functions. We can say that a Class in C++ is a blue-print representing a group of objects which shares some common properties and behaviours.

2. **Object** - An instance of a class. When a class is defined, no memory is allocated but when it is instantiated (i.e. an object is created) memory is allocated. Objects can interact without having to know details of each other’s data or code, it is sufficient to know the type of message accepted and type of response returned by the objects.

3. **Encapsulation** - Binding together the data members and functions inside a single unit. A unit's data cannot be accessed by any another unit. also leads to _data abstraction or hiding_ as using encapsulation also hides the data.

4. **Abstraction** - displaying only essential information and hiding the background details or implementation. Abstraction in classes can be performed by choosing to make some members or functions private so they are not seen by the outside world. Even in header files, we directly use a function from some other header file without knowing the real implementation.

5. **Polymorphism** - having many forms. Operator overloading and function overloading. An operator or a function with same symbol or name can do different tasks in different situations.

6. **Inheritance** - the ability of a class to derive properties and functions from another class. 
    - Subclass: the class which is inheriting
    - Superclass: the class from which the functions are inherited from
    - Reusability: use the code from some other class in the newly defined class by inheriting from it.

7. **Access Modifiers** - Used to implement data hiding/abstraction. There are three types of access modifiers in C++: private, public, protected. 
**Note**: If we do not specify any access modifiers for the members inside the class, then by default the access modifier for the members will be Private.

### [Classes and Objects](https://www.geeksforgeeks.org/c-classes-and-objects/?ref=lbp)

Member functions can either be defined inside or outside of a class. For outside, use class_name::function_name
All these member functions are by default inline (which means that they are executed at the point of function call in the main function and so the overhead time of calling the function is not there). _Inline functions_ are actual functions, which are copied everywhere during compilation, like pre-processor macro, so the overhead of function calling is reduced.

1. **Constructors**: Default functions called when an object definition is instantiated
2. **Destructors**: Default functions called when the scope of an object ends

### [Data Hiding and access specifiers](https://www.geeksforgeeks.org/access-modifiers-in-c/)

Outside functions can’t access the information (that is not entitled to them) without having either assigned privileges (such as those possessed by a friend class or an inherited class, as will be seen in this article ahead) or access to one of these 10 members who is allowed direct access to the confidential information (similar to how private members of a class can be accessed in the outside world through public member functions of the class that have direct access to private members).

1. public: the data members and functions defined as public can be accessed from anywhere in the program
2. private: the data members and functions defined as private can only be accessed by the member functions within the class definition. However, these can also be accessed indirectly using the public member functions of the class if they have access to the private members.
3. protected: the data members and functions termed as protected can only be accessed by the class member functions but they can also be accessed by the child class member functions depending on the mode of inheritance (described below)

We can also give access of private and protected members to non-member functions of a class by declaring a _friend function_ inside the class containing the private members.

### Inheritance

Modes of inheritance: