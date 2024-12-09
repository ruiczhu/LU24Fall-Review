# CDS532 Programming

## Examination Arrangement
- This examination will cover Lecture 1 – 11 
- This is a closed-book and written examination. 
- This paper carries Two Parts and 100 Marks. 
  - Part I carries 15 MC questions and 30 Marks.
  - Part II carries 4 questions and 70 Marks. 
- You are required to attempt ALL questions. 
- Please select one answer to each MC question. Selecting more than one answer
to a MC question is equivalent to a wrong answer. 
- Marking scheme for the MC question: +2 marks for a right answer and 0 mark for
a blank answer or a wrong answer. 
- You are reminded of necessity for clear and orderly presentation in your answers.

## Lecture 1 - Introduction to Data Science

### 1. Simplified Computer Architecture

<p style="display: block;">
  <img src="image_10.png" alt="image_10"/>
</p>

<p style="display: block;">
  <img src="image_11.png" alt="image_11"/>
</p>

<p style="display: block;">
  <img src="image_12.png" alt="image_12"/>
</p>

<p style="display: block;">
  <img src="image_13.png" alt="image_13"/>
</p>

<p style="display: block;">
  <img src="image_14.png" alt="image_14"/>
</p>

<p style="display: block;">
  <img src="image_15.png" alt="image_15"/>
</p>

<p style="display: block;">
  <img src="image_16.png" alt="image_16"/>
</p>

### 2. CPU Execution Cycle
![image_17.png](image_17.png)

### 3. Programs Stored in Memory
- Programming code written by programmers can be converted
  into CPU instructions by compilation and interpretation

- compilation
<p style="display: block;">
  <img src="image_18.png" alt="image_18"/>
</p>
- interpretation
![image_19.png](image_19.png)

### 4. Types of Programming Language
![image_20.png](image_20.png)

### 5. Computer program as set of primitives
- Primitive in program language: numbers, strings, Boolean values, operators,
  etc

### 6. Programming Errors
#### 1. Syntactic Errors
- Violating the rules for the structure of a programming language,
- Common and easily caught
#### 2. Static Semantic Errors
- Violating the rules that govern the meaning and relationships between the primitives of
a program that may cause unpredictable behaviour
- Some program language check these error before execution
#### 3. Semantic errors
- Unexpected meaning and behavior of a program when it is executed.
- Program crashes and stops running

## Lecture 2 - Python Basics

### 1. What is a good program X?
- Program X is described by step-by-step instructions, i.e. algorithm.
- Each instruction can be implemented easily.
- Each instruction is not ambiguous.
- Each instruction is concise.
- Program X can be executed quickly on all input instances.
  - Time is money!
- Program X should be extendable.

### 2. What can computer do?
- Most computer devices require programs to function properly.
- A computer program consists of a collection of instructions that performs a
specific task when executed by a computer.
- A computer program is usually written by a computer programmer in a
programming language.

### 3. What is Python?
- Python is a programming language which is:
  - Modern, high-level, general-purpose with multiple paradigms (e.g. procedural,
  object-oriented, and functional programming)
  - Interpreted (No compilation before execution)
  - Dynamically typed (No need to explicitly define the type of variable)
  - Garbage collection (No need to free memory)

### 4. Three Approaches of Running Python Program
- Python IDE (e.g. Jupyter notebook)
- Load the script to Python interpreter directly through command prompt
- Run Python interpreter interactively from command prompt

### 5. What Is a File Path
- Absolute path: It specifies the path to navigate and locate a file from the
root directory
- Relative path: It specifies the path to navigate and locate a file related to
the current directory
- Working directory: The directory that the command prompt is pointing to

![image_26.png](image_26.png)

![image_27.png](image_27.png)

![image_28.png](image_28.png)

## Lecture 3a - Simple Program Development

### 1. The Software Development Process
- The process of creating a program is often broken down into stages
according to the information that is produced in each phase.

![image_29.png](image_29.png)

### 2. Elements of Programs
#### 1. Names/Identifiers
- Names are given to variables (Celsius, Fahrenheit), functions
(convert(), print()), etc.
- These names are called identifiers 
- Every identifier must begin with a letter or underscore ("_"), followed by
any sequence of letters, digits, or underscores. 
- Identifiers are case-sensitive.

- Some identifiers are part of Python itself. These identifiers are known as
reserved words (or keywords). This means they are not available for you to
use as a name for a variable, etc. in your program

![image_30.png](image_30.png)

#### 2. Expressions
- The fragments of code that produce or calculate new data values are called
  expressions.

#### 3. Numeric Data Types
- There are two different kinds of numbers!
- (5, 4, 3, 6) are whole numbers – they don’t have a fractional part
- (.25, .10, .05, .01) are decimal fractions
- Inside the computer, whole numbers and decimal fractions are represented quite
differently!
- We say that decimal fractions and whole numbers are two different data types.

- Whole numbers are represented using the integer (int for short) data type. 
- These values can be positive or negative whole numbers. 
- Numbers that can have fractional parts are represented as floating point (or
float) values.

- A numeric literal without a decimal point produces an int value
- A literal that has a decimal point is represented by a float (even if the
fractional part is 0)
- Why do we need two number types?
  - Values that represent counts can’t be fractional (you can’t have 3 ½ quarters)
  - Most mathematical algorithms are very efficient with integers
  - The float type stores only an approximation to the real number being represented!

<tip>IMPORTANT！！！</tip>

#### 4. Assignment Statements
```
  <variable> = <expr>
```
- The expression on the RHS is evaluated to produce a value which is then
  associated with the variable named on the LHS

- Technically, this model of assignment is simplistic for Python.
- Python doesn't overwrite these memory locations (boxes).
- Assigning a variable is more like putting a “sticky note” on a value and
  saying, “this is x”.
<p style="display: block;">
  <img src="image_32.png" alt="image_32"/>
</p>

- Simultaneous Assignment
  - Several expressions can be evaluated at the same time
  - The results of the evaluated expressions in the RHS can be sequentially assigned to
  the variables on the LHS
  - Expressions and Values are separated by comma
![image_31.png](image_31.png)

### 3. Function
- A function is like a subprogram, a small program inside a program
![image_33.png](image_33.png)

### 4. Comment
- A comment starts with a hash symbol, #, and extends to the end of the line

### 5. Output Statement
![image_34.png](image_34.png)

### 6. Input Statement
<p style="display: block;">
  <img src="image_35.png" alt="image_35"/>
</p>
<p style="display: block;">
  <img src="image_36.png" alt="image_36"/>
</p>
<p style="display: block;">
  <img src="image_37.png" alt="image_37"/>
</p>

### 7. Simple Decisions
<p style="display: block;">
  <img src="image_38.png" alt="image_38"/>
</p>
<p style="display: block;">
  <img src="image_39.png" alt="image_39"/>
</p>
<p style="display: block;">
  <img src="image_40.png" alt="image_40"/>
</p>

### 8. Simple Conditions
- Conditions are based on Boolean expressions
- When a Boolean expression is evaluated, it produces either a bool value of
  True (the condition holds), or it produces False (it does not hold).

### 9. Two-Way Decisions
![image_41.png](image_41.png)

### 10. Multi-Way Decisions
![image_42.png](image_42.png)

### 11. Definite Loops
![image_43.png](image_43.png)

### 12. Indefinite Loops
![image_44.png](image_44.png)

### 13. Basic Python Programming Example
```python
#    A program to compute the value of an investment
#    carried 10 years into the future

def main():
    print("This program calculates the 
    future value of a 10-year investment.")

    principal = eval(input("Enter the initial principal: "))
    apr = eval(input("Enter the annual interest rate: "))

    for i in range(10):
        principal = principal * (1 + apr)

    print ("The value in 10 years is:", principal)

main()
```

## Lecture 3b - Boolean Operators
### 1. Boolean Operators
#### 1. AND
![image_45.png](image_45.png)
#### 2. OR
![image_46.png](image_46.png)
#### 3. NOT
![image_47.png](image_47.png)

<tip>IMPORTANT！！！</tip>

### 2. Boolean Expressions
<p style="display: block;">
  <img src="image_48.png" alt="image_48"/>
</p>
<p style="display: block;">
  <img src="image_49.png" alt="image_49"/>
</p>
<tip>IMPORTANT！！！</tip>

### 3. Boolean Expressions as Decisions - Example
```python
ans = input("What flavor of you want [vanilla]: ")
if ans:
    flavor = ans
else:
    flavor = "vanilla"
```

## Lecture 4 - Number Computation

### 1. Numeric Data Types
- In mixed-typed expressions Python will convert ints to floats.
- Sometimes we want to control the type conversion. This is called explicit
  typing.
![image_51.png](image_51.png)

### 2. Type Conversions & Rounding
![image_50.png](image_50.png)

### 3. Type Conversions & Rounding - Example
```python
# change.py
#   A program to calculate the value of some change in dollars
def main():
  print("Change Counter")
  print()
  print("Please enter the count of each coin type.")
  quarters = int(input("Quarters: "))
  dimes = int(input("Dimes: "))
  nickels = int(input("Nickels: "))
  pennies = int(input("Pennies: "))
  total = quarters * .25 + dimes * .10 + nickels * .05 + pennies * .01
  print()
  print("The total value of your change is", total)
```

### 4. Using the math Library
![image_52.png](image_52.png)
- Example:
```python
import math  # Makes the math library available.
def main():
  print("This program finds the real solutions to a quadratic")
  a, b, c = eval(input("Please enter the coefficients (a, b, c): "))
  discRoot = math.sqrt(b * b - 4 * a * c)
  root1 = (-b + discRoot) / (2 * a)
  root2 = (-b - discRoot) / (2 * a)
  print("The solutions are:", root1, root2 )
main()
```
<p style="display: block;">
  <img src="image_53.png" alt="image_53"/>
</p>
<p style="display: block;">
  <img src="image_54.png" alt="image_54"/>
</p>

### 5. range Function
- The range function generates a sequence of numbers.
<p style="display: block;">
  <img src="image_55.png" alt="image_55"/>
</p>

- factorial program example
```python
# Program to compute the factorial of a number
# Illustrates for loop with an accumulator
def main():
  n = eval(input("Please enter a whole number: "))
  fact = 1
  for factor in range(n,1,-1): 
    fact = fact * factor
  print("The factorial of", n, "is", fact)
main()
```

### 6. The Limits of Int
- While there are an infinite number of integers, there is a finite range of
  ints that can be represented.
- This range depends on the number of bits a particular CPU uses to
    represent an integer value.
- ![image_56.png](image_56.png)

### 7. Handling Large Numbers
![image_57.png](image_57.png)
- Floats are approximations
- Floats allow us to represent a larger range of values, but with fixed
precision. 
- Python has a solution, expanding ints!
- Python ints are not a fixed size and expand to handle whatever value it
holds. 
- Newer versions of Python automatically convert your ints to expanded form
when they grow so large as to overflow. 
- We get indefinitely large values (e.g. 100!) at the cost of speed and
memory.

## Lecture 5 - Object

### 1. What is Object-Oriented Programming
- Basic idea – view a complex system as the interaction of simpler objects
which are representations of real-world entities.
- An object is a sort of active data type that combines data and operations.
  - Field (The properties of the object, i.e. Data)
  - Method (The actions of the object, i.e. Operation)
- Objects interact by sending each other messages (i.e. calling methods just
like functions).
<p style="display: block;">
  <img src="image_58.png" alt="image_58"/>
</p>

### 2. Object-Oriented Programming
- Programmers define and
implement the fields and
methods of the object as Class
template in the source code they
wrote.
<p style="display: block;">
  <img src="image_59.png" alt="image_59"/>
</p>

- Example 2D Point Class:
```python
class Point:
  def __init__(self):
    self.X = 0
    self.Y = 0
  def setX(self, x1):
    self.X = x1
  def setY(self, y1):
    self.Y = y1
  def getX(self):
    return self.X
  def getY(self):
    return self.Y
```
<tip>
    The __init__ method is a special method that is called when an object is
        created. It initializes the object’s data.
</tip>
<tip>
    The methods of an object instance can access and update the fields by using the self keyword.
</tip>

```python
point1 = Point()
point1.setX(1.0)
point1.setY(2.0)
print("The X coordinate of Point 1:", point1.getX()) 
print("The Y coordinate of Point 1:", point1.getY())
```
<tip>
A point object is created by call the constructor function 
of the object.
</tip>

## Lecture 6 - Sequences
### 1. The String Data Type
- A string is a sequence of characters enclosed within quotation marks (")
  or apostrophes (').
![image_60.png](image_60.png)

<tip>We can access the individual characters in a string through indexing.</tip>

```python
greet = "Hello Bob"
print(greet[0])
# Output: H

greet = "Hello Bob"
print(greet[-1])
# Output: b
``` 
### 2. slicing
- Slicing is a way to extract a subsequence of a string.

- The syntax for slicing is:
```
  <string>[<start>:<end>:<step>]
```
- Example:
```python
greet = "Hello Bob"

print(greet[0:3])
# Output: Hel

print(greet[:5])
# Output: Hello

print(greet[5:])
# Output: Bob

print(greet[:])
# Output: Hello Bob

print(greet[::2])
# Output: HloBb
```
### 3. String Operation
#### 1. Concatenation
- Concatenation operator (+) glues two strings together
- The result is a new string
```python
concat_str1 = "spam" + "eggs"
concat_str2 = "Spam" + "And" + "Eggs"

print(concat_str1)
# Output: spameggs
print(concat_str2)
# Output: SpamAndEggs
```

#### 2. Repetition
- Repetition operator (*) builds up a string by multiple concatenations of a
string with itself
- The result is a new string
```python
repetition_str1 = 3 * "spam"
repetition_str2 = "spam" * 5
repetition_str3 = (3 * "spam") + ("eggs" * 5)
print(repetition_str1)
print(repetition_str2)
print(repetition_str3)

# Output: spamspamspam
# Output: spamspamspamspamspam
# Output: spamspamspameggseggseggseggseggs
```

#### 3. len() Function
- The function len() will return the length of a string.
```python
str1_len = len("spam")
print(str1_len)
# Output: 4
```

#### 4. Iterating every character in a string
- String is a sequence of characters, so its characters can be iterated with a
  for loop
```python
for ch in "Spam!":
    print(ch, end=" ")
# Output: S p a m !
```

#### 5. Summary
<p style="display: block;">
  <img src="image_61.png" alt="image_61"/>
</p>

### 4. Simple String Processing - Examples
```python
# A program to print the abbreviation of a month, given its number
def main():
    # months is used as a lookup table
    months = "JanFebMarAprMayJunJulAugSepOctNovDec"
    n = int(input("Enter a month number (1-12): "))

    # compute starting position of month n in months
    pos = (n-1) * 3
    
    # Grab the appropriate slice from months
    monthAbbrev = months[pos:pos+3]

    # print the result
    print ("The month abbreviation is", monthAbbrev + ".")

main()

# Input: 3
# Output: The month abbreviation is Mar.
```
### 5. String Representation
- The ord function returns the numeric (ordinal) code of a single character.
```python
numeric_code = ord("A")
print(numeric_code)
# Output: 65
```
- The chr function converts a numeric code to the corresponding character.
```python
character = chr(97)
print(character)
# Output: a
```

### 6. Conversion between String and Other Data Type
<p style="display: block;">
  <img src="image_62.png" alt="image_62"/>
</p>

### 7. String Methods
<p style="display: block;">
  <img src="image_63.png" alt="image_63"/>
</p>
<p style="display: block;">
  <img src="image_64.png" alt="image_64"/>
</p>

<tip>It turns out that strings are really a special kind of sequence, 
so string operations also apply to sequences!</tip>

### 8. Lists as Sequences

<tip>IMPORTANT！！！</tip>

<tip>Strings are always sequences of characters, but lists can be sequences of 
arbitrary values. <br/>
Lists can have numbers, strings, or both!</tip>

Example:
```python
# A program to print the month name, given it's number.
# This version uses a list as a lookup table.
def main():
    # months is a list used as a lookup tableab
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November",
              "December"]

    n = int(input("Enter a month number (1-12): "))
    
    print ("The month abbreviation is", months[n-1] + ".")
main()
# Input: 3
# Output: The month abbreviation is March.
```

<tip>List is mutable, meaning its 
elements CAN be changed. </tip>
<tip>Strings is immutable, meaning its 
element CANNOT be changed</tip>

### 9. append() Method
- The append method adds a new element to the end of a list.
```python
squares = []
for x in range(1,101):
  squares.append(x*x)
```

### 10. String Formatting
- To output the value of coins in the expected format, we can modify the
  print statement as follows:
```python
print("The total value of your change is ${0:0.2f}".format(total))

# ${0:0.2f} is a string format template
# .format(total) is the function for 
# specifying the expression(s) subjected to be printed
```
- The form of description is:
```
<index>:<format-specifier>
```
<tip>Index (start from 0) specific which 
parameter of the format() should be
insert into the slot. In this case, total.</tip>

- format-specifier
```
<width>.<precision><type>
```
- Example:
```python
print("Hello {0} {1}, you may have won ${2}"
 .format("Mr.", "Smith", 10000))
print('This int, {0:5}, was placed in a field of width 5'
.format(7))
print('This int, {0:10}, was placed in a field of witdh 10'
.format(10))
print('This float, {0:10.5}, has width 10 and precision 5.'
.format(3.1415926))
print('This float, {0:10.5f}, is fixed at 5 decimal places.'
.format(3.1415926))
print("Compare {0} and {0:0.20}"
.format(3.14))

# Output:
# Hello Mr. Smith, you may have won $10000
# This int,     7, was placed in a field of width 5
# This int,         10, was placed in a field of witdh 10
# This float,    3.1416, has width 10 and precision 5.
# This float,    3.14159, is fixed at 5 decimal places.
# Compare 3.14 and 3.1400000000000001243
```
- numeric values are right-justified and strings are left-justified by default
- Text justification can be explicitly specified by putting a symbol before the
  width definition.
```python
print("left justification: [{0:<5}]".format("Hi!"))
print("right justification: [{0:>5}]".format("Hi!"))
print("centered: [{0:^5}]".format("Hi!"))

# Output:
print("left justification: [{0:<5}]".format("Hi!  "))
print("right justification: [{0:>5}]".format("  Hi!"))
print("centered: [{0:^5}]".format(" Hi! "))
```
### 11. Text files: Multi-Line Strings
- A file is a sequence of data that is stored in secondary memory (disk drive). 
- A file is a sequences of bytes that can represent any types of data, but the
  easiest to work with are text. 
- A text file can contain more than one line of text, each line is delimited by a
  newline character  (\n)

### 12. File processing
- The process of opening a file involves associating a file on disk with an
  object in memory. 
- We can manipulate the file by manipulating the object to read data from a
  file or write data to a file. 
- When done with the file, it needs to be closed. Closing the file causes any
  outstanding operations and other bookkeeping for the file to be completed. 
- In some cases, not properly closing a file could result in data loss

#### 1. Reading File - General Operation Flow
<p style="display: block;">
  <img src="image_65.png" alt="image_65"/>
</p>

#### 2. Writing File - General Operation Flow
<p style="display: block;">
  <img src="image_66.png" alt="image_66"/>
</p>

#### 3. working with file in Python
<p style="display: block;">
  <img src="image_67.png" alt="image_67"/>
</p>

<p style="display: block;">
  <img src="image_68.png" alt="image_68"/>
</p>

<p style="display: block;">
  <img src="image_69.png" alt="image_69"/>
</p>

<tip>
不知道老师有没有说下面这个文件打开方式。这个方法会自动关闭文件，不用手动，更安全。
</tip>

```python
# 使用 with 语句打开文件
with open('example.txt', 'r') as file:
    content = file.read()
    print(content)
```

## Lecture 7 - Data Collection
<tip>IMPORTANT！！！</tip>

### 1. Lists and Arrays
- A list or array is a sequence of items where the entire sequence is referred
  to by a single name (i.e. s) and individual items can be selected by indexing

- Python lists are dynamic. They can grow and shrink on demand.
- Python lists are also heterogeneous, a single list can hold arbitrary data
  types.
- Python lists are mutable sequences of arbitrary objects.
<tip>string is a special version of list</tip>

### 2. Lists vs String
- Strings are always sequences of characters, but lists can be sequences of
  arbitrary values.
- Lists are mutable, meaning they can be changed. Strings cannot be
  changed.

### 3. List Operations
<p style="display: block;">
  <img src="image_70.png" alt="image_70"/>
</p>

#### 1. Membership
- can be used to check if a certain value appears anywhere in a sequence.
```python
lst = [1,2,3,4]
check_membership1 = 3 in lst
check_membership2 = 5 in lst
print(check_membership1)
print(check_membership2)
# Output: True, False
```

#### 2. append() Method
- New value can be added at the end of a sequence through append().
- Lists can be constructed one piece at a time using append().

```python
nums = []
nums.append(10)
nums.append(20)
nums.append(30)
nums.append(40)
nums.append(50)
print(nums)
# Output: [10, 20, 30, 40, 50]
```

#### 3. del
- Individual items or entire slices can be removed from a list using the del
  operator.

```python
myList=[34, 26, 0, 10]
print(myList)
del myList[1]
print(myList)
del myList[1:3]
print(myList)

# Output: [34, 26, 0, 10]
# Output: [34, 0, 10]
# Output: [34]
```

#### 4. Extra List Operations
<p style="display: block;">
  <img src="image_71.png" alt="image_71"/>
</p>

### 4. Dictionary
<tip>IMPORTANT！！！</tip>
- A dictionary is a collection that allows us to look up a piece of information
  (value) associated with an arbitrary key.
- Its working principle is similar to a locker cabinet

<p style="display: block;">
  <img src="image_72.png" alt="image_72"/>
</p>

<p style="display: block;">
  <img src="image_73.png" alt="image_73"/>
</p>

#### 1. Dictionary initialization
```python
# Create a dictionary
score = {"Peter":"80", "John":"78", "Bill":"90"}
print(score)
# Output: {'Peter': '80', 'John': '78', 'Bill': '90'}
```
#### 2. Accessing element from dictionary
```python
score = {"Peter":"80", "John":"78", "Bill":"90"}
print(score["Peter"])
# Output: 80
```

#### 3. Modifying Values in a Dictionary
```python
score = {"Peter":"80", "John":"78", "Bill":"90"}
score["Peter"] = "60"
print(score)
# Output: {'Peter': '60', 'John': '78', 'Bill': '90'}
```

#### 4. Dictionary Operations
<p style="display: block;">
  <img src="image_74.png" alt="image_74"/>
</p>

### 5. word frequency example
```python
def byFreq(pair):
    return pair[1]

def main():
    fname = input("File to analyze: ")
    text = open(fname,'r').read()
    text = text.lower()
    for ch in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~':
        text = text.replace(ch, ' ')
    words = text.split()

    counts = {}
    for w in words:
        counts[w] = counts.get(w,0) + 1

    items = list(counts.items())
    items.sort()
    items.sort(key=byFreq, reverse = True)

    for word, count in items:
        print("{0:<15}{1:>5}".format(word, count))

main()
```

## Lecture 8 - Function

- A function is like a subprogram, a small program inside of a program
- The basic idea – we write a sequence of statements and then give that
  sequence a name. We can then execute this sequence at any time by
  referring to the name. 
- The part of the program that creates a function is called a function
  definition. 
- When the function is used in a program, we say the definition is called or
  invoked.

### 1. Functions and Parameters
<p style="display: block;">
  <img src="image_75.png" alt="image_75"/>
</p>

- A function is called by using its name followed by a list of actual parameters 
or arguments.
```
<name>(<actual-parameters>)
```
### **Four-step** process
- When Python comes to a function call, it initiates a **four-step** process.
1. The calling program suspends execution at the point of the call.
2. The formal parameters of the function get assigned the values supplied
by the actual parameters in the call. 
3. The body of the function is executed. 
4. Control returns to the point just after where the function was called

<tip>IMPORTANT！！！</tip>

### 2. Functions That Return Values
- One "gotcha" – all Python functions return a value, whether they contain a
return statement or not. Functions without a return hand back a special
object, denoted None. 
- A common problem is writing a value-returning function and omitting the
return!
- If your value-returning functions produce strange messages, check to make
sure you remembered to include the return!

### 3. Functions that Modify Parameters
- The formal parameters of a function only receive the values of the actual parameters
(i.e. Each formal parameter become a extra reference to the value of the
corresponding actual parameter). 
- The function does not have access to the actual parameter variable. 
- Python is said to pass all parameters by object reference.

<p style="display: block;">
  <img src="image_76.png" alt="image_76"/>
</p>
<tip>这块是重点，Lecture 8 PPT上有完整的函数执行流程，这里就不全截图了。</tip>

### 4. Call Stack
- A call stack is a stack that stores information about the function currently
being executed in a computer program. 
- Although the call stack is important for the proper execution of any
program, the details are normally hidden and automatic in high-level
programming languages.

#### 1. What is a Stack
- A stack is a list that can only insert and delete element at the head 
- Push: Inserts an element at the head 
- Pop: Deletes (and returns) the element at the head
![image_77.png](image_77.png)

![image_78.png](image_78.png)

![image_79.png](image_79.png)

<tip>Lecture 8b PPT上有完整stack的演示，这里就不全截图了。</tip>
<tip>Stack 就是记住这句 后进先出（LIFO，Last In First Out）</tip>

#### 2. Recursion
- **Objective:** Decompose a task into smaller sub-tasks. 
  - At least one of the sub-tasks is a smaller case of the same task. 
  - The smallest case of the same task (base case) has a non-recursive solution. 
- **Recursive function:** a function that calls itself either directly or indirectly
through another function. 
- Recursion exploited the properties of the call stack such that the sub tasks
are executed in Last In First Out (LIFO) order.

<tip>Lecture 8b PPT上有完整递归的演示，这里就不全截图了。</tip>
<tip>深入到底：递归函数不断调用自身，直到满足终止条件，深入到最深层。<br/>
逐层返回：满足终止条件后，递归函数开始逐层返回，从最深层逐步回到最外层</tip>

#### Recursion Example - Factorial
```python
def factorial(n):
    if n == 0:
        return 1
    
    return n * factorial(n - 1)

factorial(10)
# Output: 3628800
```

#### Recursion Example - Fibonacci
```python
def fibonacci(n):
    if (n == 0) or (n==1):
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
fibonacci(10)
# Output: 55
```

## Lecture 9 - Packaging
### 1. Importing functions from module/package
<p style="display: block;">
  <img src="image_80.png" alt="image_80"/>
</p>

### 2. Import a function from a module directly
- To import a function in a module directly into our program
```
from <module_name> import <function_name>
```
- To import every function in the module directly into our program  
```
from <module_name> import *
```
<tip>It is not recommended in general as it often causes poorly readable code</tip>

- To import a module to our program
```
import <module_name>
```
- After import, the target function can be used by first specifying the name of
the corresponding module and then the name of function
```
<module_name>.<function_name>()
```
- Example
```python
import math
math.sqrt(100)
```

### 3. Import a function from a package directly
- To import a function in a package directly into our program
```
from <package_name>.<module_name> import <function_name>
```
- To import every function in the package directly into our program
```
from <package_name>.<module_name> import *
```
- Example
```python
from numpy.random import rand
rand(3,2)
```

### 4. Assigning new names for imported modules
- New names can be assigned to modules imported to avoid name collision
```
import <module_name> as <new_name>
```
- Example
```python
import math as mathLib
mathLib.sqrt(100)
```

### 5. Assigning new names for imported functions
- New names can be assigned to functions imported to avoid name collision
```
from <module_name> import <function_name> as <new_name>
```
- Example
```python
from math import sqrt as sqrt_imported
sqrt_imported(100)
```

## Lecture 10 - NumPy
<tip>IMPORTANT！！！</tip>

### 1. What is NumPy?
- Fundamental package for scientific computing with Python 
- Contains a powerful N-dimensional array object 
- Provides sophisticated (broadcasting) functions 
- Can integrate C/C++ and Fortran code 
- Useful linear algebra, Fourier transform, and random number capabilities

### 2. Introduction to NumPy Array
- NumPy has a multi-dimensional array object, ndarray. It consists of
  - The actual data 
  - Some meta-data describing the data 
- Ndarray behaves similarly to a vector or a matrix.
- Ndarray is homogeneous (i.e. all data elements have the same). 
- The index of Ndarrays starts from 0.

### 3. Creating Ndarray
- The array() function creates an Ndarray from an object provided. The
  object needs to be array-like, e.g., a Python list.
```python
import numpy as np
vector_1d = np.array([2, 4, 6, 8, 10, 12])
```
![image_81.png](image_81.png)

- The array() function creates an Ndarray from an object provided. The
  object needs to be array-like, e.g., a Python list.
```python
import numpy as np
matrix_2d = np.array([[3, 4, 5], [6, 7, 8]])
```
![image_82.png](image_82.png)
<tip>The matrix is specified in a row-by-row manner, 
starting from the first row to the last row</tip>

- The array() function creates an Ndarray from an object provided. The
  object needs to be array-like, e.g., a Python list.
```python
tensor_3d = np.array([[[10, 11, 12], [13, 14, 15]], [[16, 17, 
18], [19, 20, 21]]])
```

![image_83.png](image_83.png)
<tip>The 3D tensor is specified in a page-by-page manner, starting from the first 
matrix(page) to the last matrix (page)</tip>

- The **arange()** and **linspace()** functions can create arrays containing a
continuous sequence of numbers
```python
import numpy as np
vector_1 = np.arange(1, 9, 2)
vector_2 = np.linspace(0, 1, 11)

print(vector_1)
print(vector_2)
# Output: [1 3 5 7]
# Output: [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]
```

<p style="display: block;">
  <img src="image_84.png" alt="image_84"/>
</p>
<p style="display: block;">
  <img src="image_85.png" alt="image_85"/>
</p>

- The **zeros()** and **ones()** functions can create arrays with zeros and ones
respectively.
```Python
np.zeros((2, 4))
np.ones((2, 3, 4))
```

- The **eye()** functions creates an Identity matrix.
```Python
np.eye(4)
```

- The **diag()** function creates a matrix with the diagonal elements taking
from an 1-D array.
```Python
np.diag(np.array([2, 3, 4]))
```

- NumPy has build-in functions which support generating NumPy arrays with
  various kinds of distributions: uniform, standard normal, beta, Dirichlet, ...
```Python
np.random.rand(3)
np.random.standard_normal((2, 3, 4))
```

### 4. Accessing Basic Metadata of Arrays
- The **ndim** attribute of a Ndarray returns the number of dimensions
```
<Ndarray>.ndim
```

- The **shape** attribute describes the number of elements in each dimension
```
<Ndarray>.shape
```

- The **size** attribute returns the total number of elements in an array object
```
<Ndarray>.size
```

- The **dtype** attribute returns the data type of elements in an array object
```
<Ndarray>.dtype
```

### 5. Data Type of Nparray
- NumPy auto-detects the data-type from the input.
- To create a Nparray with an explicit defined data type:
```Python
  np.array([1, 2, 3, 4, 5], dtype='int8')
```
**Boolean and integer data types**
<p style="display: block;">
  <img src="image_86.png" alt="image_86"/>
</p>

**floating-point number and complex number data types**
<p style="display: block;">
  <img src="image_87.png" alt="image_87"/>
</p>

### 6. Indexing, Slicing, and Assignment

- Nparray supports indexing, slicing, and assignment operation similar to list.
- Consider the following 1d vector Nparray below:
- Accessing an element through index
- Accessing a sub array through slicing
```python
a = np.arange(10)
print(a[0])
print(a[5:7])
```

- Accessing a sub array through slicing with step
- Assigning new values to an element with a given index
- Assigning new values to several elements with slicing
```python
print(a[1:6:2])
a[0] = 10
a[5:] = -1
```

- Consider the following 2d matrix Nparray:
- Accessing an element through indexing 
- Assign new value to an element through indexing in a 2d matrix Nparray:
```python
b = np.array([[3,4,5], [6,7,8]])
print(b[1,2])
```
![image_88.png](image_88.png)

```python
print(b[0,1:3])
```
![image_89.png](image_89.png)

- Consider the following 3d tensor Nparray:
- Accessing an element through indexing in a 3d tensor Nparray
```python
c = np.array([[[10, 11, 12], [13, 14, 15]], [[16, 17, 18], [19, 
20, 21]]])
print(c[0, 0, 2])
print(c[:, 1, 1:2])
```
<p style="display: block;">
  <img src="image_90.png" alt="image_90"/>
</p>
<p style="display: block;">
  <img src="image_91.png" alt="image_91"/>
</p>

### 7. Copies and Views
- Slicing operation creates a view on the original array, which is an alternative
way of accessing array data. 
- The original array is NOT duplicated in memory!

### 8. Arithmetic Operations and Functions
- When a simple comparison and logical operation is performed on an NumPy
  array, the operation is performed on each element directly:
```python
import numpy as np
a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 2, 1])
print(a == b)
print(a > 2)

# Output: [ True  True False False]
# Output: [False False  True  True]
```

```python
import numpy as np
a = np.array([1, 1, 0, 0], dtype=bool)
b = np.array([1, 0, 1, 0], dtype=bool)
print(np.logical_or(a, b))
print(np.logical_and(a, b))

# Output: [ True  True  True False]
# Output: [ True False False False]
```

### 9. Advanced Indexing Techniques
- NumPy array supports extracting sub array with a list of indices.

```python
import numpy as np
a = np.arange(7, 0, -1)
selector = np.array([1, 3, 5])
print(a)
print(selector)
print(a[selector])

# Output: [7 6 5 4 3 2 1]
# Output: [1 3 5]
# Output: [6 4 2]
```
- NumPy array supports extracting sub array with a Boolean mask.
```python
import numpy as np
a = np.arange(8)
selector = (a % 2) == 0
print(a)
print(selector)
print(a[selector])

# Output: [0 1 2 3 4 5 6 7]
# Output: [ True False  True False  True False  True False]
# Output: [0 2 4 6]
```
<p style="display: block;">
  <img src="image_92.png" alt="image_92"/>
</p>

### 10. Sorting
- When an NumPy array with a dimension > 2 is sorted, an axis has to be
  specified as the direction (along the row/column) for sorting
- For example:
```python
a = np.array([[1, 20, 3], [40, 5, 60]])
```
![image_93.png](image_93.png)

- Sorting along the column (along the direction of axis 0)
```python
a.sort(axis=0)
```
<p style="display: block;">
  <img src="image_94.png" alt="image_94"/>
</p>

- Sorting the row (along the direction of axis 1)
```python
a.sort(axis=1)
```
<p style="display: block;">
  <img src="image_95.png" alt="image_95"/>
</p>

### 11. Aggregations

- Summary statistics of each column or row can be easily computed with
aggregation functions 
- Similar to sort(), an axis has to be specified as the direction (along the
row/column) for aggregation operation 
- Consider the following 2d matrix Nparray:
```python
a = np.array([[1, 2], [4, 3]])
```
![image_96.png](image_96.png)

- For each row (along the direction of axis = 1), calculating the sum of
element:
```python
x.sum(axis = 1)
```
<p style="display: block;">
  <img src="image_97.png" alt="image_97"/>
</p>

- For each column (along the direction of axis = 0), calculating the sum of
element:
```python
x.sum(axis = 0)
```
<p style="display: block;">
  <img src="image_98.png" alt="image_98"/>
</p>

## Lecture 11 - Pandas
### 1. What is Pandas
- Python language provides libraries for faster and more convenient handling
  of data 
- Pandas (stands for "Python Data Analysis Library") is one of the most
  commonly used tools for data analysis and manipulation

### 2. Data in Pandas
- In Panda, there are 3 types of object mainly used to represent data:
1. Series
   - 1 dimensional
   - Like a 1-D list object or a 1-D numpy array 
2. DataFrame
   - 2 dimensional
   - Table-like data
3. Panel
   - 3 or more dimensional

### 3. Series
- list, tuple, dictionary, numpy array can be directly turn into Series
- Series uses "index" to specify each element 
- By default, integer is used as the index of a Series object

```python
import numpy as np
import pandas
list_1 = [1, 3, 4, 5, 6, 7]
series_2 = pandas.Series(list_1)
dict_1 = {'Day 1':1, 'Day 2':3, 'Day 3':4, 'Day 4':5, 
'Day 5':6, 'Day 6':7}
series_3 = pandas.Series(dict_1)
print(series_2)
print(series_3)
```

### 4. DataFrame
- pandas.DataFrame() represents two-dimensional tabular data 
- It consists of rows and columns 
- It can be viewed as containing multiple Series (columns)
<p style="display: block;">
  <img src="image_99.png" alt="image_99"/>
</p>

- Similar to Series, DataFrame can be created from list, tuple, dictionary,
  numpy array
- For DataFrame, we use "index" to specify the labels of rows and "columns"
  to specify the labels of columns
```python
nparray1 = np.array([[4, 4, 5, 0], 
                     [5, 3, 0, 4], 
                     [2, 5, 2, 3]])
df2 = pandas.DataFrame(nparray1, 
         index=['Day 1', 'Day 2', 'Day 3'], 
         columns=['Item 1', 'Item 2', 'Item 3', 'Item 4'])
```
- The labels of rows can be retrieved by:
- The labels of columes can be retrieved by:
```python
df.index
df.columns
```

### 5. Selecting Data with Indexing
- In a Series, we use **index** to specify a data element:
<p style="display: block;">
  <img src="image_100.png" alt="image_100"/>
</p>
<p style="display: block;">
  <img src="image_101.png" alt="image_101"/>
</p>

- In a DataFrame, use **column** to return a column as a Series:
<p style="display: block;">
  <img src="image_102.png" alt="image_102"/>
</p>

-  Similarly, we can use **column index** to specify a data element:
<p style="display: block;">
  <img src="image_104.png" alt="image_104"/>
</p>

- We can also use **loc[]** to specify a data element:
<p style="display: block;">
  <img src="image_103.png" alt="image_103"/>
</p>

### 6. Selecting Data with Slicing
- pandas also allow slicing operation using colon ":"
<p style="display: block;">
  <img src="image_105.png" alt="image_105"/>
</p>

- Notably, for DataFrame object, slicing only refers to rows but not columns:
<p style="display: block;">
  <img src="image_106.png" alt="image_106"/>
</p>

### 7. Selecting Data with Condition Query
- Pandas support selecting specific element(s) through defining certain
condition(s):
- For example, the elements in a series that are equal to 5 can be selected by:
```python
s = pandas.Series([1, 4, 5, 7, 8])
print(s[s == 5]) 

# Output: 2    5
```
<p style="display: block;">
  <img src="image_107.png" alt="image_107"/>
</p>

- For DataFrame, you can query specific row(s) by adding condition(s) to the
  columns:
```python
nparray = np.array([[4, 4, 5, 0], 
                    [5, 3, 0, 4], 
                    [2, 5, 2, 3]])
df = pandas.DataFrame(nparray, 
         index=['Day 1', 'Day 2', 'Day 3'],
         columns=['Item 1', 'Item 2', 'Item 3', 'Item 4'])
print(df[df['Item 2'] > 3])
```

- Multiple conditions can be joined using characters "&" (and) or "|" (or):
```python
nparray = np.array([[4, 4, 5, 0], 
                    [5, 3, 0, 4], 
                    [2, 5, 2, 3]])
df = pandas.DataFrame(nparray, 
         index=['Day 1', 'Day 2', 'Day 3'], 
         columns=['Item 1', 'Item 2', 'Item 3', 'Item 4'])
print(df[(df['Item 2'] > 3) & (df['Item 4'] == 0)])
print(df[(df['Item 2'] > 3) | (df['Item 1'] == 5)])
```
<tip>Each condition should be placed inside a pair of brackets ( and )</tip>

### 8. Working with Missing Data
- One advantage of pandas is that it allows empty cells (NA values *) and
  provides relevant operations
```python
raw_data = [[1, 2, None, 4], 
            [2, 3, 4, 5], 
            [3, None, 5, 6]]
df = pandas.DataFrame(raw_data, 
           index=['Row 1', 'Row 2', 'Row 3'],
           columns=['Col 1', 'Col 2', 'Col 3', 'Col 4'])
print(df)
```
<tip>By default, "None", "np.nan" and "NaN" "NaT" in 
pandas are considered as NA values</tip>

- Functions **isna()** and **notna()** can be used to determine which value(s)
  stored in a series or a dataframe is/are missing value

- **dropna()** function can be used to removes the rows containing NA values
  (or elements with NA value in Series) and return the DataFrame (or Series)
  after removal

- By specifying axis="columns", dropna() can remove the column
  containing NA values in a given DataFrame
```python
df.dropna(axis="columns")
```
<p style="display: block;">
  <img src="image_108.png" alt="image_108"/>
</p>

- fillna() function can be used to fill NA values with a given value
  df.fillna(0)
<p style="display: block;">
  <img src="image_109.png" alt="image_109"/>
</p>

- A deliciated value can even be specified for replacing the missing values of
  each column

```python
df.fillna({'Col 1': 100, 'Col 2': 200, 'Col 3': 300, 'Col 4': 400})
```

<p style="display: block;">
  <img src="image_110.png" alt="image_110"/>
</p>

### 9. Adding new data
- series
```python
import pandas
s = pandas.Series([1, 2, 3])
print(s)
s[3] = 0
print(s)
s_2 = pandas.Series({'a':1, 'b':2, 'c':3})
print(s_2)
s_2['d'] = 9
print(s_2)
```

- DataFrame
```python
import pandas
nparray = np.array([[4, 4, 5, 0], 
                    [5, 3, 0, 4], 
                    [2, 5, 2, 3]])
df = pandas.DataFrame(nparray, 
         index=['Day 1', 'Day 2', 'Day 3'], 
         columns=['Item 1', 'Item 2', 'Item 3', 'Item 4'])
print(df)
df['Item 5'] = [9, 9, 9]
print(df)
```

### 10. Joining different series
- Function concat() can be used to join another series at the end of a given
series
```python
  s1 = pandas.Series([1, 2, 3])
  s2 = pandas.Series([9, 0])
  s3 = pandas.concat([s1, s2])
```
<note>Note: concat() function will not change 
the original series. Therefore, remember to 
assign the return value as a new variable</note>

- If the concat() function is called directly, the indexes of new data inserted
  into the series will not be updated
<p style="display: block;">
  <img src="image_111.png" alt="image_111"/>
</p>

### 11. Joining different dataframes
- Function concat() can be used to join another dataframe at the end of a
given dataframe
```python
a = np.array([[0, 1, 2, 3],
              [4, 5, 6, 7],
              [8, 9, 10, 11]])
b = np.array([[0, 0, 0, 0],
              [9, 9, 9, 9]])
df1 = pandas.DataFrame(a, index=['Row 1', 'Row 2', 'Row 3'],
         columns=['Col 1', 'Col 2', 'Col 3', 'Col 4'])
df2 = pandas.DataFrame(b, index=['Row 1', 'Row 2'] ,
         columns=['Col 1', 'Col 2', 'Col 3', 'Col 4'])
df3 = pandas.concat([df1, df2])
```
<note>Note: concat() function will not 
change the original dataframe. 
Therefore, remember to assign the 
return value as a new variable</note>

<p style="display: block;">
  <img src="image_112.png" alt="image_112"/>
</p>

- To automatically adjust the indexes, set the ignore_index attribute to
  True (default is False) as demonstrated below
```python
s3 = pandas.concat([s1, s2], ignore_index=True)
df3 = pandas.concat([df1, df2], ignore_index=True)
```

<p style="display: block;">
  <img src="image_113.png" alt="image_113"/>
</p>

### 12. Removing element from series
- drop() function can be used to remove specific element(s) from the Series
```python
import pandas
s = pandas.Series([1, 2, 3, 4, 5])
s1 = s.drop(index=[0, 1, 3])
print(s)
print(s1)
```

### 13. Removing row or column from dataframe
- drop() function can be used to remove rows or columns from a dataframe.
```python
import pandas
a = np.array([[4, 4, 5, 0], 
[5, 3, 0, 4], 
[2, 5, 2, 3]])
df = pandas.DataFrame(a, index=['Day 1','Day 2','Day 3'], 
columns=['Item 1', 'Item 2', 'Item 3', 'Item 4'])
df1 = df.drop(index=['Day 2'])
df2 = df.drop(columns=['Item 1', 'Item 3'])
print(df1)
print(df2)
```

### 14. Renaming index of a series/dataframe
- rename() function is to modify the name of column or index (row) of the
  dataFrame or series
- To modify the name of an index in a series or dataframe
- To modify the name of a column in a dataframe
```
<target>.rename(index={'old_name':'new_name'})
<target>.rename(column={'old_name':'new_name'})
```
<note>
Note: rename() function will not change the original series. 
Therefore, remember to assign the return value as a new variable
</note>

### 5. Panel

## Python Syntax Reference Sheet
![image_21.png](image_21.png)

![image_22.png](image_22.png)

![image_23.png](image_23.png)

![image_24.png](image_24.png)

![image_25.png](image_25.png)
