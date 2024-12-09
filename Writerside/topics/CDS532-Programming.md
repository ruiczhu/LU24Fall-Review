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

### 2. Boolean Expressions
<p style="display: block;">
  <img src="image_48.png" alt="image_48"/>
</p>
<p style="display: block;">
  <img src="image_49.png" alt="image_49"/>
</p>

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

### 1. Functions and Parameters
<p style="display: block;">
  <img src="image_75.png" alt="image_75"/>
</p>

- A function is called by using its name followed by a list of actual parameters 
or arguments.
```
<name>(<actual-parameters>)
```

- When Python comes to a function call, it initiates a **four-step** process.
1. The calling program suspends execution at the point of the call.
2. The formal parameters of the function get assigned the values supplied
by the actual parameters in the call. 
3. The body of the function is executed. 
4. Control returns to the point just after where the function was called

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

## Lecture 9 - Packaging

## Lecture 10 - NumPy

## Lecture 11 - Pandas


## Python Syntax Reference Sheet
![image_21.png](image_21.png)

![image_22.png](image_22.png)

![image_23.png](image_23.png)

![image_24.png](image_24.png)

![image_25.png](image_25.png)
