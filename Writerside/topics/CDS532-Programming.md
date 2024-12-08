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

### 1. Simpified Computer Architecture

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

## Lecture 5 - Object

## Lecture 6 - Sequences

## Lecture 7 - Data Collection

## Lecture 8 - Function

## Lecture 9 - Packaging

## Lecture 10 - NumPy

## Lecture 11 - Pandas


## Python Syntax Reference Sheet
![image_21.png](image_21.png)

![image_22.png](image_22.png)

![image_23.png](image_23.png)

![image_24.png](image_24.png)

![image_25.png](image_25.png)
