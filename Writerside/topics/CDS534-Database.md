# CDS534 Database

## 1. Introduction to DBMS
- A DBMS is a computerized system that enables users to create and maintain a database

### 1. Data Abstraction: three-level architecture

<p style="display: block;">
    <img src="image_164.png" alt="image_164"/>
</p>

- Levels of Abstraction
1. Physical/internal level: 
   - internal schema uses a physical data model and
   describes the complete details of data storage and access paths for the
   database. 
2. Conceptual level: 
   - conceptual schema describes the logical structure of the whole
   database for a community of users and hides the details of physical storage
   structures.
   - Entities, data types, relationships, user operations, and constraints
3. View/external level: 
   - external schema describes the part of the database that a
   particular user group is interested in and hides the rest from that group.
   - Describe how users see the data

## 2. ER Model
### 1. ER Model Concepts
- Entity: 
   - a distinguishable object with an independent existence
- Entity Set: 
   - a set of entities of the same type
- Attribute: 
   - a piece of information describing an entity
   - Each attribute can take a value from a domain
   - Types: 
     - simple (Atomic)
       - Each entity has a single atomic value for the attribute. Non-divisible.
     - composite
       - The attribute may be composed of several components.
       - Composition may form a hierarchy where some components are themselves composite
     - multi-valued ((double-lined oval)
       - An entity may have multiple values for that attribute.
     - derived-value
     - Null value
- key attribute: 
   - attribute to uniquely identify an entity in an entity set.
   - Composite key is also allowed (StudentName + PhoneNumber)

### 2. ER Diagram
<p style="display: block;">
    <img src="image_165.png" alt="image_165"/>
</p>
<p style="display: block;">
    <img src="image_166.png" alt="image_166"/>
</p>
<p style="display: block;">
    <img src="image_167.png" alt="image_167"/>
</p>
<p style="display: block;">
    <img src="image_168.png" alt="image_168"/>
</p>
<p style="display: block;">
    <img src="image_169.png" alt="image_169"/>
</p>
<p style="display: block;">
    <img src="image_170.png" alt="image_170"/>
</p>

## 3. ER Model to Relational Model
### 1. ER-to-Relational Mapping Algorithm
1. STEP 1:
<p style="display: block;">
    <img src="image_171.png" alt="image_171"/>
</p>
2. STEP 2:
<p style="display: block;">
    <img src="image_172.png" alt="image_172"/>
</p>
3. STEP 3:
<p style="display: block;">
    <img src="image_173.png" alt="image_173"/>
</p>
4. STEP 4:


## 4. Relational model
### 1. Relational Integrity Constraints
- Domain Constraint:
- Key Constraint:
- Entity Integrity Constraint:
- Referential Integrity Constraint:

- Update Operations on Relations:
- INSERT
- DELETE
- MODIFY(UPDATE)

## 5. SQL: Structured Query Language
### 1. Data Definition Language (DDL)
- CREATE
- DROP
- ALTER

#### Data Type and Domains
- CHAR(n)
- VARCHAR(n)
- INT
- FLOAT
- DATE
- TIME

#### Specifying Attribute Constraints
- NOT NULL
- UNIQUE
- PRIMARY KEY

### 2. Data Manipulation Language (DML)
- INSERT
- DELETE
- UPDATE

### 3. Data Query Language (DQL)

#### 1.The SELECT-FROM-WHERE Structure of Basic SQL Queries
- SELECT
- FROM
- WHERE

#### 2. Substring Pattern Matching
- LIKE
- %: any string of zero or more characters
- _: any single character

#### 3. Aliasing and Renaming
- AS

#### 4. Use of the Asterisk
- SELECT *

#### 5. Ordering Tuples
- ORDER BY
- ASC: ascending
- DESC: descending

#### 6. Use of DISTINCT
- DISTINCT

#### 7. Tables as Sets in SQL
- UNION
- INTERSECT
- EXCEPT
- UNION ALL
- INTERSECT ALL
- EXCEPT ALL

#### 8. Nested Queries
- IN
- SOME/ANY
- ALL
- EXISTS

#### 9. Aggregation
- COUNT
- SUM
- AVG
- MAX
- MIN

#### 10. Group By Clause
- GROUP BY

#### 11. Having Clause
- HAVING

#### 12. JOIN TABLES
- JOIN

### 4. EXPANDED Block Structure of SQL Queries
- A retrieval query in SQL can consist of up to six clauses, but only the first two— SELECT and
  FROM—are mandatory. 
- The other four clauses—WHERE, GROUP BY, HAVING, and ORDER BY—are optional.

<p style="display: block;">
    <img src="image_174.png" alt="image_174"/>
</p>
<p style="display: block;">
    <img src="image_175.png" alt="image_175"/>
</p>

## 6. Relational Algebra
### 1. Unary Relational Operations
- SELECT
- PROJECT
- RENAME
### 2. Binary Relational Operations
- JOIN
- DIVISION
### 3. Relational algebra operations from set theory
- UNION
- INTERSECTION
- DIFFERENCE or MINUS
- CARTESIAN PRODUCT
### 4. Additional Relational Operations
- Aggregate functions

### 5. Relational Algebra Operations
<p style="display: block;">
    <img src="image_176.png" alt="image_176"/>
</p>
<p style="display: block;">
    <img src="image_177.png" alt="image_177"/>
</p>


## 7. FD, MVD, Normalization
### 1. Functional Dependency
- Inference Rules for FDs: the set of all dependencies that include F as well as all dependencies that can be
  inferred from F is called the closure of F; it is denoted by F+
- Closure of a set of attributes X with respect to F is the set X+ of all attributes that are
  functionally determined by X
- If X+ consists of all attributes of R, X is a superkey for R
- X+ can be calculated by repeatedly applying IR1, IR2, IR3 using the FDs in F

| Inference Rule | Description           |
|----------------|-----------------------|
| IR1            | Reflexive Rule        |
| IR2            | Augmentation Rule     |
| IR3            | Transitive Rule       |
| IR4            | Decomposition Rule    |
| IR5            | Union Rule            |
| IR6            | Pseudotransitive Rule |

#### Canonical Cover of F
- A canonical cover of F is a set of dependencies that is equivalent to F and satisfies the following
  properties:
  - No dependency in F+ has a redundant attribute on the right side
  - No dependency in F+ has a redundant attribute on the left side
  - No dependency in F+ has both a redundant attribute on the right side and a redundant attribute on the left side

### 2. Multivalued Dependency

### 3. Normalization
1. First Normal Form (1NF)
   - A relation is in 1NF if the domain of each attribute contains only atomic values, and the value of each attribute contains only a single value from that domain.
2. Second Normal Form (2NF)
   - A relation is in 2NF if it is in 1NF and every non-key attribute is fully functionally dependent on the primary key.
3. Third Normal Form (3NF)
   - A relation is in 3NF if it is in 2NF and every non-key attribute is non-transitively dependent on the primary key.
4. Boyce-Codd Normal Form (BCNF)
   - A relation is in BCNF if it is in 3NF and for every non-trivial functional dependency X → Y, X is a superkey.
5. Fourth Normal Form (4NF)
6. Fifth Normal Form (5NF)

- Test and Remedy for Normalization
<p style="display: block;">
    <img src="image_178.png" alt="image_178"/>
</p>

## 8. Transactions, Concurrency Control, Recovery
### 1. Transaction
- A transaction = database operations + transaction operations
- Transaction Operations:
  - Begin Transaction
  - Commit
  - Rollback

### 2. ACID Properties
- Atomicity: 
  - A transaction is an atomic unit of processing. It is either performed
    completely or not performed at all (all or nothing)
- Consistency:
  - A correct execution of a transaction must take the database from one
    consistent state to another (correctness)
- Isolation: 
  - A transaction should not make its updates visible to other transactions
    until it is committed (no partial results)
- Durability:
  - Once a transaction changes the database state and the changes are
    committed, these changes must never be lost because of subsequent failure
    (committed and permanent results)

### 3. Read and write operation conflict rules
<p style="display: block;">
    <img src="image_179.png" alt="image_179"/>
</p>

### 4. Serialization Graphs
- The determination of a conflict serializable schedule can be done according to
  serialization graph (SG) or called precedence graph
- A serialization graph tells the effective execution order of a set of transactions
- The set of edges consists of all edges TiàTj for which one of the following three
  conditions holds:
  - W/R conflict: Ti executes write(x) before Tj executes read(x)
  - R/W conflict: Ti executes read(x) before Tj executes write(x)
  - W/W conflict: Ti executes write(x) before Tj executes write(x)
- Serializability theorem:
  - A schedule is serializable iff the SG is acyclic (not a circle)

### 4. Database Concurrency Control
- To preserve database consistency
  - To ensure all schedules are serializable and recoverable
- To maximize the system performance (higher concurrency)

#### Implementation of Multiple-mode Lock
- read lock operation
- write lock operation
- unlock operation

#### 1. Basic Two-Phase Locking(B2PL)

#### 2. Conservative Two-Phase Locking (C2PL)

#### 3. Strict Two-Phase Locking (S2PL)

#### 4. Comparison of B2PL, C2PL, S2PL
<p style="display: block;">
    <img src="image_180.png" alt="image_180"/>
</p>

### 5. Deadlock
- Deadlock is a situation in which two or more transactions are waiting for each other to release locks

#### 1. Deadlock Detection
- Wait-for graph
- We have a state of deadlock if and only if the wait-for graph has a cycle.
<p style="display: block;">
    <img src="image_181.png" alt="image_181"/>
</p>

#### 2. Deadlock Prevention using TS
1. Wait-Die Rule
2. Wound-Wait Rule