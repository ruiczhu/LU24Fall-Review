# CDS535 DataMining
- The exam has nine Questions
- Data contains value and knowledge

## 1. Data Mining Introduction

### 1. Definition of Data Mining
- Non-trivial extraction of implicit, previously unknown and potentially useful information from data
- Exploration & analysis, by automatic or semi-automatic means, of large quantities of data to discover meaningful patterns
- ![image.png](image.png)

### 2. Data Science compared to Data Mining
**Business Understanding**
- Seldom pre-packaged as clear data science problems.
- Creative problem formulation is needed to consider the business problem as one or more data science problems.
- High-level knowledge of the domains helps creative Data
- Scientists/Business Analysts see novel formulations.
- The design team should think carefully about the problem to be solved and about the user scenario.

**Data Understanding**
- Understand the strengths and limitations of the data.
- Historical data often are collected for other purposes unrelated to the current data science problem.

**Data Preparation**
- Many data science/data mining algorithms require data to be in a specific form (e.g., Table format).
- Some conversion will be necessary.
- Missing values should be handled.
- 
**Modeling**
- Apply data science/data mining algorithms to create models/data science results from the data

**Evaluation**
- Assess the data science results.
- Gain confidence that they are correct and reliable.
- Test the results (models) in a controlled laboratory setting.
- Ensure that the model satisfies the original business goals.

**Deployment**
- The data science results (models) are used in real problems (e.g., fake detections).
- Implement it in some information system or business process.
- Deploying a model into a production system typically requires that the model be re-coded for the production environment (e.g., program in Windows environment).

![image_2.png](image_2.png)

### 3. Data Mining Process
![image_3.png](image_3.png)

### 4. Data Mining Tasks
**Prediction Methods**
- Use some variables to predict unknown or future values of other variables.

**Description Methods**
- Find human-interpretable patterns that describe the data.

二、过程与步骤
- Prediction Methods（预测方法）
- 数据收集：收集与预测目标相关的历史数据。 
- 数据预处理：清洗数据、处理缺失值、标准化等。 
- 特征选择：选择对预测目标有影响的特征。 
- 模型训练：使用训练数据集训练预测模型。 
- 模型验证：使用验证数据集验证模型的准确性。 
- 预测：使用训练好的模型对未知数据进行预测。


- Description Methods（描述方法） 
- 数据收集：收集与目标问题相关的数据。 
- 数据探索：通过可视化、统计等方法初步了解数据。 
- 模式发现：运用数据挖掘技术发现数据中的模式、关联和异常等。 
- 模式解释：以人类可理解的方式解释发现的模式。 
- 结果呈现：通过图表、报告等形式呈现描述结果。

三、应用实例
- Prediction Methods（预测方法） 
- 金融领域：预测股票价格、汇率、信用风险等。 
- 销售领域：预测销售额、市场份额、客户行为等。 
- 医疗领域：预测疾病发生率、治疗效果、患者预后等。

- Description Methods（描述方法） 
- 市场研究：描述消费者行为、市场趋势等。 
- 社会科学：描述社会现象、人类行为等。 
- 生物信息学：描述基因表达、蛋白质结构等。

### Data Mining Tasks
- Classification [Predictive]
- Clustering [Descriptive]
- Association Rule Discovery [Descriptive]  
- Sequential Pattern Discovery [Descriptive]  
- Regression [Predictive]
- Deviation Detection [Predictive]

![image_4.png](image_4.png)

## 2. Data
- **Data:** Collection of data objects and their attributes
- **Attribute Values:** Attribute values are numbers or symbols assigned
  to an attribute
    - Distinction between attributes and attribute values
        - Same attribute can be mapped to different attribute
          values
        - Different attributes can be mapped to the same set of
          values

### 1. Types of Attributes

| Property       | Symbol | Nominal | Ordinal | Interval | Ratio |
|----------------|--------|---------|---------|----------|-------|
| Distinctness   | = ≠    | ✅       | ✅       | ✅        | ✅     |
| Order          | < >    |         | ✅       | ✅        | ✅️    |
| Addition       | + -    |         |         | ✅        | ✅️    |
| Multiplication | * /    |         |         |          | ✅     |

| Attribute Type  | Description                                                                | Examples                                                                       | Operations                                                     | Transformation                                                                                      | Comments                                                                                                                               |
|-----------------|----------------------------------------------------------------------------|--------------------------------------------------------------------------------|----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| **Categorical** |                                                                            |                                                                                |                                                                |                                                                                                     |                                                                                                                                        |
| Nominal         | Nominal attribute values only distinguish. (=, ≠)                          | zip codes, employee ID numbers, eye color, sex: {male, female}                 | mode, entropy, contingency correlation, χ² test                | Any permutation of values                                                                           | If all employee ID numbers were reassigned, would it make any difference?                                                              |
| Ordinal         | Ordinal attribute values also order objects. (<, >)                        | hardness of minerals, {good, better, best}, grades, street numbers             | median, percentiles, rank correlation, run tests, sign tests   | An order preserving change of values, i.e.,new_value = f(old_value) where f is a monotonic function | An attribute encompassing the notion of good, better, best can be represented equally well by the values {1, 2, 3} or by {0.5, 1, 10}. |
| **Numeric**     |                                                                            |                                                                                |                                                                |                                                                                                     |                                                                                                                                        |
| Interval        | For interval attributes, differences between values are meaningful. (+, -) | calendar dates, temperature in Celsius or Fahrenheit                           | mean, standard deviation, Pearson's correlation, t and F tests | new_value = a * old_value + b where a and b are constants                                           | Thus, the Fahrenheit and Celsius temperature scales differ in terms of where their zero value is and the size of a unit (degree).      |
| Ratio           | For ratio variables, both differences and ratios are meaningful. (*, /)    | temperature in Kelvin, monetary quantities, counts, age, mass, length, current | geometric mean, harmonic mean, percent variation               | new_value = a * old_value                                                                           | Length can be measured in meters or feet.                                                                                              |

### 2. Discrete and Continuous Attributes

#### Discrete Attribute 离散属性
- Has only a finite or countably infinite set of values
- Examples: zip codes, counts, or the set of words in a collection of documents
- Often represented as integer variables
- Note: binary attributes are a special case of discrete attributes

#### Continuous Attribute 连续属性
- Has real numbers as attribute values
- Examples: temperature, height, or weight
- Practically, real values can only be measured and represented using a finite number of digits
- Continuous attributes are typically represented as floating-point variables

### 3. Asymmetric Attributes
- Only presence (a non-zero attribute value) is regarded as
  important.

### 4. Critiques of the Attribute Categorization

#### Incomplete
- Asymmetric binary
- Cyclical (e.g., position on the surface of the Earth, Time)
- Multivariate (e.g., set of movies seen)
- Partially ordered
- Partial membership
- Relationships between the data

#### Real Data is Approximate and Noisy
- This can complicate recognition of the proper attribute type.
- Treating one attribute type as another may be approximately correct.

### 5. Key Messages for Attribute Types

#### Meaningful Operations
- The types of operations you choose should be "meaningful" for the type of data you have.
    - Distinctness, order, meaningful intervals, and meaningful ratios are only four (among many possible) properties of data.
    - The data type you see—often numbers or strings—may not capture all the properties or may suggest properties that are not present.
    - Analysis may depend on these other properties of the data.
        - Many statistical analyses depend only on the distribution.
    - In the end, what is meaningful can be specific to the domain.

### 6. Important Characteristics of Data

- **Dimensionality (number of attributes)**
    - High dimensional data brings a number of challenges

- **Sparsity**
    - Only presence counts

- **Resolution**
    - Patterns depend on the scale

- **Size**
    - Type of analysis may depend on size of data

### 7. Types of Data Sets

<tip>详情见 Data_Mining_week2_slides.pdf</tip>

#### Record
- **Data Matrix**
- **Document Data**
- **Transaction Data**

#### Graph
- **World Wide Web**
- **Molecular Structures**

#### Ordered
- **Spatial Data**
- **Temporal Data**
- **Sequential Data**
- **Genetic Sequence Data**

### 8. Data Quality
- Poor data quality negatively affects many data processing
  efforts
- Quality issues include:
    - Noise
        - For objects, noise is an extraneous (無關) object
        - For attributes, noise refers to modification of original values
    - Outliers
        - Outliers are data objects with characteristics that
          are considerably different from most of the other
          data objects in the data set
    - Wrong data
    - Fake data
    - Missing values
        - Reasons for missing values
            - Information is not collected
            - Attributes may not be applicable to all cases
        -  Handling missing values
        - Eliminate data objects or variables
        - Estimate missing values
        - Ignore the missing value during analysis
    - Duplicate data
        - Data set may include data objects that are
          duplicates, or almost duplicates of one another

### 9. Data Preprocessing
#### 1. Aggregation
- Combining two or more attributes (or objects) into a single
  attribute (or object)
- Purpose
    - Data reduction - reduce the number of attributes or objects
    - Change of scale
        - Cities aggregated into regions, states, countries, etc.
        - Days aggregated into weeks, months, or years
    - More “stable” data - aggregated data tends to have less variability
#### 2. Sampling
- Sampling is the main technique used for data
  reduction.
    - It is often used for both the preliminary investigation of
      the data and the final data analysis.
- Statisticians often sample because getting the
  entire set of data of interest is too expensive or
  time consuming.
- Sampling is typically used in data mining because
  processing the entire set of data of interest is too
  expensive or time consuming.
- The **key principle** for effective sampling is the
  following:
    - Using a sample will work almost as well as using the
      entire data set, if the sample is representative
    - A sample is representative if it has approximately the
      same properties (of interest) as the original set of data

    - **Simple Random Sampling**
    - There is an equal probability of selecting any particular item.

        - **Sampling without replacement**
        - As each item is selected, it is removed from the population.

        - **Sampling with replacement**
        - Objects are not removed from the population as they are selected for the sample.
        - In sampling with replacement, the same object can be picked up more than once.

    - **Stratified sampling**
    - Split the data into several partitions; then draw random samples from each partition.


#### 3. Discretization
- Discretization is the process of converting a
  continuous attribute into an ordinal attribute
    - A potentially infinite number of values are mapped into
      a small number of categories
    - Discretization is used in both unsupervised and
      supervised settings
- Unsupervised Discretization
    - Equal interval width (distance) partitioning
    - Equal-frequency (frequency) partitioning
    - K-means
- Supervised Discretization
    - Discretization is based on class labels
    - The goal is to find partitions that minimize the
      entropy (or maximize the purity) of the classes in
      each partition

#### 4. Binarization
- Binarization maps a continuous or categorical
  attribute into one or more binary variables

#### 5. Attribute Transformation
- An attribute transform is a function that maps the
  entire set of values of a given attribute to a new
  set of replacement values such that each old
  value can be identified with one of the new values

- Simple Functions

    - Power Function (幂函数):
      $x^k$
    - Logarithmic Function (对数函数):
      $log(x)$
    - Exponential Function (指数函数):
      $ e^x $
    - Absolute Value Function (绝对值函数):
      $|x|$

- Normalization
    - Refers to various techniques to adjust to
      differences among attributes in terms of frequency
      of occurrence, mean, variance, range
    - Take out unwanted, common signal, e.g.,
      seasonality
- In statistics, standardization refers to subtracting off
  the means and dividing by the standard deviation

#### 5. Dimensionality Reduction

- Curse of Dimensionality
    - Many types of data analysis become significantly harder as
      the dimensionality of the data increases
    - When dimensionality increases, data becomes increasingly
      sparse in the space that it occupies
    - Definitions of density and distance between points, which
      are critical for clustering and outlier detection, become less
      meaningful

- Purpose of Dimensionality Reduction:
    - Avoid curse of dimensionality
    - Reduce amount of time and memory required by data
      mining algorithms
    - Allow data to be more easily visualized
    - May help to eliminate irrelevant features or reduce
      noise
- Techniques
    - Principal Components Analysis (PCA)
    - Singular Value Decomposition
    - Others: supervised and non-linear techniques

#### 6. Feature subset selection
- Another way to reduce dimensionality of data
- Redundant features
    - Duplicate much or all of the information contained in
      one or more other attributes
    - Example: purchase price of a product and the amount
      of sales tax paid
- Irrelevant features
    - Contain no information that is useful for the data
      mining task at hand
    - Example: students' ID is often irrelevant to the task of
      predicting students' GPA
- Many techniques developed, especially for
  classification

#### 7. Feature creation
- Create new attributes that can capture the
  important information in a data set much more
  efficiently than the original attributes
- Three general methodologies:
    - Feature extraction
    - Feature construction
    - Mapping data to new space

## 3. Decision Tree

### 1. Decision Tree Algorithms
1. Hunt’s Algorithm (one of the earliest)
    - General Structure of Hunt’s Algorithm
    - Let Dt be the set of training
      records that reach a node t
        - General Procedure:
            - If Dt contains records that
              belong the same class yt,
              then t is a leaf node
              labeled as yt
            - If Dt contains records that
              belong to more than one
              class, use an attribute test
              to split the data into smaller
              subsets. Recursively apply
              the procedure to each
              subset.
2. CART
3. ID3, C4.5
4. SLIQ, SPRINT

<p style="display: block;">
  <img src="image_119.png" alt="image_119"/>
</p>

### 2. Design Issues of Decision Tree Induction

1. Greedy strategy
    - Split the records based on an attribute test that
      optimizes certain criterion
2. How should training records be split?
    - Method for expressing test condition
        - depending on attribute types
    - Measure for evaluating the goodness of a test
      condition
3. How should the splitting procedure stop?
    - Stop splitting if all the records belong to the same
      class or have identical attribute values
    - Early termination

### 3. Methods for Expressing Test Conditions

1. Binary
2. Nominal
    - Multi-way split:
        - Use as many partitions as distinct values.

    - Binary split:
        - Divides values into two subsets

    - ![image_6.png](image_6.png)

3. Ordinal
    - Multi-way split:
        - Use as many partitions as distinct values

    - Binary split:
        - Divides values into two subsets
        - Preserve order property among attribute values

    - ![image_7.png](image_7.png)

4. Continuous
    - ![image_8.png](image_8.png)
    - Splitting Based on Continuous Attributes
        - Discretization to form an ordinal categorical
          attribute
        - Ranges can be found by equal interval bucketing,
          equal frequency bucketing (percentiles), or
          clustering.
            - Static – discretize once at the beginning
            - Dynamic – repeat at each node
    - Binary Decision: (A < v) or (A >= v)
        - consider all possible splits and finds the best cut
        - can be more compute intensive

### 4. How to determine the Best Split

- Greedy approach:
    - Nodes with purer class distribution are preferred

### 5. Measures of Node Impurity
<tip>应该重要吧</tip>

- $p_i(t)$ 是节点 $t$ 上类 $i$ 的频率， $c$ 是类的总数。

#### 1. Gini Index

$$
Gini \ Index = 1 - \sum_{i=0}^{c-1} p_i(t)^2
$$

For 2-class problem (p, 1 – p):

$$
GINI = 1 – p2 – (1 – p)2 = 2p (1-p)
$$

- When a node $p$ is split into $k$ partitions (children)

$$
GINI_{split} = \sum_{i=1}^{k} \frac{n_i}{n} GINI(i)
$$
where,
$$
n_i = \text{number of records at child } i
$$
$$
n = \text{number of records at parent node } p
$$

- Binary Attributes: Computing GINI Index
<p style="display: block;">
  <img src="image_120.png" alt="image_120"/>
</p>
<p style="display: block;">
  <img src="image_122.png" alt="image_122"/>
</p>

- Categorical Attributes: Computing Gini Index
<p style="display: block;">
  <img src="image_121.png" alt="image_121"/>
</p>

- Continuous Attributes: Computing Gini Index
<p style="display: block;">
  <img src="image_123.png" alt="image_123"/>
</p>

<p style="display: block;">
  <img src="image_124.png" alt="image_124"/>
</p>

<tip>Gini反映的是不纯度（impurity），所以越低越好</tip>
<tip>Gain算的是增益，高的好</tip>

#### 2. Entropy (熵)

$$
Entropy = -\sum_{i=0}^{c-1} p_i(t) \log_2 p_i(t)
$$

- Computing Entropy of a Single Node
<p style="display: block;">
  <img src="image_125.png" alt="image_125"/>
</p>

- Computing Information Gain After Splitting

    - $$
      Gain_{split} = Entropy(p) - \sum_{i=1}^{k} \frac{n_i}{n} Entropy(i)
      $$

    - Parent Node, $p$ is split into $k$ partitions (children)  
      $n_i$ is the number of records in child node $i$

    - Choose the split that achieves the most reduction (maximizes GAIN)
    - Used in the ID3 and C4.5 decision tree algorithms
    - Information gain is the mutual information between the class variable and the splitting variable

#### 3. Problem with large number of partitions
- **Node impurity measures tend to prefer splits that
  result in large number of partitions, each being
  small but pure**
- Customer ID has highest information gain
  because entropy for all the children is zero

#### 4. Gain Ratio

- Gain Ratio:
  $$
  \text{Gain Ratio} = \frac{\text{Gain}_{\text{split}}}{\text{Split Info}}
  $$

- Split Info Formula:
  $$
  \text{Split Info} = - \sum_{i=1}^{k} \frac{n_i}{n} \log_2 \frac{n_i}{n}
  $$

- Definitions:
- **Parent Node**, $p$ is split into $k$ partitions (children)
- $n_i$ is the number of records in child node $i$

- Key Points:
- Adjusts Information Gain by the entropy of the partitioning (\( \text{Split Info} \)).
    - Higher entropy partitioning (large number of small partitions) is penalized!
- Used in C4.5 algorithm
- Designed to overcome the disadvantage of Information Gain

#### 5. Misclassification Error (错误分类率)

$$
Classification \ error = 1 - \max[p_i(t)]
$$

- Computing Error of a Single Node
<p style="display: block;">
  <img src="image_126.png" alt="image_126"/>
</p>

### 6. Finding the Best Split
1. Compute impurity measure (P) before splitting
2. Compute impurity measure (M) after splitting
    - Compute impurity measure of each child node
    - M is the weighted impurity of child nodes
3. Choose the attribute test condition that produces the highest gain

$$ Gain = P - M $$

or equivalently, lowest impurity (highest purity) measure after splitting (M)

### 7. Comparison among Impurity Measure
<p style="display: block;">
  <img src="image_127.png" alt="image_127"/>
</p>

### 8. Decision Tree Based Classification
- **Advantages:**
    - Relatively inexpensive to construct
    - Extremely fast at classifying unknown records
    - Easy to interpret for small-sized trees
    - Robust to noise (especially when methods to avoid overfitting are used)
    - Can easily handle redundant attributes
    - Can easily handle irrelevant attributes (unless the attributes are interacting)

- **Disadvantages:**
    - Due to the greedy nature of splitting criterion, interacting attributes (that can distinguish between classes together but not individually) may be passed over in favor of other attributes that are less discriminating.
    - Each decision boundary involves only a single attribute

#### 1. Data Fragmentation
- Number of instances gets smaller as you traverse
  down the tree
- Number of instances at the leaf nodes could be
  too small to make any statistically significant
  decision

#### 2. Search Strategy
- Finding an optimal decision tree is NP-hard
- The algorithm presented so far uses a greedy,
  top-down, recursive partitioning strategy to
  induce a reasonable solution
- Other strategies:
    - Bottom-up
    - Bi-directional

#### 3. Expressiveness

- Decision trees provide expressive representation for learning discrete-valued functions.

- **Do not generalize well to certain types of Boolean functions.**
    - **Example: Parity Function**
        - Class = 1 if there is an even number of Boolean attributes with truth value = True.
        - Class = 0 if there is an odd number of Boolean attributes with truth value = True.
    - For accurate modeling, must have a complete tree.

- **Not expressive enough for modeling continuous variables.**
    - Particularly when test condition involves only a single attribute at-a-time.

##### Decision Boundary
- Border line between two neighboring regions of different classes is
  known as decision boundary
- Decision boundary is parallel to axes because test condition involves
  a single attribute at-a-time
<p style="display: block;">
  <img src="image_128.png" alt="image_128"/>
</p>

##### Oblique Decision Trees
<p style="display: block;">
  <img src="image_129.png" alt="image_129"/>
</p>

#### 4. Tree Replication
<p style="display: block;">
  <img src="image_130.png" alt="image_130"/>
</p>

### 9. Practical Issues of Classification

- **Classification Errors**
1. Training errors: Errors committed on the training set
2. Test errors: Errors committed on the test set
3. Generalization errors: Expected error of a model over random selection of records from same distribution

#### 1. Underfitting and Overfitting
- Underfitting: when model is too simple, both training and test errors are large
- Overfitting: when model is too complex, training error is small but test error is large
    - Increasing the size of training data reduces the difference between training and
      testing errors at a given size of model
    - Reasons for Model Overfitting
        - Noisy data
        - Not enough training data
        - High model complexity
<note>
Overfitting results in decision trees that are more
complex than necessary 
</note>
<note>
Training error does not provide a good estimate
of how well the tree will perform on previously
unseen records
</note>
<note>
Need ways for estimating generalization errors
</note>

#### 2. Missing Values

- Missing values affect decision tree construction in
  three different ways:
    - Affects how impurity measures are computed
    - Affects how to distribute instance with missing
      value to child nodes
    - Affects how a test instance with missing value
      is classified

<p style="display: block;">
  <img src="image_132.png" alt="image_132"/>
</p>

#### 3.  Model Evaluation/Costs of Classification

[](#6-model-evaluation)

### 10. Model selection
- Performed during model building
- Purpose is to ensure that model is not overly
  complex (to avoid overfitting)
- Need to estimate generalization error
    - Using Validation Set
    - Divide training data into two parts:
        - Training set:
            - use for model building
        - Validation set:
            - use for estimating generalization error
            - Note: validation set is not the same as test set
        - Drawback:
            - Less data available for training
    - Incorporating Model Complexity
        - Rationale: Occam’s Razor
            - Given two models of similar generalization errors,
              one should prefer the simpler model over the more
              complex model
            - A complex model has a greater chance of being fitted
              accidentally
            - Therefore, one should include model complexity when
              evaluating a model
            - Generalization Error (泛化误差)
            - $$
              \text{Gen. Error(Model)} = \text{Train. Error(Model, Train. Data)} + \alpha \times \text{Complexity(Model)}
              $$

#### 1. Estimating the Complexity of Decision Trees
##### Pessimistic Error Estimate of Decision Tree T with k Leaf Nodes

Pessimistic: it assumes that the generalization error will be larger than the training error, so it is necessary to add the penalty term

$$
err_{gen}(T) = err(T) + \Omega \times \frac{k}{N_{train}}
$$

- **$err(T)$**: error rate on all training records
- **$\Omega$**: trade-off hyper-parameter (similar to \alpha)
    - Relative cost of adding a leaf node
- **$k$**: number of leaf nodes
- **$N_{train}$**: total number of training records

Example:
<p style="display: block;">
  <img src="image_131.png" alt="image_131"/>
</p>

- Resubstitution Estimate:
- Using training error as an optimistic estimate of
  generalization error
- Referred to as optimistic error estimate

#### 2. Pre-Pruning (Early Stopping Rule)

- **Stop the algorithm before it becomes a fully-grown tree**
- **Typical stopping conditions for a node:**
    - Stop if all instances belong to the same class
    - Stop if all the attribute values are the same
- **More restrictive conditions:**
    - Stop if number of instances is less than some user-specified threshold
    - Stop if class distribution of instances are independent of the available features (e.g., using $\chi^2$ test)
    - Stop if expanding the current node does not improve impurity measures (e.g., Gini or information gain)
    - Stop if estimated generalization error falls below certain threshold

#### 3. Post-Pruning

- **Grow decision tree to its entirety**
- **Subtree replacement**
    - Trim the nodes of the decision tree in a bottom-up fashion
    - If generalization error improves after trimming, replace sub-tree by a leaf node
    - Class label of leaf node is determined from majority class of instances in the sub-tree


## 4. Clustering
- Finding groups of objects such that the objects in a
group will be similar (or related) to one another and
different from (or unrelated to) the objects in other
groups

- What is not Cluster Analysis?
  1. Supervised classification
  2. Simple segmentation
  3. Results of a query
  4. Graph partitioning
  <tip>聚类分析是一种无监督学习方法，它基于对象的固有相似性或距离来进行分组，而不使用带有标签的数据或外部规范。</tip>

### 1. 划分聚类（Partitional Clustering）
- A division of data objects into non-overlapping subsets (clusters)
- 划分聚类是将数据对象分成非重叠的子集（即聚类），每个数据点只能属于一个聚类。这种方法通常需要预先指定聚类的数量K，然后算法会尝试找到K个聚类，使得每个数据点都被分配到一个聚类中，且每个聚类中的数据点尽可能相似。
划分聚类的典型算法包括K-means和K-中心点算法等。这些算法通过迭代的方式不断优化聚类的结果，直到达到某个终止条件（如聚类中心不再发生变化或达到最大迭代次数）。

<p style="display: block;">
  <img src="image_152.png" alt="image_152"/>
</p>
<note>在划分聚类中，最常用的算法之一是K-means算法。</note>

#### 1. K-means Clustering
- Partitional clustering approach
- Number of clusters, K, must be specified
- Each cluster is associated with a centroid (center point)
- Each point is assigned to the cluster with the closest
  centroid
- The basic algorithm is basic

<p style="display: block;">
    <img src="image_5.png" alt="image_5"/>
</p>

- Steps

1.  选择初始质心
    - 这一步通常随机从数据集中选择K个点作为初始质心。虽然随机选择简单易行，但它可能导致算法收敛到不同的局部最优解，因为每次运行算法时，初始质心的选择都可能不同。
2.  迭代过程
    1. **分配数据点到最近质心**：对于数据集中的每个点，计算其与所有质心的距离（如欧几里得距离），并将其分配给最近的质心，从而形成一个聚类。
    2. **重新计算质心**：对于每个聚类，计算其所有点的平均值（或其他中心趋势的度量），并将这个新的平均值作为该聚类的质心。
    3. **重复上述两个步骤**，直到质心的位置不再发生显著变化（即算法收敛），或者达到预设的迭代次数。

3.  收敛条件
    - 常见的收敛条件是质心停止变化，但实际应用中，也可以采用其他条件，如“直到相对较少的数据点改变其所属聚类”。这是因为，在算法的后期阶段，随着质心的逐渐稳定，很少有数据点会改变其聚类归属。

4.  算法复杂度
    - K-means聚类算法的复杂度为 O(n * K * I * d)，其中 n 是数据点的数量，K 是聚类的数量，I 是迭代次数，d 是数据点的属性数量。这个复杂度表明，算法的计算成本随着数据集的大小、聚类数量和迭代次数的增加而增加。

5.  质心的定义
    - 在 K-means 聚类中，质心通常是聚类中所有点的平均值。然而，也可以采用其他定义，如中位数或加权平均值，以适应不同的数据集和聚类需求。

6.  收敛性
    - 对于常见的距离度量（如欧几里得距离）和适当定义的质心，K-means算法将收敛。然而，由于初始质心的选择是随机的，因此算法可能收敛到不同的局部最优解。

7.  迭代次数
    - 在大多数情况下，算法的大部分收敛发生在前几次迭代中。随着迭代的进行，质心的变化逐渐减小，直到达到收敛条件。


- Simple iterative algorithm.
  - Choose initial centroids;
  - repeat {assign each point to a nearest centroid; re-compute cluster centroids}
  - until centroids stop changing.
- Initial centroids are often chosen randomly.
  -  Clusters produced can vary from one run to another
- The centroid is (typically) the mean of the points in the cluster,
  but other definitions are possible.
- K-means will converge for common proximity measures (e.g., Euclidean Distance, p.62) with appropriately defined centroid
- Most of the convergence happens in the first few iterations.
  - Often the stopping condition is changed to ‘Until relatively few points change clusters’
- Complexity is O( n * K * I * d)
  - n = number of points
  - K = number of clusters
  - I = number of iterations
  - d = number of attributes

- K-means Objective Function
  - A common objective function (used with Euclidean distance measure) is Sum of Squared Error (SSE)
    - For each point, the error is the distance to the nearest cluster center
    - To get SSE, we square these errors and sum them.
    $$
    \text{SSE} = \sum_{i=1}^{K} \sum_{x \in C_i} \text{dist}^2(m_i, x)
    $$
    - x is a data point in cluster Ci and mi is the centroid (mean) for
cluster Ci
    - SSE improves in each iteration of K-means until it reaches a local or global minima.
    
- 误差平方和（SSE）
  - 在K-means聚类算法中，SSE用于量化数据点到其所属聚类中心的距离之和。对于每个数据点，其误差定义为该点到其最近聚类中心的距离。为了得到一个非负的、易于处理的误差度量，我们通常将这些距离平方后求和
  - SSE在K-means算法中的应用
    1. 初始化：
       - 随机选择K个点作为初始质心。
       - 计算每个数据点到这些质心的距离，并将其分配给最近的质心，形成初始聚类。
    2. 迭代过程：
       - 对于每个聚类，重新计算质心（即该聚类中所有点的平均值）。
       - 重新分配每个数据点到最近的质心。
       - 计算新的SSE值。
    3. 收敛条件：
       - 重复上述迭代过程，直到SSE不再显著减少（达到局部或全局最小值），或者达到预设的迭代次数。
  - SSE的性质
    - SSE是一个非负值，它表示数据点到其所属聚类中心的总体距离。
    - 在每次迭代中，K-means算法都会尝试通过重新分配数据点和重新计算质心来减小SSE。
    - 由于初始质心的选择是随机的，K-means算法可能会收敛到不同的局部最小值。因此，SSE的值可能会因运行而异。
    - SSE的减小速度通常在前几次迭代中较快，随着迭代的进行，质心的变化逐渐减小，SSE的减小速度也逐渐放缓。

#### 2. Two different K-means Clusterings
1. **Optimal Clustering（最优聚类）**
   - 最优聚类是指K-means算法在给定数据集上能够找到的最佳聚类方案。这个方案通常满足以下条件：
   1. 聚类内数据点尽可能相似：在同一聚类内的数据点之间的距离应该尽可能小，这表示数据点之间的相似性较高。
   2. 聚类间数据点尽可能不同：不同聚类之间的数据点距离应该尽可能大，这表示聚类之间的差异性较高。
   3. SSE最小：误差平方和（SSE）是衡量聚类质量的一个重要指标。在最优聚类中，SSE应该达到最小值，表示数据点到其所属聚类中心的总体距离最小。
   - 然而，由于K-means算法对初始质心的选择敏感，且可能陷入局部最优解，因此在实际应用中很难保证总是找到全局最优的聚类方案。

2. **Sub-optimal Clustering（次优聚类）**
   -  次优聚类是指K-means算法在给定数据集上找到的聚类方案，但并非最佳方案。次优聚类可能由于以下原因产生：
   1. 初始质心选择不当：如果初始质心的选择不够合理，K-means算法可能会收敛到一个局部最优解，而不是全局最优解。
   2. 迭代次数不足：如果迭代次数设置得不够多，K-means算法可能还没有足够的时间来找到最优的聚类方案就已经停止了迭代。
   3. 数据集特性：某些数据集可能具有复杂的结构或噪声，这使得K-means算法难以找到最优的聚类方案。
   - 在次优聚类中，SSE的值可能会比最优聚类中的SSE值大，表示数据点到其所属聚类中心的总体距离较大，聚类效果较差。

3. 如何评估聚类质量
   - 为了评估K-means聚类的质量，除了SSE之外，还可以考虑以下指标：
   1. 轮廓系数（Silhouette Coefficient）：用于衡量数据点与其所属聚类之间的紧密程度，以及与其他聚类之间的分离程度。轮廓系数的值范围在-1到1之间，值越大表示聚类效果越好。
   2. Calinski-Harabasz指数：也称为方差比率准则，用于评估聚类结果的紧密性和分离性。该指数的值越大，表示聚类效果越好。

#### 3. Importance of Choosing Initial Centroids

1. 影响聚类结果
   - 初始质心的选择会直接影响K-means算法的聚类结果。不同的初始质心可能导致算法收敛到不同的局部最优解，从而产生不同的聚类结果。因此，合理的初始质心选择有助于算法更快地找到全局最优解或较好的局部最优解，提高聚类结果的稳定性和准确性。
2. 影响收敛速度
   - 初始质心的选择还会影响K-means算法的收敛速度。如果初始质心选择得当，算法可以在较少的迭代次数内收敛到稳定的聚类结果。相反，如果初始质心选择不当，算法可能需要更多的迭代次数才能收敛，甚至可能陷入无效的迭代循环中，导致计算效率低下。
3. 对噪声和异常值的敏感性
   - K-means算法对噪声和异常值较为敏感。如果初始质心选择不当，算法可能会将噪声或异常值作为质心，从而导致聚类结果偏离实际数据分布。因此，合理的初始质心选择有助于降低算法对噪声和异常值的敏感性，提高聚类结果的鲁棒性。

#### 4. Problems with Selecting Initial Points

1. 低概率选择到每个真实聚类的中心：
   1. 当数据集存在K个“真实”聚类时，随机选择初始点的策略很难确保每个聚类都能被选中一个初始点。特别是当K值较大时，这种概率会显著降低。
   2. 例如，如果每个聚类的大小都是n，且聚类之间不重叠，那么随机选择K个初始点，每个点都来自不同聚类的概率是K个点的组合数除以总数据点的组合数，即K!/(nK)!/[(n(K-1))+K-K!]（这里简化了计算，但核心意思是概率很低）。当K=10时，即使每个聚类有相同的数量n的数据点，这个概率也非常低。
      $$
      P = \frac{\text{number of ways to select one centroid from each cluster}}{\text{number of ways to select } K \text{ centroids}} = \frac{K!n^K}{(Kn)^K} = \frac{K!}{K^K}
      $$
        <p style="display: block;">
         <img src="image_154.png" alt="image_154"/>
        </p>
2. 初始点可能不稳定：
   1. 初始点的选择具有随机性，这可能导致算法每次运行的结果都不同，甚至可能收敛到不同的局部最优解。
   2. 初始点的不稳定选择会影响聚类的稳定性和可重复性。
3. 对大数据集和小数据集的挑战：
   1. 对于大数据集，随机选择初始点可能会导致计算开销大，且难以保证聚类质量。
   2. 对于小数据集，由于数据点数量有限，随机选择初始点可能更容易导致算法陷入局部最优解。

- **初始质心（centroid）选择策略**
1. 每对聚类中选择一个聚类，并在其中放置两个初始质心
   - 这种方法试图通过在每个聚类对中选择一个聚类并为其分配两个初始质心来改进聚类过程。然而，这种方法有几个潜在的问题：
   1. 质心重叠：如果所选的聚类非常紧密或具有相似的数据分布，那么两个初始质心可能会非常接近，导致它们在迭代过程中难以分离，并可能最终收敛到同一个位置。
   2. 聚类不平衡：如果其他聚类中只分配了一个初始质心，而这些聚类中的数据点数量或分布与具有两个质心的聚类不同，那么聚类结果可能会不平衡。

2. 多次运行并选择最优结果
   - 方法：通过多次运行K-means算法，每次选择不同的初始质心，然后比较聚类结果的质量（如使用SSE、轮廓系数等指标）。选择质量最好的一次作为最终聚类结果。
   - 优点：
     - 增加了找到较好聚类结果的可能性。
     - 适用于小规模数据集或计算资源有限的情况。
   - 缺点：
     - 概率上并不总是能保证找到全局最优解。
     - 多次运行算法可能增加计算开销。
3. 使用策略选择初始质心
   - 方法：
     - 选择最广泛分离的质心：首先随机选择一个点作为第一个初始质心，然后计算每个数据点与已选择的质心之间的最小距离。接下来，选择距离已选择质心最远的点作为下一个初始质心，重复此过程直到选择了k个初始质心。这种方法可以确保初始质心在数据集中分布较为均匀。
     - 使用层次聚类确定初始质心：首先使用层次聚类算法（如AGNES或DIANA）对数据集进行初步聚类，然后从这些初步聚类中选择k个聚类中心作为K-means算法的初始质心。这种方法可以基于数据的层次结构来选择初始质心，从而可能提高聚类结果的准确性。
   - 优点：
     - 通过策略选择初始质心，可以提高聚类结果的准确性和稳定性。
     - 减少了随机选择初始质心带来的不确定性。
   - 缺点：
     - 层次聚类算法本身可能具有较高的计算复杂度。
     - 需要额外的步骤来确定初始质心，可能增加算法的整体复杂度。
4. 使用K-means++算法
   - 方法：K-means++算法是K-means算法的一种改进版本，它通过智能地选择初始质心来提高聚类结果的准确性和稳定性。K-means++算法首先随机选择一个点作为第一个初始质心，然后对于每个未选择的点，计算其与已选择的初始质心之间的最小距离的平方，并根据这个距离的平方以正比于概率的方式选择下一个初始质心。这个过程重复进行，直到选择了k个初始质心。
   - 优点：
     - 显著提高了初始质心选择的合理性。
     - 减少了算法陷入局部最优解的可能性。
   - 缺点：
     - 相对于传统的K-means算法，K-means++算法可能具有稍高的计算复杂度。
     - 综上所述，针对初始质心选择的问题，我们可以采用多次运行并选择最优结果、使用策略选择初始质心（如选择最广泛分离的质心或使用层次聚类确定初始质心）以及使用K-means++算法等方法来改进K-means聚类算法的性能。在实际应用中，可以根据数据集的特点和计算资源的情况选择合适的解决方案。

#### 5. Limitations of K-means
- K-means has problems when clusters are of differing
  - Sizes
  - <p style="display: block;">
      <img src="image_155.png" alt="image_155"/>
    </p>
    
  1. 处理不同大小的簇时的局限性
     1. 假设等大小簇：
        - K-means算法假设所有簇的大小是相似的，即每个簇中的点数量大致相同。然而，在实际应用中，簇的大小往往是不等的。
        - 当簇的大小差异很大时，K-means算法可能会将较小的簇拆分或合并到较大的簇中，导致聚类结果不准确。
     2. 对簇形状的敏感性：
        - K-means算法还假设簇是紧凑和圆形的。如果簇的形状是非圆形的（如椭圆形、长条形等），K-means算法可能无法准确地捕捉簇的结构。
        - 这会导致聚类结果的边界不清晰，且可能无法准确反映数据的实际分布。
  2. 处理原始点时的局限性
     1. 对初始点的依赖性：
        - K-means算法的结果在很大程度上取决于初始点的选择。如果初始点选择不当，可能会导致聚类结果的不稳定。
        - 为了减轻这种依赖性，可以使用K-means++等算法来改进初始点的选择。然而，即使使用这些改进算法，仍然无法完全消除对初始点的依赖。
     2. 对噪声和离群点的敏感性：
        - K-means算法对噪声和离群点非常敏感。噪声和离群点会干扰簇中心的计算，从而影响聚类结果。
        - 在实际应用中，如果数据集中存在大量的噪声或离群点，可能需要先对数据进行预处理，如使用滤波、去噪等方法来减少噪声和离群点的影响。
     3. 无法处理非数值型数据：
        - K-means算法只能处理数值型数据。如果数据集中包含非数值型数据（如文本、图像等），则需要先进行适当的数据转换或预处理，才能应用K-means算法。

  - Densities
  - <p style="display: block;">
     <img src="image_156.png" alt="image_156"/>
    </p>

  - 不同密度的局限性
    1. 问题描述：K-means算法的一个主要局限性在于它难以处理密度不均匀的数据集。该算法假设所有簇的密度大致相同，即每个簇中的数据点数量和数据点之间的紧密程度相似。然而，在实际应用中，数据集的密度往往是不均匀的，某些簇可能比其他簇更密集或更稀疏。
    2. 影响：当数据集的密度不均匀时，K-means算法可能会将低密度区域的点错误地归类到高密度区域的簇中，或者在高密度区域形成过多的簇，而在低密度区域形成较少的簇。这会导致聚类结果不准确，无法反映数据的真实分布。


  - Non-globular shapes
  - <p style="display: block;">
     <img src="image_157.png" alt="image_157"/>
    </p>

- K-means has problems when the data contains outliers.
  - One possible solution is to remove outliers before clustering

#### 6. Over coming K-means Limitations
- 针对K-means算法在处理复杂数据集时可能出现的局限性，一种策略是预先设定一个较大的聚类数量，使得每个小聚类都能代表自然聚类的一部分。然而，这种方法随后需要在后处理阶段将这些小聚类进一步合并，以形成更符合实际需求的聚类结果。以下是对这一策略的详细阐述：
  1. 策略概述
     1. 初步聚类：首先，使用K-means算法对数据进行初步聚类，但此时设定的K值（聚类数量）较大，目的是确保每个自然聚类都能被分割成多个小聚类。这样做的好处是，即使自然聚类的形状、密度或大小存在差异，也能在一定程度上被捕捉到。
     2. 后处理合并：在初步聚类完成后，需要进行后处理步骤，将这些小聚类根据一定的规则或相似性度量进行合并。这一步骤可能涉及多种技术和算法，如层次聚类、图聚类或基于密度的聚类方法，以进一步整合和优化聚类结果。
  2. 后处理合并的方法
     1. 层次聚类：可以使用层次聚类方法（如凝聚层次聚类）来合并小聚类。这种方法通过计算聚类间的相似性（如距离、密度等），逐步将最相似的小聚类合并成大聚类，直到满足一定的停止条件。
     2. 图聚类：将小聚类视为图中的节点，聚类间的相似性作为节点间的权重，然后应用图聚类算法（如谱聚类、最小割算法等）来合并节点（即小聚类）。
     3. 基于密度的聚类：可以使用基于密度的聚类方法（如DBSCAN）来进一步合并小聚类。这种方法通过计算数据点的局部密度，将密度相似的点归为一类，从而可能将相邻的小聚类合并成更大的聚类。
     4. 手动合并：在某些情况下，可能需要根据领域知识或特定需求手动合并小聚类。这通常涉及对聚类结果的深入分析和理解，以及对数据集的充分了解。

### 2. 层次聚类（Hierarchical Clustering）
- A set of nested clusters organized as a hierarchical tree
- 层次聚类则是创建一组嵌套的聚类，这些聚类被组织成一个层次树（或称为聚类树）。在层次树中，不同类别的原始数据点是树的最低层，而树的顶层则是一个聚类的根节点。层次聚类可以通过自下而上的合并（凝聚法）或自上而下的分裂（分裂法）来构建聚类树。
- 层次聚类假设数据类别之间存在层次结构，通过对数据集在不同层次的划分，构造出树状结构的聚类结果。这种方法不需要预先指定聚类数量，而是逐步合并或分裂聚类，以揭示数据的内在结构和层次关系。

<p style="display: block;">
  <img src="image_153.png" alt="image_153"/>
</p>

#### 1. Strengths of Hierarchical Clustering
1. 无需预设簇的数量
   - 层次聚类的一个显著优势是，它不需要在开始时就预设数据集中簇（或群组）的数量。这与K-means等需要事先指定簇数量的聚类算法形成了鲜明对比。
   - 在层次聚类中，簇的数量是在聚类过程结束后，根据实际需求通过“切割”树状图（dendrogram）来确定的。这意味着用户可以根据数据的特性和分析的目标，灵活地选择所需的簇数量。
2. 可获得任意数量的簇
   - 通过在树状图的适当位置进行“切割”，用户可以轻松地获得所需数量的簇。这种灵活性使得层次聚类在探索性数据分析中特别有用，因为用户可以在不同的簇数量之间进行比较，以找到最佳或最有意义的聚类结果。
3. 可能对应有意义的分类体系
   - 层次聚类的结果可能对应于数据集中存在的有意义分类体系。例如，在生物科学领域，层次聚类可以用于重建动物界的分类系统（如动物王国中的物种分类）或构建生物进化树（phylogeny）。
   - 在这些应用中，树状图不仅展示了数据点之间的相似性和差异性，还揭示了它们之间的潜在关系或进化历史。
4. 生物科学中的应用实例
   - 在生物科学中，层次聚类被广泛用于分析基因表达数据、蛋白质相互作用网络、物种分类和进化研究等领域。
   - 例如，通过层次聚类，研究人员可以识别出具有相似表达模式的基因群，这些基因群可能与特定的生物过程或疾病状态相关。
   - 同样地，在物种分类和进化研究中，层次聚类可以帮助研究人员构建更准确的分类系统和进化树，从而更深入地了解生物多样性和进化历史。
- 综上所述，层次聚类具有无需预设簇数量、可获得任意数量的簇以及可能对应有意义的分类体系等显著优势。这些优势使得层次聚类在数据分析领域具有广泛的应用前景和重要的研究价值。

#### 2. Two Types of Hierarchical Clustering
1. 凝聚型层次聚类（Agglomerative Hierarchical Clustering）
   - 起始点：每个数据点被视为一个单独的簇。
   - 逐步合并：在每一步中，都会找到并合并最接近（或最相似）的一对簇。这通常基于某种距离度量（如欧几里得距离）或相似性度量（如余弦相似性）来确定。
   - 终止条件：合并过程会一直进行，直到所有数据点都被合并成一个单一的簇，或者根据需求在达到某个指定的簇数量时停止。
   - 结果表示：结果通常表示为一个树状图（dendrogram），它记录了簇的合并序列和每个合并步骤的距离或相似性度量。

   - Key Idea: Successively merge closest clusters
   - Basic algorithm
     1. Compute the proximity matrix
     2. Let each data point be a cluster
     3. Repeat
     4. Merge the two closest clusters
     5. Update the proximity matrix
     6. Until only a single cluster remains
   - Key operation is the computation of the proximity of two clusters
     - Different approaches to defining the distance between clusters distinguish the different algorithms
     
   - How to define Inter-cluster distance
     1. MIN（单链法）
        - 定义：两个簇之间的最短距离被定义为这两个簇中任意两点之间的最短距离。
        - 特点：单链法倾向于形成长链形的簇，因为它只关注最近的点对。
        - 适用场景：适用于数据点分布不均匀或存在噪声的情况，但可能对噪声和离群点敏感。
        - <p style="display: block;">
            <img src="image_158.png" alt="image_158"/>
          </p>
          
          1. 嵌套簇（Nested Clusters）
             - 层次聚类生成的簇结构是嵌套的，即一个簇可以包含其他子簇。这种嵌套结构使得层次聚类能够揭示数据的深层结构和关系。在凝聚型层次聚类中，每个数据点最初都被视为一个单独的簇，然后通过不断地合并最近的两个簇，最终形成一个包含所有数据点的单一大簇。在这个过程中，可以形成多个中间层次的簇，这些簇构成了嵌套簇结构。
          2. 树状图（Dendrogram）
             - 树状图是层次聚类结果的可视化表示，它展示了数据点或簇之间的层次关系。在树状图中，每个数据点或簇都表示为一个节点，节点之间的连线表示它们之间的层次关系。树状图通常具有以下几个特点：
             - 层次结构：树状图清晰地展示了数据点或簇之间的层次关系，使得研究者可以直观地看到哪些数据点或簇被合并在一起，以及它们合并的顺序。
             - 距离度量：树状图中的连线长度通常表示节点之间的距离或相似度。在凝聚型层次聚类中，连线长度通常表示合并时两个簇之间的距离或相似度度量。
             - 可读性：通过适当的缩放和布局，树状图可以变得易于阅读和理解。研究者可以根据需要调整树状图的显示参数，以便更好地展示数据的层次结构。
             - 在层次聚类分析中，树状图是一个非常重要的工具，它使得研究者能够直观地看到数据的聚类过程和结果，从而更深入地理解数据的结构和关系。
        - **Strength of MIN**
        - 处理非椭圆形形状：
          1. MIN方法通过计算两个簇中距离最近的两个点之间的距离来确定簇间的相似度。这种计算方式使得MIN方法能够更灵活地处理形状不规则的数据集，特别是那些不符合椭圆形分布的数据。
          2. 相比之下，其他聚类方法（如K-means）可能更适合处理椭圆形或球形分布的数据集，因为它们通常基于数据点到簇中心的距离来进行聚类。
        - 对噪声和异常值的鲁棒性：
          1. 尽管MIN方法对噪声和异常值较为敏感，但在某些情况下，这种敏感性可以被视为一种优势。例如，在数据集中存在少量噪声或异常值时，MIN方法仍然能够找到数据中的主要结构，因为这些噪声或异常值通常不会对距离最近的点对产生太大影响。
          2. 当然，如果噪声或异常值过多，MIN方法可能会受到较大影响，导致聚类结果不准确。但在实际应用中，可以通过数据预处理和参数调整来减轻这种影响。
        - 直观性和易于理解：
          1. MIN方法的聚类过程相对直观，因为它基于距离最近的点对进行聚类。这使得研究者能够更容易地理解聚类结果和数据的结构。
          2. 此外，MIN方法还可以与其他方法（如可视化技术）相结合，以提供更丰富的聚类信息和分析结果。
        
        - **Weakness of MIN**
        - 对噪声的敏感性
        - 易受噪声点影响：
          1. MIN方法通过计算两个簇中距离最近的两个点之间的距离来确定簇间的相似度。这种计算方式使得MIN方法对噪声点非常敏感。当数据集中存在噪声点时，这些点可能会成为距离最近的点对，从而导致错误的聚类结果。
          2. 噪声点可能会误导MIN方法将原本不相似的簇合并在一起，或者将相似的簇分开。这降低了聚类结果的准确性和可靠性。
        - 难以处理离群点：
          1. 离群点是指与数据集中大多数点距离较远的点。在MIN方法中，离群点可能会成为影响聚类结果的关键因素。
          2. 由于MIN方法只关注距离最近的点对，因此离群点可能会与距离较近的簇形成错误的链接，从而破坏数据的整体结构。

     2. MAX（全链法）
        - 定义：两个簇之间的最长距离被定义为这两个簇中任意两点之间的最长距离。
        - 特点：全链法倾向于形成紧凑的球形簇，因为它关注最远的点对。
        - 适用场景：适用于数据点分布较为均匀且希望形成紧凑簇的情况。
        - <p style="display: block;">
            <img src="image_159.png" alt="image_159"/>
          </p>

        - **Strength of MAX**
        - Less susceptible to noise

        - **Weakness of MAX**
        - Tends to break large clusters
        - Biased towards globular clusters
         
     3. Group Average（平均链法）
        - 定义：两个簇之间的平均距离被定义为这两个簇中所有点对距离的平均值。
        - 特点：平均链法考虑了簇内所有点对的距离，因此能够更全面地反映簇间的相似性。
        - 适用场景：适用于数据点分布较为均匀且希望形成平衡簇的情况。
        - <p style="display: block;">
            <img src="image_160.png" alt="image_160"/>
          </p>

        - **Strength of Average**
        - Less susceptible to noise

        - **Weakness of Average**
        - Biased towards globular clusters
         
     4. Distance Between Centroids（质心距离法）
        - 定义：两个簇之间的质心距离被定义为这两个簇质心之间的距离。
        - 特点：质心距离法计算简单，但可能受到簇形状和大小的影响。
        - 适用场景：适用于簇形状和大小较为一致的情况。
     5. Ward’s Method（沃德方法）
        - 定义：Ward方法通过计算合并两个簇后总体内聚类方差增加的量来确定簇间距离。
        - 特点：Ward方法倾向于形成紧凑且方差较小的簇，因为它关注合并后的SSE增量。
        - 适用场景：适用于希望形成紧凑且方差较小的簇的情况。
        - **Strength of Ward**
        - Less susceptible to noise

        - **Weakness of Ward**
        - Biased towards globular clusters

2. 分裂型层次聚类（Divisive Hierarchical Clustering）
   - 起始点：所有数据点都被视为一个整体簇。
   - 逐步分裂：在每一步中，都会选择一个簇并将其分裂成两个较小的簇。这通常基于某种差异性度量（如簇内数据点的距离）来确定。
   - 终止条件：分裂过程会一直进行，直到每个簇都只包含一个数据点，或者根据需求在达到某个指定的簇数量时停止。
   - 结果表示：与凝聚型层次聚类类似，结果也可以表示为一个树状图，但它是一个倒置的树，记录了簇的分裂序列和每个分裂步骤的差异性度量。
3. 传统层次算法中的相似性或距离矩阵
   - 在进行层次聚类之前，通常需要计算每对数据点之间的相似性或距离矩阵。这个矩阵用于衡量数据点之间的关系，并作为后续合并或分裂步骤的依据。
   - 常用的距离度量包括欧几里得距离、曼哈顿距离等，而相似性度量则可能包括余弦相似性、皮尔逊相关系数等。
4. 合并或分裂的贪婪策略
   - 无论是凝聚型还是分裂型层次聚类，都使用了一种贪婪的策略来逐步合并或分裂簇。这意味着在每一步中，都会根据当前的信息选择最优的合并或分裂操作，而不是考虑全局最优解。
   - 这种贪婪策略虽然可能无法找到全局最优的聚类结果，但由于其计算效率较高，因此在实践中得到了广泛应用。

#### 3. Time and Space requirements
<p style="display: block;">
  <img src="image_161.png" alt="image_161"/>
</p>

#### 4. Problems and Limitations
- Once a decision is made to combine two clusters, it cannot be undone
- No global objective function is directly minimized
- Different schemes have problems with one or more of the following:
  1. Sensitivity to noise
  2. Difficulty handling clusters of different sizes and non- globular shapes
  3. Breaking large clusters

#### 5. Euclidean Distance
$$
d(x,y) = \sqrt{\sum_{k=1}^{n}(x_k - y_k)^2}
$$
- where n is the number of dimensions (attributes) and xk and yk are, respectively, the kth attributes (components) or data objects x and y.

<note>Standardization is necessary if scales differ.</note>

### 3. Other Distinctions Between Sets of Clusters
- Exclusive versus non-exclusive
  - In non-exclusive clusterings, points may belong to multiple
  clusters.
    - Can belong to multiple classes or could be ‘border’ points
  - Fuzzy clustering	(one type of non-exclusive)
    - In fuzzy clustering, a point belongs to every cluster with some weight
        between 0 and 1
    - Weights must sum to 1
    - Probabilistic clustering has similar characteristics
- Partial versus complete
  - In some cases, we only want to cluster some data

## 5. Association Rule Discovery
- Given a set of records each of which contain
some number of items from a given collection
- Produce dependency rules which will predict
occurrence of an item based on occurrences of other
items.

1. Itemset
   - A collection of one or more items

   - k-itemset
     - An itemset that contains k items

2. Support count (σ)
   - Frequency of occurrence of an itemset

3. Support
   - Fraction of transactions that contain an itemset

4. Frequent Itemset
   - An itemset whose support is greater than or equal to a minsup threshold

5. Association Rule and Rule evaluation matrix

<p style="display: block;">
  <img src="image_139.png" alt="image_139"/>
</p>

<p style="display: block;">
  <img src="image_140.png" alt="image_140"/>
</p>

<p style="display: block;">
  <img src="image_141.png" alt="image_141"/>
</p>

- Given d items, there are 2d possible candidate itemsets
<p style="display: block;">
  <img src="image_142.png" alt="image_142"/>
</p>

1. Reduce the number of candidates (M)
   - Complete search: M=2d
   - Use pruning techniques to reduce M
   - Apriori principle:
     - If an itemset is frequent, then all of its subsets must also be frequent

   - Apriori principle holds due to the following property of the support measure:
     - Support of an itemset never exceeds the support of its subsets
     - This is known as the anti-monotone property of support
     <note>具体例子与步骤见week5 ppt 🫠 哥们写到这已经快死了</note>

2. Reduce the number of transactions (N)
   - Reduce size of N as the size of itemset increases
3. Reduce the number of comparisons (NM)
   - Use efficient data structures to store the candidates or transactions
   - No need to match every candidate against every transaction

### Apriori Algorithm
<p style="display: block;">
  <img src="image_143.png" alt="image_143"/>
</p>

<p style="display: block;">
  <img src="image_145.png" alt="image_145"/>
</p>

<p style="display: block;">
  <img src="image_144.png" alt="image_144"/>
</p>

<p style="display: block;">
  <img src="image_146.png" alt="image_146"/>
</p>

<p style="display: block;">
  <img src="image_147.png" alt="image_147"/>
</p>

<p style="display: block;">
  <img src="image_148.png" alt="image_148"/>
</p>

<p style="display: block;">
  <img src="image_149.png" alt="image_149"/>
</p>

<p style="display: block;">
  <img src="image_150.png" alt="image_150"/>
</p>

<p style="display: block;">
  <img src="image_151.png" alt="image_151"/>
</p>

- Factors Affecting Complexity of Apriori
  - Choice of minimum support threshold
    - lowering support threshold results in more frequent itemsets
    - this may increase number of candidates and max length of frequent itemsets
  - Dimensionality (number of items) of the data set
    - More space is needed to store support count of	itemsets
    - if number of frequent itemsets also increases, both computation and I/O costs may also increase
  - Size of database
    - run time of algorithm increases with number of transactions
  - Average transaction width
    - transaction width increases the max length of frequent itemsets
    - number of subsets in a transaction increases with its width, increasing computation time for support counting


### 1. Association Analysis: Applications

- Market-basket analysis 
  - Rules are used for sales promotion, shelf management, and inventory management

- Telecommunication alarm diagnosis 
  - Rules are used to find combination of alarms that  occur together frequently in the same time period

- Medical Informatics 
  - Rules are used to find combination of patient  symptoms and test results associated with certain  diseases

## 6. Model Evaluation

### 1. Metrics for Performance Evaluation
<tip>IMPORTANT!!!</tip>

![image_9.png](image_9.png)

### Confusion Matrix 混淆矩阵
|                 | Predicted Positive  | Predicted Negative  |
|-----------------|---------------------|---------------------|
| Actual Positive | True Positive (TP)  | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN)  |

#### 1. Accuracy 准确率
Accuracy表示分类器正确分类的样本比例。

$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$

#### 2. Precision 精确率
Precision表示被预测为正类的样本中实际为正类的比例。

$
\text{Precision} = \frac{TP}{TP + FP}
$

#### 3. Recall (Sensitivity) 召回率（灵敏度）
Recall表示实际为正类的样本中被正确预测为正类的比例。

$
\text{Recall} = \frac{TP}{TP + FN}
$

#### 4. F-measure (F1-score)
F-measure是Precision和Recall的调和平均值。

$
\text{F-measure} = 2 \times\frac{ \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$

#### 5. Specificity 特异性
Specificity表示实际为负类的样本中被正确预测为负类的比例。

$
\text{Specificity} = \frac{TN}{TN + FP}
$

#### 6. False Positive Rate (FPR) 假阳性率
FPR表示实际为负类的样本中被错误预测为正类的比例。

$
\text{False Positive Rate} = \frac{FP}{TN + FP}
$

### 2. Limitation of Accuracy

- Consider a 2-class problem
  - Number of Class 0 examples = 9990
  - Number of Class 1 examples = 10
-  If model predicts everything to be class 0,
accuracy is 9990/10000 = 99.9 %
  - Accuracy is misleading because model does
  not detect any class 1 example

### 3. Computing Cost of Classification
Cost Matrix
<p style="display: block;">
  <img src="image_133.png" alt="image_133"/>
</p>

<p style="display: block;">
  <img src="image_135.png" alt="image_135"/>
</p>

<p style="display: block;">
  <img src="image_134.png" alt="image_134"/>
</p>

<p style="display: block;">
  <img src="image_136.png" alt="image_136"/>
</p>

### 4. Methods for Performance Evaluation
Performance of a model may depend on other
factors besides the learning algorithm:
- Class distribution
- Cost of misclassification
- Size of training and test sets

### 5. Learning Curve
- Learning curve shows
  how accuracy changes
  with varying sample size
- Requires a sampling
  schedule for creating
  learning curve
- Effect of small sample
  size:
  - Bias in the estimate
  - Variance of estimate

### 6. Methods of Estimation
- Holdout
  - Reserve k% for training and (100-k)% for testing
- Random subsampling
  - Repeated holdout
- Cross validation
  - Partition data into k disjoint subsets
  - k-fold: train on k-1 partitions, test on the remaining one
  - Leave-one-out: k=n
- Stratified (分層) sampling
  - oversampling vs undersampling
- Bootstrap
  - Sampling with replacement

#### Variations on Cross-validation
- Repeated cross-validation
  - Perform cross-validation a number of times
  - Gives an estimate of the variance of the
  generalization error
- Stratified cross-validation
  - Guarantee the same percentage of class
  labels in training and test
  - Important when classes are imbalanced and
  the sample is small
- Use nested cross-validation approach for model
  selection and evaluation

### 7. Methods for Model Comparison
ROC (Receiver Operating Characteristic) Curve

- ROC curve plots TPR (on the y-axis) against FPR
(on the x-axis)
  - TPR (TP rate) = TP/(TP+FN); positive hits
  - FPR (FP rate)= FP/(FP + TN); false alarms
<p style="display: block;">
  <img src="image_137.png" alt="image_137"/>
</p>
<p style="display: block;">
  <img src="image_138.png" alt="image_138"/>
</p>
<note>AUC（曲线下面积）用于评估模型性能。AUC值在0到1之间，值越大表示模型性能越好。</note>


## 7. Ensemble Methods
- Construct a set of base classifiers learned from the training data
- Predict class label of test records by combining the predictions made by multiple classifiers (e.g., by taking majority vote)

- suppose we have 25 base classifiers
- each classifier has an error rate of 0.35
<p style="display: block;">
  <img src="image_209.png" alt="image_209"/>
</p>
- 当所有分类器都是独立的且错误不相关时，集成分类器的错误率可以显著降低。这是集成方法能够工作的重要原因之一

- Ensemble Methods work better than a single base classifier if:
  1. All base classifiers are independent of each other
  2. All base classifiers perform better than random guessing (error rate < 0.5 for binary classification)

### 1. Rationale for Ensemble Learning
- Ensemble Methods work best with unstable base classifiers
  - Classifiers that are sensitive to minor perturbations in training set, due to high model complexity
  - Examples: Unpruned decision trees, ANNs, …
  - 集成学习通过构建并结合多个学习器来完成学习任务。这些学习器可以是同种类型的（如决策树集成），也可以是不同类型的（如决策树与神经网络结合）。集成学习的目标是通过结合多个学习器的预测结果，来提高整体的泛化能力。
  - 不稳定的基分类器指的是那些对训练数据中的微小变动高度敏感的分类器。这种敏感性通常源于模型的高复杂性，如未剪枝的决策树、人工神经网络（ANNs）等。这些模型在训练过程中容易过拟合，即它们能够很好地拟合训练数据，但对于新的、未见过的数据则可能表现不佳。

- <p style="display: block;">
    <img src="image_162.png" alt="image_162"/>
  </p>

- 集成学习（Ensemble Learning）的一般方法确实包括构建多个分类器（或称为个体学习器）以及结合这些分类器的响应。
- Step 1: Build multiple classifiers（构建多个分类器） 
  - 这一步是集成学习的基础。通过训练数据集，我们可以生成多个个体学习器。这些学习器可以是同质的（例如，都是决策树或都是神经网络），也可以是异质的（例如，结合支持向量机、逻辑回归和朴素贝叶斯等不同的算法）。 
- Step 2: Combine classifier response（结合分类器响应） 
  - 在得到多个个体学习器的预测结果后，我们需要采用一种策略来结合这些结果，从而得到一个更强大的预测模型。常用的结合策略包括平均法、投票法和学习法。
- 投票法特别适用于分类问题。 
- 多数投票法：根据各个分类器的预测结果，选择获得最多票数的类别作为最终的预测类别。如果有多个类别获得相同数量的最高票数，则通常会随机选择一个作为最终类别。 
- 加权多数投票法：与多数投票法类似，但每个分类器的预测结果会乘以一个权重。最终，将各个类别的加权票数求和，选择加权票数最大的类别作为最终的预测类别。


### 2. Constructing Ensemble Classifiers
1. By manipulating training set
   - Example: bagging, boosting, random forests
2. By manipulating input features
   - Example: random forests
3. By manipulating class labels
   - Example: error-correcting output coding
4. By manipulating learning algorithm
   - Example: injecting randomness in the initial weights of	ANN

### 3. Bagging (Bootstrap AGGregatING)
- Bootstrap sampling: sampling with replacement
- Bootstrap采样
  - Bootstrap采样是从原始训练集中随机抽取多个不同的训练子集（可能包含重复样本）的过程。在每次采样中，每个训练实例被选中的概率是相同的，且采样是有放回的，这意味着同一个训练实例可能在同一个子训练集中出现多次，也可能根本不出现。
  - 训练实例在Bootstrap样本中的选中概率
  - 对于一个包含n个训练实例的原始训练集，在Bootstrap采样中，一个特定的训练实例被选中进入某个子训练集的概率可以通过以下公式计算：
  $$
  P= 1 - (1 - \frac{1}{n})^n
  $$
  - 当n很大时（即训练集包含大量实例时），这个概率可以近似为0.632。这意味着在Bootstrap采样中，大约有63.2%的训练实例会被选中进入某个子训练集，而剩下的36.8%则不会被选中。这些未被选中的实例被称为袋外数据（Out of Bag, OOB），它们可以用于评估基学习器的性能，而无需额外的验证集。
<p style="display: block;">
  <img src="image_163.png" alt="image_163"/>
</p>
<p style="display: block;">
  <img src="image_210.png" alt="image_210"/>
</p>

1. 训练基分类器：
   1. 使用自助采样法（Bootstrap Sampling）从训练集中生成多个不同的训练子集。
   2. 对于每个训练子集，训练一个决策树桩作为基分类器。
      - 决策树桩只在一个特征上进行一次分裂，以最大化分类的准确性。
2. 预测与投票：
   1. 对于测试集中的每个样本，使用所有训练好的决策树桩进行预测。
   2. 每个决策树桩都会输出一个预测值（对于二分类问题，这通常是-1或1）。
   3. 计算所有决策树桩预测值的和，并根据和的符号来确定最终的类别。
      1. 如果和为正，则样本被归类为类别1。 
      2. 如果和为负，则样本被归类为类别-1。
        3. 如果和为0，则可以通过投票法或其他规则来确定最终的类别。
<p style="display: block;">
  <img src="image_211.png" alt="image_211"/>
</p>

### 4. Boosting
- An iterative procedure to adaptively change distribution of training data by focusing more on previously misclassified records
  1. Initially, all N records are assigned equal weights (for being selected for training)
  2. Unlike bagging, weights may change at the end of each boosting round

- **Boosting中的权重调整机制**
1. 初始权重分配：
   - 在Boosting的起始阶段，所有的训练记录都会被赋予相同的权重。
2. 权重更新规则：
   1. 误分类记录：如果一个记录在当前轮次的分类中被错误地归类，那么它在下一轮训练中的权重将会被增加。这样做的目的是为了使得模型在后续的迭代中更加关注这些难以分类的记录，从而提高整体的分类准确性。
   2. 正确分类记录：相反，如果一个记录在当前轮次被正确地分类，那么它在下一轮的权重通常会降低。因为对于这些已经得到良好处理的记录，模型不需要再给予过多的关注。
3. 迭代过程：
   1. Boosting算法会进行多轮迭代，每一轮都会根据当前的权重分布来训练一个新的弱分类器（或称为基分类器）。
   2. 在每一轮迭代结束后，都会根据该轮弱分类器的性能来更新记录的权重，然后进入下一轮迭代。
example
<p style="display: block;">
  <img src="image_213.png" alt="image_213"/>
</p>

#### 1. AdaBoost (Adaptive Boosting)
- AdaBoost是一种常见的Boosting算法，它通过迭代训练多个弱分类器，并根据每个分类器的性能来调整训练数据的权重，从而提高整体的分类准确性。
<p style="display: block;">
  <img src="image_214.png" alt="image_214"/>
</p>
<p style="display: block;">
  <img src="image_215.png" alt="image_215"/>
</p>

<p style="display: block;">
  <img src="image_216.png" alt="image_216"/>
</p>

### 5. Random Forest Algorithm
- Construct an ensemble of decision trees by manipulating training set as well as features
  1. Use bootstrap sample to train every decision tree (similar to Bagging)
  2. Use the following tree induction algorithm:
     1. At every internal node of decision tree, randomly sample p attributes for selecting split criterion
     2. Repeat this procedure until all leaves are pure (unpruned tree)

- 随机森林算法
1. 构建训练集的Bootstrap样本：
   1. 随机森林算法使用Bootstrap方法从原始训练集中随机抽取多个样本集（通常与原始训练集大小相同，但允许重复抽样）。
   2. 每个Bootstrap样本都将用于训练一棵决策树。
2. 训练决策树：
   1. 对于每个Bootstrap样本，都训练一棵决策树。
   2. 这些决策树在构建过程中是独立的，彼此不产生任何影响。
3. 决策树的构建过程：
   1. 随机选择特征：在决策树的每个内部节点处，不是考虑所有特征，而是随机选择p个特征（p通常远小于总特征数）。
   2. 选择分裂准则：从随机选择的p个特征中，选择最优的特征和分裂点来构建当前节点的分裂准则。
   3. 递归构建：重复上述过程，直到达到停止条件（例如，所有叶子节点都是纯的，或者树的深度达到了预设的最大值）。
   4. 不剪枝：在随机森林中，通常不对决策树进行剪枝，以保留更多的信息并增强模型的泛化能力。
4. 集成模型：
   1. 将所有训练好的决策树组合成一个集成模型。
   2. 对于新的输入数据，每棵决策树都会给出一个预测结果。
   3. 随机森林通过投票（对于分类任务）或平均（对于回归任务）所有决策树的预测结果来得到最终的预测结果。

- Characteristics of Random Forest
<p style="display: block;">
  <img src="image_217.png" alt="image_217"/>
</p>

- 随机森林的特性
1. 基分类器是不剪枝的树：
   1. 随机森林中的基分类器通常是未剪枝的决策树。这意味着这些树会生长到最大可能的深度，直到所有叶子节点都是纯的（对于分类任务）或满足其他停止条件。
   2. 不剪枝的树往往具有较高的方差和较低的偏差，因此它们是“不稳定”的分类器，对训练数据的微小变化可能产生较大的预测差异。
2. 基分类器之间的去相关性：
   1. 随机森林通过两种方式来减少基分类器之间的相关性：
      1. 训练集的随机化：使用Bootstrap方法从原始训练集中随机抽取多个样本集来训练不同的决策树。
      2. 特征的随机化：在决策树的每个内部节点处，随机选择一部分特征来考虑作为分裂候选。
   2. 这种随机化有助于减少基分类器之间的相关性，从而提高集成模型的性能。
3. 减少不稳定分类器的方差：
   1. 随机森林通过集成多个不稳定的基分类器来减少整体模型的方差。
   2. 由于基分类器之间的去相关性，它们的预测错误往往不会在同一方向上累积，因此集成模型的预测结果更加稳定。
4. 对偏差的影响较小：
   1. 随机森林在减少方差的同时，对模型的偏差影响较小。
   2. 这是因为每个基分类器都是独立训练的，并且都有机会捕捉到数据中的真实模式。
5. 超参数p的选择：
   1. p表示在决策树的每个内部节点处随机选择的特征数。
   2. p的选择对随机森林的性能有很大影响：
      1. 较小的p值：有助于减少基分类器之间的相关性，但可能导致单个基分类器的性能下降。
      2. 较大的p值：可能提高单个基分类器的性能，但也可能增加基分类器之间的相关性。
   3. 常见的默认选择包括√d（其中d是总特征数）或log₂(d + 1)。这些选择通常能在减少相关性和保持基分类器性能之间取得良好的平衡。