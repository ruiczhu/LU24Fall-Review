# CDS535 DataMining
- The exam has 9 Questions
- Data contains value and knowledge

## 1. What is Data Mining?

### 1. Definition of Data Mining
- Non-trivial extraction of implicit, previously unknown and potentially useful information from data
- Exploration & analysis, by automatic or semi-automatic  means, of large quantities of data in order to discover  meaningful patterns
- ![image.png](image.png)

### 2. Data Science compared to Data Mining
**Business Understanding**
- Seldom pre-packaged as clear data science problems.
- Creative problem formulation is needed to consider the  business problem as one or more data science problems.
- High-level knowledge of the domains helps creative Data
- Scientists/Business Analysts see novel formulations.
- The design team should think carefully about the problem to  be solved and about the user scenario.

**Data Understanding**
- Understand the strengths and limitations of the data.
- Historical data often are collected for other purposes unrelated  to the current data science problem.

**Data Preparation**
- Many data science/data mining algorithms require data to be in a specific form (e.g. Table format).
- Some conversion will be necessary.
- Missing values should be handled.
- 
**Modeling**
- Apply data science/data mining algorithms to create  models/data science results from the data

**Evaluation**
- Assess the data science results.
- Gain confidence that they are correct and reliable.
- Test the results (models) in a controlled laboratory setting.
- Ensure that the model satisfies the original business goals.

**Deployment**
- The data science results (models) are used in real problems  (e.g. fake detections).
- Implement it in some information system or business process.
- Deploy a model into a production system typically requires  that the model be re-coded for the production environment  (e.g. program in Windows environment).

![image_2.png](image_2.png)

### 3. Data Mining Process
![image_3.png](image_3.png)

## 2. Data Mining Tasks
**Prediction Methods**
- Use some variables to predict unknown or  future values of other variables.

**Description Methods**
- Find human-interpretable patterns that  describe the data.

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

## 3. Predictive Modeling
<note>本章具体内容见以下链接</note>

### 1. predictive modeling: classification
- Find a model	for class attribute as a function of the values of other attributes
[](#7-classification)

### 2. predictive modeling: regression
- Predict a value of a given continuous valued variable based on the values of other variables, assuming a linear or nonlinear model of dependency. 
- Extensively studied in statistics, neural network fields.

### 3. predictive modeling: clustering
- Finding groups of objects such that the objects in a
group will be similar (or related) to one another and
different from (or unrelated to) the objects in other
groups

#### 1. K-means Clustering
- Partitional clustering approach 
- Number of clusters, K, must be specified 
- Each cluster is associated with a centroid (center point) 
- Each point is assigned to the cluster with the closest
centroid 
- The basic algorithm is very simple
![image_5.png](image_5.png)

#### 2. Two different K-means Clusterings
- Optimal Clustering（最优聚类）
- Sub-optimal Clustering（次优聚类）

#### 3. Importance of Choosing Initial Centroids 


### 4. Association Rule Discovery: Definition
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


#### 1. Association Analysis: Applications

- Market-basket analysis 
  - Rules are used for sales promotion, shelf management, and inventory management

- Telecommunication alarm diagnosis 
  - Rules are used to find combination of alarms that  occur together frequently in the same time period

- Medical Informatics 
  - Rules are used to find combination of patient  symptoms and test results associated with certain  diseases

## 4. Data
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
      are considerably different than most of the other
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
- Sampling is the main technique employed for data
  reduction.
  - It is often used for both the preliminary investigation of
    the data and the final data analysis.
- Statisticians often sample because obtaining the
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

## 5. Decision Trees

<note>本章具体内容见以下链接</note>

[](#1-decision-tree-algorithms)

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

## 7. Classification

- Goal: previously unseen records should be
  assigned a class as accurately as possible.

- General Approach for Building
  Classification Model

<p style="display: block;">
  <img src="image_118.png" alt="image_118"/>
</p>

### Classification Techniques
- Base Classifiers
  - Decision Tree based Methods
  - Rule-based Methods
  - Nearest-neighbor
  - Naïve Bayes and Bayesian Belief Networks
  - Support Vector Machines
  - Neural Networks, Deep Neural Nets

- Ensemble Classifiers
  - Boosting
  - Bagging
  - Random Forests

<p style="display: block;">
  <img src="image_119.png" alt="image_119"/>
</p>

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
    - Robust to noise (especially when methods to avoid overfitting are employed)
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

Resubstitution Estimate:
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



      