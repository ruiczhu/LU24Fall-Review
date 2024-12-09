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

- Classification [Predictive]
- Clustering [Descriptive]
- Association Rule Discovery [Descriptive]  
- Sequential Pattern Discovery [Descriptive]  
- Regression [Predictive]
- Deviation Detection [Predictive]

![image_4.png](image_4.png)

## 3. Predictive Modeling

### 1. predictive modeling: classification
- Find a model	for class attribute as a function of the values of other attributes

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
– Produce dependency rules which will predict
occurrence of an item based on occurrences of other
items.

#### 1. Association Analysis: Applications

- Market-basket analysis 
  - Rules are used for sales promotion, shelf management, and inventory management

- Telecommunication alarm diagnosis 
  - Rules are used to find combination of alarms that  occur together frequently in the same time period

- Medical Informatics 
  - Rules are used to find combination of patient  symptoms and test results associated with certain  diseases

## 4. Data Preprocessing

### 1. Outliers

- Outliers are data objects with characteristics that are considerably different than most of the other data objects in the data set

### 2. Missing Values

### 3. Duplicate Data

## 5. Decision Trees

### 1. Test Condition for Nominal Attributes
- Multi-way split:
- Use as many partitions as distinct values.

- Binary split:
- Divides values into two subsets

![image_6.png](image_6.png)

### 2. Test Condition for Ordinal Attributes
- Multi-way split:
- Use as many partitions as distinct values

- Binary split:
- Divides values into two subsets
- Preserve order property among attribute values

![image_7.png](image_7.png)


### 3. Test Condition for Continuous Attributes
![image_8.png](image_8.png)

## 6. Model Evaluation

### 1. Metrics for Performance Evaluation
![image_9.png](image_9.png)

### 2. Limitation of Accuracy

### 3. Computing Cost of Classification