# CDS525 DeepLearning

## Lecture 1

### Quiz 1.1

1. If we want to use a machine learning model to solve the problem of classifying the credit level of bank customers, then the input should be <########>, and the output should be <########>. Choose the most appropriate one from the following.

    A. customer information, {Yes, No}  
    **B. customer information, {Good, Average, Bad}**  
    C. bank information, {Good, Average, Bad}  
    D. bank information, {Yes, No}

2. If we want to use a machine learning model to solve the problem of classifying different pet species (Cat, Dog and Fish), then the input should be <########>, and the output should be <########>. Choose the most appropriate one from the following.

    A. Cat/Dog/Fish, {Yes, No}  
    **B. pet images, {Cat, Dog, Fish}**  
    C. None of the above.  
    D. pet images, {Yes, No}

3. If we want to use a machine learning model to predict the result of a football game for a football team, then the input should be <########>, and the output should be <########>. Choose the most appropriate one from the following.

    A. None of the above.  
    **B. (game, team), {Win, Lose}**  
    C. team, {game}  
    D. game, {Win, Lose}

4. During machine learning, which process occurs for adjusting parameters for function f?

    A. Testing  
    B. a and b  
    C. None of the above  
    **D. Training**

### Quiz 1.2

1. What is deep learning?

    A. A programming language for data science.  
    B. A subset of machine learning algorithms that focus on decision trees.  
    C. A machine learning technique that uses shallow neural networks.  
    **D. A subfield of artificial intelligence that deals with neural networks with multiple hidden layers.**

2. What is the purpose of the training process in a neural network?

    A. To design the architecture of the neural network.  
    B. To initialize the weights and biases.  
    C. To validate the model's performance  
    **D. To optimize the model's parameters to make accurate predictions on new data**

3. What is the purpose of the output layer in a neural network?

    A. To calculate the loss function  
    B. To preprocess the input data  
    C. To introduce non-linearity into the model  
    **D. To produce the final predictions or outputs based on the network's learned representations**

4. What is the input layer of a neural network responsible for?

    **A. Transforming input data into a format suitable for neural network processing**  
    B. Processing input data to extract features and patterns in data  
    C. All of the above  
    D. Making predictions and generating output

### Quiz 1.3

1. T/F: Deep networks are more suitable than shallow ones when the required functions are complex and regular.

    **True**

2. T/F: The same number of parameters, generally deeper models will have higher time complexity.

    **False**

3. Which of the following statements is true?

    **A. The deeper layers of a neural network are typically computing more complex features of the input than the earlier layers.**  
    B. The earlier layers of a neural network are typically computing more complex features of the input than the deeper layers.

4. Which of the following two networks is deeper?

    **A**  

    ![img.png](img.png)

    B

    ![img_1.png](img_1.png)

### Quiz 1.4

1. Which of the following options is not included in sequence prediction?

    **A. Image Segmentation**  
    B. POS Tagging  
    C. Machine Translation  
    D. Speech Recognition

2. Which network architecture is suitable for processing input sequences with temporal information?

    **A. RNN (recurrent neural network)**  
    B. AlexNet  
    C. CNN (convolutional neural network)  
    D. VGGNet

3. Which of the following statements accurately reflects the relationship between the learning algorithm and the input/output domains?

    **A. The learning algorithm f maps the input domain X to the output domain Y, and network design should consider the properties of the input and output domains.**  
    B. The learning algorithm f maps the input domain X to the output domain Y, and network design is independent of the properties of the input and output domains.  
    C. The learning algorithm f maps the output domain Y to the input domain X, and network design is independent of the properties of the input and output domains.  
    D. The learning algorithm f maps the output domain Y to the input domain X, and network design should consider the properties of the input and output domains.

4. Which of the following factors is considered a core factor for applied deep learning?

    A. Data: big data  
    B. Talent: design algorithms for specific problems  
    C. Hardware: GPU computing  
    **D. All of the above**

### Quiz 1.5

1. Which of the following options accurately lists the four key components needed in machine/deep learning problems?

    **A. Objective function, learning algorithm, data, model.**  
    B. Objective function, data transformation, model parameters, learning algorithm.  
    C. Model, dimensionality, learning algorithm, objective function.  
    D. Data, learning algorithm, objective function, dimensionality.

2. In supervised learning, what is the role of the label or target?

    A. The label or target represents a set of attributes used to make predictions.  
    B. The label or target is very important in capturing features.  
    **C. The label or target is a special attribute that needs to be predicted by the model.**  
    D. The label or target determines the dimensionality of the data.

3. Which of the statement is not right?

    A. For classification, the most common objective function is cross-entropy.  
    B. When we have more data, we are more possible to train more powerful models and rely less heavily on pre-conceived assumptions.  
    C. The objective function can guide the optimization process during training.  
    **D. For classification, the most common objective function is squared error.**

4. Which of the following statements accurately describes gradient descent?

    A. Gradient descent is a learning algorithm that searches for the worst possible parameters for maximizing the loss function.  
    **B. Gradient descent is a popular optimization algorithm for deep learning that minimizes the loss function by perturbing parameters in the direction that reduces the loss.**  
    C. Gradient descent is a learning algorithm that calculates the loss function by perturbing the parameters in a small amount.  
    D. Gradient descent is a method that maximizes the loss function by updating parameters in the direction that increases the loss.

## Lecture 2

### Quiz 2.1

1. What is a neural network?

    A. A biological system of interconnected neurons.  
    B. A computer program used for word processing.  
    C. A statistical tool for data visualization.  
    **D. A computational model inspired by the human brain, composed of layers of interconnected nodes.**

2. What does a neuron compute?

    **A. A neuron computes a linear function followed by an activation function.**  
    B. A neuron computes the means of all features before applying the output to an activation function.  
    C. A neuron computes a function g that scales the input linearly.  
    D. A neuron computes an activation function followed by a linear function.

3. What is the role of the weights and biases in a neural network?

    A. They determine the number of neurons in each layer  
    B. They are used to calculate the loss function  
    C. They are fixed and do not change during training  
    **D. They are adjusted during training to control the flow of information and make predictions**

4. What is Binary Classification in Machine Learning?

    A. A technique for classifying data without labels into two classes  
    B. A method to classify instances into more than two classes, but less than five  
    C. A technique where the data points are clustered into two classes  
    **D. A technique where an instance is classified into two classes**

5. Multi-class Classification differs from Binary Classification in that it:

    A. Is used only in image recognition tasks  
    B. Requires more data resources always  
    C. Involves only two classes  
    **D. Involves classifying instances into three or more classes**

### Quiz 2.2

1. Which Boolean operation on two variables cannot be represented by a single perception layer?

    **A. X1 XOR X2**  
    B. X1 NOR X2  
    C. X1 AND X2  
    D. X1 OR X2

2. What is the purpose of an activation function during training?

    A. To regularize the model  
    B. To control the learning rate  
    **C. To introduce non-linearity into the model**  
    D. To compute the gradient of the loss function

3. Which of the following descriptions of a two-layer neural network is correct?

    A. The input is multiplied by a matrix and added to a bias, then this result is multiplied by another matrix and also multiplied by the same input-plus-bias result, and finally, a non-linear function is applied to the entire expression.

    B. The input is multiplied by a matrix and added to a bias, then this result is multiplied by another matrix and added to another bias, and finally, a non-linear function is applied to the result.

    **C. The input is multiplied by a matrix and added to a bias, then a non-linear function is applied to this result. Next, the activated result is multiplied by another matrix and added to another bias.**

    D. The input is multiplied by a matrix and added to a bias, then a non-linear function is applied to this result. Next, the activated result is multiplied by another matrix and also multiplied by the non-linear function of the input-plus-bias result, and finally, another non-linear function is applied to the entire expression.

### Quiz 2.3

1. What is the purpose of the loss function in a neural network?

    A. To compute the gradient  
    B. To measure the accuracy of the model  
    C. To update the weights  
    **D. To measure the difference between predicted output and actual output**

2. What is the role of the optimizer in a neural network?

    **A. To minimize the loss function**  
    B. To set the learning rate.  
    C. To select the best activation function.  
    D. To determine the number of layers

3. Which of the following statements about Mean Squared Error (MSE) is correct?

    A. The formula for MSE is $MSE = \sum (y_i - \hat{y}_i)$, where $y_i$ is the actual value and $\hat{y}_i$ is the predicted value.  
    B. MSE gives equal weight to all errors, regardless of their size.  
    C. MSE is a metric for measuring model prediction accuracy, but it only considers the average magnitude of the differences between predicted values and actual observed values without squaring them.  
    **D. MSE measures model prediction accuracy by calculating the average of the squared differences between predicted values and actual observed values, and a lower MSE value indicates better model performance.**

## Lecture 3

### Quiz 3.1

1. T/F: During forward propagation, in the forward function for a layer l you need to know what is the activation function in a layer (Sigmoid, tanh, ReLU, etc.). During backpropagation, the corresponding backward function also needs to know what is the activation function for layer l, since the gradient depends on it.

    **True**

2. 我们有一个简单的函数 $ f(x,y,z) = (x + y)z $ 。

我们可以将其拆分为方程 $ q = x + y $ 和 $ f(x,y,z) = qz $ 。使用这种简化的表示法，我们也可以将此方程表示为计算图：

![img_2.png](img_2.png)

此外，令上游梯度（损失对函数的梯度，$ \frac{\partial L}{\partial F} $ 的值等于1。这些数值已在计算图中填充。

求解以下数值（在  $ x = -2, y = 5, z = -4, \frac{\partial L}{\partial F} = 1 $  处计算）：

$\\( \frac{\partial f}{\partial q} = -4 \\)$, $\\( \frac{\partial q}{\partial x} = 1 \\)$
, $\\( \frac{\partial q}{\partial y} = 1 \\)$, $\\( \frac{\partial f}{\partial z} = 3 \\)$, $\\( \frac{\partial f}{\partial x} = -4 \\)$, $\\( \frac{\partial f}{\partial y} = -4 \\)$

<tip>

**解题过程与解析**

1. **计算 $q$ 的值：**
    $$
    q = x + y = -2 + 5 = 3
    $$

2. **计算 $f(x, y, z)$ 的值：**
    $$
    f(x, y, z) = qz = 3 \times (-4) = -12
    $$

3. **计算各个偏导数：**

    - $ \frac{\partial f}{\partial q} = z = -4 $
    - $ \frac{\partial q}{\partial x} = 1 $
    - $ \frac{\partial q}{\partial y} = 1 $
    - $ \frac{\partial f}{\partial z} = q = 3 $

4. **链式法则求 $f$ 对 $x$ 和 $y$ 的偏导：**

    $$
    \frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial x} = (-4) \times 1 = -4
    $$
    $$
    \frac{\partial f}{\partial y} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial y} = (-4) \times 1 = -4
    $$

</tip>

### Quiz 3.2

1. Suppose we are using gradient descent with learning rate $\alpha$. For logistic regression and linear regression, $J(\theta)$ is a convex optimization problem, and thus we do not want to choose a learning rate $\alpha$ that is too large. For a neural network, however, $J(\theta)$ may not be convex, and thus choosing a very large value of $\alpha$ can only speed up convergence. (Note: $J(\theta)$ is the Cost function for the neural network (with regularization).)

    **False**

2. The following picture is of a feedforward network with a single input $X$, two hidden layers with two neurons in each layer and a single output $Y$.

    ![img_3.png](img_3.png)

    What is the update rule of weight matrix $W_1$?  
    (In other words, what is the partial derivative of $Y$ with respect to $W_1$?)

    A. $\frac{\partial y}{\partial W_1} = \frac{\partial y}{\partial h_3} \frac{\partial h_3}{\partial h_1} \frac{\partial h_1}{\partial W_1}$  
    **B. $\frac{\partial y}{\partial W_1} = \frac{\partial y}{\partial h_3} \frac{\partial h_3}{\partial h_1} \frac{\partial h_1}{\partial W_1} + \frac{\partial y}{\partial h_4} \frac{\partial h_4}{\partial h_1} \frac{\partial h_1}{\partial W_1}$**

    <tip>

    **解题过程与解析**

    1. **理解网络结构：**
       - 输入层：$X$
       - 隐藏层1：$h_1$, $h_2$
       - 隐藏层2：$h_3$, $h_4$
       - 输出层：$Y$

    2. **计算 $Y$ 对 $W_1$ 的偏导数：**
       - 使用链式法则：

       $$
       \frac{\partial Y}{\partial W_1} = \frac{\partial Y}{\partial h_3} \cdot \frac{\partial h_3}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1} + \frac{\partial Y}{\partial h_4} \cdot \frac{\partial h_4}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1}
       $$

       - 其中，$h_3$ 和 $h_4$ 都依赖于 $h_1$，因此需要分别计算它们对 $W_1$ 的偏导数。

    3. **选择正确答案：**
       - 选项B是正确的，因为它考虑了$h_3$和$h_4$对$W_1$的影响。
       - 选项A不正确，因为它只考虑了$h_3$对$W_1$的影响，忽略了$h_4$。
    </tip>

3. T/F: Suppose we have a correct implementation of backpropagation and are training a neural network using gradient descent. Suppose we plot $J(\theta)$ as a function of the number of iterations, and find that it is increasing rather than decreasing. One possible cause is that the learning rate $\alpha$ is too large. (Note: $J(\theta)$ is the Cost function for the neural network (with regularization).)

    **True**

### Quiz 3.3

1. What are word embeddings?

    **A. Vectors representing words, such that semantically similar words are represented by similar vectors.**  
    B. Vectors used to compress the meaning of a text in sequence-to-sequence problems.  
    C. A mechanism that made neural networks more efficient, leading to the birth of the Transformer neural network.  
    D. None of the above.

2. Which of these equations do you think should hold for a good word embedding?

    **A.** $e_{boy} - e_{girl} \approx e_{brother} - e_{sister}$  
    B. $e_{boy} - e_{girl} \approx e_{sister} - e_{brother}$  
    **C.** $e_{boy} - e_{brother} \approx e_{girl} - e_{sister}$  
    D. $e_{boy} - e_{brother} \approx e_{sister} - e_{girl}$

3. T/F: Suppose you learn a word embedding for vocabulary of 10000 words. Then the embedding vectors should be 10000 dimensional, to capture the full range of variation and meaning in those words.

    **False**

### Quiz 3.4

1. Word2Vec utilizes two architectures: Skip-gram and CBOW (continuous bag-of-words), please match the figures:

    A. Skip-gram:  

    ![img_4.png](img_4.png)

    B. CBOW:  

    ![img_5.png](img_5.png)

2. The table below lists several co-occurrence probabilities given words “ice” and “steam” and their ratios based on statistics from a large corpus with GloVe.

    Below is the table of co-occurrence probabilities for the words “ice” and “steam” with several context words, along with their ratios:

    | $w_k$                          | solid    | gas      | water    | fashion   |
    |--------------------------------|----------|----------|----------|-----------|
    | $p_1 = P(w_k\,|\,\text{ice})$  | 0.00019  | 0.000066 | 0.003    | 0.000017  |
    | $p_2 = P(w_k\,|\,\text{steam})$| 0.000022 | 0.00078  | 0.0022   | 0.000018  |
    | $p_1/p_2$                      | 8.9      | 0.085    | 1.36     | 0.96      |

    其中，$p_{ij} = \mathrm{P}(w_j\,|\,w_i)$ 表示在语料库中以 $w_i$ 为中心词生成上下文词 $w_j$ 的条件概率。

    请根据共现概率比值 $p_1/p_2$，为单词 $w_k$ 选择最合适的描述：

    - **solid** 是与 “ice” 相关但与 “steam” 无关的词（$p_1/p_2 \gg 1$）
    - **water** 是与 “ice” 和 “steam” 都相关的词（$p_1/p_2 \approx 1$，且概率都较大）
    - **fashion** 是与 “ice” 和 “steam” 都无关的词（$p_1/p_2 \approx 1$，且概率都很小）
    - **gas** 是与 “steam” 相关但与 “ice” 无关的词（$p_1/p_2 \ll 1$）

    对应填空：

    - <u>solid</u> is related to “ice” but unrelated to “steam”
    - <u>water</u> is related to both “ice” and “steam”
    - <u>fashion</u> is unrelated to both “ice” and “steam”
    - <u>gas</u> is related to “steam” but unrelated to “ice”

3. What is a bag of words (BOW) representation?

    A. A representation where text is represented as the set of its words, disregarding word order but considering grammar.  
    **B. A representation where text is represented as the set of its words, disregarding grammar and even word order but keeping multiplicity.**  
    C. A representation where text is represented as the ordered list of its words, disregarding grammar but keeping multiplicity.  
    D. None of above.

4. In the word2vec algorithm, you estimate $P(t\,|\,c)$, where $t$ is the target word and $c$ is a context word. How are $t$ and $c$ chosen from the training set? Please choose the best answer.

    A. $c$ is a sequence of several words immediately before $t$.  
    B. $c$ is the one word that comes immediately before $t$.  
    **C. $c$ and $t$ are chosen to be nearby words.**  
    D. $c$ is the sequence of all the words in the sentence before $t$.

### Quiz 3.5

1. What is the primary difference between "n-gram" models and neural language models?

    A. N-gram models use deep learning techniques  
    B. N-gram models are more efficient for large datasets  
    C. Neural models focus on individual word frequencies  
    **D. Neural language models may learn similar representations for semantically similar words**

2. In neural language modeling, if we want to estimate $P(\text{``this is a cat''})$, we should:

    A. Estimate $P(\text{this}) \cdot P(\text{is}) \cdot P(\text{a}) \cdot P(\text{cat})$  
    B. Estimate $P(\text{this} \mid \text{is}) \cdot P(\text{is} \mid \text{a}) \cdot P(\text{a} \mid \text{cat})$  
    **C. Estimate $P(\text{this} \mid \text{START}) \cdot P(\text{is} \mid \text{this}) \cdot P(\text{a} \mid \text{is}) \cdot P(\text{cat} \mid \text{a})$**  
    D. None of above

3. What is the primary advantage of Recurrent Neural Networks (RNNs) in handling sequential data?

    A. They don't require any training  
    B. They require less memory  
    **C. They can capture context and dependencies in sequential data**  
    D. They are faster than other neural network architectures

4. You are training the RNN language model shown in the figure:  

    ![img_6.png](img_6.png)  

    where $w_1$, $w_2$, $w_3$ refer to the weights.

    At the $n$-th time step, what is the RNN doing? Choose the best answer.

    A. Estimating $P(w_1, w_2, \ldots, w_n)$  
    B. Estimating $P(w_{n-1} \mid w_1, w_2, \ldots, w_{n-1})$  
    C. Estimating $P(w_{n-1})$  
    **D. Estimating $P(w_n \mid w_1, w_2, \ldots, w_{n-1})$**

### Quiz 3.6

1. In a Recurrent Neural Network, what is the purpose of the hidden state?

    A. To reduce the dimensionality of the input data  
    **B. To store information from previous time steps**  
    C. To introduce non-linearity  
    D. To update weights

2. In RNN, what is the purpose of the Softmax activation function?

    **A. To convert input values into probabilities**  
    B. To introduce non-linearity  
    C. To update weights  
    D. To reduce the dimensionality of the input data

3. Which layer type is responsible for making final predictions in an RNN?

    A. Input layer  
    **B. Output layer**  
    C. Activation layer  
    D. Hidden layer

4. T/F: The main and most important feature of RNN is its hidden state, which remembers some information about a sequence.

    **True**

### Quiz 3.7

1. What is 'gradient' when we are talking about RNN?

    **A. A gradient is a partial derivative with respect to its inputs**  
    B. It is how RNN calls its features  
    C. A parameter that can help you improve the algorithm's accuracy  
    D. The most important step of RNN algorithm

2. What is the purpose of the BackPropagation Through Time (BPTT) algorithm in RNN training?

    A. To adjust the learning rate during training  
    **B. To compute the gradients and update the network's parameters**  
    C. To prevent overfitting by regularizing the model  
    D. None of the above

3. You have a pet dog whose mood is heavily dependent on the current and past few days' weather. You've collected data for the past 365 days on the weather, which you represent as a sequence as $x^{\langle 1 \rangle}, \ldots, x^{\langle 365 \rangle}$. You’ve also collected data on your dog's mood, which you represent as $y^{\langle 1 \rangle}, \ldots, y^{\langle 365 \rangle}$. If you'd like to build a model to map from $x \rightarrow y$, you will choose:

    A. RNN with backpropagation, because this allows backpropagation to compute more accurate gradients.  
    **B. Unidirectional RNN, because the value of $y^{\langle t \rangle}$ depends only on $x^{\langle 1 \rangle}, \ldots, x^{\langle t \rangle}$ but not on $x^{\langle t+1 \rangle}, \ldots, x^{\langle 365 \rangle}$**  
    C. RNN with backpropagation, because this allows the prediction of mood on day $t$ to take into account more information.  
    D. Unidirectional RNN, because the value of $y^{\langle t \rangle}$ depends only on $x^{\langle t \rangle}$ and not other days' weather.

### Quiz 3.8

1. What does the term "Vanishing Gradients" refer to in the context of RNNs?

    A. The overfitting of RNN models  
    B. The vanishing of the loss function during training  
    **C. The phenomenon where gradients become too small during training, hindering learning in deep networks**  
    D. The rapid convergence of RNN training

2. What problem does the "Vanishing Gradient" cause in RNN?

    A. Gradient explosion  
    **B. Slow training**  
    C. Rapid convergence  
    D. Weight initialization

3. You are training an RNN, and find that your weights and activations are all taking on the value of NaN ("Not a Number"). Which of these is the most likely cause of this problem?

    A. ReLU activation function $g(\cdot)$ used to compute $g(z)$, where $z$ is too large.  
    B. Sigmoid activation function $g(\cdot)$ used to compute $g(z)$, where $z$ is too large.  
    **C. Exploding gradient**  
    D. Vanishing gradient

4. T/F: Saturated activation functions can be one of the causes of the vanishing gradient problem, because when these functions saturate (output values approach 0 or 1), the gradients become very small.

    **True**

## Lecture 4

### Quiz 4.1

1. Which of the following approaches could be used to solve the vanishing gradient problem of RNN?

    A. Using ReLU activation function  
    B. Long-Short Term Memory (LSTM) architecture  
    C. Gating  
    **D. All of the above**

2. Which of the following options about LSTM is true?

    A. LSTM networks are an extension for recurrent neural networks, which basically extends their memory. Therefore it is not recommended to use it, unless you are using a small Dataset.  
    B. LSTM networks are an extension for recurrent neural networks, which basically shorten their memory.  
    C. It is well suited to learn from important experiences that have very low time lags in between.  
    **D. LSTM networks are an extension for recurrent neural networks, which basically extends their memory. Therefore it is well suited to learn from important experiences that have very long time lags in between.**

3. Which type of gating mechanism is used in Long Short-Term Memory (LSTM) networks?

    A. Memory gate and update gate  
    B. Reset gate and update gate  
    **C. Forget gate, input gate, and output gate**  
    D. Hidden gate and output gate

4. What is the role of the "Forget gate" in Long Short-Term Memory (LSTM) networks?

    **A. Decides what information to forget from the previous time step**  
    B. Computes the output of the LSTM  
    C. Determines which information to keep in the cell state  
    D. Updates the cell state with new information

### Quiz 4.2

1. What role do gates with sigmoid activation functions play in Long Short-Term Memory (LSTM) networks?

    a. Control the depth of the network  
    **b. Regulate the cell state update**  
    c. Determine the output of the LSTM  
    d. Enable parallel processing

2. What is the main function of the Input Gate in LSTM networks?

    a. Decide which information to forget  
    b. Regulate the network's output  
    c. Control the flow of information  
    **d. Determine the cell state's update**

3. What is the primary advantage of Gated Recurrent Unit (GRU) RNNs over traditional LSTMs?

    a. Greater memory capacity  
    **b. Simplicity and reduced complexity**  
    c. Faster training times  
    d. Improved parallelism

4. Which of the following statements are INCORRECT about RNN/LSTM/GRU?

    a. Recurrent neural networks can handle a sequence of arbitrary length, while feedforward neural networks can not.  
    b. GRU is computationally more efficient than LSTMs if the hidden dimension size for LSTM and GRU are the same.  
    **c. Gradient clipping is an effective way of solving the vanishing gradient problem.**  
    d. Training recurrent neural networks is hard because of vanishing and exploding gradient problems.

### Quiz 4.3

1. You want to train a neural network to predict the next 30 daily prices using the previous 30 daily prices as inputs. Which model selection and explanation make the most sense?

    a. A single one-directional RNN because it considers the order of the prices, and the output length is the same as the input length.  
    b. A fully connected deep feed-forward network because it considers all input prices in the hidden layers to make the best decision.  
    c. A bidirectional RNN because the prediction benefits from future labels.  
    **d. Unidirectional RNN architecture because it can generate a sequence of future prices based on all input historical prices.**

2. What is the primary application of RNNs in natural language processing (NLP)?

    **a. Machine translation**  
    b. Speech synthesis  
    c. Image classification  
    d. Object detection

3. What kind of Sequential Output process does the following figure show?

    ![img_7.png](img_7.png)

    a. Speech Recognition  
    b. Machine Translation  
    **c. POS Tagging**  
    d. None of above

### Quiz 4.4

1. What is the primary purpose of Sequence-to-Sequence (Seq2Seq) models in deep learning?

    a. Text classification  
    **b. Handling tasks that involve variable-length sequences, such as machine translation and text summarization**  
    c. Reinforcement learning  
    d. None of above

2. In the context of machine translation, what role does the encoder in a Seq2Seq model play?

    a. Translates the source sentence to the target language directly  
    b. Regularizes the translation to make it smoother  
    **c. Compresses the source sentence into a fixed-size context vector**  
    d. Generates the final translated sentence based on the context vector

3. What is the primary architectural limitation of traditional Seq2Seq models based on recurrent neural networks (RNNs)?

    a. Excessive use of memory  
    b. Overfitting to training data  
    c. Inability to handle variable-length sequences  
    **d. Lack of parallelism in training and inference**

4. In the context of Seq2Seq, what challenge arises when relying solely on the final hidden state of the encoder to represent the entire input sequence?

    **a. Loss of sequential information from the beginning of the sequence**  
    b. Gradient instability during backpropagation  
    c. Overfitting due to excessive reliance on one state  
    d. High computational overhead

### Quiz 4.5

1. For an image recognition problem, which neural network architecture would be better suited to solve the problem?

    a. Multilayer Perceptron  
    **b. Convolutional Neural Network**  
    c. Single-Layer Perceptron  
    d. None of the above

2. Which of the following options regarding Image Classification with CNN is correct?

    a. We must use "fully connected" neural networks in image processing.  
    b. Patterns are much smaller than the whole image, so we should consider the whole image in each receptive field.  
    c. Every neuron has to see the whole image.  
    **d. Each receptive field has a set of neurons.**

3. Suppose we have an image whose size is 4x4, while we set the kernel size to 3x3, and stride = 2. How many receptive fields do we need at least to cover the whole image?

    **Answer:** 4

### Quiz 4.6

1. Answer the following questions based on the image below:

    ![img_8.png](img_8.png)

    1. What value would be in place of the question mark?  
       **Answer: 4**
    2. What is the stride value in this case?  
       **Answer: 1**

2. Given an image of size 27 x 27 and a filter of size 5 x 5 with a stride of 2 and no padding. What is the output size?

    - **Calculation formula:**  
      $$
      \text{Output size} = \left\lfloor \frac{\text{Input size} - \text{Filter size}}{\text{Stride}} \right\rfloor + 1
      $$
      $$
      = \left\lfloor \frac{27 - 5}{2} \right\rfloor + 1 = \left\lfloor \frac{22}{2} \right\rfloor + 1 = 11 + 1 = 12
      $$

    a. 23 x 23  
    b. 13 x 13  
    c. 27 x 27  
    **d. 12 x 12**

3. Given the 5 x 5 input image and 3 x 3 filter, suppose stride = 1, which is the correct output?

**5 x 5 input image:**  

![img_9.png](img_9.png)

**3 x 3 filter:**  

![img_10.png](img_10.png)

A.
```
 0 -1  1
-1  0 -1
-2  0  1
```
B.
```
 0 -2  1
 1  0 -1
-2  0  1
```
C.
```
 0 -2  1
-1  0 -1
-2  0 -1
```
**D**.
```
 0 -2 -1
-1  0  1
-2  0 -1
```

### Quiz 4.7

1. As shown in the figure, the Max Pooling filter operates on each feature map using a 2x2 filter, and the colors represent the corresponding sampling objects and outputs.

    Which is the correct output of the Max Pooling filter?

    ![img_11.png](img_11.png)
    
    ![img_12.png](img_12.png)

2. Which of the following options are the benefits of using convolutional layers instead of fully connected ones for visual tasks?

    **a. Have much fewer parameters**  
    b. Have deeper network  
    **c. Uses spatial context**  
    **d. Translation invariance**

3. Which of the following statements about Pooling layers are true?

    **a. It is applied after convolution operations.**  
    b. The more pooling layers the better because the number of parameters can be greatly reduced.  
    c. We need pooling layers since the number of hidden layers required to learn the complex relations present in the image would be large.  
    **d. It is called subsampling or downsampling.**  
    **e. It reduces the dimensionality of each feature map by retaining the most important information.**

### Quiz 4.8

1. What type of learning task is GANs often used for?

    **a. Image generation**  
    b. Reinforcement learning  
    c. Text summarization  
    d. Classification

2. What does the generator model aim to do in GANs?

    **a. Generate new data samples**  
    b. Classify real data  
    c. Create fake data  
    d. Compete with the discriminator

3. What does the discriminator model in GANs try to do?

    a. Create latent vectors  
    b. Fool the generator  
    **c. Identify real data samples**  
    d. Generate new data

4. What is the primary goal of the generator model in GANs?

    a. To compete with the discriminator  
    **b. To learn the underlying data distribution**  
    c. To classify real data samples  
    d. To perform feature extraction

## Lecture 5

### Quiz 5.1

1. What does the term "adversarial" in GANs refer to?

    A. The classification of real data  
    B. The generation of new data  
    **C. The competitive process between two sub-models**  
    D. The architecture of the generator

2. What type of learning task is generative modelling in GANs?

    A. Supervised learning  
    B. Semi-supervised learning  
    C. Reinforcement learning  
    **D. Unsupervised learning**

3. What is the latent space in GANs?

    **A. The space of random input vectors**  
    B. The space where real data resides  
    C. The space of hidden layers in the generator  
    D. The space of real data labels

4. In GANs, what happens when the discriminator becomes unable to distinguish real from fake samples?

    A. Discriminator becomes more efficient  
    B. Training stops  
    C. Discriminator becomes the generator  
    **D. The generator updates its weights**

### Quiz 5.2

1. What is mode collapse in GANs?

    A. When the generator and discriminator achieve perfect equilibrium  
    B. When the GAN training process becomes unstable  
    C. When the discriminator fails to distinguish between real and generated samples  
    **D. When the generator produces limited variations of samples**

2. Which of the following statements about LSGAN is not true?

    A. LSGANs perform more stable during the learning process.  
    B. LSGANs adopt the least squares loss function for the discriminator.  
    **C. LSGANs use the cross-entropy loss which avoids the vanishing gradients problem during the learning process.**  
    D. LSGANs are able to generate higher quality images than regular GANs.

3. Which of the following is true about Non-Saturating GAN Loss?

    **A. The generator seeks to maximize the probability of images being real, instead of minimizing the probability of an image being fake.**  
    B. The non-saturating loss will saturate when the input is relatively small.  
    **C. The non-saturating loss will saturate when the input is relatively large.**  
    **D. It avoids generator saturation through a more stable weight update mechanism.**

### Quiz 5.3

1. What is the purpose of the evaluation metrics in GANs?

    **A. To measure the quality and diversity of generated samples**  
    B. To adjust the learning rate during training  
    C. To adjust the weights and biases of the generator  
    D. None of the above

2. Which of the following statements about FID is true?

    **A. Lower FID values mean better image quality and diversity.**  
    **B. It measures the distance between fake images distributions and real images distributions.**  
    **C. It uses the Inception network to extract features from an intermediate layer.**

3. What is the purpose of the Wasserstein distance in Wasserstein GANs?

    **A. To measure the similarity between real and generated samples**  
    B. To adjust the learning rate during training  
    C. To adjust the weights and biases of the generator  
    D. None of the above

4. What is the objective of Wasserstein GAN (WGAN)?

    A. To perform image super-resolution  
    **B. To mitigate stability issues in GANs**  
    C. To introduce cycle consistency  
    D. To create diverse art content

### Quiz 5.4

### Quiz 5.5

## Lecture 6

### Quiz 6.1

### Quiz 6.2

### Quiz 6.3

### Quiz 6.4

## Lecture 7

### Quiz 7.1

### Quiz 7.2

### Quiz 7.3

### Quiz 7.4

## Lecture 8

### Quiz 8.1

### Quiz 8.2

### Quiz 8.3

### Quiz 8.4

## Lecture 9

### Quiz 9.1

### Quiz 9.2

### Quiz 9.3

### Quiz 9.4