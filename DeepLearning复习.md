# Deep Learning Review



## 1 Introduction

### Basics

​	**DL, ML, AI**

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425030731007.png" alt="image-20210425030731007" style="zoom:70%;" />

​	**Renaissance of NN**

​		More data (social media); Faster hardware (GPU, cloud computing); Improved algorithms

​	**Domains**

​		audios, images, videos, sequences, graphs, sets, probability distributions

​	**Models and Algorithms**

​		feed-forward NN (FFNN/DNN); convolutional NN (CNN); recurrent NN (RNN)

​		generative-adversarial networks (GANs); auto-encoders (AEs); variational-AEs (VAEs)

#### 	Training Regimes (制度)

​		supervised, unsupervised learning; reinforcement learning; embeddings, k-shot learning;

​		adversarial training; density estimation; generative models

#### 	Techniques

- Weight initialization and symmetry-breaking: orthogonal weight initialization; Dropout

- Optimization methods: momentum; RMSProp; AdaGrad; Adam

- Feature normalization: batchNorm; layerNorm; groupNorm

- Weight sharing: convolution, recurrence

- Variational approximation: VAEs, NFs (Normalizing Flows)

- Gradient propagation: activation functions, residual connections

- Transfer learning and domain adaptation: pretraining & fine-tuning, adversarial domain adaptation

- Attention: soft, hard, multi-head *(which inputs or activations are most important?)*

---

### Artificial Neural Networks (ANN)

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425032557666.png" alt="image-20210425032557666" style="zoom:80%;" />

​		e.g. linear regression can be considered as an NN with: $\hat{y}=\Sigma^{m}_{i=1}x_iy_i=\mathbf{x}^T\mathbf{w}$

---

### Some Linear Algebra

​		Matrices multiplication = **outer product** = $\mathbf{A}\times\mathbf{B}$

​		Matrices **inner product** = $\mathbf{A}\cdot\mathbf{B}$

​		Matrices **Hadamard product** = <u>element-wise</u> product = $\mathbf{A}\bigodot\mathbf{B}$

​		**Jacobian**:

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425035022283.png" alt="image-20210425035022283" style="zoom:60%;" />

​		**Hessian**:

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425035048693.png" alt="image-20210425035048693" style="zoom:60%;" />

---

### Some Probability Theory

PMF: probability mass function (for finite or countable sample space)

- e.g. 6-sided fair die

PDF: probability density function (for uncountable sample space)

- uniformly-distribution

- Gaussian (normal) distribution:

  <img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425044221706.png" alt="image-20210425044221706" style="zoom:60%;" />

Joint probability distributions: 

​		compute the marginal distributions: <img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425044416908.png" alt="image-20210425044416908" style="zoom:60%;" />

**Conditional probability distributions**: $P(x|y)$

​		relations: $P(x|y)P(y)=P(x,y)$;  $P(x|y,z)P(y|z)=P(x,y|z)$

​		<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425044739714.png" alt="image-20210425044739714" style="zoom:60%;" />

​		conditional independence: $P(x,y|z)=P(x|z)P(y|z)\rarr P(x|y,z)=P(x|z)$. (x and y are conditionally independent given z)

#### 	Bayes' rule

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425045425276.png" alt="image-20210425045425276" style="zoom:60%;" />

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425045514738.png" alt="image-20210425045514738" style="zoom:60%;" />

​		example:

​		<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425050138773.png" alt="image-20210425050138773" style="zoom:70%;" />

	#### 	Maximum Likelihood Estimation (MLE)

​		MLE: the value of a latent variable is estimated as the one that makes the observed data as likely as possible

​		The probability of each $h_i$ given $b$: $P(h_i|b)=b^{h_i}(1-b)^{1-h_i}$. We seek to maximize this probability by optimizing b, while it is easier to optimize the **log-likelihood**:
$$
argmax_b P(h_1,...,h_n|b)=argmax_b logP(h_1,...,h_n|b)
\\
logP(h_1,...,h_n|b)=log\prod_{i=1}^{n}P(h_i|b)
$$

#### 	Linear-Gaussian Model

​				<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425203306354.png" alt="image-20210425203306354" style="zoom:60%;" />

​		Then Y is a normal/Gaussian random variable; the expected value of Y is $x^Tw$; the variance of Y is constant ($\sigma^2$) for all x. i.e.
$$
P(y|\mathbf{w},\mathbf{x})=N(y;\mathbf{x}^T\mathbf{w},\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{(y-\mathbf{x}^T\mathbf{w})^2}{2\sigma^2})
$$
​		MLE for $\mathbf{w}$: <img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425204157013.png" alt="image-20210425204157013" style="zoom:60%;" />

​		MLE for $\sigma^2$: <img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425204324148.png" alt="image-20210425204324148" style="zoom:60%;" />

---



## 2 Shallow Architectures

### Linear Regression

> aka 2-layer NN

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425035918149.png" alt="image-20210425035918149" style="zoom:70%;" />

Loss function: MSE = $\frac{1}{2n}\Sigma^n_{i=1}(x(i)^T\mathbf{w}-y(i))^2$

Since $f_{MSE}$ is convex, we are guaranteed to get global minimum by setting the gradient to 0:

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425040640820.png" alt="image-20210425040640820" style="zoom:80%;" />

MSE gradient: <img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425041524829.png" alt="image-20210425041524829" style="zoom:60%;" />

加入bias term后: $\hat{y}=\mathbf{x}^T\mathbf{w}+b$

#### 	Gradient Descent

> **Numerical solution**: need to iterate many times to approximate the optimal value

​		Gradient update: <img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425041743498.png" alt="image-20210425041743498" style="zoom:60%;" /> with learning rate $\epsilon$, update until convergence

#### 	Stochastic Gradient Descent (SGD)

​		*idea: using gradient descent updates the weights after scanning the entire dataset, which is slow*

​		Procedure: REPEAT: Randomly sample a small mini-batch of training examples, and estimate the gradient on just the mini-batch. Then update weights based on mini-batch gradient estimate.

​		**Epoch**: a single pass through the entire training set

​		trick: the learning rates need to be <u>annealed</u> (reduced slowly by time) to guarantee convergence, one common schedule is <u>exponential decay</u>.



---

### Logistic & Softmax Regression

Two main <u>supervised learning</u> cases:

* **Regression**: predict any real number
* **Classification**: choose from a finite set

Logistic regression is used primarily for classification (for binary classification)

#### 	Sigmoid Function

​		logistic sigmoid function $\sigma$:

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210426221323099.png" alt="image-20210426221323099" style="zoom:60%;" />

​		It has some nice properties: $\sigma(-z)=1-\sigma(z)$,  $\sigma'(z)=\sigma(z)(1-\sigma(z))$

The prediction of logistic regression: $\hat{y}=\sigma(\mathbf{x^Tw})$, they are forced to be in (0,1).

The output real values are seen as <u>probabilities</u> of how confident we are in the prediction.

Gradient descent for logistic regression:

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210426222552264.png" alt="image-20210426222552264" style="zoom:70%;" />

However, because of the attenuated (衰减) gradient that "when gradient is 0 then no learning occurs", we use logarithms in $f_{MSE}$.

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210426222855240.png" alt="image-20210426222855240" style="zoom:60%;" />

Gradient descent for logistic regression with log-loss:

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210426223352295.png" alt="image-20210426223352295" style="zoom:70%;" />

#### 	Linear Regression vs Logistic Regression

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210426223603872.png" alt="image-20210426223603872" style="zoom:80%;" />

---

### Sofmax Regression

> aka multinomial logistic regression (for multiple classes)

One-hot encoding.

Output as the probabilistic of every classes.

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210426231545744.png" alt="image-20210426231545744" style="zoom:60%;" />

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210426231815226.png" alt="image-20210426231815226" style="zoom:60%;" />

#### 	Cross-entropy Loss

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210426231924473.png" alt="image-20210426231924473" style="zoom:70%;" />

Gradient descent for Softmax regression:

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210426233119898.png" alt="image-20210426233119898" style="zoom:60%;" />

---



## 3 Shallow Optimization

### $L_2$ Regularization

​	Regularization: any practice designed to improve the machine's ability to <u>generalize</u> to new data

​	One simplest regularization technique: <u>penalize</u> large weights in the cost function

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210425205558186.png" alt="image-20210425205558186" style="zoom:60%;" />

---

### Hyperparameter Tuning

​	Hyperparameters: learning rate $\epsilon$, batch size $\tilde{n}$, regularization strength $\alpha$

	#### 	Train, Validation, Test

​		Training: optimization of parameters (0.7-0.8)

​		Validation: tuning of hyperparameters (0.05-0.1)

​		Testing: evaluation of the final model (0.05-0.1)

#### 	Cross-validation

> idea: when using smaller datasets, this can use all data for training

​		Suppose the known best hyperparameters $h^*$, partition data into k equal folds.

​		Over k iterations, train on (k-1) folds and test on the remaining fold.

​		Compute the average accuracy over the k testing folds.

#### 	Double Cross-validation

> idea: to find the best hyperparameters $h^*$ for each fold in CV

​		i.e. use a CV inside another CV to determine the best hyperparameters for the kth fold.

---

### Why Optimization can Go Wrong?

- Presence of multiple local minima
- Bad initialization of the weights
- Learning rate too small or too big

---

### Convexity

​	Property of a convex function f: the second derivative of f is always non-negative. For higher-dimensional f, consider the **Hessian** of f.

​	With convex functions, <u>every local minimum is also a global minimum</u>, which are ideal for conducting gradient descent

**Positive Semi-Definite (PSD)**

> PSD is the matrix analog (类比) of being "non-negative"

​		A real symmetric matrix **A** is PSD if: all eigenvalues $\ge$0, and for every vector **v**: $\mathbf{v^TAv}\ge0$

---

### Feature Transformations

We can "spherize" the input features using a **whitening transformation**, which makes the auto-covariance matrix equal the identity matrix **I**.

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210426213554751.png" alt="image-20210426213554751" style="zoom:70%;" />

Whitening transformation is a classical ML technique, which time cost is high, but it later inspired **batch normalization** and **concept whitening**.

---

### Newton's Method

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210426214304627.png" alt="image-20210426214304627" style="zoom:70%;" />

It's an iterative root-finding method, which requires computation of **H**. The Hessian can be huge, which makes the pure form of Newton's method impractical for DL. But it later inspired **Adam** optimizer.

---



## 4 Basics of Deep Models

### Activation Functions

Logistic sigmoid, Tanh, ReLU, ...

*The non-differentiability at 0 of ReLU doesn't matter since the gradient is often non-zero.*

**Crucial role of non-linearity**: if not:

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210427013331735.png" alt="image-20210427013331735" style="zoom:70%;" />

### Forward Propagation

It produces all the intermediary $(h,z)$ and final ($\hat{y}$) network outputs

It results in a computational graph: <img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210427013620447.png" alt="image-20210427013620447" style="zoom:60%;" />

### Training Neural Networks

To find good values for the weights and bias terms automatically: Gradient Descent.

### Back Propagation

It produces the gradient terms of all the weight matrices and bias vectors. It requires forward propagation to be conducted first.

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210427014552163.png" alt="image-20210427014552163" style="zoom:80%;" />

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210427014642397.png" alt="image-20210427014642397" style="zoom:50%;" />

### Weight Initialization

Many cases discussed in class, according to these formulas:

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210427021147465.png" alt="image-20210427021147465" style="zoom:90%;" />

One common approach to initialize:

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210427021230609.png" alt="image-20210427021230609" style="zoom:70%;" />

---



## 5 Deep Learning Models

**Why deep learning?**

- each layer can represent increasingly abstract representations of the input
- the hidden units can represent the content of the input in a compact way

**Difficulty in deep learning**

- Vanishing gradients
- Exploding gradients

---

### Convolution

convolution, filter, pooling, padding, dilate (扩张): <img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210427185304108.png" alt="image-20210427185304108" style="zoom:70%;" />

multiple channels: <img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210427190138837.png" alt="image-20210427190138837" style="zoom:70%;" />

Universal Function Approximation Theorem (泛函逼近定理): 

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210427192915212.png" alt="image-20210427192915212" style="zoom:80%;" />

#### 	Convolutional Neural Networks (CNNs)

​		CNN architecture: multiple conv layers; pooling layers between some of the conv layers; FC layers at the end; non-linear activation functions between each layer.

​		Trend: less pooling, more conv layers, residual connections.

​		Consider convolution as a linear function:

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210427194058282.png" alt="image-20210427194058282" style="zoom:70%;" />

#### 	Receptive Fields

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210428014815235.png" alt="image-20210428014815235" style="zoom:70%;" />

​			Dilated convolution: <img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210428015115697.png" alt="image-20210428015115697" style="zoom:60%;" />

---

### Recurrent Neural Networks (RNNs)

> *idea*: there're problems that the input length is not fixed or unknown, i.e. variable-length

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210428030731353.png" alt="image-20210428030731353" style="zoom:70%;" />

The same weights are used in all time-steps, and we need to sum their contributions to the gradient of the loss function $J$. Moreover, we need to sum the loss value $J_t$ at every timestep.

Difficulty in training RNNs: exploding gradient which forces us to have a small learning rate; vanishing gradient which also makes learning very slow.

One strategy for preventing vanishing and exploding gradients is to use skip connections, which are used in **LSTM** and **GRU RNN**s.

#### 	Long Short-term Memory (LSTM) NN

​	Three gates: forget(f), input(i), output(o)

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210428032438833.png" alt="image-20210428032438833" style="zoom:70%;" />

​	There're 4 weight matrices and 4 bias vectors in total.

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210428032711689.png" alt="image-20210428032711689" style="zoom:70%;" />

​					This is sometimes called a *skip connection*.

#### 	Bi-directional RNNs

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210428033403082.png" alt="image-20210428033403082" style="zoom:70%;" />

---



## 6 Regularization

If there's a large divergence between training acc and testing acc, (i.e. overfitting), then try regularizing the model by:

- Increasing L1, L2 regularization strength
- Adding/increasing dropout
- Reducing number of epochs
- Synthesizing more training examples with label-preserving transformations

**L2 regularization**: the sum of its squared entries. It encourages all the entries to be small.

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210428033924318.png" alt="image-20210428033924318" style="zoom:60%;" />

​				= weight decay = Gaussian noise augmentation (for 2-layer linear NN, L2 regularization is also equivalent to augmenting the training set by adding element-wise Gaussian noise to each input)

**L1 regularization**: the sum of absolute values of each entry. It encourages some parameters to be exactly 0, which encourages sparse feature representations.

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210428033938310.png" alt="image-20210428033938310" style="zoom:60%;" />



---



## 7 Normalization

> *Idea*: it can be helpful to put every feature onto the same scale.

For features in a <u>finite range</u>, try rescaling to [0,1] or [-1,1].

For features in <u>infinite range</u>, try subtracting the mean and dividing by standard deviation.

### Data Normalization

Common methods:

- mapping each feature into a fixed range ([0,1], [-1,1] ...)
- z-scoring each feature (s.t. the mean and stddev are 0 and 1)
- whitening all features jointly (to have 0 mean and I-covariance)

Two ways to achieve this:

- Learn transformation (parameters) on training data
- Learn transformation (parameters) on training and testing data <u>separately</u>

### Data Augmentation

Data augmentation is the creation of new examples based on existing ones.

This is a method to prevent overfitting.

Common methods:

- adding noise to existing examples (e.g. Gaussian).
- geometric transformations (e.g. flip, rotate, translate (挪动目标在图像的位置)).

---



## 8 Learning Representations

### Unsupervised Pre-training

​		*intuition*: a good representation captures the essence of the raw input data

​		We can compress the data into a smaller representation, and uncompress it to reconstruct the original data.

		#### 			Auto-encoders

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210427030048967.png" alt="image-20210427030048967" style="zoom:80%;" />

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210427030127916.png" alt="image-20210427030127916" style="zoom:70%;" />

​			After training, $W^{(1)}$ and $b^{(1)}$ have hopefully learned to encode x into a representation that is useful for a variety of classification/regression problems.

​			After we combine the AE onto another network, we don't want the incoming small dataset mess up AE's weights, so we often use a small learning rate, which is **fine-tuning**.

### 	Supervised Pre-training

​		Strategy:

1. Pre-train a NN on a large dataset (e.g. ImageNet) for a general-purpose image recognition task.
2. Chop off the final layers
3. Add a secondary network in place of the deleted layers and train it for the new task

### Multi-task Learning (MTL)

> *idea*: we can train a model to solve multiple related tasks to learn a general hidden representation

<img src="D:%5CTypora%5C%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0%5Cimage-20210427035136355.png" alt="image-20210427035136355" style="zoom:80%;" />

​	With MTL, the loss function consists of those multiple components.

