# Generalized Linear Models

## The exponential family

&emsp;&emsp;指数家族分布形式如下：
$$
p(y;\eta) = b(y) \exp {(\eta^TT(y)-a(\eta))}
$$
&emsp;&emsp;大部分时候，$T(y) = y$ 。$\eta$ 为自然参数 (natural parameter)， $a(\eta)$ 为对数分割函数 (log partition function)，$b(y)$ 为基础测量 (base measure)。固定 $T、a、b$ 后，通过调整 $\eta$ 来调整分布，很多分布都可以转化成指数家族分布的形式。

## Bernoulli distribution

&emsp;&emsp;下面推导将伯努利分布转换为指数家族分布的形式。假设均值为 $\phi$ ，$y \in \{0,\ 1\}$ ，则：
$$
\begin{aligned}
p(y = 1;\phi) &= \phi \\
p(y = 0;\phi) &= 1 - \phi
\end{aligned}
$$
&emsp;&emsp;所以可以推出：
$$
p(y;\phi) = \phi ^y (1 - \phi)^{1 - y}
$$
&emsp;&emsp;所以有：
$$
\begin{aligned}
p(y;\phi) 
&= \exp {( \ln {(\phi ^y (1 - \phi)^{1 - y})} )} \\
&= \exp {(y \ln \phi +(1-y)\ln(1-\phi))} \\
&= \exp {(y\ln \frac {\phi} {1-\phi} +\ln (1-\phi))}
\end{aligned}
$$
&emsp;&emsp;因为 $\ln \frac {\phi} { 1 - \phi }$ 是一个常数，所以令 $\eta = \ln \frac {\phi} {1 - \phi}$ 有：
$$
\begin{aligned}
e^\eta &= \frac {\phi} {1 - \phi} \\
e^\eta &= (1 + e^\eta) \phi \\
\phi &= \frac {e^\eta} {1 + e^\eta} \\
&= \frac {1} {1+e^{-\eta}}
\end{aligned}
$$
&emsp;&emsp;则 $p(y;\phi)$ 可以转换为：
$$
\begin{aligned}
p(y;\phi) &= \exp (\eta^T y + \ln \frac {1} {1 + e^\eta})
\end{aligned}
$$
&emsp;&emsp;至此将该式转化为了指数家族分布的形式，其中：
$$
\begin{aligned}
b(y) &= 1 \\
T(y) &= y \\
a(\eta) &= - \ln \frac {1} {1 + e^\eta} \\
&=\ln (1 + e^\eta) \\
\end{aligned}
$$

## Gaussian distribution

&emsp;&emsp;回顾 Linear Regression，当时用高斯分布推导的时候，高斯分布的方差 $\sigma^2$ 并不会影响 $\boldsymbol \theta$ 的结果，所以在这里，不妨将其设为 $1$ 。所以，高斯分布如下：
$$
\begin{aligned}
p(y;\mu) &= \frac {1} {\sqrt{2 \pi}} \exp (- \frac {(y - \mu)^2} {2}) \\
&=\frac {1} {\sqrt{2 \pi}} \exp (- \frac {y^2} {2}) \exp (\mu y - \frac {1} {2} \mu^2) 
\end{aligned}
$$
&emsp;&emsp;至此，我们将高斯分布转换为了指数家族分布的形式，其中：
$$
\begin{aligned}
b(y) &= \frac {1} {\sqrt{2 \pi}} \exp (- \frac {y^2} {2}) \\
T(y) &= y \\
\eta &= \mu \\
a(\eta) &= \frac {1} {2} \mu^2 \\
&= \frac {1} {2} \eta^2 \\
\end{aligned}
$$
&emsp;&emsp;指数家族分布是广义线性模型，很多分布都可以转化为这种形式

## 构建 GLMs

&emsp;&emsp;首先需要进行三个假设：

$$
\begin{aligned}
1.\ \ &y|\boldsymbol x; \boldsymbol \theta \sim \text {ExponentialFamily}(\eta) \\
2.\ \ &\text {Given $\boldsymbol x$, our goal is to predict the expected value of $T(y)$. This means} \\
&\text{we would like learned hypothesis $h$ to satisfy $h(\boldsymbol x) = E[y|\boldsymbol x]$} \\
3.\ \ &\eta = \boldsymbol \theta^T \boldsymbol x \text{. (If $\eta$ is a vector-valued, then $\eta_i = \boldsymbol  \theta_i^T \boldsymbol x$}
\end{aligned}
$$
&emsp;&emsp;从这三个假设出发，我们来推导线性回归、Logistic Regression 的模型形式。

### Ordinary Least Squares(普通最小二乘法)

&emsp;&emsp;在线性回归中，我们假设了 $y$ 和 $\boldsymbol x$ 是严格的线性关系加高斯分布的随机噪声的结果，所以 $y|\theta;x \sim N(\mu,\sigma^2)$ ，在前面我们推到出高斯分布可以转化为指数家族分布，由假设 2和假设3 ，我们有：
$$
\begin{aligned}
h_{\boldsymbol \theta} (\boldsymbol x) &= E[y|\boldsymbol x]  \\
&=\mu \\
&=\eta \\
&= \boldsymbol \theta^T \boldsymbol x
\end{aligned}
$$

### Logistic Regression

&emsp;&emsp;根据前面推到的结果和假设2、假设3，我们有：
$$
\begin{aligned}
h_{\boldsymbol \theta} (\boldsymbol x) &= E[y|\boldsymbol x]  \\
&= \phi \\
&= \frac {1} {1 + e^{-\eta}} \\
&= \frac {1} {1 + e^{-\boldsymbol \theta^T \boldsymbol x}} \\
\end{aligned}
$$


### Softmax Regression

&emsp;&emsp;在分类问题中，当分类结果不是两个，而是 $k$ 个时，伯努利分布就不满足我们的需求了，这个时候我们可以假设 $y|\boldsymbol x; \boldsymbol \theta$ 服从多项式分布 (Multinomial distribution) ，我们可以使用 $\phi_1,\phi_2, \cdots,\phi_k$ 来分别表示取到某一个值的概率。因为 $\sum_{i=1}^k \phi_k = 1$ ，所以可以令：
$$
\phi_k = 1 - \sum_{i= 1}^{k-1} \phi_i
$$
&emsp;&emsp;定义如下计算，大括号中是一个逻辑表达式的结果：
$$
1\{ \text{True}\} = 1\\ 
1\{ \text{False}\} = 0​
$$
&emsp;&emsp;所以有：
$$
\begin{aligned}
p(y;\phi) = \prod_{i=1}^k\phi_i^{1\{y = i\}} 
\end{aligned}
$$
&emsp;&emsp;令 $T$ 为对 $y$ 进行的一个转换，转换结果为一个 $k-1$ 行的列向量，转换规则为第 $y$ 行为 $1$ ，其余全为零，若 $y = k$ ，则每一行都为 $0$ 。
$$
T(1) = 
\begin{bmatrix}
1 \\
0\\
0 \\
\vdots \\
0
\end{bmatrix}_{(k-1) \times 1} , \ 

T(2) = 
\begin{bmatrix}
0 \\
1\\
0 \\
\vdots \\
0
\end{bmatrix}_{(k-1) \times 1}  ,
\cdots,\ 

T(k-1) = 
\begin{bmatrix}
0\\
0\\
0 \\
\vdots \\
1
\end{bmatrix}_{(k-1) \times 1}  , \

T(k) = 
\begin{bmatrix}
0\\
0\\
0 \\
\vdots \\
0
\end{bmatrix}_{(k-1) \times 1}   
$$
&emsp;&emsp;则 $p(y;\phi)$ 可继续转化为：
$$
\begin{aligned}
p(y;\phi) 
&= \phi_1^{1\{y = 1\}} \phi_2^{1\{y = 2\}} \cdots\phi_{k-1}^{1\{y = k-1\}}\phi_k^{1 -\sum_{i=1}^{k-1}1\{y=i\}}  \\
&= \phi_1^{(T(y))_1}\phi_2^{(T(y))_2}\cdots\phi_{k-1}^{(T(y))_{k-1}}\phi_k^{1 - \sum_{i=1}^{k-1}(T(y))_i} \\
&=\exp((T(y))_1\ln \phi_1 + (T(y))_2\ln \phi_2 +\cdots + (T(y))_{k-1}\ln \phi_{k-1} +(1-\sum_{i=1}^{k-1} (T(y))_i)\ln \phi_k  ) \\
&=\exp ((T(y))_1 \ln \frac {\phi_1} {\phi_k} + \cdots + (T(y))_{k-1}\ln \frac {\phi_{k-1}} {\phi_k}+\ln \phi_k) \\
&= b(y)\exp (\eta^T T(y) - a(\eta))
\end{aligned}
$$

&emsp;&emsp;所以有：
$$
\begin{aligned}
b(y) &=1 \\
\eta &= \begin{bmatrix}
\ln \frac {\phi_1} {\phi_k} \\
\vdots \\
\ln \frac {\phi_{k-1}} {\phi_k} \\
\end{bmatrix} \\
a(\eta) &= - \ln (\phi_k)
\end{aligned}
$$
&emsp;&emsp;所以可以得到：
$$
\begin{aligned}
\eta_i &= \ln \frac {\phi_i} {\phi_k}  \\
\phi_k e^{\eta_i} &= \phi_i \\
\phi_k \sum_{i=1}^{k} e^{\eta_i} &=  \sum_{i=1}^{k} \phi_i \\
&=1 \\
\phi_k &= \frac {1} {\sum_{i=1}^{k} e^{\eta_i}} 
\end{aligned}
$$
&emsp;&emsp;将 $\phi_k$ 代回到 $\phi_k e^{\eta_i} = \phi_i $ 有：
$$
\phi_i =\frac {e^{\eta_i}} {\sum_{j=1}^{k} e^{\eta_j}}
$$
&emsp;&emsp;由假设 2 和假设 3 可以推出：
$$
\begin{aligned}
p(y=k|\boldsymbol x; \boldsymbol \theta) &= \phi_i \\
&= \frac {e^{\eta_i}} {\sum_{j=1}^{k} e^{\eta_j}}  \\
&= \frac {e^{\boldsymbol \theta_i ^ T \boldsymbol x}} {\sum_{j=1}^{k} e^{\boldsymbol \theta_j ^ T \boldsymbol x}}  \\
\end{aligned}
$$

$$
\begin{aligned}
h_{\boldsymbol \theta}(\boldsymbol x) &= E[T(y)| \boldsymbol x;\boldsymbol \theta] \\
&= 
\begin{bmatrix}
\phi_1 \\
\phi_2 \\
\vdots \\
\phi_{k-1}
\end{bmatrix} \\
&= 
\begin{bmatrix}
\frac {\exp({\boldsymbol \theta_{1} ^ T \boldsymbol x})} {\sum_{j=1}^{k} \exp({\boldsymbol \theta_j ^ T \boldsymbol x})} \\
\frac {\exp({\boldsymbol \theta_{2} ^ T \boldsymbol x})} {\sum_{j=1}^{k} \exp({\boldsymbol \theta_j ^ T \boldsymbol x})} \\
\vdots \\
\frac {\exp({\boldsymbol \theta_{k-1} ^ T \boldsymbol x})} {\sum_{j=1}^{k} \exp({\boldsymbol \theta_j ^ T \boldsymbol x})}
\end{bmatrix} \\
\end{aligned}
$$










