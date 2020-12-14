# Logistic Regression

## 基本公式推导

### 损失函数推导

&emsp;&emsp;Logistic Regression 解决的是分类的问题，我们首先来讨论一下二分类的问题，假设我们的分类标签为 $\{0, 1\}$ ，那么也就是我们 $y$ 的取值只有这两种，此时我们再使用线性回归就不太合适了，因为在线性回归中，取值偏大也会产生损失，例如 $\boldsymbol x^T \boldsymbol \theta$ 求出来一个远大于 $1$ 的数，此时我们已经可以分类出他是输入 $y = 1$ 的类别的，而且应该来说非常自信这一结果，但是线性回归会给予这样一个结果很大的损失，因为他远大于 $1$ ，所以我们可以想到将 $\boldsymbol x^T \boldsymbol \theta$ 转换到 $[0,1]$ 之间的数，我们可以使用 sigmoid 函数来进行这一转换，函数如下：
$$
g(z) = \frac {1} {1+e^{-z}}
$$
&emsp;&emsp;因此，我们的假设函数变为：
$$
\begin{aligned}
h_{\boldsymbol \theta} (\boldsymbol x) &=  g(\boldsymbol x^T\boldsymbol \theta) \\
& =\frac {1} {1 + e^{-\boldsymbol x^T \boldsymbol \theta}}
\end{aligned}
$$
&emsp;&emsp;假设在输入为 $\boldsymbol x$ 的条件下，输出 $y$ 服从伯努利分布，且：
$$
\begin{aligned}
p(y = 1|\boldsymbol x;\boldsymbol \theta) &= h_{\boldsymbol \theta} (\boldsymbol x) \\
p(y = 0|\boldsymbol x;\boldsymbol \theta) &=  1 -h_{\boldsymbol \theta} (\boldsymbol x) \\
p(y|\boldsymbol x;\boldsymbol \theta) &=(h_{\boldsymbol \theta} (\boldsymbol x))^{y}(1  - h_{\boldsymbol \theta} (\boldsymbol x))^{1-y}
\end{aligned}
$$
&emsp;&emsp;所以在这个条件下，采集得到训练集样本的概率的似然函数为：
$$
\begin{aligned}
L(\boldsymbol \theta) &= \prod_{i = 1}^mp(y^{(i)})   \\
& =\prod_{i=1}^m (h_{\boldsymbol \theta} (\boldsymbol x^{(i)}))^{y^{(i)}}(1  - h_{\boldsymbol \theta} (\boldsymbol x^{(i)}))^{1-y^{(i)}}
\end{aligned}
$$
&emsp;&emsp;对两边取对数，可以得到：
$$
\begin{aligned}
\ln L(\boldsymbol \theta) = \sum_{i = 1}^m [y^{(i)}\ln h_{\boldsymbol \theta} (\boldsymbol x) + (1 - y^{(i)}) \ln (1 - h_{\boldsymbol \theta} (\boldsymbol x))]
\end{aligned}
$$
&emsp;&emsp;所以我们的目标是最大化似然函数，即最小化损失函数 $J(\boldsymbol \theta)$ :
$$
J(\boldsymbol \theta) = - \sum_{i = 1}^m [y^{(i)}\ln h_{\boldsymbol \theta} (\boldsymbol x) + (1 - y^{(i)}) \ln (1 - h_{\boldsymbol \theta} (\boldsymbol x))]
$$

### 梯度推导

&emsp;&emsp;在推导梯度前，先来推导一下 sigmoid 函数的导数：
$$
\begin{aligned}
g'(z) &= -\frac {1} {(1 + e^{-z})^2} \frac {d(1+e^{-z})} {dz} \\
& = -\frac {1} {(1 + e^{-z})^2} (-e^{-z}) \\
& = \frac {1} {1 + e^{-z}} \frac {1 +e^{-z} - 1} {1 + e^{-z}} \\
& = g(z)(1-g(z))
\end{aligned}
$$
&emsp;&emsp;接下来推导损失函数 $J$ 的梯度，对于任意一个 $\theta_j$ 有：
$$
\begin{aligned}
\frac {\partial J(\boldsymbol \theta)} {\partial \theta_j} 
&= - \sum_{i = 1}^m [\frac {y^{(i)}}{h_{\boldsymbol \theta} (\boldsymbol x^{(i)})} \frac {\partial h_{\boldsymbol \theta} (\boldsymbol x^{(i)})} {\partial \theta_j} + \frac {1 - y^{(i)}}{ 1 -h_{\boldsymbol \theta} (\boldsymbol x^{(i)})} \frac {\partial (1 -h_{\boldsymbol \theta} (\boldsymbol x^{(i)}))} {\partial \theta_j}]  \\
&= - \sum_{i = 1}^m [\frac {y^{(i)}}{h_{\boldsymbol \theta} (\boldsymbol x^{(i)})} h_{\boldsymbol \theta} (\boldsymbol x^{(i)})(1-h_{\boldsymbol \theta} (\boldsymbol x^{(i)})) \frac {\partial (x_0^{(i)}\theta_0+\cdots+x_n^{(i)}\theta_n)} {\partial \theta_j} + \\
& \ \ \ \ \ \ \ \ \  \frac {1 - y^{(i)}}{ 1 -h_{\boldsymbol \theta} (\boldsymbol x^{(i)})} (- h_{\boldsymbol \theta} (\boldsymbol x^{(i)}))(1-h_{\boldsymbol \theta} (\boldsymbol x^{(i)})) \frac {\partial (x_0^{(i)}\theta_0+\cdots+x_n^{(i)}\theta_n)} {\partial \theta_j}  \\
&= - \sum_{i = 1}^m [ y^{(i)} (1-h_{\boldsymbol \theta} (\boldsymbol x^{(i)})) x_j^{(i)} - (1 - y^{(i)}) h_{\boldsymbol \theta}(\boldsymbol x^{(i)}) x^{(i)}_j] \\ 
&= - \sum_{i = 1}^m [ y^{(i)} - y^{(i)} h_{\boldsymbol \theta} (\boldsymbol x^{(i)}) + y^{(i)} h_{\boldsymbol \theta}(\boldsymbol x^{(i)}) - h_{\boldsymbol \theta}(\boldsymbol x^{(i)})]x^{(i)}_j \\
&= \sum_{i = 1}^m ( h_{\boldsymbol \theta}(\boldsymbol x^{(i)}) -  y^{(i)} )  x^{(i)}_j \\
\end{aligned}
$$
&emsp;&emsp;可以看到 Logistic Regression 的损失函数的梯度的公式和 Linear Regression 的梯度的公式相同，但是这是两个不同的东西，因为其中的假设函数不同。

# 感知机 (Perceptron)

&emsp;&emsp;感知机和 Logistic Regression 不同的地方在于他的 $g(z)$ 不是一个 sigmoid 函数了，而是如下一个阶跃函数：
$$
g(z) = 
\begin{cases}
1 & \text{if $z \ge 0$} \\
0 & \text{if $z \lt 0$}
\end{cases}
$$
&emsp;&emsp;因此假设函数为：
$$
h_{\boldsymbol \theta}(\boldsymbol x) = g(\boldsymbol x^T \boldsymbol \theta)
$$
&emsp;&emsp;若继续使用 Logistic Regression 在随机梯度下降中的迭代更新规则，我们就得到了感知机算法，他的参数更新方法为：
$$
\theta_j := \theta_j + \alpha(y^{(i)} - h_{\boldsymbol \theta}(\boldsymbol x^{(i)}))x_j^{(i)}
$$
















