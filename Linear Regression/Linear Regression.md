# 线性回归 (Linear Regression)

## 符号说明 (Notation)

- $\boldsymbol x$ 记为输入，是一个 $n+1$ 维的向量 (vector) ，$\boldsymbol x \in \Bbb R^{n+1}$ ，用上标 $^{(i)}$ 表示第 $i$ 个样本的输入，下标 $_j$ 表示第 $j$ 个属性的取值。例如 $x^{(5)}_2$ 表示第 5 个样本的第 2 的属性的值。
- $y$ 记为输出，也称为标签 (label) ，是一个标量，同上用上标区分不同样本的输出的取值
- $n$ 记为输入属性的个数值
- $m$ 记为样本的总个数
- $\boldsymbol \theta$ 记为参数（系数），是一个维度和 $\boldsymbol x$ 相同的向量，同 $\boldsymbol x$ 用下标区分不同属性对应的参数值。
- $h(\boldsymbol x)$ 记为假设方程，表示输入输出的关系函数
- $\alpha$ 记为学习率

## 线性回归的公式

&emsp;&emsp;在线性回归中，输入和输出呈线性关系，所以假设方程为：
$$
\begin{aligned}
    h(\boldsymbol x) &= \theta_0 + \theta_1x_1+...+\theta_nx_n \\
    &=\boldsymbol \theta^T\boldsymbol x
\end{aligned}
$$
&emsp;&emsp;其中 $x_0$ 恒等于 1，被称为偏置项。所以我们希望找到一组参数 $\boldsymbol \theta$ ，使得输入一组输入 $\boldsymbol x$ ，得到的输出 $h(\boldsymbol x)$ 越接近真实的结果越好，即对于训练集中的样本，输入某一个样本 $\boldsymbol x^{(i)}$ ，得到的结果 $h(\boldsymbol x^{(i)})$ 约接近这个样本的标签 $y^{(i)}$ 越好。所以我们训练的目标就是减少这两个值之间的差距，那么很自然的，我们可以想到可以使用二者差值的平方作为误差的度量，记为 $loss(\boldsymbol \theta)$ ，那么某一个样本的误差即为：
$$
\begin{aligned}
loss(\boldsymbol \theta) &= (h(\boldsymbol x^{(i)}) - y^{(i)})^2 \\
&=(\boldsymbol \theta^T\boldsymbol x - y^{(i)})^2
\end{aligned}
$$
&emsp;&emsp;我们不能只让我们所拟合的直线只在某一个样本上拟合的结果很多，我们希望的是直线可以在一些样本的拟合结果不好，但是在大部分样本上拟合的效果都不错，所以我们需要度量的是直线在整个训练集的所有样本上的拟合情况，不难想到可以用每个样本 $loss$ 的和或者均值来度量总体的拟合情况，我们用 $J$ 来表示总体的损失，$J$ 也被称为损失函数。这里我们采用求和的方法（事实上，可以证明求均值和求和对拟合 $\theta$ 没有影响，不仅如此，对 $J$ 进行任意常数的缩放都是没有影响的）这里为了后面的运算结果更简洁，在求和的基础上再除以常数 2 ，那么损失函数 (cost function) 为：
$$
J(\boldsymbol \theta) = \frac {1} {2} \sum _{i = 1}^{n}{(h(\boldsymbol x^{(i)}) - y^{(i)})^2}
$$
&emsp;&emsp;所以使 $J$ 最小时的 $\boldsymbol \theta$ 就是我们要求的结果，即：
$$
\boldsymbol \theta = \arg {\boldsymbol \min_\theta J(\boldsymbol \theta)}
$$

### 梯度下降

&emsp;&emsp;梯度下降的思想是向着当前 $\boldsymbol \theta$ 取值条件下，朝着能使 $J(\boldsymbol \theta)$ 下降最快的方向前进一步，即迭代更新一次 $\boldsymbol \theta$ ，在数学上可以证明，当更新的幅度足够小，这个方向是梯度的反方向。梯度用 $\nabla$ 符号表示，所以梯度向量为：
$$
\nabla_{\boldsymbol \theta} J(\boldsymbol \theta) = 
\begin{bmatrix}
\frac {\partial J(\boldsymbol \theta)} {\partial \theta_0} \\
\frac {\partial J(\boldsymbol \theta)} {\partial \theta_1} \\
\vdots \\
\frac {\partial J(\boldsymbol \theta)} {\partial \theta_n} \\
\end{bmatrix}
$$
&emsp;&emsp;其中，对任意一个 $\frac {\partial J(\boldsymbol \theta)} {\partial \theta_j}$ ，可以求得：
$$
\begin{aligned}
\frac {\partial J(\boldsymbol \theta)} {\partial \theta_j} &= \frac {1} {2} \sum_{i=1}^n \frac {\partial[(h(\boldsymbol x^{(i)}) - y^{(i)})^2] }{\partial \theta_j}  \\
&=\frac {1} {2} \sum_{i=1}^n 2(h(\boldsymbol x^{(i)}) - y^{(i)}) \frac {\partial[(h(\boldsymbol x^{(i)}) - y^{(i)})]} {\partial \theta_j}  \\
&=\sum_{i = 1} ^n (h(\boldsymbol x^{(i)}) - y^{(i)})\frac {\partial[\theta_0x_0 + \theta_1x_1+...+\theta_jx_j+...+\theta_nx_n - y^{(i)})]} {\partial \theta_j}  \\
&=\sum_{i = 1} ^n (h(\boldsymbol x^{(i)}) - y^{(i)})x_j
\end{aligned}
$$

&emsp;&emsp;注意：当 $j=0$ 时，$x_0=1$ 作为偏置项。 所以在学习率为 $\alpha$ 时，每次参数 $\boldsymbol \theta$ 的更新为：
$$
\begin{aligned}
\boldsymbol \theta &:= \boldsymbol \theta - \alpha \nabla_{\boldsymbol \theta} J(\boldsymbol \theta) 
\end{aligned}
$$
&emsp;&emsp;注意，这里的 $:=$ 表示更新，右边的 $\boldsymbol \theta$ 是更新前的参数，左边的 $\boldsymbol \theta$ 是更新后的参数。其中，对于任意一个 $\theta_j$ ，每次的更新方法为：
$$
\begin{aligned}
\theta_j &:=\theta_j - \alpha \frac {\partial h(\boldsymbol \theta)} {\partial \theta_j} \\
&:= \theta_j - \alpha \sum_{i = 1} ^n (h(\boldsymbol x^{(i)}) - y^{(i)})x_j
\end{aligned}
$$
&emsp;&emsp;所以梯度下降的算法就是迭代更新权重，直至收敛。

## 数学方法求解模型

&emsp;&emsp;上面迭代学习的过程实际是为了求解得到能使 $J(\boldsymbol \theta)$ 最小时的 $\boldsymbol \theta$ ，而事实上，$J(\boldsymbol \theta)$ 是一个只有一个极值点的函数，即求出那个极值点，这个点对应的参数就是 $\boldsymbol \theta$ 的取值，所以我们可以用数学方法来求解，过程和我们数学上求一个函数的最小值点类似。

&emsp;&emsp;首先让我们来回顾一下我们的损失函数 $J(\boldsymbol \theta)$ 的公式：
$$
J(\boldsymbol \theta) = \frac {1} {2} \sum _{i = 1}^{n}{(h(\boldsymbol x^{(i)}) - y^{(i)})^2}
$$
&emsp;若把训练集的输入 $(\boldsymbol x^{(j)})^T$ 按行排列起来，就会得到一个 $\Bbb R^{m \times n+1}$ 的矩阵，记为 $\boldsymbol X$ ，有如下形式：
$$
\boldsymbol X = 
\begin{bmatrix}
(\boldsymbol x^{(1)})^T \\
(\boldsymbol x^{(2)})^T \\
\vdots  \\
(\boldsymbol x^{(m)})^T \\
\end{bmatrix}_{m \times n+1}
$$
&emsp;&emsp;所以，$h(\boldsymbol X)$ 可以通过矩阵进行运算得到所有训练集样本的结果向量：
$$
h(\boldsymbol X) = \boldsymbol X \boldsymbol \theta
$$
&emsp;&emsp;所以 $J(\boldsymbol \theta)$ 可以通过以下形式转换为矩阵运算：
$$
\begin{aligned}
J(\boldsymbol \theta) &= \frac {1} {2} \sum _{i = 1}^{n}{(h(\boldsymbol x^{(i)}) - y^{(i)})^2} \\
&=\frac {1} {2} \sum _{i = 1}^{n} (\boldsymbol X \boldsymbol \theta - y)^2 \\
&= \frac {1} {2}(\boldsymbol X \boldsymbol \theta - y)^T(\boldsymbol X \boldsymbol \theta - y) \\
&= \frac {1} {2} (\boldsymbol \theta^T \boldsymbol X^T \boldsymbol X \boldsymbol \theta -y^T \boldsymbol X \boldsymbol \theta - \boldsymbol \theta^T \boldsymbol X^T y + y^Ty)
\end{aligned}
$$
&emsp;&emsp;其中，$(\boldsymbol X \boldsymbol \theta - y)$ 是一个列向量减一个列向量结果还是一个列向量。公式的第二行可能不太规范，表示的是对向量的所有元素求和。第二步到第三步是因为在线性代数中，若 $\boldsymbol x$ 是一个 $n$ 维列向量，那么 $\boldsymbol x^T \boldsymbol x = \sum_{i=1}^n (x_i)^2$  。其中 $\boldsymbol \theta^T \boldsymbol X^T y$ 其实是一个标量，标量的转置还是自己，所以可以化为如下形式：
$$
\begin{aligned}
J(\boldsymbol \theta)  &= \frac {1} {2} (\boldsymbol \theta^T \boldsymbol X^T \boldsymbol X \boldsymbol \theta -y^T \boldsymbol X \boldsymbol \theta - (\boldsymbol \theta^T \boldsymbol X^T y)^T + y^Ty)  \\
&=  \frac {1} {2} (\boldsymbol \theta^T \boldsymbol X^T \boldsymbol X \boldsymbol \theta - y^T \boldsymbol X \boldsymbol \theta - y^T \boldsymbol X \boldsymbol \theta + y^Ty)  \\
&=  \frac {1} {2} (\boldsymbol \theta^T \boldsymbol X^T \boldsymbol X \boldsymbol \theta - 2y^T \boldsymbol X \boldsymbol \theta + y^Ty)  \\
\end{aligned}
$$
&emsp;&emsp;对这个函数求偏导得：
$$
\begin{aligned}
\nabla_{\boldsymbol \theta}J(\boldsymbol \theta) &= \nabla_{\boldsymbol \theta}[\frac {1} {2} (\boldsymbol \theta^T \boldsymbol X^T \boldsymbol X \boldsymbol \theta - 2y^T \boldsymbol X \boldsymbol \theta + y^Ty)] \\
&= \frac {1} {2} \nabla_{\boldsymbol \theta}(\boldsymbol \theta^T \boldsymbol X^T \boldsymbol X \boldsymbol \theta) - \nabla_{\boldsymbol \theta}(y^T \boldsymbol X \boldsymbol \theta) + 0
\end{aligned}
$$
&emsp;&emsp;对于 $y^T \boldsymbol X \boldsymbol \theta$ 这一项，$y^T \boldsymbol X$ 是一个行向量，可以证明一个行向量和一个列向量矩阵相乘后，对这个列向量求偏导，得到的结果是行向量的转置。证明如下，假设 $\boldsymbol {row}$ 是一个 $n$ 维行向量，$\boldsymbol {col}$ 是一个 $n$ 维列向量，那么 :
$$
\begin{aligned}
\boldsymbol {row} \cdot \boldsymbol {col} &= row_1 \ * col_1 + row_2 \ * col_2 + \cdots + row_n \ * col_n  
\end{aligned}
$$
&emsp;&emsp;那么：
$$
\begin{aligned}
\nabla_{\boldsymbol {col}} (\boldsymbol {row} \cdot \boldsymbol {col}) &= 
\begin{bmatrix}
\frac {\partial {row \cdot col}} {\partial col_1}  \\
\frac {\partial {row \cdot col}} {\partial col_2} \\
\vdots \\
\frac {\partial {row \cdot col}} {\partial col_n} 
\end{bmatrix} \\
& = \begin{bmatrix}
row_1  \\
row_2 \\
\vdots \\
row_n
\end{bmatrix} \\
& = \boldsymbol {row}^T
\end{aligned}
$$
&emsp;&emsp;所以有如下结果：
$$
\begin{aligned}
\nabla_{\boldsymbol \theta}(y^T \boldsymbol X \boldsymbol \theta) &= (y^T \boldsymbol X)^T \\
& = \boldsymbol X^Ty
\end{aligned}
$$
&emsp;&emsp;对于 $\nabla_{\boldsymbol \theta}(\boldsymbol \theta^T \boldsymbol X^T \boldsymbol X \boldsymbol \theta)$ 这一项，其中 $(\boldsymbol X^T \boldsymbol X)^T = \boldsymbol X^T \boldsymbol X$ ，所以是一个对称矩阵，我们将其简记为 $\boldsymbol A$ 。所以：
$$
\begin{aligned}
\boldsymbol \theta^T \boldsymbol A \boldsymbol \theta &= \boldsymbol \theta^T 
\begin{bmatrix}
\sum_{i=1}^n A_{1i}\theta_i \\
\sum_{i=1}^n A_{2i}\theta_i \\
\vdots \\
\sum_{i=1}^n A_{ni}\theta_i \\
\end{bmatrix} \\
&=\theta_1(\sum_{i=1}^n A_{1i}\theta_i) + \theta_2(\sum_{i=1}^n A_{2i}\theta_i) + \cdots + \theta_n(\sum_{i=1}^n A_{ni}\theta_i) \\
& = \sum_{j=1}^n \sum_{i=1}^n A_{ji} \theta_i \theta_j
\end{aligned}
$$
&emsp;&emsp;$j$ 在前 $i$ 在后看着有点不顺眼，所以交换一下两个字母，改变符号并不会影响结果，所以有：
$$
\begin{aligned}
\boldsymbol \theta^T \boldsymbol A \boldsymbol \theta 
& = \sum_{i=1}^n \sum_{j=1}^n A_{ij} \theta_j \theta_i \\
& = \sum_{i=1}^n \sum_{j=1}^n A_{ij} \theta_i \theta_j 
\end{aligned}
$$
&emsp;&emsp;那么对 $\boldsymbol \theta$ 求偏导得：
$$
\begin{aligned}
\nabla_{\boldsymbol \theta} (\boldsymbol \theta^T \boldsymbol A \boldsymbol \theta ) &= \nabla_{\boldsymbol \theta} (\sum_{i=1}^n \sum_{j=1}^n A_{ij} \theta_i \theta_j) \\
&= 
\begin{bmatrix}
\frac {\partial(\sum_{i=1}^n \sum_{j=1}^n A_{ij} \theta_i \theta_j)} {\partial \theta_1} \\
\frac {\partial(\sum_{i=1}^n \sum_{j=1}^n A_{ij} \theta_i \theta_j)} {\partial \theta_2} \\
\vdots \\
\frac {\partial(\sum_{i=1}^n \sum_{j=1}^n A_{ij} \theta_i \theta_j)} {\partial \theta_n} \\
\end{bmatrix}
\end{aligned}
$$


  &emsp;&emsp;其中，对于任意第 $k$ 个元素有：
$$
\begin{aligned}
\frac {\partial(\sum_{i=1}^n \sum_{j=1}^n A_{ij} \theta_i \theta_j)} {\partial \theta_k} &=
\frac {\partial(\sum_{j=1}^n A_{1j} \theta_1 \theta_j+\cdots +\sum_{j=1}^n A_{kj} \theta_k \theta_j+\cdots +\sum_{j=1}^n A_{nj} \theta_n \theta_j)} {\partial \theta_k} \\
& = A_{1k}\theta_1 + A_{2k} \theta_2 + \cdots +\sum_{j=1}^{k-1}A_{kj}\theta_k + 2A_{kk}\theta_k +\sum_{j=k+1}^{n}A_{kj}\theta_k + A_{k+1\ k}\theta_{k+1} + \cdots + A_{nk} \theta_{n} \\
& = (A_{1k}\theta_1 + A_{2k} \theta_2 + \cdots + A_{kk}\theta_k  + A_{k+1\ k}\theta_{k+1} +  \cdots +A_{nk} \theta_{n} ) + (\sum_{j=1}^{k-1}A_{kj}\theta_k + A_{kk}\theta_k +\sum_{j=k+1}^{n}A_{kj}\theta_k) \\
&= \sum_{i =1}^nA_{ik}\theta_i + \sum_{i=1}^nA_{ki} \theta_i
\end{aligned}
$$
&emsp;&emsp;因为 $\boldsymbol A$ 是对称矩阵，所以有 $A_{ij} = A_{ji}$ ，所以：
$$
\begin{aligned}\frac {\partial(\sum_{i=1}^n \sum_{j=1}^n A_{ij} \theta_i \theta_j)} {\partial \theta_k} 
&= \sum_{i =1}^nA_{ik}\theta_i + \sum_{i=1}^nA_{ki} \theta_i \\
&= 2 \sum_{i =1}^nA_{ki}\theta_i \\
& = 2 \boldsymbol A_{k \cdot} \boldsymbol \theta
\end{aligned}
$$
&emsp;&emsp;所以：
$$
\begin{aligned}
\nabla_{\boldsymbol \theta} (\boldsymbol \theta^T \boldsymbol A \boldsymbol \theta ) 
&= 
\begin{bmatrix}
\frac {\partial(\sum_{i=1}^n \sum_{j=1}^n A_{ij} \theta_i \theta_j)} {\partial \theta_1} \\
\frac {\partial(\sum_{i=1}^n \sum_{j=1}^n A_{ij} \theta_i \theta_j)} {\partial \theta_2} \\
\vdots \\
\frac {\partial(\sum_{i=1}^n \sum_{j=1}^n A_{ij} \theta_i \theta_j)} {\partial \theta_n} \\
\end{bmatrix} \\
&= 
\begin{bmatrix}
2 \boldsymbol A_{1 \cdot} \boldsymbol \theta \\
2 \boldsymbol A_{2 \cdot} \boldsymbol \theta \\
\vdots \\
2 \boldsymbol A_{n \cdot} \boldsymbol \theta
\end{bmatrix} \\
&= 
2 \boldsymbol A \boldsymbol \theta
\end{aligned}
$$
&emsp;&emsp;所以，对于 $\nabla_{\boldsymbol \theta}(\boldsymbol \theta^T \boldsymbol X^T \boldsymbol X \boldsymbol \theta)$ 有：
$$
\begin{aligned}
\nabla_{\boldsymbol \theta}(\boldsymbol \theta^T \boldsymbol X^T \boldsymbol X \boldsymbol \theta)
&=  2  \boldsymbol X^T \boldsymbol X \boldsymbol \theta
\end{aligned}
$$
&emsp;&emsp;综上所述，将两个计算结果代入到计算梯度的式子中有：
$$
\begin{aligned}
\nabla_{\boldsymbol \theta}J(\boldsymbol \theta) 
&= \frac {1} {2} \nabla_{\boldsymbol \theta}(\boldsymbol \theta^T \boldsymbol X^T \boldsymbol X \boldsymbol \theta) - \nabla_{\boldsymbol \theta}(y^T \boldsymbol X \boldsymbol \theta) + 0 \\
& =\boldsymbol X^T \boldsymbol X \boldsymbol \theta - \boldsymbol X^Ty
\end{aligned}
$$
&emsp;&emsp;令 $\nabla_{\boldsymbol \theta}J(\boldsymbol \theta) = \boldsymbol 0$ ，得：
$$
\begin{aligned}
0
& =
\boldsymbol X^T \boldsymbol X \boldsymbol \theta - \boldsymbol X^Ty \\
\boldsymbol X^T \boldsymbol X \boldsymbol \theta &=\boldsymbol X^Ty  \\
\boldsymbol \theta &= (\boldsymbol X^T \boldsymbol X)^{-1}\boldsymbol X^Ty 
\end{aligned}
$$
&emsp;&emsp;所以可以得到，当 $\boldsymbol \theta = (\boldsymbol X^T \boldsymbol X)^{-1}\boldsymbol X^Ty $ 时，$J(\boldsymbol \theta)$ 取最小值。需要注意的是 $\boldsymbol X^T \boldsymbol X$ 必须是一个可逆矩阵上面的公式才成立。

## 从概率论上推导线性回归模型

&emsp;&emsp;不管怎么说，直接使用偏差的平方作为损失会让人有一丝不安，接下来从概率论上推导出前面所设定的损失函数，给它补上数学理论的支撑。

&emsp;&emsp;首先，我们需要进行一些假设。我们的第一个假设就是：输出 $y$ 和输入 $\boldsymbol x$  是呈严格的线性关系的，而实际样本可能不是严格在直线上而是在直线附近是因为还有其他因素微弱的影响着 $y$ 的结果。第二个假设就是：那些其他未考虑的因素对 $y$ 的影响我们认为是对 $y$ 加了一个零均值的正态分布的噪声，第三个假设是：数据集中，样本与样本之间是无关的。所以有：
$$
y-h_{\boldsymbol \theta} (\boldsymbol x) \sim N(0, \sigma^2)
$$
&emsp;&emsp;所以有：
$$
p(y-h_{\boldsymbol \theta} (\boldsymbol x)) = \frac {1} {\sqrt {2 \pi} \sigma} \exp(- \frac {(y-h_{\boldsymbol \theta} (\boldsymbol x))^2} {2 \sigma^2})
$$
&emsp;&emsp;所以样本数据被采集到的概率 $P$ 为：
$$
\begin{aligned}
p(\boldsymbol y | \boldsymbol X; \boldsymbol \theta) &= \prod_{i=1}^mp(y^{(i)} - h_{\boldsymbol \theta}(\boldsymbol x^{(i)}))  \\
& = \prod_{i=1}^m \frac {1} {\sqrt {2 \pi} \sigma} \exp(- \frac {(y^{(i)}-\boldsymbol x^{(i)T} \boldsymbol \theta)^2} {2 \sigma^2})
\end{aligned}
$$
&emsp;&emsp;该函数也称为似然函数，记为 $L$ ，因为 $L$ 是关于 $\boldsymbol \theta$ 的函数，所以有：
$$
L(\boldsymbol \theta) = p(\boldsymbol y | \boldsymbol X; \boldsymbol \theta) = \prod_{i=1}^m \frac {1} {\sqrt {2 \pi} \sigma} \exp(- \frac {(y^{(i)}-\boldsymbol x^{(i)T} \boldsymbol \theta)^2} {2 \sigma^2})
$$
&emsp;&emsp;对两边取以自然数为底数的对数有：
$$
\ln {L(\boldsymbol \theta)} = m \ln {\frac {1} {\sqrt {2 \pi} \sigma}} - \sum_{i=1}^m\frac {(y^{(i)}-\boldsymbol x^{(i)T} \boldsymbol \theta)^2} {2 \sigma^2}
$$
&emsp;&emsp;在上式中，$\sigma$ 是一个常数，因为他是样本的方差，样本确定后方差就确定了，$m$ 也是一个常数，所以 $\ln {L(\boldsymbol \theta)}$ 只受 $(y^{(i)}-\boldsymbol x^{(i)T} \boldsymbol \theta)^2$ 部分影响。我们希望我们的似然函数能取到最大值，即极大似然估计，因此我们希望 $-\frac {1} {2}\sum_{i=1}^m (y^{(i)}-\boldsymbol x^{(i)T} \boldsymbol \theta)^2$ 越大越好，即：
$$
\boldsymbol \theta = \arg {\boldsymbol \min_\theta \frac {1} {2} \sum_{i=1}^m (y^{(i)}-\boldsymbol x^{(i)T} \boldsymbol \theta)^2}
$$
&emsp;&emsp;由此，我们从概率论出发推导出了我们的求解目标，从中也可以得到我们的损失函数 $J(\boldsymbol \theta) = \frac {1} {2} \sum_{i=1}^m (y^{(i)}-\boldsymbol x^{(i)T} \boldsymbol \theta)^2$ 。 

## 局部加权线性回归 (Locally Weighted Linear Regression)

&emsp;&emsp;局部加权线性回归根据需要预测的点的位置，未每一个训练集的样本的损失值增加了一个权重 $w$，离预测的点约近的权重越高，加权后，损失函数变为：
$$
J(\boldsymbol \theta) = \sum_{i = 1}^mw^{(i)}(y^{(i)} - \boldsymbol x^{(i)T}\boldsymbol \theta)^2
$$
&emsp;&emsp;权重的计算方式为：
$$
w^{(i)} = \exp(- \frac {(\boldsymbol x^{(i)} - \boldsymbol x)^T(\boldsymbol x^{(i)} - \boldsymbol x)} {2 \tau ^2})
$$
&emsp;&emsp;其中 $\boldsymbol x$ 是待预测的点，$\tau$ 是带宽，控制了受影响的范围。局部线性加权回归的思想就是更希望能在预测点周围的拟合情况更好，而不在乎较远的地方，这样依据局部线性的思想可以拟合非线性的曲线。但是每对一个新样本就要计算一次权重，然后才能再计算它的拟合曲线，再去得到结果。















