## 第1章  绪论（习题）

### 本章学习目标

### 本章思维导图

### 本章概念地图



> [!NOTE] 练习 1.1（★☆☆）线性回归的最优解
> 考虑式(1.2)给出的平方和误差函数，其中函数 $y(r,w)$ 由式(1.1)给出。证明最小化误差函数的系数 $\boldsymbol{w}=\{w_{i}\}$ 由下列线性方程的集合给出
> $$
> \sum_{j=0}^{M} A_{ij}w_j = T_i
> \tag{1.122}
> $$
> 其中
> $$
> \quad A_{ij} = \sum_{n=1}^{N} (x_n)^{i+j}, \quad T_i = \sum_{n=1}^{N} (x_n)^{i} t_{n} \tag{1.123}
> $$
> 这里 $i, j$ 表示元素的下标，而 $(x)^{i}$ 表示 $x$ 的次幂。

**解答**

观察式(1.122)和式(1.123)，我们发现其形式与矩阵乘法类似，因此考虑用矩阵乘法重新表达 $E(\boldsymbol{w})$。将式(1.1)写成向量乘法的形式并代入多项式（式(1.2)），得到
$$
E(\boldsymbol{w}) = \frac{1}{2} \sum\limits_{n=1}^{N} \Big[ y(x_{n}, \boldsymbol{w}) - t_{n}\Big]^{2} = \frac{1}{2} \sum\limits_{n=1}^{N} \Big[ \boldsymbol{x}_{n}^{\top}\boldsymbol{w} -t_{n} \Big]^{2} 
$$
其中 $\boldsymbol{x}_{n} = [1, x_{n}, x_{n}^{2}, \dots, x_{n}^{M}]^{\top} \in \mathbb{R}^{M+1}$，$M$ 为多项式的阶。现在构造数据矩阵 $\boldsymbol{X}$ 和目标向量 $\boldsymbol{t}$：
$$
\boldsymbol{X} = \begin{bmatrix}
\boldsymbol{x}_{1}^{\top}\\\boldsymbol{x}_{2}^{\top}\\\vdots \\\boldsymbol{x}_{N}^{\top}
\end{bmatrix} = \begin{bmatrix}
1 & x_{1} & x_{1}^{2} & \cdots  & x_{1}^{M}\\
1 & x_{2} & x_{2}^{2} & \cdots  & x_{2}^{M}\\
\vdots & \vdots & \vdots & & \vdots \\
1 & x_{N} & x_{N}^{2} & \cdots  & x_{N}^{M}
\end{bmatrix}, \quad \boldsymbol{t} = \begin{bmatrix}
t_{1}\\t_{2}\\\vdots \\t_{N}
\end{bmatrix} \implies \left\| \boldsymbol{X}\boldsymbol{w} - \boldsymbol{t} \right\|_{2}^{2} = \sum\limits_{n=1}^{N} \Big[ \boldsymbol{x}_{n}^{\top}\boldsymbol{w} - t_{n} \Big]^{2} \tag{*1.1}
$$
因此损失函数 $E(\boldsymbol{w})$ 根据式 (\*1.1) 又可以写为
$$
E(\boldsymbol{w}) = \frac{1}{2} \left\| \boldsymbol{X}\boldsymbol{w} - \boldsymbol{t} \right\|_{2}^{2}.
$$
可见 $E(\boldsymbol{w})$ 对 $\boldsymbol{w}$ 是可微的凸函数，因此只需令 $E(\boldsymbol{w})$ 对 $\boldsymbol{w}$ 求导的结果为零即可：
$$
\displaystyle \frac{ \partial E(\boldsymbol{w}) }{ \partial \boldsymbol{w} } = \boldsymbol{X}^{\top}(\boldsymbol{X}\boldsymbol{w} - \boldsymbol{t}) \implies \boxed{\boldsymbol{X}^{\top}\boldsymbol{X}\boldsymbol{w}^{*} = \boldsymbol{X}^{\top}\boldsymbol{t}}. \tag{*1.2}
$$
此时令 $\boldsymbol{X}^{\top}\boldsymbol{X} = \boldsymbol{A} \in \mathbb{R}^{(M+1)\times(M+1)}$，$\boldsymbol{X}^{\top}\boldsymbol{t} = \boldsymbol{T} \in \mathbb{R}^{M+1}$，则
$$
\begin{align}
A_{i, j} &= \sum\limits_{n=1}^{N} X_{n,i} X_{n,j} = \sum\limits_{n=1}^{N} x_{n}^{i}x_{n}^{j},\\
T_{i} &= \sum\limits_{n=1}^{N} X_{n,i}t_{n} = \sum\limits_{n=1}^{N} x_{n}^{i}t_{n};\\
\end{align}
$$
其中 $X_{i,j}$ 表示数据矩阵 $\boldsymbol{X}$ 的第 $i$ 行第 $j$ 列的元素。最后由等式 $\boldsymbol{A}\boldsymbol{w}^{*} = \boldsymbol{T}$，得到
$$
\sum\limits_{j=0}^{M} A_{i,j} w_{j} = T_{i}.
$$


> [!NOTE] 练习 1.2（★☆☆）岭回归的最优解
> 写下能够使由公式 (1.4) 给出的正则化的平方和误差函数取得最小值的系数应该满足的与公式 (1.122) 类似的一组线性方程. 


公式 (1.4) 为加上 L2 正则项后的平方损失，可以仿照练习1.1中式 (\*1.1) 的矩阵写法，得到下面的形式：
$$
\tilde{E}(\boldsymbol{w}) = \frac{1}{2}\| \boldsymbol{X}\boldsymbol{w} - \boldsymbol{t} \|_{2}^{2} + \frac{\lambda}{2}\|\boldsymbol{w}\|_{2}^{2}.
$$
对 $\boldsymbol{w}$ 求偏导，有
$$
\displaystyle \frac{ \partial E(\boldsymbol{w}) }{ \partial \boldsymbol{w} } = \boldsymbol{X}^{\top}(\boldsymbol{X}\boldsymbol{w} - \boldsymbol{t}) + \lambda \boldsymbol{w} \tag{*1.3}
$$
令 (\*1.3) 为零，整理后有
$$
(\boldsymbol{X}^{\top}\boldsymbol{X} + \lambda \boldsymbol{I})\boldsymbol{w} = \boldsymbol{X}^{\top}\boldsymbol{t}. \tag{*1.4}
$$
因此练习1.1中的 $\displaystyle \sum\limits_{j=0}^{M} A_{i,j} w_{j} = T_{i}$ 将变为
$$
\lambda w_{i} + \sum\limits_{j=0}^{M} A_{i,j} w_{j} = T_{i}. \tag{*1.5}
$$
式 (\*1.5) 中多加的一项 $\lambda w_{i}$ 来自于式 (\*1.4) 括号中的 $\lambda \boldsymbol{I}$。除此外的两个线性方程和式 (1.123) 相同。

> [!NOTE] 练习 1.3（★★☆）全概率公式和 Bayes 公式的应用
> 假设我们有三个彩色的盒子：$r$（红色）、$b$（蓝色）、$g$（绿色）. 盒子 $r$ 里有 3 个苹果，4 个橘子，3 个酸橙；盒子 $b$ 里有 1 个苹果，1 个橘子，0 个酸橙；盒子 $g$ 里有 3 个苹果，3 个橘子和 4 个酸橙. 如果盒子随机被选中的概率为 $p(r)=0.2$，$p(b)=0.2$，$p(g)=0.6$. 选择一个水果从盒子中拿走（盒子中选择任何水果的概率都相同），那么选择苹果的概率是多少？如果我们观察到选择的水果实际上是橘子，那么它来自绿色盒子的概率是多少？

对于第一问选择苹果的概率，由贝叶斯定理推论（式 (1.13)），有
$$
\begin{align}
P(苹果) &= \underbrace{ P(苹果|r)P(r) }_{ 选择红色盒子 } + \underbrace{ P(苹果|b)P(b) }_{ 选择蓝色盒子 } + \underbrace{ P(苹果|g)P(g) }_{ 选择绿色盒子 }\\
&= \frac{3}{10} \cdot \frac{1}{5} +  \frac{1}{2} \cdot \frac{1}{5} + \frac{3}{10} \cdot \frac{3}{5}
= 0.34
\end{align}
$$
第二问求已知选的是橘子，它来自绿色盒子的概率。它可以写成 $P(g|橘子)$，我们使用贝叶斯定理（式 (1.12)），有
$$
\begin{align}
P(g|橘子) &= \frac{P(橘子|g)p(g)}{P(橘子|r)P(r) + P(橘子|b)P(b) + P(橘子|g)P(g)}\\
&= \frac{\frac{3}{10}\cdot \frac{3}{5}}{\frac{4}{10}\cdot \frac{1}{5} + \frac{5}{10}\cdot \frac{1}{5} + \frac{3}{10}\cdot \frac{3}{5}} = \frac{1}{2}
\end{align}
$$


> [!NOTE] 练习 1.4（★★☆）密度函数的变换和最大密度点的关系
> 考虑一个定义在连续变量 $x$ 上的概率密度 $p_x(x)$，假设我们使用 $x=g(y)$ 做了一个非线性变量变换，从而概率密度将按照公式 (1.27) 发生变化. 通过对公式 (1.27) 取微分，请证明 $y$ 的概率密度最大的位置 $\hat{y}$ 与 $x$ 的概率密度最大的位置 $\hat{x}$ 的关系通常不是简单的函数关系 $\hat{x} = g(\hat{y})$. 这说明概率密度（与简单的函数不同）的最大值取决于变量的选择. 请证明，在线性变换的情况下，最大值位置的变换方式与变量本身的变换方式相同. 

根据一维随机变量密度函数的变量替换公式（式 (1.27)），一维连续变量 $x$ 的概率密度函数做变换 $x = g(y)$ 后得到关于 $y$ 的密度函数为
$$
p_{y}(y) = p_{x}(g(y)) |g'(y)| \tag{*1.6}
$$
假设 $p_y(y)$ 和 $p_{x}(x)$ 都是连续可微的，则 $\hat{x}$ 满足 $p_{x}'(\hat{x}) = 0$。对式 (\*1.6) 两侧求导并忽略绝对值函数的不可微点，有
$$
\begin{align}
\frac{ \mathrm{d} }{ \mathrm{d}y } p_{y}(y) &= |g'(y)| \frac{ \mathrm{d}  }{ \mathrm{d}y } p_{x}(g(y)) + p_{x}(g(y)) \frac{ \mathrm{d}  }{ \mathrm{d}y } |g'(y)|\\
&= |g'(y)|p_{x}'(g(y)) + p_{x}(g(y)) \frac{ \mathrm{d}  }{ \mathrm{d}y } |g'(y)|
\end{align} \tag{*1.7}
$$
令 $\hat{y} = g^{-1}(\hat{x})$，代入式 (1.7) 后得到
$$
\begin{align}
\left.\frac{ \mathrm{d} }{ \mathrm{d}y } p_{y}(y) \right|_{y=\hat{y}} &= |g'(\hat{y})|p_{x}'(g(\hat{y})) + p_{x}(g(\hat{y})) \frac{ \mathrm{d} }{ \mathrm{d}y } |g'(\hat{y})|\\
&= p_{x}(\hat{x}) \frac{ \mathrm{d} }{ \mathrm{d}y } |g'(\hat{y})| \neq 0.
\end{align} \tag{*1.8}
$$
可以看出直接代入 $g^{-1}(\hat{x})$ 后式 (\*1.8) 非零，因此变换后的概率密度的极值点位置和变换前的极值点位置并非简单的函数关系 $\hat{x} = g(\hat{y})$，而概率密度最大的位置总是概率密度函数的极值点，因此也不满足上述的简单函数关系。

如果 $g$ 是线性变换，即 $x = g(y) = ky$，其中 $k$ 是常数，变换后的密度函数可以写为
$$
p_{y}(y) = p_{x}(g(y)) |g'(y)| = kp_{x}(ky). \tag{*1.9}
$$
与之相应地，先前对 $p_{y}(y)$ 求导过程（式 (\*1.7)）中的 $\displaystyle \frac{ \mathrm{d}  }{ \mathrm{d}y } |g'(y)|$ 将消失不见，此时成立
$$
p_{x}'(x) = 0 \implies p'_{y}(g^{-1}(x)) = 0.
$$
我们可以看出当 $g$ 取线性函数时，$p_{x}$ 的极值点与 $p_y$ 的极值点存在 $x = g(y)$ 的关系. 另外从变换后的密度函数的形式（式 (1.9)）上看，这个线性变换将原来的密度函数 $p_{x}$ 在横轴上做伸缩后再进行归一化，因此它们的极值点之间存在 $\hat{x} = g(\hat{y})$ 的对应关系。

<font color="red">待与官方英文版答案对比. </font>

> [!NOTE] 练习 1.5（★☆☆）随机变量的方差
> 使用定义 (1.38) 证明 $\text{var}[f(x)]$ 满足公式 (1.39). 

这是方差的经典性质。由方差的定义（式 (1.38)），有
$$
\begin{align}
\text{var}[f(x)] &= \mathbb{E}\Big[ \big(f(x) - \mathbb{E}[f(x)]\big)^{2} \Big] \\
&= \mathbb{E}\Big[ [f(x)]^{2} - 2f(x)\mathbb{E}[f(x)] + \underbrace{ [\mathbb{E}[f(x)]]^{2} }_{ 常数 } \Big] & 期望的线性性\\
&= \mathbb{E}[f^{2}(x)] - \cancel{ 2 }\mathbb{E}^{2}[f(x)] + \cancel{ \mathbb{E}^{2}[f(x)] }\\
&= \mathbb{E}[f^{2}(x)] - \mathbb{E}^{2}[f(x)].
\end{align}
$$

> [!NOTE] 练习 1.6（★☆☆）独立的一维随机变量的协方差为零
> 请证明，如果两个随机变量 $x$ 和 $y$ 是独立的，那么它们的协方差为零. 

如果两个随机变量 $x$ 和 $y$ 是独立的，那么其联合分布等于边缘分布的乘积，即 $p(x,y) = p(x)p(y)$。首先证明这两个随机变量乘积的期望等于各自期望的乘积：
$$
\begin{align}
\mathbb{E}_{x,y}[xy] &= \iint xyp(x,y) \, \mathrm{d}x\,\mathrm{d}y \\
&= \iint xp(x)yp(y) \, \mathrm{d}x\,\mathrm{d}y & p(x,y) = p(x)p(y)\\
&= \int xp(x)  \, \mathrm{d}x \cdot \int yp(y) \, \mathrm{d} y\\
&= \mathbb{E}_{x}[x] \mathbb{E}_{y}[y]
\end{align} \tag{*1.10}
$$
根据协方差的定义，有 $\text{cov}[x,y] = \mathbb{E}_{x,y}[xy] - \mathbb{E}_{x}[x] \mathbb{E}_{y}[y]$，由上面的结果，立即得到 $\text{cov}[x, y] = 0$。协方差衡量两个随机变量之间的关系，如果两个随机变量独立，那么它们无关，协方差为零。

> [!NOTE] 练习 1.7（★☆☆）一元高斯分布的归一化性质
> 在本练习中，我们证明公式 (1.48) 给出的一元高斯分布的归一化条件. 为了证明这一点，我们考虑下面的积分
> $$
> I = \int_{-\infty}^{\infty} \exp\left(-\frac{x^2}{2\sigma^2}\right) \mathrm{d}x \tag{1.124}
> $$
> 这个积分可以这样计算：首先将它的平方写成下面的形式
> $$
> I^2 = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right) \mathrm{d}x \, \mathrm{d}y \tag{1.125}
> $$
> 现在使用笛卡尔坐标 $(x,y)$ 到极坐标 $(r,\theta)$ 的坐标变换，然后替换 $u=r^2$. 请证明，通过对 $\theta$ 和 $u$ 积分，然后两边取平方根，我们可以得到
> $$
> I = (2\pi\sigma^2)^{1/2} \tag{1.126}
> $$
> 最后，使用这个结果，证明高斯分布 $N(x|μ,\sigma^2)$ 是归一化的. 

首先讨论式 (1.124) 和式 (1.125) 的积分技巧，原式的平方为
$$
\begin{align}
I^{2} &= \left[ \int_{-\infty}^{+\infty} \exp\left( -\frac{x^{2}}{2\sigma^{2}} \right) \, \mathrm{d}x \right] \left[ \int_{-\infty}^{+\infty} \exp\left( -\frac{y^{2}}{2\sigma^{2}} \right) \, \mathrm{d}y \right]\\
&= \int_{-\infty}^{+\infty} \int_{-\infty}^{+\infty} \exp\left( -\frac{x^{2}+y^{2}}{2\sigma^{2}} \right) \, \mathrm{d} x \, \mathrm{d}y 
\end{align}
$$
然后做笛卡尔坐标到极坐标的变换：
$$
\left\{ \begin{align}
x &= r\cos \theta\\
y &= r \sin \theta
\end{align} \right. \implies J = \begin{bmatrix}
\cos \theta & -r\sin \theta\\ \sin \theta & r\cos \theta
\end{bmatrix}, |J| = r
$$
因此根据微积分的变量替换公式，有
$$
\begin{align}
I^{2} &= \int_{0}^{2\pi} \int_{0}^{+\infty} r\exp\left( -\frac{r^{2}}{2\sigma^{2}} \right) \, \mathrm{d}r  \, \mathrm{d}\theta \\
&= 2\pi\int_{0}^{+\infty} r\exp\left( -\frac{r^{2}}{2\sigma^{2}} \right) \, \mathrm{d}r \\
&= \pi\int_{0}^{+\infty} \exp\left( -\frac{r^{2}}{2\sigma^{2}} \right) \, \mathrm{d}r^{2} & 换元积分\\
&= \pi \left[ -2\sigma^{2}\exp\left( -\frac{r^{2}}{2\sigma^{2}} \right) \right]^{r=+\infty}_{r=0} = 2\pi \sigma^{2}.
\end{align}
$$
因此得到 $I = (2\pi \sigma^{2})^{1/2}$。

最后验证高斯分布的归一化性质（式 (1.48)），只需对 $x$ 做标准化 $y = \displaystyle \frac{x - \mu}{\sigma}$ 即可（假设 $\sigma > 0$）：
$$
\begin{align}
&\, \int_{-\infty}^{+\infty} \frac{1}{\sqrt{ 2\pi \sigma^{2} }} \exp\left(  - \frac{(x - \mu)^{2}}{2\sigma^{2}} \right) \, \mathrm{d}x \\
=&\, \int_{-\infty}^{+\infty} \frac{1}{\sqrt{ 2\pi \sigma^{2} }} \exp\left(  - \frac{(y - 0)^{2}}{2\cdot 1^{2}} \right) \, \mathrm{d}(\sigma y + \mu) & y = \displaystyle \frac{x - \mu}{\sigma}\\
=&\, \frac{1}{\sqrt{ 2\pi }}  \int_{-\infty}^{+\infty} \exp\left(  - \frac{y^{2}}{2} \right) \, \mathrm{d}y = \frac{1}{\sqrt{ 2\pi }} \cdot (2\pi \cdot 1)^{1/2} = 1.\\
\end{align}
$$
最后一步使用了刚刚证明得到的式 (1.125) 的积分值，其中令 $\mu = 0$，$\sigma = 1$。

> [!NOTE] 练习 1.8（★★☆）一元高斯分布的均值和方差
> 通过使用变量替换，证明由公式 (1.46) 给出的一元高斯分布满足公式 (1.49). 接下来，通过对下面的归一化条件
> $$
> \int_{-\infty}^{+\infty} \mathcal{N}(x|μ,\sigma) \,\mathrm{d}r = 1 \tag{1.127}
> $$
> 两侧关于 $\sigma^2$ 求微分，证明高斯分布满足公式 (1.50). 最后，证明公式 (1.51) 成立. 

这个问题使用到了积分常见的技巧. 即通过换元配凑等方法凑出 $\mathcal{N}(x|\mu, \sigma)$ 的积分，将其直接替换为 $1$，从而大大简化计算。先计算一阶矩
$$
\begin{align}
\mathbb{E}[x] &= \int_{-\infty}^{+\infty} \frac{1}{\sqrt{ 2\pi \sigma^{2} }} x \exp\left( -\frac{(x-\mu)^{2}}{2\sigma^{2}} \right) \, \mathrm{d}x \\
&= \int_{-\infty}^{+\infty} \frac{1}{\sqrt{ 2\pi \sigma^{2} }} (y+\mu) \exp\left( -\frac{y^{2}}{2\sigma^{2}} \right) \, \mathrm{d}y & y = x - \mu\\
&= \underbrace{ \int_{-\infty}^{+\infty} \frac{1}{\sqrt{ 2\pi \sigma^{2} }} y \exp\left( -\frac{y^{2}}{2\sigma^{2}} \right) \, \mathrm{d}y }_{ 奇函数，积分为零 } + \mu\underbrace{ \int_{-\infty}^{+\infty} \frac{1}{\sqrt{ 2\pi \sigma^{2} }}\exp\left( -\frac{y^{2}}{2\sigma^{2}} \right) \, \mathrm{d}y }_{ {}=1 }\\
&= \mu.
\end{align}
$$
再计算二阶矩
$$
\begin{align}
\mathbb{E}[x^{2}] &= \int_{-\infty}^{+\infty} \frac{1}{\sqrt{ 2\pi \sigma^{2} }} x^{2} \exp\left( -\frac{(x-\mu)^{2}}{2\sigma^{2}} \right) \, \mathrm{d}x\\
&= \frac{1}{\sqrt{ 2\pi \sigma^{2} }}\int_{-\infty}^{+\infty}  (y+\mu)^{2} \exp\left( -\frac{y^{2}}{2\sigma^{2}} \right) \, \mathrm{d}y & y = x - \mu\\
&= \frac{1}{\sqrt{ 2\pi \sigma^{2} }} \cdot (-\sigma^{2}) \cdot \int_{-\infty}^{+\infty}  y \cdot \underbrace{ \left[ -\frac{y}{\sigma^{2}} \exp\left( -\frac{y^{2}}{2\sigma^{2}} \right) \right] }_{  \frac{ \mathrm{d}  }{ \mathrm{d}x } \exp\left( -\frac{y^{2}}{2\sigma^{2}} \right) } \, \mathrm{d}y + \mu^{2}\\
&= \frac{1}{\sqrt{ 2\pi \sigma^{2}}}\left\{ \cancel{ \left[ y \exp\left( -\frac{y^{2}}{2\sigma^{2}} \right) \right]^{+\infty}_{-\infty}  }+ \sigma^{2}\int_{-\infty}^{+\infty}  \exp\left( -\frac{y^{2}}{2\sigma^{2}} \right) \, \mathrm{d} y \right\} + \mu^{2}\\[0.2em]
&= \mu^{2}+\sigma^{2}
\end{align}
$$
所以最后高斯分布的方差为 $\text{var}[x] = \mathbb{E}_{x}[x^{2}] - \mathbb{E}_{x}^{2}[x] = \sigma^{2}$。

> [!NOTE] 练习 1.9（★☆☆）高斯分布的众数
> 证明由公式 (1.46) 给出的高斯分布的众数（即最大值）为 $\mu$. 类似地，证明由公式 (1.52) 给出的多元高斯分布的众数为 $\boldsymbol{\mu}$. 

对于一元情形，
多元情形，
$$
\begin{align}
\nabla_{\boldsymbol{x}} \mathcal{N}(\boldsymbol{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) &= 
\frac{1}{\sqrt{ (2\pi)^{D}|\boldsymbol{\Sigma}| }} \nabla_{\boldsymbol{x}} \exp \left\{ -\frac{1}{2}(\boldsymbol{x} - \boldsymbol{\mu})^{\top} \boldsymbol{\Sigma}^{-1}(\boldsymbol{x} - \boldsymbol{\mu}) \right\}\\
&= \square \cdot \nabla_{\boldsymbol{x}} \left[ (\boldsymbol{x} - \boldsymbol{\mu})^{\top} \boldsymbol{\Sigma}^{-1}(\boldsymbol{x} - \boldsymbol{\mu}) \right] & \square 为恒大于零的标量项 \\
&= \square \cdot (\boldsymbol{\Sigma}^{-1} + \boldsymbol{\Sigma}^{-\top})(\boldsymbol{x} - \boldsymbol{\mu}) & \nabla_{\boldsymbol{x}} \boldsymbol{x}^{\top}\boldsymbol{A}\boldsymbol{x} = (\boldsymbol{A} + \boldsymbol{A}^{\top})\boldsymbol{x}\\
&= \square \cdot \boldsymbol{\Sigma}^{-1}(\boldsymbol{x} - \boldsymbol{\mu}) & \boldsymbol{\Sigma}^{\top} = \boldsymbol{\Sigma},\text{ 常数项 2 方块吸收}\\
\end{align}
$$
于是想要 $\nabla_{\boldsymbol{x}} \mathcal{N}(\boldsymbol{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = 0$，即需 $\boldsymbol{\Sigma}^{-1}(\boldsymbol{x} - \boldsymbol{\mu}) = 0$. 由于 $\boldsymbol{\Sigma}$ 可逆，因此 $\boldsymbol{\Sigma}$ 满秩，该方程只有零解，即 $\boldsymbol{x} = \boldsymbol{\mu}$. 这样就证明了多元高斯分布的密度函数有且仅有一个极值点 $\boldsymbol{\mu}$，这个极值点也是极大值点或众数. 

> [!NOTE] 练习（★☆☆）1.10 均值和方差对独立变量和的加性
> 假设两个变量 $r$ 和 $z$ 是统计独立的. 证明它们的和的均值和方差满足
> $$
> \mathbb{E}[x+z] = \mathbb{E}[x] + \mathbb{E}[z] \tag{1.128}
> $$
> $$
> \text{var}[x+z] = \text{var}[x] + \text{var}[z] \tag{1.129}
> $$

首先证明均值的加性：
$$
\begin{align}
\mathbb{E}_{x,z}[x+z] &= \iint (x + z)p(x,z) \, \mathrm{d}x\,\mathrm{d}z \\
&= \int p(z) \int xp(x) \, \mathrm{d}x\,\mathrm{d}z + \int zp(z) \int p(x)\, \mathrm{d}x\,\mathrm{d}z \\
&= \int p(z) \mathbb{E}[x] \,\mathrm{d}z + \int zp(z) 1\,\mathrm{d}z & \int p(x) \, \mathrm{d} x = \int p(z) \, \mathrm{d} z = 1  \\
&= \mathbb{E}[x] + \mathbb{E}[z]
\end{align}
$$
然后利用均值的加性证明方差的加性：
$$
\begin{align}
\text{var}[x+z] &= \mathbb{E}_{x,z}[ (x+z)^{2} ] - \mathbb{E}^{2}_{x,z}[x+z]\\
&= \mathbb{E}_{x}[x^{2}] + 2\mathbb{E}_{x,z}[xz] + \mathbb{E}_{z}[z^{2}] - (\mathbb{E}_{x}[x] + \mathbb{E}_{z}[z])^{2} & \mathbb{E}_{x,z}[x+z] = \mathbb{E}_{x}[x] + \mathbb{E}_{z}[z]\\
&= {\color{red} \mathbb{E}_{x}[x^{2}] }  + \cancel{ 2\mathbb{E}_{x,z}[xz] } + {\color{blue} \mathbb{E}_{z}[z^{2}] }  {\color{red} {}- \mathbb{E}_{x}^{2}[x] }  - \cancel{ 2\mathbb{E}_{x}[x]\mathbb{E}_{z}[z] } {\color{blue} {}- \mathbb{E}_{z}^{2}[z] } & \mathbb{E}_{x,z}[x,z] = \mathbb{E}_{x}[x]\mathbb{E}_{z}[z]\\
&= {\color{red} \text{var}[x] }  + {\color{blue} \text{var}[z] } .
\end{align}
$$

> [!NOTE] 练习 1.11（★☆☆）均值和方差的极大似然解
> 通过令对数似然函数 (1.54) 关于 $μ$ 和 $σ^2$ 的导数等于零，证明公式 (1.55) 和公式 (1.56). 

回忆对数似然函数
$$
L = \ln p(\boldsymbol{x}|\mu, \sigma^{2}) = -\frac{1}{2\sigma^{2}} \sum\limits_{n=1}^{N} (x_{n} - \mu)^{2} - \frac{N}{2} \ln \sigma^{2} - \frac{N}{2} \ln (2\pi)
$$
注意 $L$ 对 $\mu$ 和 $\sigma^{2}$ 都是连续的凸函数，因此导数为零处的极小值都是最小值. 首先对 $\mu$ 求偏导，有
$$
\begin{align}
\displaystyle \frac{ \partial L }{ \partial \mu } &= - \frac{1}{\sigma^{2}}\sum\limits_{n=1}^{N} (x_{n} - \mu) = \frac{1}{\sigma^{2}}\left[ N\mu - \sum\limits_{n=1}^{N} x_{n} \right] 
\end{align}
$$
令其为零，得到 $\mu_{\text{ML}} = \bar{x} = \displaystyle \sum\limits_{n=1}^{N}x_{n}$. 
现在给定 $\mu = \mu_{ML}$ 考虑方差的极大似然解，依旧是对 $\sigma^{2}$ 求偏导：
$$
\displaystyle \frac{ \partial L }{ \partial \sigma^{2} } = \frac{1}{2(\sigma^{2})^{2}} \sum\limits_{n=1}^{N} (x_{n} - \mu_{\text{ML}})^{2} - \frac{N}{2\sigma^{2}}
$$
令其为零，得到 $\displaystyle \sigma_{\text{ML}}^{2} = \frac{1}{N} \sum\limits_{n=1}^{N} (x_{n} - \mu_{\text{ML}})^{2}$. 

> [!NOTE] 练习 1.12（★★☆）
> 使用公式 (1.49) 和公式 (1.50) 的结果，证明
> $$
> E[x_nx_m] = μ^2 + I_{nm}\sigma^2 \tag{1.130}
> $$
> 其中 $x_n$ 和 $x_m$ 表示从均值为 $μ$ 方差为 $σ^2$ 的高斯分布中采样的数据点. 当 $n=m$ 时，$I_{nm}=1$，否则 $I_{nm}=0$. 从而证明了公式 (1.57) 和公式 (1.58) 的结果. 

当 $n = m$ 时，$x_{n} = x_{m}$，$\mathbb{E}[x_{n}x_{m}] = \mathbb{E}[x_{n}^{2}] = \mu^{2} + \sigma^{2}$
当 $n \neq m$ 时，$x_{n}$ 和 $x_{m}$ 独立同分布，有 $\mathbb{E}[x_{n}x_{m}] = \mathbb{E}[x_{n}]\mathbb{E}[x_{m}] = \mu^{2}$

> [!NOTE] 练习 1.13（★☆☆）
> 假设高斯分布的方差由公式 (1.56) 进行估计，但是估计时将均值的最大似然估计 $μ_{ML}$ 替换为真实的均值 $μ$. 证明，此时对于方差的估计的期望等于真实的方差. 

我们要证明 $\mathbb{E}[\sigma_{\text{ML}}^{2}|\mu] = \sigma^{2}$，利用练习 1.11 的结果，有
$$
\begin{align}
\mathbb{E}[\sigma_{\text{ML}}^{2}] &= \mathbb{E}\left[ \frac{1}{N} \sum\limits_{n=1}^{N} (x_{n} - \mu)^{2} \right] = \frac{1}{N}\sum\limits_{n=1}^{N} \mathbb{E}[(x_{n} - \mu)^{2}] = \frac{1}{N} \sum\limits_{n=1}^{N} \text{var}[x_{n}]
\end{align}
$$
注意 $x_{1}, \dots, x_{n}$ 是来自同一高斯分布 $\mathcal{N}(x;\mu, \sigma^{2})$ 的独立同分布随机变量，因此有 $\text{var}[x_{n}] = \sigma^{2}$. 由此观之，对方差的极大似然解是各样本随机变量方差的平均，因此也有 $\mathbb{E}[\sigma_{\text{ML}}^{2}] = \sigma^{2}$.

> [!NOTE] 练习 1.14（★★☆）
> 证明任意的方阵的元素 $u$ 都可以写成 $w_j = w_{i,j}^{S} + w_{i,j}^{A}$ 的形式，其中 $w$ 和 $u$ 分别是对称矩阵和反对称矩阵，即对于所有的 $i$ 和 $j$ 都有 $w = w$ 和 $u = -u$. 现在考虑 $D$ 维空间高阶多项式中的二阶项，由下式给出
> $$
> \sum_{i=1}^{D} \sum_{j=1}^{D} w_{ij}x_i x_j \tag{1.131}
> $$
> 证明
> $$
> \sum_{i=1}^{D} \sum_{j=1}^{D} w_{ij}^{S} x_i x_j = \sum_{i=1}^{D} \sum_{j=1}^{D} w_{ij}x_i x_j \tag{1.132}
> $$
> 从而来自反对称矩阵的贡献消失了. 于是，我们看到，不失一般性，系数 $u_i$ 的矩阵可以选择成对称的，并且这个矩阵中并非所有 $D^2$ 个元素都可以独立选取. 证明，在矩阵 $u$ 中，独立参数的个数为 $\frac{P(D+1)}{2}$. 

任意方阵 $\boldsymbol{W}$ 都可以构造出对称和反对称的方阵：
$$
\boldsymbol{S} = \frac{\boldsymbol{W} + \boldsymbol{W}^{\top}}{2},\quad \boldsymbol{A} = \frac{\boldsymbol{W} - \boldsymbol{W}^{\top}}{2}
$$
显然有 $\boldsymbol{W} = \boldsymbol{S} + \boldsymbol{A}$，$\boldsymbol{W}$ 中的元素 $W_{i,j}$ 也就可以写成 $S_{i,j} +A_{i,j}$ 的形式，前者来自对称矩阵，后者来自反对称矩阵. 

现在考虑高阶多项式的二阶项。我们构造的系数矩阵是 $\boldsymbol{W} \in \mathbb{R}^{D \times D}$，我们让它有和上面一样形式的分解，即 $\boldsymbol{W} = \boldsymbol{S} + \boldsymbol{A}$，因此多项式 (1.131) 可以写成
$$
\sum\limits_{i=1}^{D} \sum\limits_{j=1}^{D} \left[ \frac{w_{i,j} + w_{j,i}}{2} + \frac{w_{i,j} - w_{j,i}}{2} \right] x_{i}x_{j} = \sum\limits_{i=1}^{D} \sum\limits_{j=1}^{D} \left[ w_{i,j}^{S} + w_{i,j}^{A} \right] x_{i}x_{j}. 
$$
接下来自然就是要证明 $\displaystyle \sum\limits_{i=1}^{D} \sum\limits_{j=1}^{D} w_{i,j}^{A}x_{i}x_{j} = 0$。回忆反对称矩阵的定义，即 $\boldsymbol{A}^{\top} = -\boldsymbol{A}$，我们知道它的对角线一定为零，而除对角线外的上下三角部分有关系 $a_{i,j} = -a_{j,i}$，因此我们可以将这个求和拆成三部分：
$$
\begin{align}
\sum_{i=1}^{D} \sum_{j=1}^{D} w_{i, j}^{A}x_i x_j &= \underbrace{ \sum\limits_{i=1}^{D} \sum\limits_{j=1}^{i-1} w_{i,j}^{A}x_{i}x_{j} }_{ 下三角 } + \underbrace{ \sum\limits_{i=1}^{D} w_{i,i}^{A}x_{i}x_{i} }_{ 对角线, \,=\,0 } + \underbrace{ \sum\limits_{j=1}^{D} \sum\limits_{i=1}^{j-1} w_{i,j}^{A} x_{i}x_{j} }_{ 上三角 }\\
&= \sum\limits_{i=1}^{D} \sum\limits_{j=1}^{i-1} w_{i,j}^{A}x_{i}x_{j} + \sum\limits_{j=1}^{D} \sum\limits_{i=1}^{j-1} (-w_{j,i}^{A}) x_{i}x_{j} & w_{i,j}^{A} = -w_{j,i}^{A}\\
&= \sum\limits_{i=1}^{D} \sum\limits_{j=1}^{i-1} w_{i,j}^{A}x_{i}x_{j} - \sum\limits_{i=1}^{D} \sum\limits_{j=1}^{i-1} w_{i,j}^{A}x_{i}x_{j} = 0.
\end{align}
$$
因此 $\displaystyle \sum_{i=1}^{D} \sum_{j=1}^{D} w_{i, j}x_i x_j = \sum_{i=1}^{D} \sum_{j=1}^{D} w_{i, j}^{S}x_i x_j$，这说明不重复的求和项数为 $\displaystyle \frac{D(D-1)}{2}$。这是二次多项式的情形，下一个练习将考虑一般情形。

> [!NOTE] 练习 1.15（★★★）
> 在这个练习和下一个练习中，我们研究多项式函数的独立参数的数量与多项式阶数 $M$ 以及输入空间维度 $D$ 之间的关系. 首先，我们写下 $D$ 维空间多项式的 $M$ 阶项，形式为
> $$
> \sum_{i_1=1}^{D} \sum_{i_2=1}^{D} \cdots \sum_{i_M=1}^{D} w_{i_1i_2\cdots i_M} x_{i_1}x_{i_2}\cdots x_{i_M} \tag{1.133}
> $$
> 系数 $w_{i_1i_2\cdots i_M}$ 由 $D^M$ 个元素组成，但是独立参数的数量远小于此，因为因子 $x_{i_1}x_{i_2}\cdots x_{i_M}$ 有很多互换对称性. 证明系数的冗余性可以通过把 $M$ 阶项写成下面的形式的方法消除. 
> $$
> \sum_{i_1=1}^{D} \sum_{i_2=1}^{i_{1}} \cdots \sum_{i_M=1}^{i_{M-1}} u_{i_1i_2\cdots i_M} x_{i_1}x_{i_2}\cdots x_{i_M} \tag{1.134}
> $$
> 注意，$\bar{w}$ 系数和 $w$ 系数之间的关系不需要显式表示. 使用这个结果证明，$M$ 阶项的独立参数的数量 $n(D,M)$ 满足下面的递归关系
> $$
> n(D, M) = \sum_{i=1}^{D} n(i, M-1) \tag{1.135}
> $$
> 接下来，使用归纳法证明下面的结果成立
> $$
> \sum\limits_{i=1}^{D} \frac{(i+M-2)!}{(M-1)!(i-1)!} = \frac{(D+M-1)!}{(D-1)!M!} \tag{1.136}
> $$
> 可以这样证明：首先证明 $D=1$ 的情况下，对于任意的 $M$，这个结果成立. 证明的过程中会使用 $0! = 1$. 然后假设这个结论对于 $D$ 维成立，证明它对于 $D+1$ 维也成立即可. 最后，使用之前的两个结果，以及数学归纳法，证明
> $$
> n(D,M) = \frac{(D + M - 1)!}{(D - I)!M!} \tag{1.137}
> $$
> 可以这样证明：首先证明这个结果对于 $M=2$ 且任意的 $D \geq 1$ 成立，这可以通过对比练习 1.14 的结果得出. 然后使用公式 (1.135) 和公式 (1.136)，证明，如果结果对于 $M-1$ 阶成立，那么对于 $M$ 阶也成立. 

(1) 首先解释这里的冗余项的意思。如果 $i_{1} = 1$，$i_{2}, \dots, i_{M}$ 都等于 $2$，则对应的多项式项为 $x_{1}x_{2}^{M-1}$；如果 $i_{2} = 1$，其余都等于 $2$，得到的是 $x_{2}x_{1}x_{2}^{M-2} = x_{1}x_{2}^{M-1}$，冗余就出现了。因此自然的去重办法就是引入一个排序约束：
$$
D \geqslant i_{1} \geqslant i_{2} \geqslant i_{3} \geqslant \cdots \geqslant i_{M} \geqslant 1
$$
这不难理解，这个链中的 $i_k$ 可以理解为从 $D$ 个装了无限个球的不同盒子中拿出 $M$ 个，每个盒子中的球都是同一种，不同盒子的球不同；其中的第 $k$ 个球是来自哪个盒子。为了在求和中反应这样的序关系，原来的求和就可以写成
$$
\sum\limits_{i_{1}=1}^{D} \sum\limits_{i_{2}=1}^{i_{1}} \cdots \sum\limits_{i_{M}=1}^{i_{M-1}} \bar{w}_{i_{1},i_{2},\dots,i_{M}} \cdot x_{i_1}x_{i_2}\cdots x_{i_M}.
$$
这就是不会重复的取法。

(2) 现在考虑不重复的取法有多少种，如果将上式的项数写成 $n(D, M)$，我们能观察到天然的递归结构：
$$
\begin{align}
{\color{red} \sum\limits_{i_{1}=1}^{D} x_{i_1} } & {\color{blue} \sum\limits_{i_{2}=1}^{i_{1}} \cdots \sum\limits_{i_{M}=1}^{i_{M-1}} \bar{w}_{i_{1},i_{2},\dots,i_{M}} \cdot x_{i_2}\cdots x_{i_M} } .\\
D \geqslant {\color{red} i_{1} }  & {\color{blue} \geqslant i_{2} \geqslant i_{3} \geqslant \cdots \geqslant i_{M} }  \geqslant 1
\end{align}
$$
蓝色部分公式的项数按照上面的定义就是 $n(i_{1}, M-1)$，于是就很自然地得到下面的递推式
$$
n(D, M) = \sum\limits_{i=1}^{D} n(i, M-1).
$$

(3) 这一小问要证明的关于组合的等式有助于求解 $n(D, M)$。首先考虑 $D=1$ 的情形
$$
\begin{align}
左边 &= \frac{(M-1)!}{(M-1)!} = 1\\
右边 &= \frac{M!}{M!} = 1
\end{align}
$$
显然成立。现在假设等式在 $D = k-1$ 时成立，要证明 $D=k$ 时也成立：
$$
\begin{align}
\sum\limits_{i=1}^{k} \frac{(i+M-2)!}{(i-1)!(M-1)!} &= \sum\limits_{i=1}^{k-1} \frac{(i+M-2)!}{(i-1)!(M-1)!} + \frac{(k+M-2)!}{(k-1)!(M-1)!} \\
&= \frac{(k+M-2)!}{(k-2)!M!} + \frac{(k+M-2)!}{(k-1)!(M-1)!} & 归纳假设\\
&= \frac{(k+M-1)(k+M-2)!}{(k-1)!M!} = \frac{(k+M-1)!}{(k-1)!M!}
\end{align} 
$$
结论成立。最后令 $n(D, M) = \displaystyle \frac{(D+M-1)!}{(D-1)!M!}$ 即可验证上面的递归式。

> [!danger] 排列组合视角
> 从 $n(D, M)$ 的长相来看，我们可以用插板法理解：有 $M$ 个球，用 $D-1$ 个隔板将其分成 $D$ 份，即一共有 $D+M-1$ 个位置，其中 $M$ 个位置放球，其余位置放隔板，因此就有 $\displaystyle n(D, M) = \binom{D+M-1}{M} = \frac{(D+M-1)!}{(D-1)!M!}$.


> [!NOTE] 练习 1.16（★★★）$N(D, M)$ 的阶
> 在练习 1.15 中，我们证明了 $D$ 维多项式 $M$ 阶项的独立参数的个数满足公式 (1.135) 给出的关系. 我们现在寻找阶数小于等于 $M$ 阶的所有项的独立参数的总数 $N(D,M)$. 首先，证明 $N(D,M)$ 满足
> $$
> N(D, M) = \sum_{m=0}^{M} n(D, m) \tag{1.138}
> $$
> 其中 $n(D,m)$ 是 $m$ 阶项的独立参数的数量. 现在，使用公式 (1.137) 的结果，以及数学归纳法证明
> $$
> N(D,M) = \frac{(D+M)!}{D!M!} \tag{1.139}
> $$
> 可以这样证明：首先证明结果对于 $M=0$ 以及任意的 $D \geq 1$ 成立，然后假设它对于 $M$ 阶成立，证明它对于 $M+1$ 阶也成立即可. 最后，使用下面的 Stirling 近似
> $$
> n! \approx n^n e^{-n} \sqrt{2\pi n} \tag{1.140}
> $$
> 这个近似关系对于大的 $n$ 成立. 证明，对于 $D \gg M$，$N(D,M)$ 的增长方式类似于 $D^M$，对于 $M \gg D$，它的增长方式类似于 $M^D$. 考虑 $D$ 维的立方（$M=3$）多项式，计算下面两种情形的独立参数的总数：（1）$D=10$ 和（2）$D=100$，这对应于典型的小规模和中规模的机器学习应用问题. 

上个问题中，我们可以将 $n(D, M)$ 看成 $D$ 维多项式 $M$ 阶项的独立参数总数，因此不超过 $M$ 阶的 $D$ 维多项式独立参数总数自然就是从 $0$ 阶开始一直到 $M$ 阶的独立参数总数之和：
$$
N(D, M) = \sum\limits_{m=0}^{M} n(D, m).
$$
现在已知 $\displaystyle n(D, m) = \binom{D+m-1}{m}$，证明 $N(D, M) = \displaystyle \binom{D + M}{M}$，即证明
$$
\displaystyle \sum\limits_{m=0}^{M} \binom{D+m-1}{m} = \binom{D+M}{D} 
$$
同样用归纳法。假设 $M = 0$，左边是 $\displaystyle \binom{D}{0}$，右边是 $\displaystyle \binom{D}{D}$，显然相等。假设 $M=k-1$ 时等式成立，当 $M = k$ 时，有
$$
\begin{align}
\sum\limits_{m=0}^{k} \binom{D+m-1}{m} &= \sum\limits_{m=0}^{k-1}  \binom{D+m-1}{m} + \binom{D+k-1}{k}\\
&= \binom{D+k-1}{D} + \binom{D+k-1}{k}\\
&= \binom{D+k-1}{D} + \binom{D+k-1}{D-1} = \binom{D+k-1}{D-1}. & 组合恒等式 
\end{align}
$$
最后我们用 Stirling 公式估计 $N(D, M)$ 的阶. 首先我们有 $N(D, M) = \displaystyle \binom{D+M}{D} = \frac{(D+M)!}{D!M!}$.
* 当 $D \gg M$ 时，有 $N(D, M) \sim \displaystyle \frac{(D+M)^{D+M}e^{-(D+M)}}{D^{D}e^{-D}} \sim \frac{D^{M}}{e^{M}} \sim D^{M}$.
* 当 $M \gg D$ 时，由对称性，可知 $N(D, M) \sim M^{D}$.
当 $M=3$ 时
* 若 $D=10$，$n(D, M) = n(10, 3) = \displaystyle \binom{13}{3} = \frac{13 \cdot 12 \cdot 11}{3 \cdot 2 \cdot 1} = 286$,
* 若 $D = 100$，则 $n(D, M)$ 大致是 $D^{M} = 10^{6}$.

> [!NOTE] 练习 1.17（★★☆）
> Gamma 函数的定义为
> $$
> \Gamma(c) = \int_0^\infty u^{c-1}e^{-u} \mathrm{d}u \tag{1.141}
> $$
> 使用分部积分法，证明 $\Gamma(c+1) = c\Gamma(c)$. 并且证明，$\Gamma(1) = 1$，因此当 $c$ 为整数时，$\Gamma(c+1) = c!$. 

直接计算：
$$
\begin{align}
\Gamma(c+1) &= \int_{0}^{\infty} u^{c}e^{-u} \, \mathrm{d}u = -\int_{0}^{\infty} u^{c} \left( \displaystyle \frac{ \mathrm{d}  }{ \mathrm{d}u }  e^{-u} \right) \,\mathrm{d}u\\
&= c\int_{0}^{\infty} u^{c-1}e^{-u} \, \mathrm{d}u - \cancel{ \Big[u^{c}e^{-u}\Big]^{u=\infty}_{u=0 }} & 分部积分\\
&= c\Gamma(c).
\end{align}
$$
当 $c = 1$ 时，有 $\Gamma(1)=\displaystyle \int_{0}^{\infty} e^{-u} \, \mathrm{d}u = 1$.

> [!NOTE] 练习 1.18（★★☆）
> 我们可以使用公式 (1.126) 的结果来推导 $D$ 维空间中单位半径的球体的表面积 $S_D$ 和体积 $V_D$. 为了完成这一点，考虑下面的结果. 这个结果是通过从笛卡尔坐标系到极坐标系的坐标变换的方式得到的. 
> $$
> \prod\limits_{i=1}^{D}  \int_{-\infty}^{\infty} e^{-x_{i}^2} \mathrm{d}x_i = S_D \int_0^\infty e^{-r^2} r^{D-1} \mathrm{d}r \tag{1.142}
> $$
> 使用 Gamma 函数的定义 (1.141) 以及公式 (1.126)，计算方程的两侧，从而证明
> $$
> S_D = \frac{2\pi^{D/2}}{\Gamma(D/2)} \tag{1.143}
> $$
> 接下来，通过对半径从 0 到 1 进行积分，证明 $D$ 维单位球体的体积为
> $$
> V_D = \frac{S_D}{D} \tag{1.144}
> $$
> 最后，使用结果 $\Gamma(1) = 1$ 和 $\Gamma(3/2) = \sqrt{\pi}/2$，证明对于 $D=2$ 和 $D=3$ 的情形，公式 (1.143) 和公式 (1.144) 就是通常的结果. 

考虑公式 (1.142)，左侧可以使用练习 1.7 的结果，左边等于 $(\pi^{1/2})^{D} = \pi^{D/2}$，右边的积分式做换元 $s = r^{2}$：
$$
\int_{0}^{\infty} e^{-r^{2}}r^{D-1} \, \mathrm{d}r = \int_{0}^{\infty} e^{-s}s^{(D-1)/2} \cdot \frac{1}{2\sqrt{ s }}\, \mathrm{d} s = \frac{1}{2}\int_{0}^{\infty} e^{-s}s^{D/2 - 1}\, \mathrm{d} s = \frac{1}{2} \Gamma\left( \frac{D}{2} \right).
$$
因此公式 (1.142) 等价于 $2\pi^{D/2} = S_{D}\Gamma(D/2)$，也就是公式 (1.143).

$D$ 维球体的表面的 “维度” 是 $D-1$ 维，因此半径为 $r$ 的 $D$ 维球体的球面表面积为 $S_{D} \cdot r^{D-1}$。对其进行积分，有
$$
\int_{0}^{1} S_{D}r^{D-1} \, \mathrm{d}r = S_{D} \cdot \left[ \frac{1}{D} \cdot r^{D} \right]^{1}_{0} = \frac{S_{D}}{D}.  
$$
当 $D = 2$ 时，有 $S_{D} = \displaystyle \frac{2\pi^{2/2}}{1} = 2\pi$，$V_{D} = \displaystyle \frac{2\pi}{2} = \pi$. 
当 $D=3$ 时，有 $S_{D} = \displaystyle \frac{2\pi^{3/2}}{\pi^{1/2}/2}=4\pi$，$V_{D}=\displaystyle \frac{4\pi}{3}$. 
这和我们中小学所学的结果相吻合。

> [!NOTE] 练习 1.19（★★☆）
> 考虑 $D$ 维空间的一个半径为 $a$ 的球体和一个同心的边长为 $2a$ 的超立方体，球面与超立方体的每个面的中心接触. 通过使用练习 1.18 的结果，证明球与超立方体的体积比为
> $$
> \frac{\text{球的体积}}{\text{超立方体的体积}} = \frac{\pi^{D/2}}{D 2^{D-1} \Gamma(D/2)} \tag{1.145}
> $$
> 接下来使用下面形式的 Stirling 公式
> $$
> \Gamma(x+1) \approx (2\pi)^{1/2} e^{-x} x^{x+1/2} \tag{1.146}
> $$
> 对于 $x \gg 1$ 的情况成立. 证明，对于 $D \to \infty$，比值 (1.145) 趋于零. 并且证明，超立方体从中心到某个角的距离与从中心到某条边的垂直距离的比值为 $\sqrt{D}$，从而对于 $D \to \infty$，这个比值也趋于无穷大. 从这些结果中，我们可以看到，在高维空间中，立方体的大部分体积集中在数量众多的角上，这些角本身有着非常长的"尖刺"！

当 $D \rightarrow \infty$ 时，我们直接使用 Stirling 公式对 (1.145) 进行近似：
$$
\begin{align}
\frac{\pi^{D/2}}{D \cdot 2^{D-1} \Gamma(D/2)} &\sim \frac{\pi^{D/2}}{D \cdot 2^{D-1} (2\pi)^{1/2}e^{-(D/2-1)} \cdot (D/2-1)^{(D-1)/2}} \sim \frac{(e\pi)^{D/2}}{2^{D/2} \cdot D^{1+D/2}} \rightarrow  0.
\end{align}
$$

接着我们看超立方体中心到角点和到边中点的距离。角点距离非常容易，就是 $\displaystyle \sqrt{ \sum\limits_{n=1}^{D} \left( \frac{1}{2} \right)^{2}} = \frac{\sqrt{ D }}{2}$，而中心到面的距离就是边长的一半，即 $\displaystyle \frac{1}{2}$。于是中心到角点的距离是中心到面的距离的 $\sqrt{ D }$ 倍，当 $D \rightarrow \infty$ 时，该比值趋于零。

<font color="red">这里是否有误，中心到角点的距离是中心到面心根号D倍，而不是与中心到边的距离。后者的距离是 sqrt(D-1)/2 ？</font>

> [!NOTE] 练习 1.20（★★☆）
> 在本练习中，我们研究高维高斯空间的高斯分布的行为. 考虑 $D$ 维空间的一个高斯分布，形式如下
> $$
> p(\boldsymbol{x}) = \frac{1}{(2\pi\sigma^2)^{D/2}} \exp\left(-\frac{\|\boldsymbol{x}\|^2}{2\sigma^2}\right) \tag{1.147}
> $$
> 我们想要找到关于极坐标半径的概率密度，其中方向变量已经被积分出去. 为了完成这一点，证明，概率密度在一个半径为 $r$ 且厚度为 $e$ 的球壳上的积分为 $p(r)e$，其中 $e \ll 1$，且
> $$
> p(r) = \frac{S_D}{(2\pi\sigma^2)^{D/2}} r^{D-1} \exp\left(-\frac{r^2}{2\sigma^2}\right) \tag{1.148}
> $$
> 这里，$S_D$ 是 $D$ 维单位球体的表面积. 证明，对于大的 $D$ 值，函数 $p(r)$ 有一个驻点位于 $r \approx \sqrt{D}\sigma$ 处. 通过考虑 $p(r+\epsilon)$，其中 $\epsilon \ll r$，证明对于大的 $D$ 值
> $$
> p(r+\epsilon) = p(r) \exp\left(-\frac{\epsilon^2}{2\sigma^2}\right) \tag{1.149}
> $$
> 这表明，$r$ 是径向概率密度的最大值点，且远离最大值点 $r$ 时，$p(r)$ 会指数衰减，长度缩放因子为 $\sigma$. 我们已经看到，对于大的 $D$ 值，$\sigma \ll r$，因此我们看到大部分的概率质量都集中于大半径的薄球壳上. 最后，证明概率密度 $p(\boldsymbol{x})$ 在原点处的值大于在半径 $r$ 处的值，二者的差别是一个值为 $\exp(D/2)$ 的因子. 于是我们看到，高维高斯分布的概率质量最大的位置不同于半径上概率密度最大的位置. 当我们在后续章节中考虑模型参数的贝叶斯推断时，高维空间中的高斯分布的这个性质将会起重要的作用. 

我们考虑在 $D$ 维空间中作极坐标变换，将 $(x_{1}, \dots, x_{D})$ 转为 $(r, \theta_{1}, \dots, \theta_{D-1})$，有
$$
\left\{ \begin{align}
x_{1} &= r\cos \theta_{1}\\
x_{2} &= r\sin \theta_{1} \cos \theta_{2} \\
& \vdots \\
x_{D-1} &= r\sin \theta_{1}\sin \theta_{2} \cdots \sin \theta_{D-2} \cos \theta_{D-1}\\
x_{D} &= r\sin \theta_{1}\sin \theta_{2} \cdots \sin \theta_{D-2} \sin \theta_{D-1}
\end{align} \right.
$$
$$
\begin{align}
J &= \begin{bmatrix}
C_{1} & -rS_{1}\\
S_{1}C_{2} & rC_{1}C_{2} & -rS_{1}S_{2}\\
\vdots  & \vdots  & \ddots  & \ddots \\
S_{1:D-2}C_{D-1} &rC_{1}S_{2:D-2}C_{D-1} & \cdots & rS_{1:D-3}C_{D-2}C_{D-1} & -rS_{1:D-1}\\
S_{1:D-2}S_{D-1} &rC_{1}S_{2:D-1} &  \cdots & rS_{1:D-3}C_{D-2}S_{D-1} & rS_{1:D-2}C_{D-1}
\end{bmatrix} = r^{D-1}|J_{1}|
\end{align}
$$
因此球坐标变换后的密度函数为
$$
\begin{align}
p(r, \boldsymbol{\theta}) = \frac{r^{D-1}|J_{1}(\boldsymbol{\theta})|}{(2\pi\sigma^2)^{D/2}} \exp\left(-\frac{r^2}{2\sigma^2}\right)
\end{align}
$$
把 $\boldsymbol{\theta}$ 积掉，得到
$$
p(r) = \frac{r^{D-1}}{(2\pi\sigma^2)^{D/2}} \exp\left(-\frac{r^2}{2\sigma^2}\right)\underbrace{ \int |J_{1}(\boldsymbol{\theta})| \, \mathrm{d} \boldsymbol{\theta} }_{ 常数 C}
$$
我们只需要利用分布的归一化条件就可以反解出 $\displaystyle \int |J_1(\boldsymbol{\theta})| \, \mathrm{d}\boldsymbol{\theta}$ 的值：
$$
\begin{align}
\frac{1}{C} &= \int_{0}^{\infty} \frac{r^{D-1}}{(2\pi \sigma^{2})^{D/2}} \exp\left( -\frac{r^{2}}{2\sigma^{2}} \right) \, \mathrm{d}r \\
&= \frac{1}{(2\pi \sigma^{2})^{D/2}}\int_{0}^{\infty} (2\sigma^{2}s)^{(D-1)/2} \exp\left( -s\right) \, \mathrm{d}(\sqrt{ 2\sigma^{2}s }) & s = \frac{r^{2}}{2\sigma^{2}}\\
&= \frac{1}{(2\pi \cancel{ \sigma^{2} })^{D/2}} \cdot (2\cancel{ \sigma^{2} })^{(D-1)/2}\cdot \frac{\cancel{ \sigma }}{\sqrt{ 2}} \int_{0}^{\infty} s^{D/2-1} \exp\left( -s\right) \, \mathrm{d}s\\
&= \frac{1}{2\pi^{D/2}} \cdot \Gamma\left( \frac{D}{2} \right) = \frac{1}{S_{D}}.
\end{align}
$$
其中 $S_{D}$ 是 $D$ 维单位球体的表面积。我们用这个方法逃过了恐怖的行列式计算，最后得到径向边缘分布
$$
p(r) = \frac{r^{D-1}S_{D}}{(2\pi\sigma^2)^{D/2}} \exp\left(-\frac{r^2}{2\sigma^2}\right) = \frac{2r^{D-1}}{(2\sigma^{2})^{D/2}\Gamma(D/2)}\exp\left(-\frac{r^2}{2\sigma^2}\right).
$$
可对 $r$ 求导得到边缘分布的驻点：
$$
\displaystyle \frac{ \mathrm{d}  }{ \mathrm{d}r } p(r) \propto \left[ (D-1)r^{D-2}\exp\left(-\frac{r^2}{2\sigma^2}\right) - \frac{r^{D}}{\sigma^{2}}\exp\left(-\frac{r^2}{2\sigma^2}\right) \right]
$$
导函数的零点为 $\hat{r} = \sqrt{ D-1 }\sigma$，当 $D$ 充分大时，零点近似为 $\sqrt{ D }\sigma$.

现在考虑径向边缘分布驻点的一个向远处的小偏离 $r + \varepsilon$，可以计算
$$
\begin{align}
p(\hat{r}+\varepsilon) &= \frac{(\hat{r}+\varepsilon)^{D-1}S_{D}}{(2\pi\sigma^2)^{D/2}} \exp\left[-\frac{(\hat{r}+\varepsilon)^2}{2\sigma^2}\right]\\
&= p(r) \cdot \left( \frac{\hat{r}+\varepsilon}{r} \right)^{D-1} \exp\left[ \frac{\hat{r}^{2} - (\hat{r}+\varepsilon)^{2}}{2\sigma^{2}} \right]\\
&= p(r)  \exp\left[ (D-1)\ln\left( 1 + \frac{\varepsilon}{\hat{r}} \right)-\frac{2\hat{r}\varepsilon + \varepsilon^{2}}{2\sigma^{2}} \right]\\
&\approx p(r) \exp\left[ (D-1)\left( \frac{\varepsilon}{\hat{r}} - \frac{\varepsilon^{2}}{2\hat{r}^{2}} + O(\varepsilon^{3}) \right) -\frac{\hat{r}\varepsilon}{\sigma^{2}} - \frac{\varepsilon^{2}}{2\sigma^{2}} \right]  & 展开 \ln 至二阶\\
&= p(r) \exp\left[ \cancel{ \frac{(D-1)\varepsilon}{\hat{r}} -\frac{\hat{r}\varepsilon}{\sigma^{2}} } - \frac{(D-1)\varepsilon^{2}}{2\hat{r}^{2}}  - \frac{\varepsilon^{2}}{2\sigma^{2}}  + O(\varepsilon^{3})\right] & 驻点条件\\
&\approx p(r) \exp\left[ - \frac{\varepsilon^{2}}{\sigma^{2}}\right].
\end{align}
$$
最后讨论高斯分布概率函数 $p(\boldsymbol{x})$ 和距离原点 $\hat{r}$ 处的概率密度。首先有零处的概率密度
$$
p_{\boldsymbol{x}}(0) = \frac{1}{(2\pi \sigma^{2})^{D/2}}
$$
然后计算距离原点 $\hat{r}$ 处的概率密度：
$$
p_{\boldsymbol{x}}(\| \boldsymbol{x} \| = r) =  \frac{1}{(2\pi\sigma^2)^{D/2}} \exp\left(-\frac{r^2}{2\sigma^2}\right) \approx \frac{1}{(2\pi\sigma^2)^{D/2}} \exp\left(-\frac{D}{2}\right).
$$
可见高斯分布在直角坐标下的概率密度函数和径向概率密度函数的极大值并不是同一回事。

> [!NOTE] 练习 1.21（★★☆）
> 考虑两个非负数 $a$ 和 $b$，证明，如果 $a \leq b$，那么 $a \leq \sqrt{ab}$. 使用这个结果证明，如果二分类问题的决策区域被选择为最小化误分类的概率，那么这个概率满足
> $$
> p(\text{误分类}) \leq \int \sqrt{p(x,C_1)p(x,C_2)} \, \mathrm{d}x \tag{1.150}
> $$

先证明 $a \leqslant b$ 时，$a \leqslant (ab)^{1/2}$

因为$a \leqslant b$，两边同时乘以 $a$（因为 $a \geq 0$，这个操作保持不等式方向）得到 $a^2 \leqslant ab$. 对两边开平方根（因为 $a, b \geq 0$，开平方根操作也是合法的）得到 $a \leqslant (ab)^{1/2}$. 这个结论显而易见. 

误分类概率表达式：
$$
 p(\text{错误}) = \int p(x) \min \left[ p(C_1|x), p(C_2|x) \right] dx 
$$

利用前面的结论：对于任意 $x$，有 $\min \left[ p(C_1| x), p(C_2| x) \right] \leq \left( p(C_1| x) p(C_2| x) \right)^{1/2}$. 代入得到：
$$
 p(\text{错误}) \leq \int p(x) \left( p(C_1| x) p(C_2| x) \right)^{1/2} dx 
$$

利用全概率公式 $p(x) = p(x, C_1) + p(x, C_2)$，可以得到：
$$
 p(\text{错误}) \leq \int \left( p(x, C_1) p(x, C_2) \right)^{1/2} dx 
$$

> [!NOTE] 练习 1.22
> 给定一个损失矩阵，其元素为 $L_{ki}$，如果对于每个 $c$，我们都选择使公式 (1.81) 取得最小值的类别，那么期望风险会最小. 证明，如果损失矩阵为 $L_{kj}=1-I_{kj}$，其中 $I_k$ 是单位矩阵的元素，那么选择类别的方法就变成了选择具有最大后验概率的类别. 这种形式的损失矩阵的意义是什么？

将 $L_{kj} = 1 - \delta_{kj}$ 代入式 (1.81)，并利用后验概率之和为一的事实，我们发现对于每个 $x$，我们应该选择使得 $1 - p(C_j| x)$ 最小的类别 $j$，这等价于选择使得后验概率 $p(C_j| x)$ 最大的类别 $j$. 该损失矩阵在样本被错误分类时赋予损失为 1，在正确分类时损失为 0，因此最小化期望损失等价于最小化误分类率. 

> [!NOTE] 练习 1.23
> 对于一般的损失矩阵和一般的类先验概率，推导最小化期望损失的准则. 

这个问题很有代表性. 让我们考虑一个一般的模式识别分类问题，假设在这个问题中有K个类别$C_1,C_2,\dots,C_K$. 在观测数据为x的条件下，后验概率为$p(C_k|x)$. 那么，对于决策j，每一类的损失为$L_{kj}$. 故而整体的平均损失为：
$$
f = \sum_{k=1}^K L_{kj}p(C_k|x)
$$

我们不难看出，在这个问题中的决策变量正是j，是一个目标函数求极小值的问题. 由此我们得到了更为一般的贝叶斯决策准则：

$$
\begin{align*}
\boxed{
\text{决定} x \in C_{j}, \quad \text{其中} j^* = \arg\min_j \sum_{k=1}^K L_{kj} \, p(C_k \mid x)
}
\end{align*}
$$

这便是在一般损失矩阵和任意先验概率下，使期望损失最小化的最优决策准则. 

特例验证: 对于0–1 损失：$L_{kj} = 1 - \delta_{kj}$，此时最小化风险等价于最大化后验概率，即最大后验（MAP）准则. 
非对称损失：例如在医学诊断中，将病人误判为健康（漏诊）的损失远大于将健康人误判为病人（误报），此时 $L_{\text{病},\text{健康}} \gg L_{\text{健康},\text{病}}$，决策边界会向高风险方向偏移. 

> [!NOTE] 练习 1.24
> 考虑一个分类问题. 这个问题中，把来自类别 $C_k$ 的输入向量分类为类别 $C_j$ 所造成的损失由损失矩阵 $L_{kj}$ 给出. 并且，选择拒绝选项所造成的损失为 $\lambda$. 找到最小化期望损失的决策准则. 证明，当损失矩阵为 $L_{kj}=1-I_{kj}$ 时，这个结果就变成了 1.5.3 节讨论的拒绝准则. $\lambda$ 和拒绝阈值之间的关系是什么？

向量 x 属于类别 $C_k$ 的概率为 $p(C_k|x)$. 如果我们决定将 x 分配给类别 $C_j$，我们将产生期望损失 $\sum_k L_{kj}p(C_k|x)$；而如果我们选择拒绝选项，则会产生损失 $\lambda$. 因此，如果

$$
j = \arg\min_l \sum_k L_{kl}p(C_k|x)
$$

那么当我们采取以下行动时，就可以最小化期望损失：

$$
\text{choose}= \left\{
\begin{array}{ll}
\text{class } j, & \text{if } \min_l \sum_k L_{kl}p(C_k \boldsymbol{x}) < \lambda; \\
\text{reject}, & \text{otherwise}.
\end{array}
\right.
$$

对于损失矩阵 $L_{kj} = 1 - I_{kj}$，我们有 $\sum_k L_{kl}p(C_k|x) = 1 - p(C_l|x)$，因此除非 $1 - p(C_l|x)$ 的最小值小于 $\lambda$，否则我们不拒绝；等价地，即当 $p(C_l|x)$ 的最大值小于 $1 - \lambda$ 时拒绝. 在标准的拒绝准则中，当最大后验概率小于某个阈值 $\theta$ 时我们拒绝. 因此，只要 $\theta = 1 - \lambda$，这两个拒绝准则就是等价的. 

> [!NOTE] 练习 1.25
> 考虑将一元目标变量 $t$ 的平方和损失函数 (1.87) 推广到多元目标变量 $\boldsymbol{t}$. 推广后的形式为
> $$
> E[L(\boldsymbol{t},y(\boldsymbol{x}))] = \iint \|\boldsymbol{y}(\boldsymbol{x})-\boldsymbol{t}\|^2 p(\boldsymbol{x},\boldsymbol{t}) \, \mathrm{d}\boldsymbol{x} \, \mathrm{d}\boldsymbol{t} \tag{1.151}
> $$
> 使用变分法，证明使得这个期望损失取得最小值的函数 $y(\boldsymbol{x})$ 为 $y(\boldsymbol{x}) = E[\boldsymbol{t}|\boldsymbol{x}]$. 证明，对于一元目标变量 $t$，这个结果就变成了公式 (1.89) 给出的结果. 

所谓变分法求预期损失最小化，本质上是想求一个极值点的存在：
$$
\frac{\partial E}{\partial \boldsymbol y} = 0
$$

而我们若对这个二重积分求导，可以拆解成一个一重积分：由于y是有关x的函数，因而消除掉dx：
$$
\frac{\partial E}{\partial \boldsymbol{y(x)}} =\int 2(\boldsymbol{y(x)-t})p(\boldsymbol{t,x}) d\boldsymbol{t} = 0
$$

常数2没什么用可以拿掉，然后将减号拆开、移项可以得到：
$$
\int \boldsymbol{y(x)}p(\boldsymbol{t,x}) d\boldsymbol{t} = \int \boldsymbol{t}p(\boldsymbol{t,x}) d\boldsymbol{t} 
$$

注意到y(x)和t没什么关系也可以拿出来，于是移项又变成了：
$$
\boldsymbol{y(x)} = \frac{\int \boldsymbol{t}p(\boldsymbol{t,x}) d\boldsymbol{t}}{\int p(\boldsymbol{t,x}) d\boldsymbol{t}} = \int \boldsymbol{t}p(\boldsymbol{t|x}) d\boldsymbol{t}
$$

这个结果事实上正是$\boldsymbol y(\boldsymbol x)=\mathbb E_{\boldsymbol t}[\boldsymbol t|\boldsymbol x]$. 在单一目标变量 t 的情况下，这一结果也就显然可简化为式（1.89）. 

> [!NOTE] 练习 1.26
> 通过将公式 (1.151) 中的平方项展开，推导类似于公式 (1.90) 的结果，证明，对于目标变量组成向量 $\boldsymbol{t}$ 的情形，最小化期望平方损失的函数 $y(\boldsymbol{x})$ 仍然是 $\boldsymbol{t}$ 的条件期望. 

题目要求我们推导在单目标变量 $t$ 的情况下，使预期平方损失最小化的函数 $y(x)$ 的表达式. 具体步骤如下：

- 定义预期平方损失函数：
$$
E[L] = \iint || y(x) - t ||^2 p(x, t) \, dx \, dt
$$

其中 $y(x)$ 是我们要求的函数，$t$ 是目标变量，$p(x, t)$ 是联合概率密度函数. 

- 将平方项展开：这里玩了个小技巧就是先借再还，也是我们解代数问题的时候经常使用的一个trick. 至于为什么借条件期望，也是我们想往问题的结论上凑配. 
$$
\{ y(x) - t \}^2 = \{ y(x) - E[t| x] + E[t| x] - t \}^2
$$
这可以进一步展开为完全平方公式：
$$
\{ y(x) - E[t| x] \}^2 + \{ E[t| x] - t \}^2 + 2 \{ y(x) - E[t| x] \} \{ E[t| x] - t \}
$$

- 代入损失函数：
$$
E[L] = \iint \left( || y(x) - E[t| x] ||^2 + || E[t| x] - t ||^2 + 2 \{ y(x) - E[t| x] \} \{ E[t| x] - t \} \right) p(x, t) \, dx \, dt
$$

- 分别计算每一项的期望：

  1. 第一项：
  $$
  \iint || y(x) - E[t| x] ||^2 p(x, t) \, dx \, dt = \int || y(x) - E[t| x] ||^2 p(x) \, dx
  $$

  2. 第二项：
  $$
  \iint \{ E[t| x] - t \}^2 p(x, t) \, dx \, dt
  $$
  和y没关系，可以不管. 

  3. 第三项：
  $$
  \int 2 \{ y(x) - E[t| x] \} \{ E[t| x] - t \} p(x, t) \, dx \, dt = 0
  $$
  （因为 $E \{ E[t| x] - t \} = 0$）

- 最小化损失函数：
为了使 $E[L]$ 最小，我们需要最小化：
$$
\int \{ y(x) - E[t| x] \}^2 p(x) \, dx
$$
这显然在 $y(x) = E[t| x]$ 时取得最小值. 

> [!NOTE] 练习 1.27
> 考虑回归问题的期望损失，损失函数为公式 (1.91) 给出的 $L_q$. 写出为了最小化 $E[L_q]$，$y(\boldsymbol{x})$ 必须满足的条件. 证明，对于 $q=1$，这个解表示条件中位数，即函数 $y(\boldsymbol{x})$ 使 $t<y(\boldsymbol{x})$ 的概率质量与 $t \geq y(\boldsymbol{x})$ 的概率质量相同. 并且证明，对于 $q \to 0$，最小的期望 $L_q$ 误差为条件众数，即函数 $y(\boldsymbol{x})$ 等于最大化 $p(t|\boldsymbol{x})$ 的 $t$ 值. 

由于我们可以对每个 $x$ 的值独立地选择 $y(x)$，因此期望 $L_q$ 损失的最小值可以通过最小化如下被积函数来找到：

$$
\int |y(\boldsymbol{x}) - t|^q p(t|\boldsymbol{x}) \, dt \tag{42}
$$

对于每个 $\boldsymbol{x}$ 的值. 将式 (42) 对 $y(\boldsymbol{x})$ 求导并令其为零，得到平稳性条件：

$$
\int q|y(\boldsymbol{x}) - t|^{q-1} \text{sign}(y(\boldsymbol{x}) - t) p(t|\boldsymbol{x}) \, dt
$$

$$
= q \int_{-\infty}^{y(\boldsymbol{x})} |y(\boldsymbol{x}) - t|^{q-1} p(t|\boldsymbol{x}) \, dt - q \int_{y(\boldsymbol{x})}^{\infty} |y(\boldsymbol{x}) - t|^{q-1} p(t|\boldsymbol{x}) \, dt = 0
$$

这也可以通过直接将式 (1.91) 关于 $y(\boldsymbol{x})$ 的泛函导数设为零得到. 由此可知，$y(\boldsymbol{x})$ 必须满足：

$$
\int_{-\infty}^{y(\boldsymbol{x})} |y(\boldsymbol{x}) - t|^{q-1} p(t|\boldsymbol{x}) \, dt = \int_{y(\boldsymbol{x})}^{\infty} |y(\boldsymbol{x}) - t|^{q-1} p(t|\boldsymbol{x}) \, dt \tag{43}
$$

当 $q = 1$ 时，上式简化为：

$$
\int_{-\infty}^{y(\boldsymbol{x})} p(t|\boldsymbol{x}) \, dt = \int_{y(\boldsymbol{x})}^{\infty} p(t|\boldsymbol{x}) \, dt \tag{44}
$$

这表明 $y(\boldsymbol{x})$ 必须是 $t$ 的条件中位数. 

当 $q \to 0$ 时，我们注意到，作为 $t$ 的函数，量 $|y(\boldsymbol{x}) - t|^q$ 在除 $t = y(\boldsymbol{x})$ 附近的小邻域外几乎处处接近 1，在 $t = y(\boldsymbol{x})$ 附近趋近于零. 因此，式 (42) 的值将接近 1（因为密度 $p(t)$ 是归一化的），但会因靠近 $t = y(\boldsymbol{x})$ 处的“凹陷”而略微减小. 我们通过将凹陷的位置与 $p(t)$ 的最大值重合来实现式 (42) 的最大减小，即取条件众数. 

> [!NOTE] 练习 1.28
> 在 1.6 节，我们介绍了熵 $h(c)$ 的思想，即观察到概率分布为 $p(c)$ 的随机变量 $c$ 的值之后所获得的信息. 我们看到，对于独立的变量 $r$ 和 $y$，有 $p(c,y)=p(c)p(y)$，且熵函数是可加的，即 $h(c,y)=h(c)+h(y)$. 在这个练习中，我们推导 $h$ 和 $p$ 的函数关系 $h(p)$. 首先证明 $h(p^2)=2h(p)$，因此通过数学归纳法，有 $h(p^n)=nh(p)$，其中 $n$ 是正整数. 因此，证明 $h(p^m)=(\frac{m}{n})h(p)$，其中 $m$ 也是一个正整数. 这表明 $h(p^x)=xh(p)$，其中 $x$ 是一个正有理数. 从而根据连续性，这个结果对于 $r$ 是正实数的情形也成立. 最后，证明上述结果表明了 $h(p)$ 的形式一定为 $h(p) \propto \ln p$. 

先解决子问题1. 

既然题目的那个汇总提到：$h(x,y)=h(x)+h(y)$，不失一般性，令 $x=y$，那么$h(p^2)=h(p,p)=h(p)+h(p)=2h(p)$. 进一步，使用数学归纳法：
- 当n=2时，$h(p^2)=2h(p)$成立
- 假设当n=k的时候成立，即$h(p^k)=kh(p)$成立
- 那么当n=k+1时，$h(p^{k+1})=h(p,p^k)=h(p)+h(p^k)=(k+1)h(p)$，即$h(p^{k+1})=(k+1)h(p)$成立. 

然后解决子问题2. 

- 设$q=p^{n/m}$，那么$q^m=p^n$，即$h(q^m)=h(p^n)$
- 根据子问题1的推论，$h(q^m)=mh(q)$成立，$h(p^n)=nh(q)$成立
- 因此$nh(q)mh(p^n)$成立，得到$h(q) = (n/m)h(p)$，问题得证

最后解决子问题3. 

在解决子问题3时，我们会使用幂函数求导一个非常典型的套路，就是指数化. 考虑到$p=e^{\ln p}$，那么我们有：
$h(p)=h(e^{\ln p})=h(e)\ln p = C\ln p$

> [!NOTE] 练习 1.29
> 考虑一个 $M$ 状态的离散随机变量 $c$，使用公式 (1.115) 给出的 Jensen 不等式证明概率分布 $p(c)$ 的熵满足 $H[c] \leq \ln M$. 

Jensen不等式是指对于一个凸函数f(x)而言，有：
$$
\frac{1}{n} \sum_{i=1}^n f(x_i) \le f(\frac{1}{n}\sum_{i=1}^n x_i)
$$

那么$y=\ln x$显然是个凸函数. 让我们回到问题：
一个具有 $M$ 个状态的离散变量 $x$ 的熵可以表示为：

$$
H(x) = -\sum_{i=1}^{M} p(x_i) \ln p(x_i) = \sum_{i=1}^{M} p(x_i) \ln \frac{1}{p(x_i)}
$$

函数 $\ln(x)$ 是凹函数，因此我们可以应用 Jensen 不等式（形式如 (1.115)，但不等式方向反转），从而得到：

$$
H(x) \leq \ln \left( \sum_{i=1}^{M} p(x_i) \cdot \frac{1}{p(x_i)} \right) = \ln M
$$

> [!NOTE] 练习 1.30
> 计算两个高斯分布 $p(c)=N(c|μ,σ^2)$ 和 $q(c)=N(c|m,s^2)$ 之间的由公式 (1.113) 给出的 Kullback-Leibler 散度. 

高斯概率密度函数为：
$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right), \quad
q(x) = \frac{1}{\sqrt{2\pi s^2}} \exp\left( -\frac{(x - m)^2}{2s^2} \right).
$$

计算：
$$
\log \frac{p(x)}{q(x)} =
\frac{1}{2} \log \frac{s^2}{\sigma^2}
\frac{(x - m)^2}{2s^2} - \frac{(x - \mu)^2}{2\sigma^2}.
$$

然后对 $p(x)$ 求期望：
$\mathbb{E}_p[(x - \mu)^2] = \sigma^2$,
$\mathbb{E}_p[(x - m)^2] = \mathbb{E}_p[(x - \mu + \mu - m)^2] = \sigma^2 + (\mu - m)^2$.

代入后整理即得：

$$
\boxed{
\mathrm{KL}\big(p \parallel q\big)
= \frac{1}{2} \left( \log \frac{s^2}{\sigma^2} + \frac{\sigma^2 + (\mu - m)^2}{s^2} - 1 \right)
}
$$

> [!NOTE] 练习 1.31
> 考虑两个变量 $c$ 和 $y$，联合概率分布为 $p(c,y)$. 证明这对变量的微分熵满足
> $$
> H[c,y] \leq H[c] + H[y] \tag{1.152}
> $$
> 当且仅当 $c$ 和 $y$ 统计独立时等号成立. 

我们首先利用在 (1.121) 中得到的关系式 $I(\boldsymbol{x}; \boldsymbol{y}) = H(\boldsymbol{y}) - H(\boldsymbol{y}|\boldsymbol{x})$，并注意到互信息满足 $I(\boldsymbol{x}; \boldsymbol{y}) \geq 0$，因为它是KL散度的一种形式. 最后，我们利用关系式 (1.112) 来得出所需的结果 (1.152). 

为了证明统计独立性是等式成立的充分条件，我们将 $p(\boldsymbol{x}, \boldsymbol{y}) = p(\boldsymbol{x})p(\boldsymbol{y})$ 代入熵的定义中，得到：

$$
\begin{aligned}
H(\boldsymbol{x}, \boldsymbol{y}) 
&= \iint p(\boldsymbol{x}, \boldsymbol{y}) \ln p(\boldsymbol{x}, \boldsymbol{y}) \, d\boldsymbol{x}\,d\boldsymbol{y} \\
&= \iint p(\boldsymbol{x})p(\boldsymbol{y}) \left\{ \ln p(\boldsymbol{x}) + \ln p(\boldsymbol{y}) \right\} \, d\boldsymbol{x}\,d\boldsymbol{y} \\
&= \int p(\boldsymbol{x}) \ln p(\boldsymbol{x}) \, d\boldsymbol{x} + \int p(\boldsymbol{y}) \ln p(\boldsymbol{y}) \, d\boldsymbol{y} \\
&= H(\boldsymbol{x}) + H(\boldsymbol{y}).
\end{aligned}
$$

为了证明统计独立性是必要条件，我们将等式条件：
$$
H(\boldsymbol{x}, \boldsymbol{y}) = H(\boldsymbol{x}) + H(\boldsymbol{y})
$$
与结果 (1.112) 结合，可得：
$$
H(\boldsymbol{y}|\boldsymbol{x}) = H(\boldsymbol{y}).
$$

我们现在注意到，右边与 $\boldsymbol{x}$ 无关，因此左边也必须关于 $\boldsymbol{x}$ 是常数. 利用 (1.121)，可以推出互信息 $I[\boldsymbol{x}; \boldsymbol{y}] = 0$. 最后，根据 (1.120)，我们知道互信息是一种 KL 散度的形式，而它仅在两个分布相等时为零，因此有 $p(\boldsymbol{x}, \boldsymbol{y}) = p(\boldsymbol{x})p(\boldsymbol{y})$，即满足要求. 

> [!NOTE] 练习 1.32
> 考虑一个连续向量 $\boldsymbol{x}$，概率分布为 $p(\boldsymbol{x})$，对应的熵为 $H[\boldsymbol{x}]$. 假设我们对 $\boldsymbol{x}$ 进行了一个非奇异的线性变换，得到一个新的变量 $\boldsymbol{y} = A\boldsymbol{x}$. 证明对应的熵为 $H[\boldsymbol{y}] = H[\boldsymbol{x}] + \ln|\det(A)|$，其中 $\det(A)$ 表示 $A$ 的行列式的值. 

要证明在对连续变量向量 $\boldsymbol{x}$ 进行非奇异线性变换 $\boldsymbol{y} = \boldsymbol{A}\boldsymbol{x}$ 后，相应的熵 $H[\boldsymbol{y}]$ 与原熵 $H[\boldsymbol{x}]$ 之间的关系为 $H[\boldsymbol{y}] = H[\boldsymbol{x}] + \ln \boldsymbol{A} $，其中 $ \boldsymbol{A} $ 表示矩阵 $\boldsymbol{A}$ 的行列式，我们可以按照以下步骤进行：
步骤 1: 定义熵

对于连续随机变量 $\boldsymbol{x}$，其微分熵定义为：
$$
H[\boldsymbol{x}] = -\int p(\boldsymbol{x}) \ln p(\boldsymbol{x}) d\boldsymbol{x}
$$
步骤 2: 变换后的概率密度函数

给定 $\boldsymbol{y} = \boldsymbol{A}\boldsymbol{x}$，其中 $\boldsymbol{A}$ 是一个非奇异矩阵（即 $\det(\boldsymbol{A}) \neq 0$），则 $\boldsymbol{x} = \boldsymbol{A}^{-1}\boldsymbol{y}$. 根据概率密度函数的变换规则，$\boldsymbol{y}$ 的概率密度函数 $p(\boldsymbol{y})$ 与 $\boldsymbol{x}$ 的概率密度函数 $p(\boldsymbol{x})$ 之间的关系为：
$$
p(\boldsymbol{y}) = p(\boldsymbol{A}^{-1}\boldsymbol{y}) \det(\boldsymbol{A}^{-1})
$$
由于 $\det(\boldsymbol{A}^{-1}) = 1 / \det(\boldsymbol{A})$，我们有：
$$
p(\boldsymbol{y}) = p(\boldsymbol{A}^{-1}\boldsymbol{y}) \frac{1}{ \det(\boldsymbol{A}) }
$$
步骤 3: 计算变换后的熵

变换后 $\boldsymbol{y}$ 的熵 $H[\boldsymbol{y}]$ 为：
$$
H[\boldsymbol{y}] = -\int p(\boldsymbol{y}) \ln p(\boldsymbol{y}) d\boldsymbol{y}
$$
将 $p(\boldsymbol{y})$ 的表达式代入上式，得到：
$$
H[\boldsymbol{y}] = -\int p(\boldsymbol{A}^{-1}\boldsymbol{y}) \frac{1}{ \det(\boldsymbol{A}) } \ln \left(p(\boldsymbol{A}^{-1}\boldsymbol{y}) \frac{1}{ \det(\boldsymbol{A}) }\right) d\boldsymbol{y}
$$
利用对数的性质，可以将其拆分为两部分：
$$
H[\boldsymbol{y}] = -\int p(\boldsymbol{A}^{-1}\boldsymbol{y}) \frac{1}{ \det(\boldsymbol{A}) } \left(\ln p(\boldsymbol{A}^{-1}\boldsymbol{y}) - \ln \det(\boldsymbol{A}) \right) d\boldsymbol{y}
$$
$$
= -\int p(\boldsymbol{A}^{-1}\boldsymbol{y}) \frac{1}{ \det(\boldsymbol{A}) } \ln p(\boldsymbol{A}^{-1}\boldsymbol{y}) d\boldsymbol{y} + \ln \det(\boldsymbol{A}) \int p(\boldsymbol{A}^{-1}\boldsymbol{y}) \frac{1}{ \det(\boldsymbol{A}) } d\boldsymbol{y}
$$

注意到第二项中的积分等于 1（因为它是概率密度函数的积分），所以我们有：
$$
H[\boldsymbol{y}] = -\int p(\boldsymbol{A}^{-1}\boldsymbol{y}) \frac{1}{ \det(\boldsymbol{A}) } \ln p(\boldsymbol{A}^{-1}\boldsymbol{y}) d\boldsymbol{y} + \ln \det(\boldsymbol{A})
$$
步骤 4: 变量替换

令 $\boldsymbol{z} = \boldsymbol{A}^{-1}\boldsymbol{y}$，则 $d\boldsymbol{y} = \det(\boldsymbol{A}) d\boldsymbol{z}$，代入上式得：
$$
H[\boldsymbol{y}] = -\int p(\boldsymbol{z}) \ln p(\boldsymbol{z}) d\boldsymbol{z} + \ln \det(\boldsymbol{A})
$$
$$
= H[\boldsymbol{x}] + \ln \det(\boldsymbol{A})
$$

因此，我们证明了 $H[\boldsymbol{y}] = H[\boldsymbol{x}] + \ln \boldsymbol{A}$. 

> [!NOTE] 练习 1.33
> 假设两个离散随机变量 $r$ 和 $y$ 的条件熵 $H[y|x]$ 为零. 证明，对于所有的满足 $p(x)>0$ 的 $x$，变量 $y$ 一定是 $c$ 的函数. 换句话说，对于每个 $r$，只有一个 $y$ 的值使得 $p(y|x) \neq 0$. 

条件熵定义  
$$
H[y|x]=\sum_{x}p(x)\,H[y|x=x].
$$

对任意 $x$，条件分布熵  
$$
H[y|x=x]=-\sum_{y}p(y|x)\log p(y|x)\geq 0.
$$
因此整体和式 $H[y|x]\geq 0$. 

给定 $H[y|x]=0$，而每一项非负，故对任意 $p(x)>0$ 的 $x$，必有  
$$
H[y|x=x]=0.
$$

熵为零当且仅当分布是确定性的，即存在唯一 $y_x$ 使得  
$$
p(y_x|x)=1,\quad \forall y\neq y_x,\;p(y|x)=0.
$$
这意味着对每个 $p(x)>0$ 的 $x$，$y$ 只能取一个值 $y_x$. 

因此 $y$ 是 $x$ 的函数：  
$$
y=y_x,\quad \text{当}\;x\;\text{给定且}\;p(x)>0.
$$
换句话说，对每个 $x$ 满足 $p(x)>0$，只有一个 $y$ 使 $p(y|x)\neq 0$. 

> [!NOTE] 练习 1.34
> 使用变分法证明公式 (1.108) 之前的泛函的驻点由公式 (1.108) 给出. 然后使用限制条件 (1.105)、(1.106) 和 (1.107)，消去拉格朗日乘数，从而证明最大熵的解由高斯分布 (1.109) 给出. 

获得所需的泛函导数可以通过直接观察来完成. 然而，如果需要更正式的方法，我们可以使用附录 D 中介绍的技术进行推导. 首先考虑如下泛函：

$$
I[p(x)] = \int p(x) f(x) \, dx.
$$

在小扰动 $ p(x) \to p(x) + \epsilon \eta(x) $ 下，有：

$$
I[p(x) + \epsilon \eta(x)] = \int p(x) f(x) \, dx + \epsilon \int \eta(x) f(x) \, dx,
$$

因此，根据 (D.3)，可以推出泛函导数为：

$$
\frac{\delta I}{\delta p(x)} = f(x).
$$

类似地，如果我们定义：

$$
J[p(x)] = \int p(x) \ln p(x) \, dx,
$$

那么在小扰动 $ p(x) \to p(x) + \epsilon \eta(x) $ 下，有：

$$
\begin{aligned}
J[p(x) + \epsilon \eta(x)] 
&= \int p(x) \ln p(x) \, dx \\
&\quad + \epsilon \left\{ \int \eta(x) \ln p(x) \, dx + \int p(x) \frac{1}{p(x)} \eta(x) \, dx \right\} + O(\epsilon^2)
\end{aligned}
$$

因此，

$$
\frac{\delta J}{\delta p(x)} = \ln p(x) + 1.
$$

利用这两个结果，我们得到如下泛函导数的表达式：

$$
-\ln p(x) - 1 + \lambda_1 + \lambda_2 x + \lambda_3 (x - \mu)^2.
$$

重新整理后即得 (1.108). 

为了消去拉格朗日乘子，我们将 (1.108) 依次代入三个约束条件 (1.105)、(1.106) 和 (1.107). 该解最简便的方法是与高斯分布的标准形式进行比较，并注意到结果确实满足三个约束条件：

$$
\lambda_1 = 1 - \frac{1}{2} \ln(2\pi\sigma^2)
$$

$$
\lambda_2 = 0
$$

$$
\lambda_3 = \frac{1}{2\sigma^2}
$$


> [!NOTE] 练习 1.35
> 使用公式 (1.106) 和公式 (1.107) 的结果，证明一元高斯分布 (1.109) 的熵为 (1.110). 

$$
\begin{aligned}
H[x] &= -\int p(x) \ln p(x) \, dx \\
&= -\int p(x) \left( -\frac{1}{2} \ln(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2} \right) dx \\
&= \frac{1}{2} \left( \ln(2\pi\sigma^2) + \frac{1}{\sigma^2} \int p(x)(x - \mu)^2 dx \right) \\
&= \frac{1}{2} \left( \ln(2\pi\sigma^2) + 1 \right),
\end{aligned}
$$

> [!NOTE] 练习 1.36
> 一个严格凸函数的定义为：每条弦都位于函数图像上方的函数. 证明，这等价于函数的二阶导数为正. 

严格凸函数的定义是：如果对于函数 $f$ 的任意两点 $x_1$ 和 $x_2$ ($x_1 \neq x_2$)，以及任意 $\lambda \in (0, 1)$，都有

$$
f(\lambda x_1 + (1 - \lambda) x_2) < \lambda f(x_1) + (1 - \lambda) f(x_2),
$$

则称 $f$ 是严格凸的. 

要证明这个定义等同于二阶导数为正（即 $f''(x) > 0$ 对所有 $x$ 都成立），我们可以从两个方向进行证明：
方向一：假设 $f$ 是严格凸的，证明 $f''(x) > 0$

由于 $f$ 是严格凸的，对于任何两点 $x_1$ 和 $x_2$，以及 $\lambda \in (0, 1)$，上面的不等式都成立. 考虑固定 $x_1$ 和 $x_2$，并且让 $\lambda$ 接近 0 或 1. 在这种情况下，不等式表明函数图形上的点 $(\lambda x_1 + (1 - \lambda)x_2, f(\lambda x_1 + (1 - \lambda)x_2))$ 总是在连接点 $(x_1, f(x_1))$ 和 $(x_2, f(x_2))$ 的直线之下. 

现在，让我们考虑函数在某一点 $x_0$ 处的泰勒展开. 如果我们只取到二阶项，我们有：

$$
f(x) = f(x_0) + f'(x_0)(x - x_0) + \frac{1}{2}f''(x_0)(x - x_0)^2 + o((x - x_0)^2).
$$

如果 $f''(x_0) < 0$，那么在 $x_0$ 附近存在某些点使得函数值会低于通过该点的切线，这与严格凸性的定义相矛盾. 因此，为了满足严格凸性，必须有 $f''(x) > 0$ 对所有 $x$ 都成立. 
方向二：假设 $f''(x) > 0$ 对所有 $x$ 成立，证明 $f$ 是严格凸的

假设 $f''(x) > 0$ 对所有 $x$ 都成立. 这意味着函数 $f$ 在其定义域内总是向上弯曲的. 为了证明 $f$ 是严格凸的，我们需要证明对于任意不同的 $x_1$ 和 $x_2$ 以及任意 $\lambda \in (0, 1)$，都有上述的严格不等式成立. 

根据拉格朗日中值定理，对于任意 $x_1 < x < x_2$，存在一个 $\xi \in (x_1, x_2)$ 使得

$$
f'(x) = f'(x_1) + f''(\xi)(x - x_1).
$$

因为 $f''(x) > 0$，我们知道 $f'$ 是严格增加的. 所以，对于任何 $\lambda \in (0, 1)$，设 $x = \lambda x_1 + (1 - \lambda)x_2$，我们有

$$
f'(\lambda x_1 + (1 - \lambda)x_2) < \lambda f'(x_1) + (1 - \lambda)f'(x_2),
$$

这是因为 $f'$ 是严格增加的，且 $\lambda x_1 + (1 - \lambda)x_2$ 在 $x_1$ 和 $x_2$ 之间. 接着应用拉格朗日中值定理两次，分别对区间 $[x_1, x]$ 和 $[x, x_2]$，可以得到

$$
f(\lambda x_1 + (1 - \lambda)x_2) < \lambda f(x_1) + (1 - \lambda)f(x_2),
$$

这就证明了 $f$ 是严格凸的. 因此，我们完成了两个方向的证明，确认了严格凸性和二阶导数为正之间的等价关系. 

> [!NOTE] 练习 1.37
> 使用定义 (1.111) 以及概率的乘积规则，证明公式 (1.112) 的结果. 

$$
\begin{align*}
    H[y|x] &= -\iint p(y,x) \left[\ln p(y|x) \right]dydx \\
    &= -\iint p(y,x) \left[ \ln \frac{p(y,x)}{p(x)} \right]dydx \\
    &= -\iint p(y,x) \left[ \ln p(y,x) - \ln p(x) \right]dydx \\
    &= H[y,x] - (-\iint p(y,x)\ln p(x) dydx)\\
    &= H[y,x] - H[x]
\end{align*}
$$

> [!NOTE] 练习 1.38
> 使用归纳法，证明从凸函数的不等式 (1.114) 可以推导出公式 (1.115). 

由公式 (1.114) 可知，当 $ M = 1 $ 时，结果 (1.115) 成立. 我们现在假设该结果对某个一般值 $ M $ 成立，并证明它必然也对 $ M + 1 $ 成立. 

考虑公式 (1.115) 左边的表达式：

$$
f\left( \sum_{i=1}^{M+1} \lambda_i x_i \right)
= f\left( \lambda_{M+1} x_{M+1} + \sum_{i=1}^{M} \lambda_i x_i \right) \tag{50}
$$

可以改写为：

$$
= f\left( \lambda_{M+1} x_{M+1} + (1 - \lambda_{M+1}) \sum_{i=1}^{M} \eta_i x_i \right) \tag{51}
$$

其中我们定义了：

$$
\eta_i = \frac{\lambda_i}{1 - \lambda_{M+1}} \tag{52}
$$

我们现在应用公式 (1.114)，得到：

$$
f\left( \sum_{i=1}^{M+1} \lambda_i x_i \right) \leq \lambda_{M+1} f(x_{M+1}) + (1 - \lambda_{M+1}) f\left( \sum_{i=1}^{M} \eta_i x_i \right). \tag{53}
$$

我们现在注意到，根据定义，系数 $\lambda_i$ 满足：

$$
\sum_{i=1}^{M+1} \lambda_i = 1 \tag{54}
$$

因此有：

$$
\sum_{i=1}^{M} \lambda_i = 1 - \lambda_{M+1}. \tag{55}
$$

接着，利用公式 (52)，我们可以看出 $\eta_i$ 满足性质：

$$
\sum_{i=1}^{M} \eta_i = \frac{1}{1 - \lambda_{M+1}} \sum_{i=1}^{M} \lambda_i = 1. \tag{56}
$$

因此，我们可以对阶数 $ M $ 应用结果 (1.115)，于是 (53) 变为：

$$
f\left( \sum_{i=1}^{M+1} \lambda_i x_i \right) \leq \lambda_{M+1} f(x_{M+1}) + (1 - \lambda_{M+1}) \sum_{i=1}^{M} \eta_i f(x_i) = \sum_{i=1}^{M+1} \lambda_i f(x_i), \tag{57}
$$

其中我们使用了公式 (52). 

> [!NOTE] 练习 1.39
> 考虑两个变量 $r$ 和 $y$，每个变量只有两个可能的取值. 它们的联合概率分布在表 1.3 中给出. 计算下面各式的值，画一个图说明这些量之间的关系. 
> $$
> H[x] \quad H[y|x] \quad H[x,y] \quad H[y] \quad H[x|y] \quad I[x,y]
> $$
> **表 1.3**：练习 1.39 使用的两个二值变量 $x$ 和 $y$ 的联合概率分布. 行表示 $r$ 的值，列表示 $y$ 的值. 
> $$
> \begin{array}{c|cc}
>  & y=0 & y=1 \\
> \hline
> x=0 & 1/3 & 1/3 \\
> x=1 & 0 & 1/3
> \end{array}
> $$

(a) H(x) = -1/3 * log(1/3) - 2/3 * log(2/3) = 0.9183

(b) H(y) = -1/3 * log(1/3) - 2/3 * log(2/3) = 0.9183

(c) H(y|x) = p(x=0)H(y|x=0) + q(x=1)H(y|x=1)

p(y=0|x=0) = p(y=1|x=0) = $\frac{1/3}{2/3}$ = 0.5

H(y|x=0) = -(0.5*log(0.5) + 0.5*log(0.5)) = 1

H(y|x=1) = 0

H(y|x) = 2/3 * 1 + 1/3 * 0 = 2/3

(d) H(x|y) = 2/3，方法类似

(e) H(x,y) = -3*(1/3*log(1/3)) = 1.5850

(f) I(x,y) = H(x) + H(y) - H(x,y) = 0.2516

> [!NOTE] 练习 1.40
> 使用 Jensen 不等式 (1.115)，其中 $f(c)=\ln c$，证明一组实数的算术平均值永远不小于它们的几何平均值. 

第一步：确认 $f(x) = \ln x$ 是凹函数

函数 $f(x) = \ln x$ 在定义域 $x > 0$ 上是严格凹函数，因为其二阶导数为：
$$
f''(x) = -\frac{1}{x^2} < 0 \quad (\forall x > 0).
$$

第二步：应用 Jensen 不等式（凹函数版本）

对于凹函数 $f$，Jensen 不等式（见式 (1.115)）指出：
$$
f\!\left( \sum_{i=1}^n \lambda_i x_i \right) \;\ge\; \sum_{i=1}^n \lambda_i f(x_i),
$$
其中 $\lambda_i \ge 0$ 且 $\sum_{i=1}^n \lambda_i = 1$. 

取均匀权重：$\lambda_i = \frac{1}{n}$，则有：
$$
\ln\!\left( \frac{1}{n} \sum_{i=1}^n x_i \right) \;\ge\; \frac{1}{n} \sum_{i=1}^n \ln x_i.
$$

第三步：化简右边

利用对数的性质：
$$
\frac{1}{n} \sum_{i=1}^n \ln x_i = \ln \left( \prod_{i=1}^n x_i \right)^{1/n}.
$$

因此不等式变为：
$$
\ln\!\left( \frac{1}{n} \sum_{i=1}^n x_i \right) \;\ge\; \ln \left( \prod_{i=1}^n x_i \right)^{1/n}.
$$

第四步：指数函数单调递增，两边取指数

由于指数函数是单调递增的，对两边取指数得：
$$
\frac{1}{n} \sum_{i=1}^n x_i \;\ge\; \left( \prod_{i=1}^n x_i \right)^{1/n}.
$$

这就是算术–几何平均不等式（AM ≥ GM）. 

第五步：等号成立条件

因为 $\ln x$ 是严格凹函数，Jensen 不等式中的等号成立 当且仅当 所有 $x_i$ 相等，即：
$$
x_1 = x_2 = \cdots = x_n.
$$

> [!NOTE] 练习 1.41
> 使用概率的加和规则和乘积规则，证明互信息 $I(c,y)$ 满足关系 (1.121). 

$$
\begin{aligned}
I(\boldsymbol{x}; \boldsymbol{y}) 
&= -\iint p(\boldsymbol{x}, \boldsymbol{y}) \ln p(\boldsymbol{y}) \, d\boldsymbol{x}\,d\boldsymbol{y} + \iint p(\boldsymbol{x}, \boldsymbol{y}) \ln p(\boldsymbol{y}|\boldsymbol{x}) \, d\boldsymbol{x}\,d\boldsymbol{y} \\
&= -\int p(\boldsymbol{y}) \ln p(\boldsymbol{y}) \, d\boldsymbol{y} + \iint p(\boldsymbol{x}, \boldsymbol{y}) \ln p(\boldsymbol{y}|\boldsymbol{x}) \, d\boldsymbol{x}\,d\boldsymbol{y} \\
&= H(\boldsymbol{y}) - H(\boldsymbol{y}|\boldsymbol{x}). 
\end{aligned}
$$
