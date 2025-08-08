# 第二章 概率论速成
## 2.1 基本定义
让我们从一个谜题开始。

### 2.1.1 Bertrand 悖论
> **问题 2.1（Bertrand 悖论）** 假如平面上有一个长为 $2$ 的的圆，然后我们 <font color="purple">随机地</font> 选一条弦。与以该圆同圆心，半径为 $1$ 的圆相交的概率是多少？

* 解法 1：由于每条弦由其中点唯一确定，因此与小圆相交的概率为小圆面积除以大圆面积，即 $\displaystyle \frac{1}{4}$.
* 解法 2：由于圆的旋转对称性，对于每一条弦，我们都能旋转整个图形，使得该弦是垂直的。大圆直径为 $4$，小圆直径为 $2$，弦与小圆相交，那它一定与（旋转后的）小圆直径相交（且垂直），因此概率为小圆直径除以大圆直径，即 $\displaystyle\frac{1}{2}$.
* 解法 3：由于圆的旋转对称性，我们总能旋转整个图形，使得它的一个端点位于大圆的最左侧。考虑弦与大圆水平直径的夹角 $\theta$，可知它落在 $\displaystyle\left[ -\frac{\pi}{2}, \frac{\pi}{2} \right]$ 之间，由几何关系可得，弦与小圆相交时，$\theta$ 落在 $\displaystyle \left[ -\frac{\pi}{6}, \frac{\pi}{6} \right]$ 之间。因此概率为这两个区间长度之比，也即 $\displaystyle \frac{1}{3}$
![600](Pasted%20image%2020250601153030.png)

真是离谱到家了，同样的问题居然有三个不同的答案！

### 2.1.2 概率空间
一个很自然的疑问就是，上面到底出了什么问题？（如果上面没问题，那数学大厦就要塌了）注意到我把”随机地“三个字用颜色标出来了，问题就出在这里：我们没有良好的定义这里的随机到底是”怎么随机“的。为了解决这样的问题，我们引入概率空间的概念。

我们先有一个非空集合 $\Omega$，其中的子集我们称之为”事件“。

> **定义 2.2（$\sigma$-代数）** 一个 $\sigma$ 代数指的是非空集合 $\Omega$ 上的一个子集族 $\mathcal{U}$，并满足下面的条件
> * $\varnothing, \Omega \in \mathcal{U}$
> * 如果 $A \in \mathcal{U}$，那么其补集 $A^{c} \in \mathcal{U}$
> * 如果一列集合 $A_{1}, \dots \in \mathcal{U}$，那么有 $$\displaystyle \bigcup_{k=1}^{\infty} A_{k},  \bigcap_{k=1}^{\infty} A_{k} \in \mathcal{U}.$$

> **定义 2.3 （概率测度）** 设 $\mathcal{U}$ 是 $\Omega$ 上的一个 $\sigma$-代数，我们称 $$P: \mathcal{U} \rightarrow [0, 1]$$是一个概率测度，如果它满足下面的条件
> 1. $P(\varnothing) = 0$, $P(\Omega) = 1$
> 2. 如果一列集合 $A_{1}, \dots \in \mathcal{U}$，则 $$P\left[\bigcup_{k=1}^{\infty} A_{k} \right] \leqslant \sum\limits_{k=1}^{\infty}P(A_{k}) $$
> 3. 如果一列集合  $A_{1}, \dots \in \mathcal{U}$ 互不相交，则 $$P\left[\bigcup_{k=1}^{\infty} A_{k} \right] = \sum\limits_{k=1}^{\infty}P(A_{k}) $$

从这个定义我们得到：假如 $A \subset B$，就有 $P(A) \leqslant P(B)$，因为 $P(B) = P(B) + P(B - A) \geqslant P(A)$。

> **定义 2.4（概率空间）** $\Omega$ 是一个非空集合，$\mathcal{U}$ 是其上的一个 $\sigma$-代数，$P$ 是 $\mathcal{U}$ 上的概率测度；我们称三元组 $(\Omega, \mathcal{U}, P)$ 是一个概率空间。

> **术语约定 2.5**
> 1. 集合 $A \in \mathcal{U}$ 被称为一个**事件**，$\omega \in \Omega$ 被称为**样本点**
> 2. $P(A)$ 是事件 $A$ 的**概率**
> 3. 如果一个属性，对除了概率为零得事件以外均为真，我们称其为 **几乎必然成立 (almost surely, 简写为 a.s.)**。

概率空间是概率论的必要设定，在讨论或解决任何问题之前，我们都要明确它所指的概率空间是什么。现在回头看之前的 Bertrand 悖论，我们不难发现三个解法对悖论中的”随机“作出了不同的解读，对应着三个不同的概率空间。

下面给出一些典型的概率空间的示例。

> **示例 2.6 （Euclid 空间及其上的Borel $\sigma$-代数和概率密度函数）**

> **示例 2.7 （Dirac 测度）**

> **示例 2.8 （Buffon 投针问题）**


### 2.1.3 随机变量

> **定义 2.9 （随机变量，$\mathcal{U}$-可测）** 设 $(\Omega, \mathcal{U}, P)$ 是一个概率空间，一个映射 $$\boldsymbol{X}: \Omega \rightarrow \mathbb{R}^{n}$$被称为一个 $n$ 维随机变量，如果对任意 $B \in \mathcal{B}$，我们有 $$\boldsymbol{X}^{-1}(B) \in \mathcal{U}.$$我们也称 $\boldsymbol{X}$ 是 $\mathcal{U}$-可测的。

我们注意这里的 $X^{-1} \in \mathcal{U}$，这样定义是比像要更好的，需要记住的是，$\mathcal{U}$ 和 $\mathcal{B}$ 都是 $\sigma$-代数，所以必须满足定义 2.2 中所述的若干条件。考虑 $B_{1}, B_{2} \in \mathcal{B}$，我们理应有
$$
\begin{align}
\boldsymbol{X}^{-1}(B_{1} \cup B_{2}), \boldsymbol{X}^{-1}(B_{1} \cap B_{2}), \boldsymbol{X}^{-1}(B_{1}^{c})\in \mathcal{U}.
\end{align}
$$
事实上，我们有集合的交并补可以穿透原像算子，也即
$$
f^{-1}(A \,\square\, B) = f^{-1}(A) \,\square\, f^{-1}(B),\quad f^{-1}(A^{c}) = [f^{-1}(A)]^{c}
$$
其中 $\square \in \{ \cup, \cap \}$。假如我们将原像算子换成看似更加自然的求像集操作，也即对任意 $U \in \mathcal{U}$，有 $f(U) \in \mathcal{B}$。由类似地逻辑，我们需要验证对任意的 $U, V \in \mathcal{U}$，都有
$$
\boldsymbol{X}(U \,\square\, V) \in \mathcal{B}, \boldsymbol{X}(U^{c}) \in \mathcal{B}.
$$
而我们可以适当地构造反例让上面的条件不再成立。

> **术语约定 2.10**
> 1. 我们一般写 $\boldsymbol{X}$，而不写 $\boldsymbol{X}(\omega)$
> 2. 我们一般写 $P(\boldsymbol{X} \in B)$，而不写 $P(\boldsymbol{X}^{-1}(B))$

> **示例 2.11 （示性函数，简单函数）** 考虑集合 $A \in \mathcal{U}$，它的示性函数定义为 $$\chi_{A}(\omega) := \begin{cases} 1, & \omega \in A\\ 0, & \omega \notin A, \end{cases}$$这是一个随机变量。更一般地，考虑 $A_{1}, \dots, A_{m} \in \mathcal{U}$，并有 $\displaystyle \Omega = \bigcup_{i=1}^{m} A_{i}$，$a_{i} \in \mathbb{R}$，于是 $$X = \sum\limits_{i=1}^{m} a_{i}\chi_{A_{i}}$$是一个随机变量，我们称其为简单函数（simple function）

> **引理 2.12 （随机变量生成的 $\sigma$-代数）** 设 $\boldsymbol{X}: \Omega \rightarrow \mathbb{R}^{n}$ 是一个随机变量，则 $$\mathcal{U}(\boldsymbol{X}) := \{ \boldsymbol{X}^{-1}(B): B \in \mathcal{B} \}$$是一个 $\sigma$-代数，被称为随机变量 $\boldsymbol{X}$ 生成的 $\sigma$-代数。

这是使得 $\boldsymbol{X}$ 可测的最小的 $\mathcal{U}$ 的子$\sigma$-代数。

**证明.** 不难发现，这个引理需要用到前面提到的集合运算能够穿透映射原像的一个结果。
首先，空集和全集 $\Omega$肯定是 $\mathcal{U}(\boldsymbol{X})$的集合。对于前者，考虑 $\boldsymbol{X}^{-1}(\varnothing) = \{ \omega \in \Omega: f(\omega) \in \varnothing \}$，因为 $f(\omega) \in \varnothing$ 永远不可能为真，因此只有 $X^{-1}(\varnothing) = \varnothing$。另一方面，考虑 $\boldsymbol{X}^{-1}(\mathbb{R}^{n})$（由拓扑学知识，我们知道 $\mathbb{R}^{n}$ 是它自己的的开子集），我们得到 $\boldsymbol{X}^{-1}(\mathbb{R}^{n}) = \Omega$。
然后我们证明第二条，也即对任意 $B \in \mathcal{B}$，都有 $\boldsymbol{X}^{-1}(B)^{c} \in \mathcal{U}(\boldsymbol{X})$。事实上，我们来证明 $\boldsymbol{X}^{-1}(B)^{c}$ 事实上就是 $\boldsymbol{X}^{-1}(B^{c})$：
$$
\begin{align}
\boldsymbol{X}^{-1}(B)^{c} &= \{ \omega \in \Omega: f(\omega) \in B \}^{c}\\
&= \{ \omega \in \Omega: f(\omega) \in B^{c} \} = \boldsymbol{X}^{-1}(B^{c})
\end{align}
$$
用类似的方法，我们也可以证明该集族对可数并和可数交都封闭。于是 $\mathcal{U}(\boldsymbol{X})$ 是一个 $\sigma$-代数，且它不能再小了：从里面拿出一个元素都会使得 $\boldsymbol{X}$ 在拿掉元素后的集族下不可测。    Q.E.D.

> <font color="red"><b>注释 2.13</b></font> 很重要的一点是，我们从概率论的角度可以说 “$\mathcal{U}(\boldsymbol{X})$ 包含了 $\boldsymbol{X}$ 的相关信息”。假如另一个随机变量 $\boldsymbol{Y}$ 可以写成 $\boldsymbol{Y} = \Phi(\boldsymbol{X})$，其中 $\Phi$ 是某个 “合理的” 函数，那 $\boldsymbol{Y}$ 也是 $\mathcal{U}(\boldsymbol{X})$-可测的。另一方面，假如存在一个 $\mathcal{U}(\boldsymbol{X})$-可测的随机变量 $\boldsymbol{Y}$，那么一定存在一个函数 $\Phi$，使得 $\boldsymbol{Y} = \Phi(\boldsymbol{X})$。
> 事实上，这样的 $\Phi$ 称为 **可测映射**，这是了可测空间范畴的态射。

### 2.1.4 随机过程
本节中我们介绍取决于时间的随机变量。

> **定义 2.14 （随机过程，采样路径）** 一个随机变量的集合 $\{ \boldsymbol{X}(t): t \geqslant 0 \}$ 被称为一个随机过程。对每个 $\omega \in \Omega$，映射 $t \mapsto \boldsymbol{X}(t, \omega)$ 称为一个采样路径。

> <font color="red"><b>术语约定 2.15</b></font>
> 有的地方也使用 $\boldsymbol{X}_{t}$ 而不是 $\boldsymbol{X}(t)$ 指称一个关于变量 $t$ 的随机过程 

## 2.2 期望和方差
### 2.2.1 关于测度的积分

> **定义 2.16 （随机变量 $X$ 关于概率测度 $P$ 的积分）**
> 设 $(\Omega, \mathcal{U}, P)$ 是概率空间，$\displaystyle X = \sum\limits_{i=1}^{k} a_i\chi_{A_{i}}$ 是一个实值简单随机变量，定义 $X$ 的积分为 $$\int_{\Omega} X \, \mathrm dP := \sum\limits_{i=1}^{k} a_{i}P(A_{i}) $$如果 $X$ 是非负的随机变量，我们用简单随机变量积分的上确界定义它的积分 $$\int_{\Omega} X \, \mathrm dP := \sup_{Y \leqslant X, Y\text{ simple}} \int_{\Omega} Y \, \mathrm dP.$$对于一般的实值随机变量 $X:\Omega \rightarrow \mathbb{R}$，定义其积分值为正部的积分减去负部的积分（若二者至少有一个有限）：$$\int_{\Omega} X \, \mathrm dP := \int_{\Omega} X^{+} \, \mathrm dP - \int_{\Omega} X^{-} \, \mathrm dP,$$其中 $X^{+} := \max\{ X, 0 \}$，$X^{-} := \max\{ -X, 0 \}$；容易验证 $X = X^{+} - X^-$。

> **注解 2.16**
> 1. 事实上，这里的积分就是随机变量 $X$ 在概率测度 $P$ 下的 **Lebesgue 积分**，不难发现我们可以将其理解为对 “值域进行分割”。
> 2. 我们假设后面出现的所有积分都存在且有限

> **术语约定 2.17**
> 对于 $n$ 维随机变量 $\boldsymbol{X}: \Omega \rightarrow \mathbb{R}^{n}$，其中 $\boldsymbol{X} = (X^{1}, \dots, X^{n})$，我们将其积分写为 $$\int_{\Omega} \boldsymbol{X} \, \mathrm dP = \left[ \int_{\Omega} X^{1} \, \mathrm d{P}, \cdots, \int_{\Omega} X^{n} \, \mathrm d{P}  \right]$$

> **定义 2.18 （期望，方差）**
> 假设 $\boldsymbol{X}: \Omega \rightarrow \mathbb{R}^{n}$ 是一个向量值的随机变量，称 $$E(\boldsymbol{X}):= \int_{\Omega} \boldsymbol{X} \, \mathrm dP $$为它的期望（或均值），称 $$V(\boldsymbol{X}) := \int_{\Omega} \|\boldsymbol{X} - E(\boldsymbol{X})\|_{2}^{2}  \, \mathrm dP $$为它的方差。不难验证 $V(\boldsymbol{X}) = E(\|\boldsymbol{X}\|^{2}_{2}) - \|E(\boldsymbol{X})\|^{2}_{2}$.

### 2.2.2 分布函数

## 2.3 独立性
### 2.3.1 条件概率

### 2.3.2 独立事件

### 2.3.3 独立随机变量


## 2.4 一些概率论中的工具和方法
### 2.4.1 Chebyshev 不等式

### 2.4.2 Borel–Cantelli 引理

考虑定义在某个概率空间的随机变量序列 $\{ X_{k} \}_{k=1}^{\infty}$ **依概率收敛** 到某一随机变量 $X$ 指的是对任意 $\epsilon > 0$，都有 $$\lim_{ k \to \infty } P(|X_{k} - X| > \epsilon) = 0.$$
事实上我们有下面的定理，揭示了上述例子和 Borel-Cantelli 引理之间的关系

> **定理 4.4（依概率收敛的随机变量子列几乎必然收敛到原极限）**
> 如果 $X_{k}$ 依概率收敛到 $X$（记为 $X_{k} \xrightarrow{P} X$）则存在一个子序列 $\{ X_{k_{j}} \}_{j = 1}^{\infty} \subset \{ X_{k} \}_{k = 1}^{\infty}$，使得 $$X_{k_{j}} \rightarrow X\quad a.s.$$

**证明.** 对每个正整数 $j$，我们找一个很大的 $k_{j}$，使其满足 
$$P\left( |X_{k_{j}} - X| > \frac{1}{j} \right) \leqslant \frac{1}{j^{2}},$$
同时保证 $k_{1} < \cdots < k_{j-1} < k_{j} < \cdots$ 因此显然 $k_{j} \rightarrow \infty$。令 $\displaystyle A_{j} := \left\{  |X_{k_{j}} - X| > \frac{1}{j} \right\}$，由于每个 $A_j$ 发生概率的上界构成的级数 $\displaystyle\sum \frac{1}{j^{2}}  < \infty$，因此可以使用 Borel-Cantelli 引理，就有 $P(A_{j}\quad \text{i.o.}) = 0$。这句话的意思是，几乎每个样本空间的样本点 $\omega$，$A_{j}$ 发生的次数是有限的（否则如果可能发生无限次，$A~ ~\text{i.o.}$ 的概率就不为零了，另一方面 $A_{j}$ 发生无限次的样本点集合的概率测度为零），因此我们能找到 $A_{j}$ 最后一次发生的指标，然后选取一个比它大的 $J(\omega)$，对所有 $j > J(\omega)$，有 $\displaystyle |X_{k_{j}}(\omega) - X(\omega)| \leqslant \frac{1}{j}$，这时候把 $j$ 推向无穷大，就有 $X_{k_{j}}(\omega) \rightarrow X(\omega)~ ~\text{a.s.}$。 

### 2.4.3 特征函数

> **定义 4.5 （随机变量的特征函数）**
> 假设 $\boldsymbol{X}$ 是一个 $n$ 维实值随机变量，定义其特征函数为 $$\phi_{\boldsymbol{X}}(\lambda) := E(e^{\text{i}\left\langle \boldsymbol{\lambda}, \boldsymbol{X} \right\rangle}), \quad \boldsymbol{\lambda} \in \mathbb{R}^{n}$$

> **示例 4.6 （Gaussian 随机变量的特征函数）**
> 考虑服从 $N(0, 1)$ 的随机变量，其特征函数是 
> $$\begin{align}\phi_{X}(\lambda) &= \int_{-\infty}^{\infty} e^{\text{i}\lambda x} \frac{1}{\sqrt{ 2\pi }} e^{-x^{2}/2} \, \mathrm d{x}\\&= \frac{e^{-\lambda^{2}/2}}{\sqrt{ 2\pi }}\int_{-\infty}^{\infty} e^{-(x - \text{i}\lambda)^{2}/2} \, \mathrm d{x}\end{align}$$
> 此时我们会想做变换 $t \leftarrow x + \text{i}\lambda$。而复分析给了我们依据。积分里面的函数在复平面上是解析的，根据 Cauchy 积分定理，我们将积分路径（实轴）往虚轴正方向移 $\lambda$ 得到的两条路径是同伦的（考虑两条路径在 Riemann 球面上的投影），因此积分值相同，而我们知道 $\displaystyle \int_{-\infty}^{\infty} e^{-x^{2}/2} \, \mathrm d{x} = \sqrt{ 2\pi }$，就得到 $\phi_{X}(\lambda) = e^{-\lambda^{2}/2}$。
> 一般的 Gaussian 随机变量也同理，最后得到的结果是 $\phi_{X}(\lambda) = e^{\text{i}m\lambda - \lambda^{2}\sigma^{2}/2}$

> **引理 4.7 （特征函数继承 Fourier 变换的性质）**
> （ 和的特征函数等于各自特征函数的积 ）如果 $\boldsymbol{X}_{1}, \dots, \boldsymbol{X}_{m}$ 是独立随机变量，对所有 $\boldsymbol{\lambda} \in \mathbb{R}^{n}$，有 $$\phi_{\boldsymbol{X}_{1} + \cdots + \boldsymbol{X}_{m}}(\boldsymbol{\lambda}) = \phi_{\boldsymbol{X}_{1}}(\boldsymbol{\lambda})\cdots \phi_{\boldsymbol{X}_{m}}(\boldsymbol{\lambda})$$
> （ 特征函数的 $k$ 阶导相当于将 $X^{k}$ 的特征函数在复平面内旋转 $k$ 次）如果 $X$ 是实值随机变量，则对任意非负整数 $k$，有 $$\phi^{(k)}(0) = \text{i}^{k} E(X^{k})$$
> （ 特征函数决定分布函数 ）$\boldsymbol{X}$ 和 $\boldsymbol{Y}$ 是两个随机变量，则 $$\phi_{\boldsymbol{X}}(\boldsymbol{\lambda}) = \phi_{\boldsymbol{Y}}(\boldsymbol{\lambda}) \implies F_{\boldsymbol{X}}(\boldsymbol{x}) = F_{\boldsymbol{Y}}(\boldsymbol{x})$$

**证明.** 
（1）
$$
\begin{align}
\phi_{\boldsymbol{X}_{1} + \cdots + \boldsymbol{X}_{m}}(\boldsymbol{\lambda}) &= E(e^{\text{i}\left\langle \boldsymbol{\lambda}, \boldsymbol{X}_{1} + \cdots + \boldsymbol{X}_{m} \right\rangle })\\
&= E\big( e^{\text{i}\left\langle \boldsymbol{\lambda}, \boldsymbol{X}_{1} \right\rangle } \cdots e^{\text{i}\left\langle \boldsymbol{\lambda}, \boldsymbol{X}_{m} \right\rangle } \big) & 内积线性性\\
&= E\big( e^{\text{i}\left\langle \boldsymbol{\lambda}, \boldsymbol{X}_{1} \right\rangle }\big) \cdots E\big(e^{\text{i}\left\langle \boldsymbol{\lambda}, \boldsymbol{X}_{m} \right\rangle } \big)&独立性\\
&=\phi_{\boldsymbol{X}_{1}}(\boldsymbol{\lambda})\cdots \phi_{\boldsymbol{X}_{m}}(\boldsymbol{\lambda})
\end{align}
$$
（2）（3）略

> **示例 4.8 （相互独立的 Gaussian 随机变量之和）**
> 两个随机变量 $X \sim N(m_{1}, \sigma_{1}^{2})$，$Y \sim N(m_{2}, \sigma_{2}^{2})$ ，则 $$X + Y \sim N(m_{1}+m_{2}, \sigma_{1}^{2} + \sigma^{2}_{2}),$$因为 $$\phi_{X+Y}(\lambda) = \phi_{X}(\lambda)\phi_{Y}(\lambda) = e^{\text{i}(m_{1}+m_{2})\lambda - \lambda^{2}(\sigma_{1}^{2}+\sigma_{2}^{2})/2}$$

## 2.5 大数定律和中心极限定理
### 2.5.1 独立同分布的随机变量

> **定义 5.1 （同分布）**
> 一族随机变量 $\boldsymbol{X}_{1}, \dots, \boldsymbol{X}_{n}, \dots$ 若对所有 $x$，满足 $$F_{\boldsymbol{X}_{1}}(x) = F_{\boldsymbol{X}_{2}}(x) = \cdots = F_{\boldsymbol{X}_{n}}(x) = \cdots$$则称它们是同分布的。

如果我们在同分布的基础上，嘉定这些随机变量是相互独立的，我们就可以将它们视为一系列结果可观测的独立重复试验，给定样本空间的点 $\omega \in \Omega$，我们可以得到随机变量序列在其上的取值序列。下面的内容揭示了这样的序列遵循的规律

### 2.5.2 强大数定律

> **定理 5.2 （强大数定律）**
> 考虑同一概率空间下一个独立同分布且可积的随机变量序列 $\boldsymbol{X}_{1}, \dots, \boldsymbol{X}_{n}, \dots$ ，记 $m = E(X_{i}), i=1, \dots$，则有 $$P\left( \lim_{ n \to \infty } \frac{\boldsymbol{X}_{1} + \cdots + \boldsymbol{X}_{n}}{n} = m \right) = 1.$$

**证明.**  不失一般性，我们假设这些随机变量都是实值的、四阶矩存在，且均值为零（假设不为零，我们就考虑 $\boldsymbol{X}_{i} - m$）。接下来考虑随机变量部分和的四阶矩：
$$
\begin{align}
E\left[ \left( \sum\limits_{i=1}^{n}{X}_{i} \right)^{4} \right] &= \sum\limits_{i,j,k,l} E(X_{i}X_{j}X_{k}X_{l}) 
\end{align}
$$
当求和指标中的 $i$ 和 $j,k,l$ 都不同时，根据独立性，有 $$E(X_{i}X_{j}X_{k}X_{l}) = \underbrace{ E(X_{i}) }_{ 0 } E(X_{j}X_{k}X_{l}) = 0$$由于 $i,j,k,l$ 这四个指标的地位是等同的，可以发现求和项里面只会剩下 $XXXX$ 或者 $XXYY$ 这样的组合了，因为其余的组合根据独立性都会变成零。因此四阶矩可以化简：
$$
\begin{align}
E\left[ \left( \sum\limits_{i=1}^{n}{X}_{i} \right)^{4} \right] &= \binom{4}{4} \sum\limits_{i=1}^{n} E(X_{i}^{4}) + \binom{4}{2} \sum\limits_{i,j=1, i \neq j} E(X_{i}^{2}X_{j}^{2})\\ 
&= \sum\limits_{i=1}^{n} E(X_{i}^{4}) + 3\sum\limits_{i,j=1, i \neq j} E(X_{i}^{2}X_{j}^{2})\\ 
\end{align}
$$

### 2.5.3 扰动和 Laplace-De Moivre 定理

### 2.5.4 中心极限定理

## 2.6 条件期望
### 2.6.1 动机

### 2.6.2 条件分布的第一种构造

### 2.6.3 条件分布的第二种构造

### 2.6.4 条件分布的诸性质

## 2.7 鞅
### 2.7.1 定义

### 2.7.2 鞅的诸不等式



