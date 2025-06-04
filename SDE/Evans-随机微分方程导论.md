# 第一章 导引
## 1.1 确定性和随机微分方程
考虑下面的常微分方程
$$
\begin{cases}
\dot{\boldsymbol{x}}(t) = \boldsymbol{b}(\boldsymbol{x}(t)) & (t > 0)\\
\boldsymbol{x}(0) = x_{0}
\end{cases} \tag{ODE}
$$
其中 $\boldsymbol{b}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{n}$ 是一个光滑向量场。它的解 $\boldsymbol{x}: \mathbb{R}_{\geqslant_{0}} \rightarrow \mathbb{R}^{n}$ 是 $\mathbb{R}^{n}$ 中的一条路径，我们称 $\boldsymbol{x}(t)$ 是**系统在 $t$ 时刻的状态** ($t \geqslant 0$)。但实际上我们发现路径大体上是和ODE 得出的路径差不多，但显然还带有噪声。这样我们就需要对原来的 (ODE) 模型进行修改，形式上的我们可以写成
$$
\begin{cases}
\dot{\boldsymbol{X}}(t) = \boldsymbol{b}(\boldsymbol{X}(t)) + \boldsymbol{B}(\boldsymbol{X}(t))\boldsymbol{\xi}(t) & (t > 0)\\
\boldsymbol{X}(0) = x_{0}
\end{cases} \tag{1}
$$
其中 $\boldsymbol{B}: \mathbb{R}^{n} \rightarrow \mathbb{M}^{n \times m}$，$\boldsymbol{\xi}(\cdot)$ 是 $m$ 维白噪声。自然地我们有下面这些问题需要解决
1. 定义白噪声 $\boldsymbol{\xi}$ 
2. 定义解 $X(\cdot)$ 的含义
3. 考虑确定了 $x_{0}$，$\boldsymbol{b}$，$\boldsymbol{B}$ 等参数后，方程 (1) 是否有解，有解的话解是否唯一，以及它的渐进性质

## 1.2 随机微分
我们首先考虑比较简单的情况，$m = n$，$x_{0} = 0$，$\boldsymbol{b} \equiv {0}$，$\boldsymbol{B} \equiv I$。这样的设定下得出的解是 **$n$维 Brown 运动**，记为 $\boldsymbol{W}(\cdot)$。形式上，我们可以把 (1) 写成 $\dot{\boldsymbol{W}}(\cdot) = \boldsymbol{\xi}(\cdot)$，这个式子的意义是 Brown 运动对时间的导数是白噪声。

我们现在考虑一般的情况，并使用微分符号 $\displaystyle \frac{\mathrm{d}}{\mathrm{d}t}$ 代替 $\dot{\square}$：
$$
\frac{ \mathrm{d}\boldsymbol{X}(t) }{ \mathrm{d}t } = \boldsymbol{b}(\boldsymbol{X}(t)) + \boldsymbol{B}(\boldsymbol{X}(t)) \frac{\mathrm{d}\boldsymbol{W}(t)}{\mathrm{d}t}
$$
两边 "乘以" $\mathrm{d}t$，得到
$$
\begin{cases}
\mathrm{d}\boldsymbol{X}(t) = \boldsymbol{b}(\boldsymbol{X}(t))\mathrm{d}t + \boldsymbol{B}(\boldsymbol{X}(t))\mathrm{d}\boldsymbol{W}(t)\\
\boldsymbol{X}(0) = x_{0}
\end{cases} \tag{SDE}
$$
其中我们称 $\mathrm{d}X$，$\boldsymbol{B}\mathrm{d}\boldsymbol{W}$ 这样的项为**随机微分**，而 SDE 的全称就是**随机微分方程 (stochastic differential equation)**。假如我们找到了形如下面形式的 $\boldsymbol{X}(\cdot)$，我们称这是 (SDE) 的一个解：
$$
\boldsymbol{X}(t) = x_{0} + \int_{0}^{t} \boldsymbol{b}(\boldsymbol{X}(s)) \, \mathrm d{s} + \int_{0}^{t} {\boldsymbol{B}(\boldsymbol{X}(s))} \, \mathrm d{\boldsymbol{W}}, \quad \forall t > 0  \tag{3}
$$
为了让上面的式子有意义，我们需要弄清楚这几件事情
* 构造一个 Brown 运动 $\boldsymbol{W}(\cdot)$ （第三章）
* 定义随机积分 $\displaystyle \int_{0}^{t} {\cdots} \, \mathrm d{\boldsymbol{W}}$ （第四章）
* 证明 (3) 有解 （第五章）
在解决完上面的问题之后，我们还需要解决下面两个问题
* (SDE) 是否真实的刻画的系统的物理状态？
* $\boldsymbol{\xi}(\cdot)$ 是“真的”白噪声还是一些光滑但高度震荡函数的组合？（第六章）
这些问题十分微妙，它们的不同答案将会导致 (SDE) 有完全不同的解。

## 1.3 Itô 链式法则
我们将遇到的一个麻烦是随机微积分中的奇怪的链式法则。令 $m = n = 1$，我们得到 $X$ 满足的等式
$$
\mathrm{d}X = b(X)\mathrm{d}t = \mathrm{d}W
$$
假设 $u: \mathbb{R} \rightarrow \mathbb{R}$ 是一个已知的光滑函数，我们要求 $Y(t) := u(X(t)), t \geqslant 0$ 的解满足什么条件。根据学过微积分知识中的链式法则，我们猜它会是下面的形式
$$
\mathrm{d}Y = u'\mathrm{d}X = u'b\mathrm{d}t + u'\mathrm{d}W,
$$
<big><font color="red">很遗憾，这是错的！</font></big>事实上，我们将会看到布朗运动的随机微分有下面这样诡异的形式：
$$
\mathrm{d}W = (\mathrm{d}t)^{1/2}.
$$
事实上如果我们计算 $\mathrm{d}Y$，然后保留所有 $\mathrm{d}t$ 和 $(\mathrm{d}t)^{1/2}$ 的幂次（我们用了这个”事实“：$(\mathrm{d}W)^{2} = \mathrm{d}t$）我们可以从 (4) 得到
$$
\begin{align}
\mathrm{d}Y &= u'\mathrm{d}X + \frac{1}{2}u''(\mathrm{d}X)^{2} + \cdots & \text{微分的泰勒展开}\\
&= u'(b\mathrm{d}t + \mathrm{d}W) + \frac{1}{2}u''(b\mathrm{d}t + \mathrm{d}W)^{2} + \cdots & \text{代入 }\mathrm{d}X = b\mathrm{d}t + \mathrm{d}W\\
&= \left[ u'b + \frac{1}{2}u'' \right] \mathrm{d}t + u'\mathrm{d}W + \{ \text{一堆 }(\mathrm{d}t)^{3/2} \text{及以上的高阶项} \}
\end{align}
$$
假如我们丢掉后面的高阶项，就得到了被称为 **Itô 链式法则** 或是 **Itô 公式** 的东西
$$
\mathrm{d} u(X) = \left[ u'b + {\color{red} \frac{1}{2}u'' }  \right] \mathrm{d}t + u'\mathrm{d}W.
$$
其中红色的部分是它和常规微积分不同的地方。

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

### 2.4.3 特征函数



## 2.5 大数定律和中心极限定理
### 2.5.1 独立同分布的随机变量

### 2.5.2 强大数定律

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




# 第三章 Brown 运动和 “白噪声”
## 3.1 动机

## 3.2 定义和基本性质

## 3.3 Brown 运动的构造

## 3.4 采样路径的性质

## 3.5 Markov 性


# 第四章 随机积分
## 4.1 预备知识

## 4.2 Itô 积分

## 4.3 Itô 链和乘积法则

## 4.4 高维空间中的 Itô 积分


# 第五章 随机微分方程
## 5.1 定义和例子

## 5.2 解的存在性和唯一性

## 5.3 解的性质

## 5.4 线性随机微分方程

# 第六章 应用