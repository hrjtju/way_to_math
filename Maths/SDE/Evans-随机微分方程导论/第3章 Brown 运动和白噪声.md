# 第三章 Brown 运动和白噪声

> 如果你还没有 Robert Brown 的新书《植物花粉的显微观察》，那我送你一本。
>                                                             —— George Eliot《Middlemarch》

## 3.1 动机
### 3.1.1 历史背景
早在 1826 年左右 R. Brown 就发现了花粉粒子在水中的不规则运动。他发现

1. 每个粒子在水中的运动路径非常不规律
2. 两个粒子的路径之间似乎没有联系

到了 1990 年左右，L. Bachelier 开始用数学工具刻画股市价格变化的波动，他的研究结果后来被 A. Einstein 在 1905 年重新发现并推广。后者是这样研究 Brown 发现的现象的：考虑一个装满水的长管道，在 $t=0$ 时刻我们在 $x=0$ 处注入单位体积的墨水。令 $u = u(x, t)$ 表示在时刻 $t >0$ 时在水管的 $x \in \mathbb{R}$ 位置处墨水粒子的密度。因此我们能得到最开始时
$$
u(x, 0) = \delta_{0}
$$
其中 $\delta_{0}$ 是 $0$ 处的 Dirac 质量。接着，假设某个墨水粒子从 $x$ 在短时间 $\tau$ 内迁移到 $x+y$ 的概率密度函数是 $f(y, \tau)$，则有
$$
\begin{align}
u(x, t + \tau) &= \int_{-\infty}^{\infty} u(x-y, t)f(y, \tau) \, \mathrm dy\\ \\
&= \int_{-\infty}^{\infty} \left( u - u_{x}y + \frac{1}{2}u_{xx} y^{2} + \cdots  \right) f(y, \tau)\, \mathrm d{y} 
\end{align} \tag{3.1}
$$
其中 $\displaystyle u_{x} =\displaystyle \frac{ \partial u }{ \partial x }$，$u_{xx} = \displaystyle \frac{ \partial ^{2}u }{ \partial x^{2} }$。第二步是将 $y$ 看做主元，变成 $u_{x,t}(y)$，然后在 $y=0$ 处展开的结果。由于 $f$ 是概率密度函数，我们知道 $\displaystyle \int_{-\infty}^{\infty} f \, \mathrm d{y}  = 1$；根据对称性，还可以得到 $f(-y, \tau) = f(y, \tau)$，因此 $\displaystyle \int_{-\infty}^{\infty} yf \, \mathrm d{y} = 0$。我们进一步假设 $f$ 的方差关于 $\tau$ 是线性的，即存在某个常数 $D > 0$，使得 $$\int_{-\infty}^{\infty} y^{2}f(y, \tau) \, \mathrm d{y} = D\tau$$我们把这些都代入 (3.1)，得到
$$
\begin{align}
u(x, t+\tau) &= u(x,t)\int_{-\infty}^{\infty} f(y, \tau) \, \mathrm dy - u_{x}(x, t)\int_{-\infty}^{\infty} yf(y, \tau) \, \mathrm d{y} + \frac{1}{2} u_{xx}(x, t)\int_{-\infty}^{\infty} y^{2}f(y, \tau) \, \mathrm{d}{y} + \cdots \\ \\
&= u(x, t) + 0 + \frac{D\tau}{2} u_{xx}(x, t) + \cdots 
\end{align}
$$
我们考虑下面的极限
$$
\displaystyle \frac{ \partial u }{ \partial t } = u_t = \lim_{\tau \to 0 } \frac{u(x, t+\tau) - u(x, t)}{\tau} = \lim_{ \tau \to 0 } \left[ \frac{D}{2}u_{xx}(x, t) + \{ 高阶项 \} \right] = \frac{D}{2}u_{xx} \tag{3.2}
$$
这就是所谓的 **扩散方程** 或是 **热传导方程**。它是一个偏微分方程，以为初值条件 $u(x,0) = \delta_{0}$ 的解是 $$
u(x, t) = \frac{1}{(2\pi Dt)^{1/2}} e^{-\frac{x^{2}}{2Dt}} \tag{3.3}
$$可见墨水在水管中 $t$ 时刻的分布是 Gauss 分布，均值为 $0$，方差为 $Dt$。Einstein 还计算了常数 $D$ 的值：$$
D = \frac{RT}{N_{A}f},
$$其中 $R$ 是气体常数，$T$ 是绝对温度，$f$ 是摩擦系数，$N_{A}$ 是 Avogadro 常数。Brown 运动和这个式子一起促成了 J. Perrin 计算 Avogadro 常数的值，对物质的原子理论非常重要。N. Wiener 在 1920 年代将这个问题严格化，他的思想构成了 3.3 至 3.5 节内容的核心。

### 3.1.2 随机游走

上文中 Einstein 所讨论的问题有一个变种。考虑一个二维方形网格$$
\{ (m\Delta x, n\Delta t): m = 0, \pm{1}, \pm 2, \dots; n=1, 2, \dots \}
$$其中 $\Delta x$ 和 $\Delta t$ 都是给定的正数。一个粒子在零时刻位于 $x=0$ 处，$n\Delta t$ 时刻时，它分别有 $\displaystyle \frac{1}{2}$ 选择往左走或往右走，到了 $(n+1)\Delta t$ 时刻到达新的位置。令 $p(m,n)$ 是该粒子在 $n\Delta t$ 时刻位于 $m\Delta x$ 的概率，则有 $$p(m, 0) = \begin{cases}
0 & m \neq 0 \\
1 & m = 0
\end{cases}$$以及$$p(m, n+1) = \frac{1}{2}p(m-1, n) + \frac{1}{2}(m+1, n)$$因此 $$p(m, n+1) - p(m, n) = \frac{1}{2}\big[ p(m+1, n) - 2p(m, n) +p(m-1, n) \big] $$此时假设 $\displaystyle\frac{(\Delta x)^{2}}{\Delta t}$ 是某个正常数 $D$，用类似的技法，也有 $$\frac{p(m, n+1) - p(m, n)}{\Delta t}= \frac{D}{2} \cdot \frac{p(m+1, n) - 2p(m, n) +p(m-1, n)}{(\Delta x)^{2}}$$令 $\Delta t \rightarrow 0$，$\Delta x \rightarrow 0$，$m\Delta x \rightarrow x$，$n\Delta t \rightarrow t$，并保证 $\displaystyle\frac{(\Delta x)^{2}}{\Delta t} = D$ 不变，我们会发现 $p(m, n)$ 将会收敛到 $u(x, t)$（因为上式右边取极限后就变成了扩散方程里面的 $u_{xx}$），也就是粒子在 $t$ 时刻位置 $x$ 处的概率密度。这样我们又回到了上文的扩散方程。

### 3.1.3 理论修正

对上述问题中求极限过程更加严谨的讨论需要借助 Laplace-De Moivre 定理。假设每到 $n\Delta t$ 时刻，粒子以等概率向左或向右移动 $\Delta x$。定义$$S_{n} \coloneqq \sum\limits_{i=1}^{n} X_{i}$$其中对 $i=1, \dots$， $X_{i}$ 满足 $$\begin{cases}
\displaystyle P(X_{i} = 0) = \frac{1}{2} \\
\displaystyle P(X_{i} = 1) = \frac{1}{2}
\end{cases}$$因此 $\displaystyle V(X_i) = \frac{1}{4}$。记 $X(t)$ 为粒子在 $t = n\Delta t$ 时刻到达的位置，就有$$X(t) = S_{n}\Delta x + (n-S_n) (-\Delta x) = (2S_{n} - n)\Delta x$$其方差为$$
\begin{align}
V(X(t)) &= (\Delta x)^{2} V(2S_{n} - n) = 4(\Delta x)^{2}V(S_{n}) = 4n(\Delta x)^{2}V(X_{1}) \\
 & =(\Delta x)^{2}n = \frac{(\Delta x)^{2}}{\Delta t}t
\end{align}
$$假设 $\displaystyle \frac{(\Delta x)^{2}}{\Delta t}t = D$，有 $$\begin{align}
X(t)  & = (2S_{n} - n)\Delta x = \left[ \frac{S_{n} - n / 2}{\sqrt{ n / 4 }} \right] \sqrt{ n } \Delta x \\  \\
& = \left[ \frac{S_{n} - n / 2}{\sqrt{ n / 4 }} \right] \sqrt{ n\Delta t D } = \left[ \frac{S_{n} - n / 2}{\sqrt{ n / 4 }} \right] \sqrt{ t D } 
	\end{align}$$注意方括号中已经凑成了可以使用 Laplace-De Moivre 定理的形式。我们用它，得到
$$
\begin{align}
\lim_{ {\scriptstyle n \to \infty}\atop{\scriptstyle t = n\Delta t, \frac{(\Delta x)^{2}}{\Delta t} = D}  } P(a \leqslant X(t) \leqslant b)  & = \lim_{ n \to \infty } \left[ \frac{a}{\sqrt{ tD }} \leqslant \frac{S_{n} - n/2}{\sqrt{ n/4 }} \leqslant \frac{b}{\sqrt{ tD }}\right] \\
&= \frac{1}{\sqrt{ 2\pi }}\int_{\frac{a}{\sqrt{ tD }}}^{\frac{b}{\sqrt{ tD }}} e^{-x^{2}/2} \, \mathrm{d}x = \frac{1}{\sqrt{ 2\pi Dt }}\int_{a}^{b} e^{-\frac{x^{2}}{2Dt}} \, \mathrm{d}x. 
\end{align} 
$$
这样我们就严格证明了离散网格上的随机游走给定时间下粒子的概率密度是 Gauss 函数。


## 3.2 定义和基本性质
### 3.2.1 Brown 运动的定义
上面的内容自然地引出了 Brown 运动，我们在此假设 $D=1$，然后对它下个定义

> **定义 3.1 （一维 Brown 运动）**
> 一个一维 Brown 运动（或 Wiener 过程）是指这样的实值随机过程 $W(\cdot)$，它满足下面的几个条件
> 1. $W(0) = 0\quad\text{a.s.}$
> 2. 对所有 $t \geqslant s \geqslant 0$，$W(t) - W(s)$ 服从 $N(0, t-s)$
> 3. 对所有 $0 < t_{1} < t_{2} < \cdots < t_{n}$，随机变量 $W(t_{1}), W(t_{2}) - W(t_{1}), \dots, W(t_{n}) - W(t_{n-1})$ 是相互独立的（增量相互独立）

中心极限定理可以给我们更多启发，Brown 运动中粒子的每次独立随机扰动的加和经过适当缩放调整后就会得出 Gauss 分布（定义中的第二条）。

### 3.2.2 联合概率的计算

根据定义里面的第一条和第二条，可以得出对任意 $t > 0$，$W(t) = N(0, t)$。进而对任意 $a \leqslant b$，有 $$P(a \leqslant W(t) \leqslant b) = \frac{1}{\sqrt{ 2\pi t }} \int_{a}^{b} e^{-\frac{x^{2}}{2t}} \, \mathrm{d}x $$这时我们想知道，如果确定一列时刻 $0 < t_{1} < \cdots < t_{n}$ 和一系列实的区间端点 $a_{i} < b_{i}, i=1, \dots, n$ 联合概率 $P\Big(a_{1} \leqslant W(t_{1}) \leqslant b_{1}, \dots, a_{n} \leqslant W(t_{n}) \leqslant b_n\Big)$ 应该是什么。换句话说，我们想知道满足上述条件的一段采样序列的概率。

我们可以猜一下。我们已经知道给定一个采样点 $t_{1}$ 和区间 $[a_{1}, b_{1}]$，我们有 $$
P(a_{1} \leqslant W(t) \leqslant b_{1}) =  \int_{a_{1}}^{b_{1}} \frac{1}{\sqrt{ 2\pi t }}\exp\left\{-\frac{x^{2}}{2t}\right\} \, \mathrm{d}x .
$$给定 $W(t_{1}) = x_{1} \in [a_{1}, b_{1}]$，根据定义的第二条，$W(t_{2}) - W(t_{1})$ 服从 $N(x_{1}, t_{2} - t_{1})$。因此
$$
P(a_{2} \leqslant W(t_{2}) \leqslant b_{2} | W(t_{1}) = x_{1}) = \int_{a_{2}}^{b_{2}} \frac{1}{\sqrt{ 2\pi(t_{2}-t_{1}) }} \exp\left\{ -\frac{|x_{2} - x_{1}|^{2}}{2(t_{2} - t_{2})} \right\}  \, \mathrm{d}x_{2} 
$$
所以两个采样点的联合概率应该是$$
\begin{align}
P(a_{1} \leqslant W(t_{1}) \leqslant b_{1}, a_{2} \leqslant W(t_{2}) \leqslant b_{2}) = \int_{a_{1}}^{b_{1}} \int_{a_{2}}^{b_{2}} g(x_{1}, t_{1}|0)g(x_{2}, t_{2}-t_{1}|x_{1}) \, \mathrm{d}x_{2}  \, \mathrm{d}x_{1} 
\end{align}
$$其中 $\displaystyle g(x, t|y) = \frac{1}{\sqrt{ 2\pi t }}\exp\left\{ -\frac{(x-y)^{2}}{2t} \right\}$。上面的推导还可以这样理解，联合概率 $P(W(t_{1}), W(t_{2}))$ 可以写成 $P(W(t_{1}))P(W(t_{2})|W(t_{1}))$，而根据定义，我们知道它们的密度函数分别为 $g(x_{1}, t_{1}|0)$ 和 $g(x_{2}, t_{2}-t_{1}|x_{1})$，在 $[a_{1}, b_{1}] \times [a_{2}, b_{2}]$ 上积分就得到了上面的结果。推而广之，我们就可以猜出一般的形式：
$$
\begin{align}
&P\Big(a_{1} \leqslant W(t_{1}) \leqslant b_{1}, \dots, a_{n} \leqslant W(t_{n}) \leqslant b_n\Big)\\[0.5em]
=&\int_{a_{1}}^{b_{1}} \cdots \int_{a_{n}}^{b_{{n}}} g(x_{1}, t_{1}|0)g(x_{2}, t_{2}-t_{1}|x_{1}) \cdots g(x_{n}, t_{n} - t_{n-1}|x_{n-1}) \, \mathrm{d}x_{n} \cdots   \, \mathrm{d}x_{1}. 
\end{align}\tag{3.4}
$$
> **定理 3.2**
> $W(\cdot)$ 是一个一维 Brown 运动，对每一个实值函数 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$ 和采样时间序列 $0 = t_{0} < t_{1} < \cdots < t_{n}$，有 $$\begin{align}&E\Big[f\big(W(t_{1}), \dots, W(t_{n})\big)\Big] \\[0.5em]= &\int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} f(x_{1}, \dots, x_{n})g(x_{1}, t_{1}|0)g(x_{2}, t_{2}-t_{1}|x_{1})\cdots g(x_{n}, t_{n} - t_{n-1}|x_{n-1}) \, \mathrm{d}x_{n} \cdots  \, \mathrm{d}x_{1} \end{align}$$特别地，令 $$f(x_{1}, \dots, x_{n}) = \chi_{[a_{1}, b_{1}]}(x_{1})\cdots \chi_{[a_{n}, b_{n}]}(x_{n}),$$则该定理给出 (3.4)。

**证明.** 主要思路是用换元把每一采样间隔的随机变量剥离，得到一组相互独立的随机变量。
记 $X_{i} = W(t_{i}), i = 1, \dots, n$，然后做线性的变量替换
$$
\begin{align}
Y_{1} &= X_{1}\\
Y_{2} &= X_{2} - X_{1}\\
&\quad\vdots\\
Y_{n} &= X_{n} - X_{n-1}
\end{align} \iff \boldsymbol{Y} = \begin{bmatrix}
Y_{1}\\Y_{2}\\\vdots \\Y_{n}
\end{bmatrix} = \begin{bmatrix}
1\\
-1 & 1\\
&\ddots &\ddots \\
&&-1 & 1
\end{bmatrix}
\begin{bmatrix}
X_{1}\\X_{2}\\\vdots \\X_{n}
\end{bmatrix} = \boldsymbol{J}\boldsymbol{X}
$$
因此有 $|\boldsymbol{J}| \equiv 1$，这样一来我们在变量替换时就非常容易了。这样我们就从原来的随机变量造出了一列相互独立且都服从 Gauss 分布的随机变量。然后令 $$h(y_{1}, y_{2}, \dots, y_{n}) = f(y_{1}, y_{1}+y_{2}, \dots, y_{1}+\cdots +y_{n}).$$我们目标期望，就可以证明定理的内容： 
$$
\begin{align}
&E\Big[f\big(W(t_{1}), \dots, W(t_{n})\big)\Big] = E\Big[ h\big( y_{1}, \dots, y_{n} \big)  \Big] \\[0.5em]
= &\int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} h(y_{1}, \dots, y_{n})p(y_{1}, \dots, y_{n}) \, \mathrm{d}y_{n} \cdots  \, \mathrm{d}y_{1} \\[0.5em]
= &\int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} h\big( y_{1}, \dots, y_{n} \big)g(y_{1}, t_{1}|0)g(y_{2}, t_{2}-t_{1}|0)\cdots g(y_{n}, t_{n} - t_{n-1}|0) \, \mathrm{d}y_{n}  \cdots \, \mathrm{d}y_{1} & 相互独立\\[0.5em]
= &\int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} f(x_{1}, \dots, x_{n})g(x_{1}, t_{1}|0)g(x_{2}, t_{2}-t_{1}|x_{1})\cdots g(x_{n}, t_{n} - t_{n-1}|x_{n-1}) \, \mathrm{d}x_{n} \cdots  \, \mathrm{d}x_{1} & 变量替换
\end{align}
$$
### 3.2.3 白噪声
现在来看看白噪声是怎么回事。

> **引理 3.3**
> $W(\cdot)$ 是一个一维 Brown 运动，对 $s, t > 0$，有 $$E(W(t)) = 0, \quad  E(W^{2}(t)) = t$$以及$$E(W(t)W(s)) = t \wedge s = \min~\{ s, t \}$$(注意，这里的 $\wedge$ 跟微分形式里面那玩意儿没关系，为约定俗成)

**证明.**
前两个结果容易得到。我们计算最后一个。不妨假设 $t > s$，得到 
$$
\begin{align}
E\big( W(t)W(s) \big) &= E\Big[ (W(s) + W(t) - W(s))W(s) \Big]\\
&= E\Big[ W(s)^{2} \Big] + E\Big[ (W(t)-W(s))W(s) \Big]\\
&= E\Big[ W(s)^{2} \Big] + \underbrace{ E\Big[W(t)-W(s)\Big]E\Big[W(s) \Big] }_{ 0 } & 增量独立性\\
&= s = t \wedge s
\end{align}
$$
**一些直觉推导**
我们在介绍 SDE 的时候提到了对 Brown 运动 “求导” 的结果 $\dot{W}(t) = \xi(t)$ 是 “一维白噪声”。但事实上几乎对所有的 $\omega$ 对应的采样路径 $t \mapsto W(t, \omega)$ 对所有 $t \geqslant 0$ 都是不可微的。因此 $\dot{W}(\cdot) = \xi(t)$ 这样的东西实际并不存在。但我们有这个启发式公式 $\text{“}E\Big[ \xi(t)\xi(s) \Big] = \delta_{0}(s - t)\text{”}$。我们可以 “证明” 这件事情。考虑 $h > 0$，固定 $t > 0$，然后算一下两个微商的乘积 
$$
\begin{align}
\phi_{h}(s) &\coloneqq E\left[ \left( \frac{W(t+h) - W(t)}{h} \right)\left( \frac{W(s+h) - W(s)}{s} \right) \right]\\
&= \frac{1}{h^{2}} \Big[ E(W(t+h)W(s+h)) - E(W(t+h)W(s)) - E(W(t)W(s+h)) + E(W(t)W(s)) \Big] \\
&= \frac{1}{h^{2}} \Big[ (t+h)\wedge(s+h) - (t+h)\wedge s - t \wedge (s+h) + t \wedge s \Big] \\
&= \frac{1}{h^{2}}\Big[ 2(t \wedge s) +h  - (t+h)\wedge s - ((t - h) \wedge (s) +h)\Big]\\
&= \frac{1}{h^{2}}\Big[ 2(t \wedge s) - (t+h)\wedge s - (t - h) \wedge s \Big]\\
&= \begin{cases}
0,& s < t - h\\
h^{-2}[s-(t-h)],& t-h < s < t\\
h^{-2}[(t+h) - s],& t < s < t+h\\
0,& s > t + h
\end{cases}
\end{align}
$$
可以看到这是一个连续的 “脉冲”，它的高是 $\displaystyle \frac{1}{h}$，因此曲线下的面积总是 $1$。令 $h \rightarrow 0$，函数 $\phi_{h}(s)$ 就会 ”趋于“ Dirac 质量 $\delta_{0}(s - t)$，也即 $E\Big[ \xi(t)\xi(s) \Big]$。

**为什么 $\dot{W}(\cdot) = \xi(\cdot)$ 叫做 “白噪声”**
假设实值随机过程 $X(\cdot)$ 满足对任意 $t \geqslant 0$，有 $E(X^{2}(t)) < \infty$，我们定义 *自相似函数* 为
$$
r(t, s) \coloneqq E(X(t)X(s)), \quad t, s \geqslant 0
$$
如果存在某个函数 $c: \mathbb{R}\rightarrow \mathbb{R}$，使得 $r(t, s) = c(t - s)$，除此之外随机过程还满足对任意 $t, s \geqslant 0$ 有 $E(X(t)) = E(X(s))$，则我们称 $X(\cdot)$ 是 *广义平稳的*。白噪声 $\xi(\cdot)$ 至少在形式上是 Gauss 广义平稳的，这时 $c = \delta_{0}$。$\xi(\cdot)$ 叫做白噪声的原因是频域中它是常函数，这说明每种 “颜色” （频率）的 “光” 的贡献是相同的。定义对 $c$ 的 Fourier 变换 
$$f(\lambda) = \frac{1}{2\pi}\int_{-\infty}^{\infty} e^{-\text{i}\lambda t}c(t) \, \mathrm{d}t, \quad \lambda \in \mathbb{R} $$
将 Dirac 点质量代入，我们有
$$
f(\lambda) = \frac{1}{2\pi}\int_{-\infty}^{\infty} e^{-\text{i}\lambda t} \delta_{0}\, \mathrm{d}t = \frac{1}{2\pi}, \quad \forall\lambda \in \mathbb{R} .
$$

## 3.3 Brown 运动的构造

现在我们的主要目标是证明 Brown 运动真的存在 —— 构造一个 Brown 运动。我们首先会用区间 $[0, 1]$ 上的所有实值平方可积函数构成的线性空间 $L^{2}(0,1)$ 的一个精心选择的标准正交基构造白噪声 $\xi(\cdot)$ 在它上面的展开。接着我们将这个展开对时间积分，然后证明它是收敛的，这样我们就构造出了 Brown 运动。

### 3.3.1 标准正交基下的展开
$L^{2}(0, 1)$ 上的一个标准正交基是指这个函数序列 $\{ \psi_{n} \}_{n=1}^\infty$ 其中 $\psi_{n} = \psi_{n}(t)$ 是 $[0,1]$ 上的函数（不是随机变量）标准正交的意思是任意两个基函数的内积不是 $0$ 就是 $1$：对所有 $m, n$，满足
$$
\int_{0}^{1} \psi_{n}(s)\psi_{m}(s) \, \mathrm{d}s = \delta_{m,n} = \begin{cases}
1, & m = n\\
0, & m \neq n
\end{cases} 
$$
由于 $\xi(\cdot)$ 是随机过程，我们把它写成下面的形式展开：
$$
\xi(t) = \sum\limits_{n=0}^{\infty} A_{n}\psi_{n}(t), \quad 0 \leqslant t \leqslant 1
$$
其中 $A_{n}$ 相互独立的随机变量（为什么要假设相互独立？）。我们容易看到 
$$
A_{n} = \int_{0}^{1} \xi(t)\psi_{n}(t) \, \mathrm{d} t = \left\langle \xi, \psi_{n} \right\rangle_{L^{2}(0, 1)}. 
$$
我们可以看一下 $A_{n}A_{m}, m \neq n$ 的期望：

$$
\begin{align}
E(A_{n})E(A_{m})= E(A_{n}A_{m}) &= \int_{0}^{1} \int_{0}^{1} {\color{red}E(\xi(t)\xi(s))}\psi_{n}(t)\psi_{m}(s) \, \mathrm{d} t \, \mathrm{d}s\\ 
&= \int_{0}^{1} \int_{0}^{1} {\color{red} \delta_{0}(s-t) }  \psi_{n}(t)\psi_{m}(s) \, \mathrm{d} t \, \mathrm{d}s \\
&= \int_{0}^{1} \psi_{n}(s)\psi_{m}(s) \, \mathrm{d}s = \left\langle \psi_{n}, \psi_{m} \right\rangle_{L^{2}(0, 1)} = \delta_{m,n} = 0.
\end{align}
$$

假如 $m=n$，就得到 $\displaystyle E(A_{n}^{2}) = \int_{0}^{1} \psi_{n}^{2}(s) \, \mathrm{d}s = 1.$ 这说明
1. 系数两两不相关
2. 系数的方差必须是 $1$
因此我们 *希望*  $A_{n}$ 服从标准 Gauss 分布 $N(0, 1)$，这样的良好性质让我们期待上述那样的 Fourier 变换是合理的。根据我们的假设，就有 
$$
W(t) \coloneqq \int_{0}^{t} \xi(s) \, \mathrm{d}s = \sum\limits_{n=0}^{\infty} A_{n} \int_{0}^{t} \psi_{n}(s) \, \mathrm{d}s.  
$$下面我们会选一个比较好的基，然后证明上面的积分收敛。

> **注解 3.4**
> 通过白噪声的性质我们并不能判定其做 Fourier 变换后的随机变量系数是服从标准正态分布的，但标准正态分布是使得后续推导较为简单，且比较自然的选择
> （把 $\xi(\cdot)$ 看成一个泛函，Fourier 变换做的就是某种 “对角化”）


### 3.3.2 Brown 运动的构造

接下来给出的 Brown 运动的构造属于 Lévy 和 Ciesielski。

> **定义 3.5 （Haar 小波基）**
> Haar 小波基 $\{ h_{k}(\cdot) \}_{k=0}^{\infty}$ 是指下面这些函数：
> $$\begin{align}h_{0}(t) &\coloneqq 1 & 0 \leqslant t \leqslant 1\\h_{1}(t) &\coloneqq \begin{cases}\displaystyle 1 & \displaystyle 0 \leqslant t \leqslant \frac{1}{2}\\\displaystyle -1 & \displaystyle \frac{1}{2} < t \leqslant 1\end{cases}\\h_{k}(t) &\coloneqq \begin{cases}\displaystyle 2^{n/2} & \displaystyle \frac{k-2^{n}}{2^{n}} \leqslant t \leqslant \frac{k-2^{n}+1/2}{2^{n}}\\\displaystyle -2^{n/2} &\displaystyle  \frac{k-2^{n}+1/2}{2^{n}} \leqslant t \leqslant \frac{k-2^{n}+1}{2^{n}}\\0 & \text{otherwise}\end{cases} & 2^{n} \leqslant k \leqslant 2^{n+1}, n = 1, 2, \dots\end{align}$$

接下来我们验证 $\{ h_{k}(\cdot) \}_{k=1}^{\infty}$ 是 $L^{2}(0, 1)$ 上的一个性质很好的正交基。

> **引理 3.6 （Haar 基是 $L^{2}(0, 1)$ 的完备标准正交基）**
> Haar 小波基 $\{ h_{k}(\cdot) \}_{k=0}^{\infty}$ 是 $L^{2}(0, 1)$ 的一个完备的标准正交基

**证明.**
首先验证标准（normal）的部分。$h_{0}$ 和 $h_{1}$ 显然符合条件。
$$
\left\langle h_{k}, h_{k} \right\rangle_{L^{2}(0, 1)} = \int_{0}^{1} [h_{k}(t)]^{2} \, \mathrm{d}t= 2^{n/2} \cdot 2^{n/2} \cdot \frac{1}{2^{n}} \equiv 1
$$
接下来验证正交（ortho-）的部分。注意如果 $l > k$ 那么 $h_{l}$ 的支撑集下 $h_{k}$ 总是常数。因此
$$
\int_{0}^{1} h_{k}(t)h_{l}(t)  \, \mathrm{d}t = \pm 2^{n/2} \int_{0}^{1} h_{l}(t) \, \mathrm{d}t \equiv 0 
$$
最后证明完备性。我们需要证明，对 $k = 0, 1, \dots$ 和 $f \in L^{2}(0, 1)$，有 
$$\int_{0}^{1} f(t)h_{k}(t) \, \mathrm{d} t = 0 \iff f = 0\quad \text{a.e.}$$当 $n=0$ 时，有 $\displaystyle \int_{0}^{1} f(t) \, \mathrm{d}t = 0$。
当 $n=1$ 时，有 $\displaystyle\int_{0}^{1/2} f \, \mathrm{d}t = \int_{1/2}^{1} f \, \mathrm{d}t$；结合 $n=0$ 的情况，我们有 $\displaystyle\int_{0}^{1/2} f \, \mathrm{d}t = \int_{1/2}^{1} f \, \mathrm{d}t = 0$。
接着这样做下去，我们能得到对所有 $0 \leqslant k \leqslant 2^{k+{1}}$，都有 $\displaystyle \int_{k/2^{n+1}}^{(k+1)/2^{n+1}} f \, \mathrm{d}t = 0$。因此对所有的二进制有理数 $0 \leqslant s \leqslant r \leqslant 1$，有 $\displaystyle\int_{s}^{r} f \, \mathrm{d}t = 0$，这说明 $f$ 几乎处处为零，因为 $[0, 1]$ 上的任意区间 $[a, b]$ 都可以使用二进制有理数逼近。


> **定义 3.7 （Schauder 函数）**
> 对 $k = 0, 1, 2, \dots$ 定义 $k$ 阶 Schauder 函数为 
> $$s_{k}(t) \coloneqq \int_{0}^{t} h_{k}(s) \, \mathrm{d}s\quad 0 \leqslant t \leqslant 1 $$

我们马上揭示 Schauder 函数和 Brown 运动之间的关系。

> **引理 3.8**
> 对任意 $0 \leqslant s, t \leqslant 1$，有 
> $$\sum\limits_{k=0}^{\infty} s_{k}(s)s_{k}(t) = t \wedge s$$

**证明.**
对 $0 \leqslant s \leqslant 1$，定义 
$$\phi_{s}(\tau) = \begin{cases}
1, &0 \leqslant \tau \leqslant s\\
0, & s < \tau \leqslant 1.
\end{cases}$$
不妨设 $0 \leqslant s \leqslant t \leqslant 1$，根据 Haar 小波基的完备性，可以将 $\phi_{s}$ 和 $\phi_{t}$ 做 Fourier 变换（小波变换），然后用内积线性性和一些实分析结果（在此原书中未提及，后续将陆续补充，现在先不严谨的空在这里）求和号从内积里面扯出来，接下来就简单了
$$
\begin{align}
s &= \int_{0}^{s} 1 \, \mathrm{d}s = \int_{0}^{1} \phi_{s}\phi_{t} \, \mathrm{d}\tau\\
&= \left\langle \phi_{s}, \phi_{t} \right\rangle_{L^{2}(0, 1)} = \left\langle \sum\limits_{k=0}^{\infty} a_{k}h_{k}, \sum\limits_{n=0}^{\infty} b_{n}h_{n} \right\rangle_{L^{2}(0, 1)} = \sum\limits_{k=0}^{\infty} \sum\limits_{n=0}^{\infty} a_{k}b_{n}{\color{red} \left\langle h_{k}, h_{n} \right\rangle_{L^{2}(0, 1)} } \\
&= \sum\limits_{k=0}^{\infty} a_{k}b_{k} \cdot {\color{red} 1} = \sum\limits_{k=0}^{\infty} a_{k}b_{k}
\end{align}
$$
其中 $a_{k}$ 和 $b_{k}$ 就分别是 $\phi_{s}$ 和 $\phi_{t}$ 到 Haar 小波基函数的投影，而且恰好就是我们要的 Schauder 函数： $$\begin{align}
a_{k} &= \int_{0}^{1} \phi_{s}h_{k} \, \mathrm{d}\tau = \int_{0}^{s} h_{k} \, \mathrm{d}\tau = s_{k}(s),\\
b_{k} &= \int_{0}^{1} \phi_{t}h_{k} \, \mathrm{d}\tau = \int_{0}^{t} h_{k} \, \mathrm{d}\tau = s_{k}(t).
\end{align}$$这样我们就证明了这个引理。

Schauder 函数 $s_{k}$ 的这个性质让我们想起 Brown 运动，因为它满足 $E(W(t)W(s)) = t \wedge s$。假设 $W(\cdot)$ 真的可以写成 $\displaystyle \sum\limits_{k=0}^{\infty} A_{k}s_{k}(t)$，我们可以检查一下。
$$
\begin{align}
E(W(t)W(s)) &= E\left[ \left( \sum\limits_{k=0}^{\infty} A_{k}s_{k}(t) \right)\left( \sum\limits_{n=0}^{\infty} A_{n}s_{n}(s) \right) \right]\\
&= \sum\limits_{k=0}^{\infty} \sum\limits_{n=0}^{\infty} s_{k}(t)s_{n}(s) {\color{red} E(A_{k}A_{n}) } \\
&= \sum\limits_{k=0}^{\infty} s_{k}(t)s_{k}(s) = t \wedge s
\end{align}
$$
我们似乎走在正确的路上。下一个问题就是级数 $\displaystyle \sum\limits_{k=0}^{\infty} A_{k}s_{k}(t)$ 是否收敛。

> **引理 3.9**
> $\{ a_{k} \}_{k=0}^{\infty}$ 是一个实序列，假如存在某个常数 $C$ 以及 $\displaystyle 0 \leqslant \delta \leqslant \frac{1}{2}$，序列满足 
> $$|a_{k}| \leqslant Ck^{\delta}, \quad k = 1, 2, \dots$$则级数 $\displaystyle  \sum\limits_{k=0}^{\infty} a_{k}s_{k}(t)$ 对 $0 \leqslant t \leqslant 1$ 一致收敛。

**证明.**
要证明函数项级数一致收敛，只需证明当 $N$ 足够大时，$\sum\limits_{k=N}^{\infty} a_{k}s_{k}(t)$ 可以任意小。设 
$$b_{n} := \max_{2^{n} \leqslant k < 2^{n+1}}|a_{k}| \leqslant C(2^{n+1})^{\delta}$$
然后我们设置一个足够大的数 $m$，然后分段估计级数去掉前 $2^{m}$ 项后的无穷求和：
$$
\begin{align}
\sum\limits_{k=2^{m}}^{\infty} |a_{k}| |s_{k}(t)| &= \sum\limits_{n=m}^{\infty} \sum\limits_{k=2^{n}}^{2^{n+1}-1} |a_{k}| |s_{k}(t)| \\
&\leqslant \sum\limits_{n=m}^{\infty} \sum\limits_{k=2^{n}}^{2^{n+1}-1} b_n |s_{k}(t)| = \sum\limits_{n=m}^{\infty} b_n\sum\limits_{k=2^{n}}^{2^{n+1}-1} |s_{k}(t)| \\
&\leqslant \sum\limits_{n=m}^{\infty} b_n \cdot \max_{{\scriptstyle 2^{n} \leqslant k < 2^{n+1}}\atop{\scriptstyle 0 \leqslant t \leqslant 1}}|s_{k}(t)| \cdot 1(t)\\
&\leqslant C\sum\limits_{n=m}^{\infty} (2^{n+1})^{\delta}2^{-n/2-1}
\end{align}
$$

由于 $\displaystyle \delta < \frac{1}{2}$，上式的指数部分被 $2^{-pn}$ 控制。当 $m$ 足够大时，后面的无穷求和可以任意小。上面推导的第三行使用了 Schauder 函数在 $2^{n} \leqslant k < 2^{n+1}$ 时的支持集（有非零函数值的集合）彼此不交的特点，因此得到了一个一致的上界，也就是常函数 $\displaystyle \max_{{\scriptstyle 2^{n} \leqslant k < 2^{n+1}}\atop{\scriptstyle 0 \leqslant t \leqslant 1}}|s_{k}(t)| \cdot 1(t)$。

> **引理 3.10**
> 考虑相互独立的标准 Gauss 随机变量序列 $\{ A_{k} \}_{k=1}^{\infty}$，对几乎所有 $\omega \in \Omega$，当 $k \rightarrow \infty$ 时，有 
> $$|A_{k}(\omega)| = O(\sqrt{ \log k })$$

**证明.** 
首先估计一下 $P(|A_{k}|) > x$ 的概率 
$$
\begin{align}
P(|A_{k}| > x) &= \frac{2}{\sqrt{ 2\pi }}\int_{x}^{\infty} {e^{-s^{2}/2}} \, \mathrm d{s}\\
&= \frac{2}{\sqrt{ 2\pi }}e^{-x^{2}/4}\int_{x}^{\infty} {e^{-s^{2}/4}} \, \mathrm d{s} \leqslant Ce^{-x^{2}/4}.
\end{align}
$$
令 $x := 4\sqrt{ \log k }$，就有 
$$
P(|A_{k}| > 4\sqrt{ \log k }) \leqslant Ce^{-4\log k} = \frac{C}{k^{4}}.
$$
此时注意到 $\displaystyle \sum \frac{1}{k^{4}} < \infty$，就可以用 Borel-Cantelli 引理，得到
$$P(|A_{k}| > 4\sqrt{ \log k }\quad\text{i.o.}) = 0$$

因此对于几乎所有 $\omega$，都可以找到一个 $K(\omega)$，当 $k \geqslant K(\omega)$ 时，有 $|A_{k}(\omega)| \leqslant 4\sqrt{ \log k } \leqslant Ck^{1/4}$。

> **定理 3.11 （构造 $[0, 1]$ 上的 Brown 运动）**
> 考虑相互独立的标准 Gauss 随机变量序列 $\{ A_{k} \}_{k=1}^{\infty}$，和 Schauder 函数 $s_{k}(t)$，级数
> $$W(t, \omega) \coloneqq \sum\limits_{k=0}^{\infty} A_{k}(\omega)s_{k}(t), \quad  0 \leqslant t \leqslant 1$$在概率空间中几乎处处对 $t$ 一致收敛，且满足下面两个条件
> 1. $W(\cdot)$ 在 $0 \leqslant t \leqslant 1$ 上是 Brown 运动
> 2. 几乎所有 $\omega$ 确定的采样路径 $t \mapsto W(t, \omega)$ 连续

**证明.**
由我们的设定 $A_{k} \sim N(0, 1)$，根据引理 3.10，它可以被 $Ck^{1/4}$ 控制；因此目标级数一致收敛，因此它对几乎所有样本点 $\omega$ 也是连续的。

下面验证这个函数项级数确实就是 Brown 运动 $W(\cdot)$。首先根据 Schauder 函数的定义立刻有 $W(0) = 0\quad\text{a.s.}$。接着我们验证对所有 $0 \leqslant s \leqslant t \leqslant 1$ 都有 $W(t) - W(s) \sim N(0, t - s)$，我们计算它的特征函数：
$$
\begin{align}
\mathscr{F}[W(t)-W(s)] &= E(\exp \left\{ \text{i}\lambda(W(t)-W(s)) \right\} )\\
&= E\left(\exp \left\{ \text{i}\lambda\left[ \sum\limits_{k=0}^{\infty} A_{k}(s_{k}(t)-s_{k}(s)) \right]\right\}\right)\\
&= \prod\limits_{k=0}^{\infty} E\left(\exp \left\{ \text{i}\lambda\left[  A_{k}(s_{k}(t)-s_{k}(s)) \right]\right\}\right) & 独立性\\
&= \prod\limits_{k=0}^{\infty} \exp \left\{ -\frac{\lambda^{2}}{2}\left[ s_{k}(t)-s_{k}(s) \right]^{2}\right\} & A_{k} \sim N(0, 1)\\
&=  \exp \left\{- \frac{\lambda^{2}}{2} \sum\limits_{k=0}^{\infty} \left[ (s_{k}(t))^{2}+(s_{k}(s))^{2}+2s_{k}(t)s_{k}(s) \right]\right\} \\
&=  \exp \left\{- \frac{\lambda^{2}}{2}\left[ t-2s+s \right]\right\} & 引理 3.8\\
&= \exp \left\{- \frac{\lambda^{2}}{2}\left[ t-s \right]\right\} = \mathscr{F}[N(0, t-s)].
\end{align}
$$


最后证明它符合 Brown 运动定义中的增量无关性质。还是一样，我们只需要证明
$$
\begin{align}
\mathscr{F}\left[ W(t_{1}), W(t_{2})-W(t_{1}), \dots, W(t_{m})-W(t_{m-1}) \right]  
&= \prod\limits_{j=1}^{m} \mathscr{F}[W(t_{j})-W(t_{j-1})]
\\&= \prod\limits_{j=1}^{m} \mathscr{F}[N(0, t_{j}-t_{j-1})]
\end{align}
$$
只需继续计算对应的特征函数，然后重复用引理 3.8 的结果即可。

> **定理 3.12 （一维 Brown 运动的存在性）**
> 考虑概率空间 $(\Omega, \mathcal{U}, P)$ 上定义的可数个相互独立的标准 Gauss 随机变量 $\{ A_{n} \}_{n=1}^{\infty}$，则存在一个定义在 $\Omega$ 和 $t \geqslant 0$ 上的一维 Brown 运动 $W(\cdot)$

大概的思路是，有可列个随机变量 $\{ A_{n} \}_{n=1}^{\infty}$ 就可以用对角线方法造出可列个可列长度的随机变量 $\{ \{ A_{k_{n}} \}_{n=1}^{\infty} \}_{k=1}^{n}$：
$$
\begin{matrix}
\vdots\\ 
\color{orange}A_{10} & \vdots\\
\color{green}A_{4} & \color{orange}A_{9} & \hspace{-6pt}{⋰}\\
\color{blue}A_{3} & \color{green}A_{5} & \color{orange}A_{8} & \cdots\\
\color{red} A_{1} & \color{blue} A_{2} &\color{green} A_{6} & \color{orange}A_{7} & \cdots 
\end{matrix}
$$
所以每一个 $k$，都可以造出一个定义在 $[0, 1]$ 上的 Brown 运动 $W_{k}(t)$。接下来的问题就是怎么把它们接起来，其实只要新的接在老的后面，这样一直接下去就可以了。这就得到了下面的递归表达式
$$
W(t) = W(n-1) + W_{n}(t - (n-1))
$$
我们可以验证它满足 Brown 运动的定义。

### 3.3.3 $\mathbb{R}^{n}$ 中的 Brown 运动

最后我们把 Brown 运动拓展到 $\mathbb{R}^{n}$。

> **定义 3.13 （ 中的 Brown 运动）**
> 一个 $\mathbb{R}^{n}$ 值随机过程 $\boldsymbol{W}(\cdot) = (W_{1}(\cdot), \dots, W_{n}(\cdot))$ 如果满足对所有 $k = 1, \dots, n$
> 1. $W_{k}(\cdot)$ 是一个 Brown 运动
> 2. $\sigma$-代数 $\mathcal{W}_{k} = \mathcal{U}[W_{k}(t): t \geqslant 0]$ 是相互独立的
> 它就是一个 $n$-维 Brown 运动

$n$-维 Brown 运动和一维的版本有类似地性质。

> **引理 3.14**
> $\boldsymbol{W}(\cdot)$ 是一个 $n$ 维 Brown 运动，对所有 $k,l = 1, \dots, n$，有 
> $$\begin{align}E\big[ W_{k}(t)W_{l}(s) \big] &= (t \wedge s)\delta_{k,l}\\E\big[ (W_{k}(t)-W_{k}(s))(W_{l}(t)-W_{l}(s)) \big] &= (t - s)\delta_{k,l}, \quad t \geqslant s \geqslant 0\end{align}$$

以及和一维版本十分相似的 $\boldsymbol{W}(t)$ 和联合分布的结果：

> **定理 3.15**
>  是一个  维 Brown 运动，则对任意 $t>0$，有 $\boldsymbol{W}(t) \sim N(\boldsymbol{0}, t\boldsymbol{I})$，因此对任意 Borel 集 $A  \subset \mathbb{R}^{n}$，有 
>  $$P(\boldsymbol{W}(t) \in A) = \frac{1}{(2\pi t)^{n/2}} \int_{A} \exp\left\{ -\frac{\|x\|^{2}}{2t} \right\}  \, \mathrm{d}\boldsymbol{x} $$
>  一般地，对每一个 $m = 1, \dots$ 和每一个函数 $f: \underbrace{ \mathbb{R}^{n} \times \cdots \times \mathbb{R}^{n} }_{ m~个 } \rightarrow \mathbb{R}$，有 
>  $$\begin{align}&E\Big[ f(\boldsymbol{W}(t_{1}), \dots, \boldsymbol{W}(t_{m})) \Big] \\=&\int_{\mathbb{R}^{n}} \cdots \int_{\mathbb{R}^{n}} f(\boldsymbol{x}_{1}, \dots, \boldsymbol{x}_{m}) g(\boldsymbol{x}_{1}, t_{1}|\boldsymbol{0})g(\boldsymbol{x}_{2}, t_{2}-t_{1}|\boldsymbol{x}_{1}) \cdots g(\boldsymbol{x}_{m}, t_{m}-t_{m-1}|\boldsymbol{x}_{m-1}) \, \mathrm{d}\boldsymbol{x}_{m}  \, \mathrm{d}\boldsymbol{x}_{1} \end{align}$$
>  其中 $\displaystyle g(\boldsymbol{x}, t|\boldsymbol{y}) = \frac{1}{(2\pi t)^{n/2}}exp\left\{ -\frac{\|\boldsymbol{x}-\boldsymbol{y}\|^{2}}{2t} \right\}$。

**证明.**
对每个 $t > 0$，随机变量 $W_{1}(t), \dots, W_{n}(t)$ 都是相互独立的，因此有
$$
\begin{align}
f_{\boldsymbol{W}(t)} &= \prod\limits_{k=1}^{n} f_{W_{k}(t)}(\boldsymbol{x}_{k}) \\&=  \prod\limits_{k=1}^{n} \frac{1}{(2\pi t)^{1/2}} \exp\left\{ -\frac{x_{k}^{2}}{2t} \right\} \\&= \frac{1}{(2\pi t)^{n/2}}\exp\left\{ -\frac{\|\boldsymbol{x}\|^{2}}{2t} \right\} = g(\boldsymbol{x}, t|\boldsymbol{0}). 
\end{align}
$$

接下来的步骤和一维情况类似，不再赘述。

## 3.4 采样路径的性质

本节中我们将见到，对样本空间 $\Omega$ 中几乎所有样本点 $\omega$，采样路径 $t \mapsto \boldsymbol{W}(t, \omega)$ 在指数 $\displaystyle \gamma \in \left(0, \frac{1}{2}\right)$ 时都是一致 Hölder 连续的，而对任意 $\gamma > \displaystyle\frac{1}{2}$ 是无处 Hölder 连续的。除此之外，采样路径是几乎一定处处不可微的，且在任意区间的变差都是无穷大。

### 3.4.1 采样路径的连续性

首先给出一致 Hölder 连续的定义

> **定义 3.16 （ $\gamma$-Hölder 连续）**
> 函数 $f:[0, T] \rightarrow \mathbb{R}$ 若在 $s$ 处满足对任意 $t \in [0, T]$，都存在 $K$，有 
> $$|f(t)-f(s)| \leqslant K|t-s|^{\gamma}$$我们称它在 $s$ 处 $\gamma$-Hölder 连续。
 
> **定义 3.17 （一致 $\gamma$-Hölder 连续）**
> 函数 $f:[0, T] \rightarrow \mathbb{R}$ 若对任意 $s,t \in [0, T]$，都存在 $K$，有 
> $$|f(t)-f(s)| \leqslant K|t-s|^{\gamma}$$我们称它在 $[0, T]$ 上一致 $\gamma$-Hölder 连续。

接下来的定理属于 Kolmogorov，是判定 Hölder 连续性的重要工具

> **定理 3.18 （Kolmogorov，采样路径的 Hölder 连续性）**
> 几乎所有采样路径都连续的随机过程 $\boldsymbol{X}(\cdot)$ 若满足存在 $\alpha, \beta > 0$，$C \geqslant 0$，对任意 $t, s \geqslant 0$，它的几乎所有采样路径都满足 
> $$E(|\boldsymbol{X}(t) - \boldsymbol{X}(s)|^{\beta}) \leqslant C|t-s|^{1+\alpha}$$则对几乎所有 $\omega$、任意 $T > 0$ 和任意 $\displaystyle \gamma \in \left(0, \frac{\alpha}{\beta}\right)$，都存在一个常数 $K=K(\omega, \gamma, T)$ 使得对任意 $s, t \in [0. T]$，有  
> $$|\boldsymbol{X}(s, \omega) - \boldsymbol{X}(t, \omega)| \leqslant K|t-s|^{\gamma}$$
> 也即 $\boldsymbol{X}(\cdot, \omega)$ 在 $[0, T]$ 上一致 $\gamma$-Hölder 连续。

定理证明从略。其证明过程稍作修改可以揭示下面这件事。如果存在一个常数 $C$，使得随机过程 $\boldsymbol{X}(\cdot)$ 满足 $$E(|\boldsymbol{X}(t)-\boldsymbol{X}(s)|^{\beta}) \leqslant C|t-s|^{1+\alpha}, \quad  \alpha, \beta >0$$那么它的一个**版本**（概率意义下和 $\boldsymbol{X}(\cdot)$ 相等）$\tilde{\boldsymbol{X}}(\cdot)$ 满足对几乎所有采样路径，以及所有 $\displaystyle \gamma \in \left(0, \frac{\alpha}{\beta}\right)$ 都是 $\gamma$-Hölder 连续的。

最后我们来看 Brown 运动的 Hölder 连续性。

> **定理 3.19 （Brown 运动采样路径的连续性）**
> 对所有 $\displaystyle \gamma \in \left( 0, \frac{1}{2} \right)$ ，Brown 运动的采样路径 $t \mapsto \boldsymbol{W}(t, \omega)$ 对几乎所有 $\omega$ 在 $[0, T]$ 上一致 $\gamma$-Hölder 连续

我们要用到定理 3.18。首先想到 $\boldsymbol{W}(t) - \boldsymbol{W}(s) \sim N(\boldsymbol{0}, (t-s)\boldsymbol{I})$，因此不等式左边的期望就比较好处理，我们考虑整数 $m=1, \dots$，然后让左边的指数是 $2m$，方便操作：
$$
\begin{align}
E(|\boldsymbol{W}(t)-\boldsymbol{W}(s)|^{2m}) &= \frac{1}{(2\pi r)^{n/2}} \int_{\mathbb{R}^{n}} \| \boldsymbol{x} \| ^{2m} \exp\left\{ -\frac{\|\boldsymbol{x}\|^{2}}{2r} \right\}  \, \mathrm{d} \boldsymbol{x} & r \leftarrow  t - s\\
&= r^{m} \cdot {\color{red} \frac{1}{(2\pi)^{n/2}} \int_{\mathbb{R}^{n}} \| \boldsymbol{y} \| ^{2m} \exp\left\{ -\frac{\|\boldsymbol{y}\|^{2}}{2} \right\}  \, \mathrm{d} \boldsymbol{y}}  & \boldsymbol{y} \leftarrow  \frac{\boldsymbol{x}}{\sqrt{ r }} \\
&= {\color{red} C } r^{m} = C|t-s|^{m}
\end{align}
$$
所以上面的定理在这里用的话，对所有的 $\alpha = m-1$，$\beta=2m$ 都成立，因此 Brown 运动在 $\mathbb{R}_{\geqslant {0}}$ 上对所有 $\displaystyle 0 < \gamma < \frac{\alpha}{\beta}= \frac{1}{2} - \frac{1}{2m}$ 都 $\gamma$-Hölder 连续。因为对所有 $m$ 成立，取极限就得到了结果。

### 3.4.2 无处可微性

最后我们来证 Brown 运动的采样序列在 $\displaystyle \gamma > \frac{1}{2}$ 时依概率处处 $\gamma$-Hölder 不连续。

> **定理 3.20**
> Brown 运动的采样路径在 $\displaystyle \frac{1}{2} <\gamma \leqslant1$ 时依概率处处 $\gamma$-Hölder 不连续，不可微，且在定义域中的任意区间上有无限变差。

**证明.**
我们只证一维的情形，不失一般性，我们只考虑 $[0, 1]$ 上的情况。先找一个很大的 $N$，使得 $$N\left(\gamma-\frac{1}{2}\right) > 1$$假设 $t\mapsto W(t, \omega)$ 在某点 $s$ 处 $\gamma$-连续，按照定义，我们知道对任意 $t \in [0, 1]$，存在某个尝试 $K$，使得
$$
|W(t, \omega) - W(s, \omega)| \leqslant K|t-s|^{\gamma}
$$
然后考虑一个大数 $n \gg 1$，和指标 $i = [ns]+1$，以及 $j = i, i+1, \dots, i+N-1$。这是为了离散化时间轴，将 $[0, 1]$ 切成 $n$ 份，然后将 $s$ 所在的那一份的右侧离散点开始分析，再往右数 $N-1$ 个离散点。接下来考虑某个采样路径在两个相邻离散点之处的函数值差的绝对值

$$
\begin{align}
\left| W\left( \frac{j}{n}, \omega \right)  - W\left( \frac{j+1}{n}, \omega \right)  \right| &\leqslant \left| W\left( s, \omega \right)  - W\left( \frac{j}{n}, \omega \right)  \right| + \left| W\left( s, \omega \right)  - W\left( \frac{j+1}{n}, \omega \right)  \right|& 三角不等式\\
&\leqslant K\left[ \left| s-\frac{j}{n} \right|^{\gamma} + \left| s-\frac{j+1}{n} \right|^\gamma  \right] \leqslant \frac{M}{n^{\gamma}}
\end{align}
$$
其中 $M$ 是某个常数。因此这个样本点 $\omega$ 位于下面的事件中：
$$
A_{M, n}^{(i)} \coloneqq \left\{ \left| W\left( \frac{j}{n} \right)- W\left( \frac{j+1}{n} \right) \right| \leqslant \frac{M}{n^{\gamma}}, j = i, \dots, i+N-1 \right\} 
$$

这个集合的意义是，给定划分细度 $n$ 下，从 $s$ 右侧第一个离散点开始（包括第一个）的 $N$ 个离散点分割得到的若干区间中，采样序列在每个小区间的增量绝对值 $\displaystyle\frac{M}{n^{\gamma}}$。 所以样本空间中满足采样路径在 $[0, 1]$ 上某点 $s$ 处 $\gamma$-Hölder 连续的所有样本点构成的集合一定**被包含在**下面的集合中：

$$
\bigcup_{M=1}^{\infty}\bigcup_{k=1}^{\infty}\bigcap_{n=k}^{\infty}\bigcup_{i=1}^{n} A_{M, n}^{(i)}
$$

* 第一层表示从 $i$ 开始往后数 $N-1$ 个格子，采样路径每格变化有限的样本点；它包含了所有在离散点 $i$ 处满足 $\gamma$-Hölder 连续（且增量被 $M$ 相关的函数控制）的样本点。
* 第二层和第三层收集所有从不同的 $k$ 开始一直加细到无穷大，满足每格增量都有限的样本点，也就是加细分割时 $\displaystyle \bigcup_{i=1}^{n} A_{M, n}^{(i)}$ 稳定发生的样本点，这一步是在调整包含集合，被 $M$ 控制的使得采样路径 $\gamma$-Hölder 连续的样本点仍然被包含在内。这两步是服务后续估计的。
* 最外一层逐渐放开上界 $M$，这样就包含了所有的 $\gamma$-Hölder 连续的函数

接下来要做的读者也许就能猜到，就是要证这个包含集合的概率是零。对所有 $k$ 和 $M$，估计里面两层的概率

$$
\begin{align}
P\left( \bigcap_{n=k}^{\infty}\bigcup_{i=1}^{n} A_{M, n}^{(i)} \right) &\leqslant \liminf_{ n \to \infty } P\left( \bigcup_{i=1}^{n} A_{M, n}^{(i)} \right) \\
&\leqslant \liminf_{ n \to \infty } \sum_{i=1}^{n} P\left(A_{M, n}^{(i)} \right) \\
&\leqslant \liminf_{ n \to \infty } n \cdot \left[ P\left( \left|W\left( \frac{1}{n}   \right) \leqslant \frac{M}{n^{\gamma}} \right| \right)  \right] ^{n} & W\left( \frac{j+1}{n} \right) - W\left( \frac{j}{n} \right) \sim W\left( \frac{1}{n} \right)
\end{align}
$$

现在来看括号里面的东西。
$$
\begin{align}
P\left( \left|W\left( \frac{1}{n}   \right) \leqslant \frac{M}{n^{\gamma}} \right| \right) &= \frac{1}{\sqrt{ 2\pi n^{-1} }}\int_{-Mn^{-\gamma}}^{Mn^{-\gamma}} \exp\left\{ -\frac{nx^{2}}{2} \right\}  \, \mathrm{d}x \\[0.5em]
&= \frac{1}{\sqrt{ 2\pi }}\int_{-Mn^{1/2-\gamma}}^{Mn^{1/2-\gamma}} \exp\left\{ -\frac{y^{2}}{2} \right\}  \, \mathrm{d}y & y \leftarrow  \sqrt{ n } \cdot x\\[0.5em]
&\leqslant Cn^{1/2-\gamma}
\end{align}
$$

所以可以计算上面概率的一个上界：
$$
\begin{align}
P\left( \bigcap_{n=k}^{\infty}\bigcup_{i=1}^{n} A_{M, n}^{(i)} \right) &\leqslant \liminf_{ n \to \infty } nC[n^{1/2-\gamma}]^{N} = 0
\end{align}
$$

由于这对所有 $k$ 和 $M$ 都成立，因此由概率的可数可加性，有
$$
P\left( \bigcup_{M=1}^{\infty}\bigcup_{k=1}^{\infty}\bigcap_{n=k}^{\infty}\bigcup_{i=1}^{n} A_{M, n}^{(i)} \right) \leqslant 0. 
$$

因此出现 $\gamma$-Hölder 连续轨道的概率至多是零。 

假如轨道在某处可微，那它至少是 $1$-Hölder 连续的，而这件事发生的概率为零。假如它在某个子区间的变差的有界的，那它一定在那个区间上几乎处处可导，因此这也是几乎不可能的。

## 3.5 Markov 性

> **定义 3. 21 （条件概率）**
> 假设 $\mathcal{V} \subset \mathcal{U}$ 是一个子 $\sigma$-代数，对事件 $A \in \mathcal{U}$，定义 
> $$P(A|\mathcal{V}) \coloneqq E(\chi_{A}|\mathcal{V})$$ 为事件 $A$ 在给定 $\mathcal{V}$ 下的条件概率，它是一个随机变量。


> **定义 3. 22 （Markov 过程）**
> 一个取值于 $\mathbb{R}^{n}$ 的随机过程 $\boldsymbol{X}(\cdot)$ 如果对一切 $0 \leqslant s \leqslant t$ 和一切 Borel 集 $B \in \mathbb{R}^{n}$，都满足 
> $$P(\boldsymbol{X}(t) \in B|\mathcal{U}(s)) = P(\boldsymbol{X}(t) \in B|\boldsymbol{X}(s))\quad\text{a.s.}$$我们称其为一个 Markov 过程。

可以看出 Markov 过程中 $\boldsymbol{X}(s)$ 所包含的 “信息” 和给定一段历史 $\mathcal{U}(s)$ 所包含的信息相当：它只 “记得” 当前的状态，而不记得来时的路。

> **定理 3. 23 （$n$ 维 Brown 运动是 Markov 过程）**
> $n$ 维 Brown 运动 $\boldsymbol{W}(\cdot)$ 是一个 Markov 过程，且对所有 $0 \leqslant s \leqslant t$ 和所有 Borel 集 $B$，都有 
> $$P(\boldsymbol{W}(t) \in B|\boldsymbol{W}(s)) = \frac{1}{(2\pi(t-s))^{n/2}} \int_{B} \exp\left\{ -\frac{|\boldsymbol{x} - \boldsymbol{W}(s)|^{2}}{2(t-s)} \right\} \, \mathrm d{\boldsymbol{x}}\quad\text{a.s.}$$

**证明.**
找一个 Borel 集合 $A$，考虑函数
$$
\Phi_{B}(\boldsymbol{y}) \coloneqq \frac{1}{(2\pi(t-s))^{n/2}} \int_{B} \exp\left\{ -\frac{|\boldsymbol{x} - \boldsymbol{y}|^{2}}{2(t-s)} \right\} \, \mathrm d{\boldsymbol{x}}
$$
$\Phi(\boldsymbol{y})$ 是 $\mathcal{U}(\boldsymbol{W}(s))$-可测的。为了证明这个定理，我们需要证明对任意 $C \in \mathcal{U}(\boldsymbol{W}(s))$，有
$$
\int_{C} \chi_{\{ \boldsymbol{W}(t) \in A \}} \, \mathrm{d}P = \int_{C} \Phi(\boldsymbol{W}(s)) \, \mathrm{d}P 
$$
由于 $\mathcal{U}(\boldsymbol{W}(s)) = \{ \boldsymbol{W}^{-1}(s,B), B \in \mathcal{B}\}$，其中 $\mathcal{B}$ 是 $\mathbb{R}^{n}$ 中的所有 Borel 集。因此 $\boldsymbol{W}(s)$ 生成的 $\sigma$-代数 $\mathcal{U}(\boldsymbol{W}(s))$ 中的元素 $C$ 就对应着 $\boldsymbol{W}(s)$ 在某个 Borel 集 $B$ 下的原像。

一方面，
$$
\begin{align}
I_{1} = \int_{C} \chi_{\{ \boldsymbol{W}(t) \in A \}} \, \mathrm{d}P &= P(\boldsymbol{W}(s) \in B, \boldsymbol{W}(t) \in A)\\
&= \int_{B} \int_{A} g(\boldsymbol{y}, s, |\boldsymbol{0})g(\boldsymbol{x}, t-s|\boldsymbol{y}) \, \mathrm{d} \boldsymbol{x} \, \mathrm{d}\boldsymbol{y} & \text{Brown 运动的联合概率公式} \\
&= \int_{B} g(\boldsymbol{y}, s, |\boldsymbol{0}) {\color{red} \int_{A} g(\boldsymbol{x}, t-s|\boldsymbol{y}) \, \mathrm{d} \boldsymbol{x} }  \, \mathrm{d}\boldsymbol{y}\\
&= \int_{B} g(\boldsymbol{y}, s|\boldsymbol{0}) {\color{red} \Phi_{A}(\boldsymbol{y}) }  \, \mathrm{d}\boldsymbol{y} 
\end{align}
$$
另一方面，
$$
\begin{align}
I_{2} = \int_{C} \Phi(\boldsymbol{W}(s)) \, \mathrm{d}P &= \int_{\color{blue}\Omega} \chi_{B}(\boldsymbol{W}(s)) \cdot \Phi(\boldsymbol{W}(s)) {\color{red} \, \mathrm{d}P}\\
&=\int_{\color{blue}\mathbb{R}^{n}} \chi_{B}(\boldsymbol{y}) \cdot \Phi(\boldsymbol{y}) \cdot {\color{red} \frac{1}{(2\pi s)^{1/2}} \exp\left\{ -\frac{|\boldsymbol{y}|^{2}}{2s} \right\} \, \mathrm{d} \boldsymbol{y} } & 换元\\
&= \int_{B} g(\boldsymbol{y}, s|\boldsymbol{0}) \Phi(\boldsymbol{y}) \, \mathrm{d}\boldsymbol{y} = I_{1}.
\end{align}
$$


> **注释**
> Brown 运动的 Markov 性一定程度上解释了其采样路径的无处可微性。

假如我们知道 $W(s, \omega) = b$ 则该采样路径在此之后的未来行为只取决于当前这个值，而与它如何达到这个地方无关，这说明了其 “无记忆性”。

