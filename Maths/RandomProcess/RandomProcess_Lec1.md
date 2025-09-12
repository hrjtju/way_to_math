# 0 Overview
* 只需交课堂笔记
* 可能中间有大作业
* 更偏向直观和应用

# 概率论

**概率模型 / 随机模型 / 随机试验**

**样本空间（集合）  $\Omega$**

**事件类（ $\sigma$-代数 ）$\mathcal{F} \subset 2^{\Omega}$**
* $\Omega \in \mathcal{F}$ （必然事件）
* $A \in \mathcal{F}$，则 $A^{c} \in \mathcal{F}$ （对补集封闭）
* 若 $A_{i} \in \mathcal{F}, I$ 可数，则 $\bigcup_{i \in I} A_{i} \in \mathcal{F}$ （对可数并封闭）
注意
* $A \cap B = ((A \cap B)^{c})^{c} = (A^{c} \cup B^{c})^{c}$ 可知 $\mathcal{F}$ 对有限交封闭
* 由于 $\mathbb{R}$ 上有不可测集，不能取 $\mathcal{F} = 2^{\mathbb{R}}$。考虑 $\mathbb{R}/_{\sim}$，其中 $\sim$ 定义为 $a \sim b \iff a - b \in \mathbb{Q}$，然后将 $\mathbb{R}/_{\sim}$ 中每个等价类取一个位于 $[0, 1]$ 中的代表元，得到集合 $S$ 这是一个不可测集

**概率测度 $\mathbb{P}: \mathcal{F} \rightarrow [0, 1]$**
* $\mathbb{P}(\Omega) = 1$ （规范性）
* $\mathbb{P}(A) \geqslant 0, \forall A \in \mathcal{F}$ （非负性）
* $A_{i} \in \mathcal{F}$ 两两不交，则 $\mathbb{P}\left[ \bigsqcup\limits_{i \in I} A_{i} \right] = \sum\limits_{i \in I} \mathbb{P}(A_{i})$ （可数可加性 / $\sigma$-可加性 ）
定义类似长度、面积、体积和级数规则

**Toy Example** 投掷均匀薄硬币
* $\Omega = \{ 正面, 反面 \}$
* $\mathcal{F} = 2^{\Omega}$
* $\mathbb{P}(\Omega) = 1$，$\displaystyle \mathbb{P}(\varnothing) = 0, \mathbb{P}(正面)=\mathbb{P}(反面) = \frac{1}{2}$

* 不均匀的硬币：考虑两次投掷后 “正反” 和 “反正”，不考虑其他情况

**Toy Example** 掷骰子

**概率的基本运算规律**
* （容斥原理）$\mathbb{P}(A \cup B) = \mathbb{P}(A) + \mathbb{P}(B) - \mathbb{P}(A\cap B)$；一般地，有 $\displaystyle \mathbb{P}\left[ \bigcup_{i = 1}^{n}A_{i} \right] = \sum\limits_{i=1}^{n}\mathbb{P}(A_{i}) - \left[ \sum\limits_{i\ne j} \mathbb{P}(A_{i} \cap A_{j}) \right] + \left[ \sum\limits_{i \neq j \neq k} \mathbb{P}(A_{i} \cap A_{j} \cap A_{k}) \right] - \cdots$
* $\mathbb{P}(A\cup B) \leqslant \mathbb{P}(A) + \mathbb{P}(B)$
* （Bayes 公式）$\displaystyle \mathbb{P}(B | A) := \frac{\mathbb{P}(A, B)}{\mathbb{P}(A)}$
* （独立性）$\mathbb{P}(A, B) = \mathbb{P}(A)\mathbb{P}(B) \iff \mathbb{P}(B|A) = \mathbb{P}(B)$ ，也即 $A$ 对 $\Omega$ 做了较好的切分，也记作 $A \perp B$
    * 两两独立不一定全部独立：两个骰子，第一个是偶数、第二个是偶数、两个加起来是偶数
* （链式法则）$\mathbb{P}(A_{1}, A_{2}, \dots, A_{n}) = \mathbb{P}(A_{1})\mathbb{P}(A_{2}|A_{1}) \cdot \cdots \mathbb{P}(A_{n}|A_{n-1},\dots,A_{1})$
* $\mathbb{P}(A^{c}) = 1- \mathbb{P}(A)$
* $\mathbb{P}(A-B) = \mathbb{P}(A) - \mathbb{P}(A, B)$
* （全概率公式）$\mathbb{P}(A) = \sum\limits_{i \in I} \mathbb{P}(A|B_{i})\mathbb{P}(B_{i})$ 其中 $B_{i}$ 是 $\Omega$ 的一个（可数）划分

**羊车门问题**

**实值随机变量** $X: \Omega \rightarrow \mathbb{R}$
满足对任意开区间 $(-\infty, x) \subset \mathbb{R}$，有 $X^{-1}[(-\infty, x)] \in \mathcal{F}$.

**累积分布函数** $F_{X}:= \mathbb{P}(X(\omega) \leqslant x) = \mathbb{P}(X \leqslant x)$
* 最多有可数个不连续点

**概率密度函数** 
若 $F_{x}$ 可以写成 $\displaystyle \int_{-\infty}^{x} {p_{X}(x)} \, \mathrm d{x}$，则称 $p_{X}$ 是分布密度函数，显然有
* $\displaystyle \int_{\mathbb{R}}p_{X}(x) \, \mathrm d{x} = 1$
* $\displaystyle \lim_{ x \to \infty } p_{X}(x) = 0$

---
**离散分布**
* Bernoulli 分布
* 二项分布
* 几何分布
* Poisson 分布

**连续分布**
* 指数分布
* Gauss 分布
* 均匀分布

* 期望、方差
* 矩母函数

尾部概率
* Markov 不等式
* Chebyshev 不等式
* Chernov 界

---

**例 0.1**

**例 0.2**

2. 不是，$A^{c}$ 不在 $\{ \Omega, A, \varnothing \}$ 中
3. 是的，它关于补集和可数并封闭，$\Omega$ 和空集也都在其中
4. 是的，$\Omega$ 是有限集，$2^{\Omega}$ 是最 “精细” 的事件类
5. 有限样本空间 $\Omega$ 的大小为 $|\Omega|$，其子集个数是 $$\underbrace{ \binom{|\Omega|}{0} }_{ 空集 } + \underbrace{ \binom{|\Omega|}{1} }_{ 单点集 } + \binom{|\Omega|}{2} + \cdots + \underbrace{ \binom{|\Omega|}{|\Omega|-1} }_{ 单点集的补集 } + \underbrace{ \binom{|\Omega|}{|\Omega|} }_{ 全集 } = 2^{|\Omega|}$$
6. 概率测度 $\mathbb{P}$ 是从 $\mathscr{F}$ 到 $\mathbb{R}_{\geqslant_{0}}$ 的映射

**例 0.3** 从 $M$ 个互不相同的球中抽取 $n$ 个
1. 有放回抽取
	1. 有序：所有情况为 $111\cdots111, 111\cdots112, \dots, 211\cdots111, \dots, M M M \cdots M M M$，因此 $|\Omega| = M^{n}$
	2. 无序：考虑一般的情形，拿出了 $n$ 个球，一共有 $k$ 种。由于是无序，规定从小到大排序结果相同的两次实验结果等价，其一般形式为 $\underbrace{ 11\cdots122\cdots 2\cdots kk \cdots k }_{ n 个 }$，这样的情况相当于在 $n$ 个格子的 $n-1$ 个空隙中插 $k-1$ 个隔板，总共情况数量为 $\displaystyle \binom{n-1}{k-1}$。因此遍历所有的情况就得到 $|\Omega| = \displaystyle \sum\limits_{k=1}^{\min\{n, M\}} \binom{M}{k} \binom{n-1}{k-1}$
2. 不放回抽取
	1. 有序：$\displaystyle |\Omega| = \binom{M}{n} n!$
	2. 无序：$\displaystyle |\Omega| = \binom{M}{n}$

**例 0.4** 把 $n$ 个球扔进 $M$ 个箱子
1. 球可区分，箱子可区分，不限箱中球数：相当于一个每列都只有一个 $1$ 的形状为 $M \times n$ 的 $0-1$ 矩阵的总数，即 $|\Omega| = M^{n}$
2. 球不可区分，箱子可区分，不限箱中球数： 相当于先在每个盒子中放一个球，然后用隔板法，有 $|\Omega| = \displaystyle \binom{M+n-1}{M-1}$
3. 球可区分，箱子可区分，箱中球数至多一个：显然有 $M \geqslant n$。相当于每行每列最多只有一个 $1$ 的 $M \times n$ 的 $0-1$ 矩阵总数，即 $|\Omega| = M(M-1)\cdots(M-n+1) =\displaystyle \frac{M!}{(M-n)!}$
4. 球不可区分，箱子可区分，箱中球数至多一个：显然有 $M \geqslant n$，即为 $M$ 个箱子中选 $n$ 个装 $1$ 个球，即 $|\Omega| = \displaystyle \binom{M}{n}$
5. 球不可区分，箱子不可区分，不限箱中球数：分拆数。记 $f(i,j)$ 是把 $i$ 个相同的球放在 $j$ 个相同的盒子中（都不空）的方案总数。因此有 $f(i, 1) = 1$，$f(0, j) = 1$，$f(1,j) = 1$。有递推式 $$f(m,n) = f(m-n, n) + f(m-1, n-1)$$并规定当 $i < j$ 时 $f(i,j) = 0$。因此 $|\Omega| = \displaystyle \sum\limits_{k=1}^{M} f(n, k)$

**例 0.5**
由容斥原理，有
$$
\begin{align}
|A \cup B \cup C| &= |A| + |B| + |C| - |A\cap B| - |B\cap C| - |C \cap A| + |A \cap B \cap C|\\
&= 67 + 41 + 29 - 14 - 6 - 10 + 2 = 109
\end{align}
$$

**例 0.6**
$$
\begin{align}
P(B \cup C) &= P(A, B \cup C) + P(\bar{A}, B \cup C)\\
&= 0.3 + P(\bar{A}) - P(\bar{A}, \overline{B \cup C})\\
&= 0.3 + P(\bar{A}) - P(\bar{A}, \bar{B}, \bar{C})\\
&= 0.3 + 0.6 - 0.1 = 0.8.
\end{align}
$$
**例 0.7**
任取正数 $\varepsilon > 0$，总存在正整数 $N > 0$，使得当 $n \geqslant N$ 时，成立
$$
\begin{align}
|P(A_{n}) - 1| = 1-P(A_{n}) &< \varepsilon\\
|P(B_{n}) - 1| = 1-P(B_{n}) &< \varepsilon
\end{align}
$$
因此 $1 \geqslant P(A_{n} \cup B_{n}) \geqslant P(A_{n}) > 1-\varepsilon$，进而有 $\displaystyle \lim_{ n \to \infty } P(A_{n} \cup B_{n}) = 1$。由容斥原理，有
$$
P(A_{n} \cup B_{n}) = P(A_{n}) + P(B_{n}) - P(A_{n} \cap B_{n})
$$
两侧取极限，就有 $\displaystyle \lim_{ n \to \infty } P(A_{n} \cap B_{n}) = 1$。

**例 0.8**
首先
$$
\begin{align}
P(A|B) + P(A|\bar{B}) &= \frac{P(A, B)}{P(B)} + \frac{P(A,\bar{B})}{P(\bar{B})}\\
&= \frac{P(A, B)}{P(B)} + \frac{P(A,\bar{B})}{1-P(B)}\\
&= \frac{P(A, B) + P(B)[P(A, \bar{B})-P(A, B)]}{P(B)(1-P(B))}
\end{align}
$$
上式无法化简，显然不等于 $1$。假如 $A$ 是掷骰子掷出偶数，$B$ 是掷骰子掷出 $3$ ，那么 $P(A|B) = 0$，$P(A|\bar{B}) = 0.6$。
$P(A)$ 和 $P(A|B)$ 可以有任一大小关系。继续假设 $A$ 是掷骰子掷出偶数
* 若 $B$ 是掷骰子掷出 $3$，则 $P(A|B) < P(A)$
* 若 $B$ 是掷骰子掷出 $2$，则 $P(A|B) > P(A)$
* 若 $B$ 是掷骰子掷出 $1, 2, \dots, 6$ 中的其中一个，则 $P(A|B) = P(A)$

**例 0.9**
* 第一个骰子和第二个骰子互不影响，因此 $A$ 和 $B$ 独立
* $\displaystyle P(AC) = 第一个为奇数，第二个为偶数 = \frac{1}{4}$，两个骰子和为奇数有两种情况，第一个是奇数，第二个是偶数；第一个是偶数，第二个是奇数。因此 $P(AC) = P(A)P(C)$，因此 $A$ 和 $C$ 独立。由对称性知 $B$ 和 $C$ 也独立
* 显然如果第一个和第二个都是奇数，那和不可能为奇数，即 $P(ABC)= 0$，因此 $A,B,C$ 不独立

**例 0.10**
(1) $A$ 和 $B \cap C$ 独立，但 $A$ 和 $B$，以及 $A$ 和 $C$ 不独立。
![[Pasted image 20250912123722.png]]
考虑这样的概率空间，取子集与全空间面积之比作为概率。全空间是矩形 $ABCD$，其中线段 $FG$ 和 $IJ$ 分别与矩形的两边垂直。令 $A$ 事件是矩形 $ABGF$，$B$ 事件是矩形 $IBGH$，$C$ 事件是矩形 $HGCJ$。显然 $A$ 和 $B \cap C$ 独立，但 $A$ 和 $B$，以及 $A$ 和 $C$ 不独立。

(2) $P(A \cap B) = P(A) + P(B) - P(A \cup B) \geqslant P(A) + P(B) - 1 > \displaystyle \frac{1}{2}$

**例 0.11**
选中黑球的概率是
$$
\begin{align}
P(B) &= \sum\limits_{i} P(B|i)P(i) = \frac{1}{3}\sum\limits_{i} \frac{b_{i}}{b_{i} + w_{i}} 
\end{align}
$$
已知是黑球，是从第 $i$ 个袋子抽出的概率为
$$
P(i|B) = \frac{3P(i,B)}{\sum_{j} P(B|j)P(j)} = \left( \frac{3b_{i}}{b_{i} + w_{i}} \right) \Big/ \left(\sum\limits_{j} \frac{b_{j}}{b_{j} + w_{j}} \right)
$$

**例 0.12**
记 Alice 发出字符 $i$ 的事件为 $A_{i}$，Bob 收到字符 $j$ 的事件为 $B_{j}$。
$$
\begin{align}
P(B_{1}|A_{0}) &= \frac{P(B_{1}, A_{0})P(A_{0})}{P(B_{1}, A_{0})P(A_{0}) + P(B_{1}, A_{1})P(A_{1})}\\
&= \frac{p_{e}p_{0}}{p_{e}p_{0} + (1-p_{e})(1-p_{0})}
\end{align}
$$

**例 0.13**
不妨假设学生偏好做线性代数试题。
* 假设学生选到的是线性代数试题，那么助教不论打开哪一个试题都是随机过程试题
* 假设学生选到的是随机过程试题，那么助教只能打开剩下的那一个随机过程试题（没有打开的那个就是线性代数试题）
所以在概率为 $\displaystyle \frac{1}{3}$ 的第一种情境中，学生不能换；在概率为 $\displaystyle \frac{2}{3}$ 的第二种情境中，学生应该换。综上所述学生应该更换试题。

**例 0.14**
(1) 学生 $A$ 被选中的概率为 $\displaystyle \binom{4}{2} \Big/ \binom{5}{2} = \frac{3}{5}$
(2) 老师是随机公布结果的，因此概率变为 $\displaystyle \binom{3}{1} \Big/ \binom{4}{2} = \frac{1}{2}$
(3) 
$$
\begin{align}
P(A|E) &= \frac{P(A, E)}{P(A, E) + P(\bar{A}, B)}\\
&= \left( \frac{1}{2} \cdot \frac{3}{10} \right) \Big/ \left( \frac{1}{2}\cdot \frac{3}{10} + \frac{1}{3} \cdot \frac{3}{10} \right) = \frac{3}{5}
\end{align}
$$
**例题 0.16**
（1）$\mathcal{F}_{1}$ 和 $\mathcal{F}$ 是事件类，$\mathcal{F}_{0}$ 不是事件类
（2）考虑以 $\mathcal{F}_{1}$ 为事件类的概率模型，以及 $X^{-1}(\{ X \leqslant 2 \}) = \{ 1, 2 \} \notin \mathcal{F}_{1}$，因此 $X$ 不是该概率模型上的随机变量。如果考虑 $\mathcal{F}$ 为事件类，则对任意的 $B = \{ X \leqslant x \}$ 都有 $X^{-1}(B) \subset \mathcal{F}$，因此 $X$ 此时是该概率模型上的随机变量
（3）由于 $Y = X \text{mod} 2$，因此 $Y$ 的取值为 $\{ 0, 1 \}$。考虑 $\{ X \leqslant 0 \}$，则 $X^{-1}(\{ X \leqslant 0 \})$ $= \{ 2, 4, 6 \}$ $= \bar{A} \in \mathcal{F}_{1}$，而 $X^{-1}(\{ X \leqslant 1 \})$ 以及其他情形就不再赘述，因此 $Y$ 是随机变量。更细的事件类 $\mathcal{F}$ 就更不用说，显然 $Y$ 也是随机变量。

**例题 0.18**
(1) $P(X=2|X_{1}=6) = 0$
(2) $P(X_{1}=2|X_{2}=6) = 1/36$
(3) $P(X_{1} = 1|X = 2) = 1$，因为骰子的点数最小就是 $1$

**例题 0.19**
按照错误模型为 $Y = (X+Z) \text{ mod } 3$ 处理。

| 乙接收<br>甲发送                     | 0                                                             | 1                                                             | 2                                                             |
| ------------------------------ | ------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------- |
| $\displaystyle 0, \frac{1}{2}$ | $\displaystyle \frac{1}{2} \cdot \frac{7}{8} = \frac{28}{64}$ | $\displaystyle \frac{1}{2} \cdot \frac{1}{16} = \frac{2}{64}$ | $\displaystyle \frac{1}{2} \cdot \frac{1}{16} = \frac{2}{64}$ |
| $\displaystyle1, \frac{1}{4}$  | $\displaystyle \frac{1}{4} \cdot \frac{1}{16} = \frac{1}{64}$ | $\displaystyle \frac{1}{4} \cdot \frac{7}{8} = \frac{14}{64}$ | $\displaystyle \frac{1}{4} \cdot \frac{1}{16} = \frac{1}{64}$ |
| $\displaystyle2, \frac{1}{4}$  | $\displaystyle \frac{1}{4} \cdot \frac{1}{16} = \frac{1}{64}$ | $\displaystyle \frac{1}{4} \cdot \frac{1}{16} = \frac{1}{64}$ | $\displaystyle \frac{1}{4} \cdot \frac{7}{8} = \frac{14}{64}$ |
* 假如乙收到的字符是 $0$，那么有 $\displaystyle \frac{14}{15}$ 的概率甲发的是 $0$，其他情况为 $\displaystyle \frac{1}{30}$
* 假如乙收到的字符是 $1$，有 $\displaystyle \frac{14}{17}$ 的概率甲发的是 $1$，有 $\displaystyle \frac{2}{17}$ 的概率是 $0$，$\displaystyle \frac{1}{17}$ 的概率是 $2$
* 假如乙收到的字符是 $2$，有 $\displaystyle \frac{14}{17}$ 的概率甲发的是 $2$，有 $\displaystyle \frac{2}{17}$ 的概率是 $0$，$\displaystyle \frac{1}{17}$ 的概率是 $1$

**例题 0.21**
有卷积公式：
$$
p_{Z}(x) = \int_{-\infty}^{\infty} p_{X}(y)p_{Y}(x - y) \, \mathrm{d}y 
$$

**例题 0.22**
(1)
$$
\begin{align}
F_{\max}(k) &= P(\max\{ X_{1}, \dots, X_{n} \} \leqslant k)\\
&= P(X_{1} \geqslant k, \dots, X_{n} \geqslant k)\\
&= \prod\limits_{i=1}^{n} P(X_{i} \geqslant k)\\&=
\prod\limits_{i=1}^{n} F_{X_{i}}(x_{i} \leqslant k)  
\end{align}
$$
(2)
$$
\begin{align}
F_{\min}(k) &= P(\min\{ X_{1}, \dots, X_{n} \} \leqslant k)\\
&= 1- P(\min\{ X_{1}, \dots, X_{n} \} \geqslant k)\\
&= 1 - P(X_{1} \geqslant k, \dots, X_{n} \geqslant k)\\
&= 1 - \prod\limits_{i=1}^{n} P(X_{i} \geqslant k)\\
&= 1- \prod\limits_{i=1}^{n} (1- F_{X_{i}}(k)) 
\end{align}
$$
(3)
$$
Z = \max\{ X_{i} \} - \min\{ X_{i} \}
$$
所以我们首先假设极差是 $k$，最小值是 $m$，最大值是 $m+k$ 。则所有其他的随机变量的值一定位于 $[m, m+k]$ 之间，由于最大值和最小值可以出现在任意两个位置，所以有
$$
\begin{align}
F_{Z}(k) 
&= \int_{-\infty}^{k}  \int_{-\infty}^{\infty} \binom{n}{2} \cdot 2! \cdot \big[F_{X}(m+t) - F_{X}(m)\big]^{n-2} f_{X}(m)f_{X}(m+t)\, \mathrm{d}m \, \mathrm{d}t \\
&= \int_{-\infty}^{k}  \int_{-\infty}^{\infty} n(n-1) \big[F_{X}(m+t) - F_{X}(m)\big]^{n-2} f_{X}(m)\, \mathrm{d}m \, \mathrm{d}(F_{X}(m+t) - F_{X}(m)) \\
&= \int_{-\infty}^{\infty}f_{X}(m)\int_{-\infty}^{k} n(n-1) \cdot \big[F_{X}(m+t) - F_{X}(m)\big]^{n-2}  \, \mathrm{d}(F_{X}(m+t) - F_{X}(m))\, \mathrm{d}m \\
&= \int_{-\infty}^{\infty}nf_{X}(m)\int_{-\infty}^{k} (n-1) \cdot y^{n-2}  \, \mathrm{d}y \, \mathrm{d}m \\
&= \int_{-\infty}^{\infty}nf_{X}(m) \cdot \big[F_{X}(m+t) - F_{X}(m)\big]^{n-1} \big|^{t=k}_{t=-\infty} \cdot \, \mathrm{d}m \\
&= \int_{-\infty}^{\infty}nf_{X}(m)  \big[F_{X}(m+k) - F_{X}(m)\big]^{n-1}  \, \mathrm{d}m \\
\end{align}
$$

**例题 0.30**
记第 $i$ 个人的随机变量为 $X_{i}$，拿到正确礼物的值为 $1$，拿到错误礼物的值为 $0$。因此
$$
\mathbb{E}[拿到正确礼物的人数] = \sum\limits_{i} \mathbb{E}[X_{i}] = 1.
$$

