## 补遗

> **定理 （扩展 Chebyshev 不等式）**
> 考虑随机变量 $X: \Omega \rightarrow \mathbb{R}$，和连续增函数 $g: \mathbb{R}_{\geqslant 0} \rightarrow \mathbb{R}_{\geqslant 0}$，如果 $g(a) > 0$ 且 $\mathbb{E}(g(|X|))$ 存在，有 
> $$\mathbb{E}(|X|\geqslant a) \leqslant \frac{\mathbb{E}(g(|X|))}{g(a)}$$

用图示法证明。考虑集合 $E = \{ X: |X| \geqslant a \}$，令 $Z = \chi_{E}$，则有
$$
Z \leqslant \frac{g(|X|)}{g(a)}
$$
两边取期望，有
$$
\mathbb{E}(Z) = P(|X| \geqslant a) \leqslant \frac{\mathbb{E}[g(|X|)]}{g(a)}
$$

* 取 $X = Y - \mathbb{E}[Y]$，$g(x) = x^{2}$，得到 Chebyshev 不等式： $\displaystyle \mathbb{E}(|Y-\mathbb{E}[Y]|\geqslant a) \leqslant \frac{\text{Var}(Y)}{a^{2}}$
* 取 $g(x) = x$，得到 Markov 不等式： $\displaystyle \mathbb{E}(|X| \geqslant a) \leqslant \frac{\mathbb{E}(|X|)}{a}$
* 取 $g(x) = e^{\lambda x}$，其中 $\lambda \geqslant 0$，得到 Chernoff 界： $\displaystyle \mathbb{E}(|X| \geqslant a) \leqslant \frac{\mathbb{E}(\exp\{ \lambda|X| \})}{\exp\{ a\lambda  \}}$
* 取 $g(x) = x^{p}$，其中 $p \geqslant 1$，得到下面另一形式的 Chebyshev 不等式： $\displaystyle \mathbb{E}(|X| \geqslant \lambda) \leqslant \frac{\mathbb{E}(|X|^{p})}{\lambda^{p}}.$

---
## 0.5 典型概率模型
### 0.5.1 采样模型（盒中有 $b$ 个黑球，$w$ 个白球）
**例题 0.31** 记 $B_{k}$ 为第 $k$ 次抽到黑球，$W_{k}$ 是第 $k$ 次抽到白球
1. $P(B_{2}|W_{1})$
    1. 有放回采样：$\displaystyle P(B_{2}|W_{1}) = P(B_{2}) = \frac{b}{b + w}$ 
    2. 无放回采样：$\displaystyle P(B_{2}|W_{1}) = \frac{b}{b+w-1}$
    3. Polya 采样：$\displaystyle P(B_{2}|W_{1}) = \frac{b}{b+w+r-1}$
2. $P(B_{1}B_{2})$
    1. 有放回采样：$\displaystyle P(B_{2}B_{1}) = P(B_{2})P(B_{1}) = \left( \frac{b}{b + w} \right)^{2}$
    2. 无放回采样：$\displaystyle P(B_{2}B_{1}) = P(B_{2}|B_{1})P(B_{1}) = \frac{b-1}{b+w-1} \cdot \frac{b}{b+w}$
    3. Polya 采样：$\displaystyle P(B_{2}B_{1}) = P(B_{2}|B_{1})P(B_{1}) = \frac{b+r-1}{b+w+r-1} \cdot \frac{b}{b+w}$
3. 定义随机变量 $X_{t}$，若第 $t$ 次取到黑球，令 $X_{t} = 1$，否则 $X_{t} = 0$。

### 0.5.2 球箱模型（均匀投 $m$ 个球到 $n$ 个箱子）
> **例题 0.32** 
> 生日悖论。将 $n$ 个球放入 $M$ 个箱子，求有一个箱子内球数超过 $1$ 的概率（事件 $A$）

考虑 $\bar{A}$：即所有箱子的球数都不超过 $1$。
当 $n \geqslant M$ 时，那么 $P(A) = 1$；
当 $n < M$ 时，有 $\displaystyle P(A) = 1-P(\bar{A}) = 1 - \binom{M}{n} \Big/ M^{n}$

> **例题 0.33** 
> 集卡问题。有 $n$ 类不同的优惠券，买一件商品可以获取其中某类的 $1$ 张。平均购买多少商品才能集齐？

购买商品收集到第  类优惠券定义的随机变量为 $X_{k}$，收集到位 $1$，反之为 $0$。考虑平均收集到第 $k$ 类优惠券所需购买的商品件数 $n_{k} = \displaystyle \frac{1}{\mathbb{E}[X_{k}]} = n$。不难发现 $X_{k}$ 是相互独立的。因此总共需要 $N = n_{1} + n_{2} + \cdots + n_{n} = n^{2}$。

> **例题 0.34** 
> 最大负载问题。若 $m = n$，则最多球的箱子里面有多少球？

记球最多的箱子中有 $k$ 个球。其概率为 
$$P(n_{\max} = k) = \binom{n-1}{n-k} \Big / {n^{n}} = \frac{(n-1)\cdots(n-k+1)}{n^{n}}$$

### 0.5.3 独立重复投掷硬币（正面朝上概率为 $p$ ）

> **例题 0.35**
> 使用自然语言描述：两点分布、二项分布、几何分布。给出概率质量函数、数学期望和矩生成函数。

* 两点分布：掷一次硬币的结果
概率质量函数：$\displaystyle p(X=k) = \begin{cases} p, & k=1\\1-p, & k=0 \end{cases}$
期望：$\mathbb{E}[X] = p$
矩生成函数：$M_{X}(t) = \mathbb{E}[e^{tX}] = e^{t} \cdot p + e^{0} \cdot (1-p) = pe^{t} - p  + 1$

* 二项分布：连续掷 $n$ 次硬币，统计正面朝上的次数
概率质量函数：$\displaystyle p(X = k) = \binom{n}{k}p^{k}(1-p)^{n-k}$
期望：
$$
\begin{align}
\mathbb{E}[X] 
&= \sum\limits_{k=0}^{n} k \binom{n}{k} p^{k}(1-p)^{n-k} = \sum\limits_{k=1}^{n} k \binom{n}{k} p^{k}(1-p)^{n-k}\\
&= np\underbrace{ \sum\limits_{k=1}^{n} \binom{n-1}{k-1} p^{k-1}(1-p)^{(n-1)-(k-1)} }_{ =\,1 } = np
\end{align}
$$
矩生成函数：
$$
\begin{align}
M_X(t) 
&= \mathbb{E}[e^{tX}] = \sum\limits_{k=0}^{n} e^{tk} \binom{n}{k} p^{k}(1-p)^{n-k}\\
&= \sum\limits_{k=0}^{\infty} \binom{n}{k} (pe^{t})^{k}(1-p)^{n-k} = (pe^{t} + 1-p)^{n}
\end{align}
$$

* 几何分布：连续掷同一枚硬币，直到出现正面为止，投掷的次数
概率质量函数：$p(X=k) = (1-p)^{k-1}p$
期望：
$$
\begin{align}
\mathbb{E}[X] &= \sum\limits_{k=1}^{\infty} k (1-p)^{k-1} p 
= p \sum\limits_{k=1}^{\infty} k (1-p)^{k-1}\\
&= - p \sum\limits_{k=1}^{\infty}  \frac{\mathrm{d}}{\mathrm{d} p}  (1-p)^{k} = -p \frac{\mathrm{d}}{\mathrm{d}p} (1-p) \sum\limits_{k=0}^{\infty} (1-p)^{k}\\
&= -p \frac{\mathrm{d}}{\mathrm{d}p} \frac{1-p}{p} = \frac{1}{p}.\\
\end{align}
$$
矩生成函数：
$$
\begin{align}
M_{X}(t) 
&= \mathbb{E}[e^{tX}]\\
&= \sum\limits_{k=1}^{\infty} e^{tk} (1-p)^{k-1} p\\
&= e^{t} \cdot \sum\limits_{k=1}^{\infty} e^{t(k-1)} (1-p)^{k-1} p\\
&= e^{t} \left[ p + (1-p)\sum\limits_{k=1}^{\infty} e^{tk} (1-p)^{k-1} p \right] = e^{t}[p + (1-p)M_{X}(t)]
\end{align}
$$
于是
$$
M_{X}(t) = \frac{pe^{t}}{1 - e^{t}(1-p)}
$$

> **例题 0.36**
> 观察到 $n$ 次正面朝上时，一共掷了多少次？

考虑随机变量 $X$ 为 $m$ 次投掷中刚好 $n$ 次正面朝上的次数。则有
$$
p(X = n) = \binom{m-1}{n-1} p^{n}(1-p)^{m-n} 
$$

### 0.5.4 概率树、概率转移图、Bayes 推断

> **例题 0.37**
> 一家有两个孩子，设男孩女孩机会均等，问
> 1. 已知老大是女孩，求老二也是女孩的概率
> 2. 已知有一个是女孩，另一个也是女孩的概率

1. 两者独立，$\displaystyle P = \frac{1}{2}$
2. 老二是女孩：（男，女）、（女、女），两者概率分别为 $\displaystyle \frac{1}{4}$；老大是女孩：（女，男）概率是 $\displaystyle \frac{1}{4}$。因此 $$P(另一个是女孩|有一个是女孩) = \frac{1}{3}$$
> **例题 0.38**
> 有三个文件柜，一个文件有可能放在某个文件柜里。设快速翻阅一个会有该文件的文件柜，发 现该文件的概率为 $α$。现翻阅文件柜 $A$ 后，未发现。问文件在 $A$ 中的概率。

设 $R$ 为文件存在的位置的随机变量，$O_{X}$ 为在第 $X$ 个文件柜中发现文件的随机变量。
$$
\begin{align}
&\, P(R = A|O_{A} = 0)\\
=&\, \frac{P(R = A, O_{A} = 0)}{\displaystyle  \sum\limits_{X = \{ A, B, C \}} P(O_{A} = 0|R =X)P(R=X) }\\
=&\, \frac{P(R = A, O_{A} = 0)}{P(R = A, O_{A} = 0) + P(R = B, O_{A} = 0) + P(R = C, O_{A} = 0)}\\
=&\, \frac{1-\alpha}{1-\alpha + 1 + 1} = \frac{1-\alpha}{3-\alpha}
\end{align}
$$

> **例题 0.39**
> 单选题设置几个选项合适？



> **例题 0.40**
> 试着解释临床检查中的“假阳性与假阴性”。一个阳性者真有病的概率受哪些因素影响?



## 0.6 课后作业
