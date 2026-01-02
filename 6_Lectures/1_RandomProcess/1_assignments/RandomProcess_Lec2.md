> 何瑞杰 
> 25110801

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

# §0.6 课后作业

1. 从 $[1, 1\,000\,000]$ 范围中随机抽取一个数。请运用容斥定理计算这个数能被 4、6 和 9 中一个或多个整除的概率。

记 $A_{i, j, k}$ 为该数字可以被 $i$ 或 $j$ 或 $k$ 整除的事件。于是

$$
\begin{align}
P(A_{4,6,9}) &= P(A_{4}) + P(A_{6}) + P(A_{9}) - P(A_{12}) - P(A_{18}) - P(A_{36}) + P(A_{36})\\
&= \frac{1000000 \left( \frac{1}{4} + \frac{1}{6} + \frac{1}{9} - \frac{1}{24} - \frac{1}{54} - \frac{1}{36} + \frac{1}{36} \right)}{1000000}\\
&= \frac{250000 + 166666 + 111111 - 83333 - 27777 - 55555 + 27777}{1000000}\\
&= \frac{388889}{1000000}
\end{align}
$$

2. 有一枚均匀的硬币和一枚两面都是头像（正面）的硬币，以相同概率从这两枚硬币中随机选择一枚并投掷。已知投出结果是出现正面，那么投掷的是两面均是头像的硬币的概率是多少？

![[Pasted image 20250924114559.png|400]]

可见
$$
\begin{align}
P(头像硬币|正面) &= \frac{P(头像硬币, 正面)}{P(头像硬币|正面) + P(正常硬币|正面)} = \frac{2}{3}
\end{align}
$$

3. 连续地抛掷一枚均匀的硬币。
    - (1) 求抛掷的前四次是下列情况的概率：
        - H, H, H, H。
        - T, H, H, H。
    - ==(2) 求模式 T、H、H、H 出现在模式 H、H、H、H 之前的概率。==

（1）两者都是 $\displaystyle \frac{1}{16}$
（2）两个模式之间有相互重叠的部分。

注意抛掷序列中的最后若干次结果，有下面这些状态：
$$
\varepsilon, T, H, TH, HH, T H H, H H H, TH H H, H H H H
$$
定义 $P(X)$ 为从 $X$ 状态起，$THH H$ 先于 $H H H H$ 出现的概率。因此有
$$
\begin{align}
P(\varepsilon) &= \frac{1}{2}P(T) + \frac{1}{2}P(H)\\
P(T) &= \frac{1}{2}P(TH) + \frac{1}{2}P(T) & P(T) = P(TH)\\
P(TH) &= \frac{1}{2}P(THH) + \frac{1}{2}P(T) & P(TH) = P(T H H)\\
P(TH H) &= \frac{1}{2} P(T H H H) + \frac{1}{2} P(T) = \frac{1}{2} + \frac{1}{2}P(T) & P(T H H) = P(T H H H) = 1\\
P(H) &= \frac{1}{2}P(H H) + \frac{1}{2}P(T) \\
P(H H) &= \frac{1}{2}P(H H H) + \frac{1}{2}P(T) \\
P(H H H) &= \frac{1}{2}P(H H H H) + \frac{1}{2}P(T) = \frac{1}{2}P(T)\\
\end{align}
$$
因此
$$
\begin{align}
P(H) &= \frac{1}{2}P(H H) + \frac{1}{2}\\
&= \frac{1}{4} P(H H H) + \frac{1}{4} + \frac{1}{2}\\
&= \frac{1}{8} + \frac{1}{4} + \frac{1}{2} = \frac{7}{8}
\end{align}
$$
所以有
$$
P(\varepsilon) = \frac{1}{2}P(T) + \frac{1}{2}P(H) = \frac{15}{16}
$$
因此模式 $T H H H$ 比模式 $H H H H$ 先出现的概率为 $\displaystyle \frac{15}{16}$。

4. 甲乙两人比赛，规定只要中间有一人赢了 $n$ 局，比赛立即结束。假定比赛在两人间公平进行，即每人赢得一局比赛的概率都是 $1/2$，与其他不同局的结果无关。那么比赛结束时，失败一方已经赢得 $k$ 局的概率是多少？

比赛结束时的模式一定是 $XXXXXXXXA$，其中 $A$ 为某一方，且该序列中 $A$ 出现了 $n$ 次，$B$ 出现了 $k < n$ 次。于是
$$
P(失败方赢得 k 局) = 2 \cdot \left( \frac{1}{2} \right)^{k+n} \cdot \binom{k+n-1}{k} 
$$

5. ==投掷 10 枚标准的六面体骰子，假定投掷每枚骰子是独立的。它们的点数之和能被 6 整除的概率是多少？==

归纳法。首先一个骰子掷出的点数能被 $6$ 整除的概率为 $\displaystyle \frac{1}{6}$。可以验证两枚骰子掷出点数之和能被 $6$ 整除的概率也是 $\displaystyle \frac{1}{6}$，事实上，两枚骰子掷出点数之和除以 $6$ 模 $n$ （$n \in \{ 0, \dots, 5 \}$ ）的概率都是 $\displaystyle \frac{1}{6}$。现在假设 $n$ 枚不相关骰子掷出点数模 $n$ 的概率都是 $\displaystyle \frac{1}{6}$，需要证明 $n+1$ 枚骰子也满足这个规律。事实上，前 $n$ 枚骰子点数之和模 $6$ 的行为相当于一枚骰子掷出点数模 $6$ 的结果。这相当于两枚骰子点数之和模 $6$ 的分布，也就是均匀分布。所以十枚骰子掷出结果被 $6$ 整除的概率是 $\displaystyle \frac{1}{6}$.

6. 对 $n$ 个人进行核酸检测，每个人可以单独检测，但费用过高。合并检测可以减少费用。把 $k$ 个人的样本合起来同时分析，如果检测结果呈阴性，对这 $k$ 个人的组，这一次检测就好了。如果检测结果呈阳性，则这 $k$ 个人需要再进行单独检测，因此这 $k$ 个人需要进行 $k+1$ 次检测。假定我们产生了 $n/k$ 个不同的组，每组 $k$ 个人 ($k$ 能整除 $n$)，并用合并法进行检测。假设对于独立检测，每个人呈阳性的概率为 $p$。
    - (1) 对 $k$ 个人的合并样本，检测呈阳性的概率是多少？
    - (2) 需要检测的期望次数是多少？
    - (3) 描述如何求最优的 $k$ 值。
    - (4) 给出一个不等式，说明对什么样的 $p$ 值，合并检测比每个人单独检测更好。

（1）$P(+) = (1-p)^{k}$
（2）$\mathbb{E}[N] = \frac{n}{k} [1 \cdot P(-) + (k+1) \cdot P(+)] = n[(1-p)^{k} + (k+1)(1-(1-p)^{k})]$
（3）记 $q = 1-p$，有单组期望检测数 $\mathbb{E}[N_{0}] = q^{k} + (k+1)(1-q^{k}) = (k+1) - kq^{k} = f(k)$。全组有 $\mathbb{E}[N] = \frac{n}{k}\mathbb{E}[N_{0}] = \displaystyle n[1 + \frac{1}{k} - q^{k}] = g(k)$，求导有 $g'(k) = \displaystyle n\left[ - \frac{1}{k^{2}} - \log k \cdot q^{k} \right]$，然后求 $g$ 极小值即可（如果极小值存在）
（4）合并检测比单人检测好时，需要 $\displaystyle q^{k} > \frac{1}{k}$，即 $k \log q > - \log k$，即 $q > \displaystyle \frac{1}{\sqrt[k]{ k }}$.

7. 设 $A, B$ 是两个事件。证明如下示性变量之间的关系并回答有关问题。
    - (1) $\mathbb{I}_\Omega = 1, \mathbb{I}_\varnothing = 0$；
    - (2) $\mathbb{I}_{A} = 1 - \mathbb{I}_{A^c}$；
    - (3) $\mathbb{I}_{A \cup B} = \max(\mathbb{I}_A, \mathbb{I}_B)$，$\mathbb{I}_{A \cap B} = \min(\mathbb{I}_A, \mathbb{I}_B)$；
    - (4) $\mathbb{I}_A + \mathbb{I}_B \mod 2$ 对应的事件是什么？$\mathbb{I}_A \mathbb{I}_B$ 呢？

（1）和（2）属于同一种。现讨论（2）。
$$
\mathbb{I}_{A^{c}} =\begin{cases}
1 & \omega \in A^{c}\\
0 & \omega \in A
\end{cases} = 1 - \begin{cases}
1 & \omega \in A\\
0 & \omega \in A^{c}
\end{cases} = 1 - \mathbb{I}_{A}
$$
（3）
$$
\mathbb{I}_{A \cup B} = \begin{cases}
1 & \omega \in A \text{ or } \omega \in B\\
0 & \text{otherwise}
\end{cases}
$$
$$
\max \{ \mathbb{I}_{A}, \mathbb{I}_{B} \} = \begin{cases}
1 & \omega \in A \text{ or } \omega \in B\\
0 & \text{otherwise}
\end{cases}
$$
$\mathbb{I}_{A \cap B}$ 同理。

（4）

$$
\begin{align}
\mathbb{I}_A + \mathbb{I}_B \mod 2 = \begin{cases}
1 & \omega \in A - B \text{ or } \omega \in B - A\\
0 & \omega \in (A \cup B)^{c} \text{ or } \omega \in A \cap B
\end{cases} = \mathbb{I}_{A \Delta B}
\end{align}
$$
其中 $\Delta$ 表示对称差。由于示性函数的值只取零或一，且相乘均为 $1$ 结果才是 $1$，因此 $\mathbb{I}_{A}\mathbb{I}_{B} = \mathbb{I}_{A \cap B}$

8. 在《万里归途》电影中有一个情景，穆夫塔刁难宗大伟，发起“轮盘赌”。现设枪中子弹数服从概率为 $1/2$ 的 0-1 两点分布。请细化概率模型，分析随着“轮盘赌”进行，枪中无子弹的概率是如何变化的？具体地，无子弹的初始概率为 $1/2$。选择至少两个概率模型（必要时可以改变两轮之间的规则），分析第 $i \geq 1$ 枪后无子弹的概率。

如果规则为每开一枪后按相同方向拨动一次轮盘（转动一格）那么当第 $6$ 枪之后枪内没有子弹。当 $0 \leqslant i < 6$ 时，枪内有子弹的概率是 $1 - 0.5^{6-i}$。

开一枪后拨动五格的情况和第一种相同。固定拨动其他数量格数需要讨论 $nk \text{ mod }6$ 的具体行为。