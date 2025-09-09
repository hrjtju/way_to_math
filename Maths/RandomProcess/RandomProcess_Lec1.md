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
* CHebyshev 不等式
* Chernov 界