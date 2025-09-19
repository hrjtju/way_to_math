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



## 0.6 课后作业
