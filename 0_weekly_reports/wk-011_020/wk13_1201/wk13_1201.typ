#import "@preview/algorithmic:1.0.5"
#import algorithmic: (
  style-algorithm, 
  algorithm-figure
)
#import "@preview/ctheorems:1.1.3": *
#import "@preview/mitex:0.2.4": *
#import "@preview/numbly:0.1.0": numbly

#show strong: set text(blue)
#show: thmrules.with(qed-symbol: $square$)

#set text(lang: "zh", font: ("New Computer Modern", "Kai", "KaiTi"))

// Snippets
#let const = "constant"
#let bs = $bold(s)$
#let bf = $bold(f)$
#let bF = $bold(F)$
#let bg = $bold(g)$
#let bG = $bold(G)$
#let bx = $bold(x)$
#let bX = $bold(X)$
#let bw = $bold(w)$
#let bW = $bold(W)$
#let by = $bold(y)$
#let bY = $bold(Y)$
#let bz = $bold(z)$
#let bZ = $bold(Z)$
#let mtxId = $bold(I)$
#let vec0 = $bold(0)$
// #let fg = $frak(g)$
// #let fh = $frak(h)$
// #let fu = $frak(u)$
// #let fp = $frak(p)$
#let mx = $macron(bx)$
#let pd = $p_("data")$
#let ps = $p_(sigma)$
#let qs = $q_(sigma)$
#let pt = $p_(theta)$
#let dd = "d"
#let be = $bold(epsilon)$
#let prod = $product$
#let int = $integral$

#let leq = $lt.slant$
#let geq = $gt.slant$
#let tensor = $times.o$

#let cdots = $dots.c$

#let KL = $D_("KL")$
#let argmin = $op("arg min", limits: #true)$
#let argmax = $op("arg max", limits: #true)$

#let normal = $cal(N)$
#let prior = $p_"prior"$
#let data = $p_"data"$
#let score = $s_theta$

#let ito = $"It"hat("o")$
#let schrodinger = "Schrödinger"

// Theorem environments
#let theorem = thmbox("theorem", "定理", fill: rgb("#eeffee"), base_level: 1)
#let proposition = thmbox("proposition", "命题", fill: rgb("#e9f6ff"), base_level: 1)
#let corollary = thmplain(
  "corollary",
  "推论",
  base: "theorem",
  titlefmt: strong
)
#let definition = thmbox("definition", "定义", inset: (x: 1.2em, top: 1em), base_level: 1)
#let example = thmplain("example", "示例").with(numbering: none)
#let proof = thmproof("proof", "证明")
#let redText(t) = {text(red)[$#t$]}
#let blueText(t) = {text(blue)[$#t$]}
#let greenText(t) = {text(green)[$#t$]}
#let orangeText(t) = {text(orange)[$#t$]}
#let tab = {h(2em)}

#show strong: set text(blue)
#show figure.caption: emph
#show: thmrules.with(qed-symbol: $square$)
#show link: underline

#set par(
  first-line-indent: 2em,
  justify: true,
)

#show figure.where(
  kind: table
): set figure.caption(position: top)

#set underline(offset: 2.5pt, stroke: 0.5pt)

#show figure.caption: it => [
  #underline[
    #it.supplement #context it.counter.display(it.numbering)
  ]
  #h(5pt)#it.body
]

#set math.equation(numbering: "(1)")
#set math.mat(delim: ("[", "]"), align: center)
#set heading(numbering: "1.")
#set math.cases(gap: 0.5em)

#let pfw(n, k) = {[$p^(#n)_(#k+1|#k) (x_(#k+1)|x_#k)$]}
#let pbk(n, k) = {[$p^(#n)_(#k|#k+1) (x_#k|x_(#k+1))$]}
#let qfw(n, k) = {[$q^(#n)_(#k+1|#k) (x_(#k+1)|x_#k)$]}
#let qbk(n, k) = {[$q^(#n)_(#k|#k+1) (x_#k|x_(#k+1))$]}

// metadata
#let wk_report_name = "2025年12月1日至12月7日周报"
#let name_affiliation = "何瑞杰 | 中山大学 & 大湾区大学"

#set page(
  paper: "a4",
  numbering: "1",
  header: wk_report_name + " | " + name_affiliation,
)

#align(
  center, 
  text(17pt)[#wk_report_name\ ] 
        + text(12pt)[\ 何瑞杰\ 中山大学, 大湾区大学]
)

#outline(depth: 2)

#linebreak()
#grid(columns: (100%), align: center, text(size: 12pt)[速 览])

#tab 本周大部分时间花在了阅读 #schrodinger 桥论文#cite(<debortoli2023diffusionschrodingerbridgeapplications>)，它给出了一个以 #schrodinger 桥问题观察生成模型的视角。一般的薛定谔桥问题（或静态薛定谔桥问题）没有闭式解，作者转而通过修改 IPF 算法使之适配分数匹配生成模型的框架，并采用迭代的方式学习薛定谔桥。

在生命游戏项目方面，本周实现了一个极小的朴素规则提取函数，在 `B3678/S34678` 数据集上可以成功提取规则。

最后本周学习了 Stein 实分析中开篇的少量内容。

#pagebreak()

= 项目进展
== 使用神经网络学习生命游戏的演化动力学

#tab 本周对训练网络在全部数据集上的统计结果套用上简单的要求从而获取提取的规则序列。具体的说，提取统计中转换频数（如统计得到的 $5$ 个存活邻居下从死亡转换为存活的案例数）超过全体转换频数乘以预测错误率的百分之六十作为“入围”标准，然后再在这个入围标准下分别筛选当前时刻下死亡或存活，并在下一时刻预测存活的频数超过预测为死亡频数的 $10$ 倍的规则作为最终得到的规则。对应的代码片段如下

```python
# class RuleSimulatorStats
def infer_rule_str(self, counters, acc) -> Tuple[List, List]:
    list_str = lambda x:list(map(lambda k:str(int(k)), x))

    dd, dl = sum(counters[0].values()), sum(counters[1].values())
    ld, ll = sum(counters[2].values()), sum(counters[3].values())

    d_th = (1-acc)*(dd+dl)
    l_th = (1-acc)*(ld+ll)

    d_all = counters[0] + counters[1]
    l_all = counters[2] + counters[3]

    filtered_b = sorted(list(filter(lambda x:x[1]>0.6*d_th, d_all.items())), 
                        key=lambda x:x[0])
    filtered_s = sorted(list(filter(lambda x:x[1]>0.6*l_th, l_all.items())), 
                        key=lambda x:x[0])

    born = []
    survive = []

    for i,_ in filtered_b:
        if counters[1][i] > 10 * counters[0][i]:
            born.append(i)
    for i,_ in filtered_s:
        if counters[3][i] > 10 * counters[2][i]:
            survive.append(i)
    return list_str(born), list_str(survive)
```

这样朴素的规则提取函数在 `B3678/S34678` 可以成功推出真实的规则。

#pagebreak()
= 文献阅读

// == Score-based Generative modeling through SDE 补遗

// *#link("http://arxiv.org/abs/2011.13456")[ICLR 2021] | Yang Song et al. *

// === 概率流 ODE 和 Fokker-Plank 方程



// #pagebreak()


== Diffusion Schrödinger Bridge with Application to Score-Based Generative Modeling #cite(<debortoli2023diffusionschrodingerbridgeapplications>)

*#link("http://arxiv.org/abs/2106.01357")[NIPS 2021] | Valentin De Bortoli et al.*

本文以 #schrodinger 桥 (SB) 的视角构建了一个新的生成模型。本质上是建立从人为规定的先验分布 $p_"prior"$ 到未知数据分布 $p_"data"$ 的最优传输。由于一般形式的 SB 难以求解，本文作者于是退而求其次，考虑迭代求解的 IPF 算法，而最终又将它转化为分数匹配模型可以使用的范畴。除此之外，本文的大部分篇幅和附录给出了有关分数匹配模型的拟合能力以及 SB 和离散情况 IPF 算法的收敛性证明，为模型提供理论保证。除此之外，本文指出该方法相比于扩散模型需要更少的步数。

=== 记号

#align(center,
  table(
    columns: 3,
    align: (center, center, left),
    stroke: none,
    table.hline(),
    table.header([记号], [表达式], [意义]),
    table.hline(stroke: 0.5pt),
    [$cal(C)$], [$scr(C)([0, T], RR^d)$], [从 $[0, T]$ 到 $RR^d$ 的连续函数全体],
    [$cal(B)(cal(C))$], [$-$], [$cal(C)$ 上的所有 Borel 集],
    [$scr(P)(mono(E))$], [$-$], [可测空间 $mono(E)$ 上的概率测度的全体],
    [$scr(P)_ell$], [$scr(P)((RR^d)^ell)$], [---#v(1em)],
    [$H(p)$], [$display(-int_(RR^d)) p(x) log p(x) dd x$], [$p$ 的熵#v(1.5em)],
    [$KL(p|q)$], [$display(int_(RR^d)) p(x) log display(p(x)/q(x)) dd x$], [$p$ 和 $q$ 的 Kullback-Leiber (KL) 散度#v(1.5em)],
    [$J(mu, nu)$], [$KL(mu|nu) + KL(nu, mu)$], [概率测度 $mu$ 和 $nu$ 的 Jeffrey 散度],
    [$norm(dot)_"TV"$], [$  $], [全变差范数],
    [$bb(P)^R$], [$  $], [路径测度 $bb(P)$ 的时间逆],
    table.hline(),
  )
)


=== 去噪扩散模型、分数匹配和逆时 SDE 回顾
==== 离散情形

考虑从数据分布 $p_"data"$ 出发的 Markov 链，它的前向转移概率是 $p_(k+1|k), k in {0, 1, ..., N-1}$。于是 ${x_k}_(k=1)^N$ 的联合分布可以写为 

$
p(x_(0:N)) = p_0 (x_0) prod_(k=0)^(N-1)) p_(k+1|k)(x_(k+1)|x_k)
$

同样，我们也可以将其表示为反向转移概率是乘积：

$
display(p(x_(0:N)) = p_N (x_N) prod_(k=0)^(N-1)) p_(k|k+1)(x_(k)|x_(k+1))
$<reversed-time-mc>

我们希望 $p_N$ 逼近先验分布 $p_"prior"$，从而可以从先验分布出发，沿着#ref(<reversed-time-mc>)演化得到未知数据分布 $p_"data"$，这就是 DDPM 中提出的*祖先采样 (Ancestral Sampling)*。一般地，我们假定该 Markov 链的正向转移概率为 $p_(k+1|k)(x_(k+1)|x_k) = cal(N)(x_(k+1); x_k + gamma_(k+1)f(x_k), 2gamma_(k+1)mtxId)$，要得到逆向转移概率 $p_(k|k+1)$，根据 Bayes 公式和一些其他技巧，可以得到
$
  & p_(k|k+1)(x_k|x_(k+1)) \ 
  &= (p_(k+1|k)(x_(k+1)|x_k)p_(k)(x_(k))) / (p_(k+1)(x_(k+1)))   
  = p_(k+1|k)(x_(k+1)|x_k) dot exp lr([log p_(k)(x_(k)) - log p_(k+1)(x_(k+1))])\  
  &= p_(k+1|k)(x_(k+1)|x_k) dot exp lr([log p_(redText(k+1))(x_(k)) - log p_(k+1)(x_(k+1))]) #tab p_k approx p_(k+1)\
  &= p_(k+1|k)(x_(k+1)|x_k) dot exp lr(chevron.l nabla log p_(k+1) (x_(k+1)), x_k - x_(k+1) chevron.r) tab "在 " x_(k+1) "处 Taylor 展开到一阶"\
  &= C exp lr({- (norm(x_(k+1) - x_k - gamma_(k+1) f(x_k))^2 + 2 chevron.l redText(2 gamma_(k+1) nabla log p_(k+1) (x_(k+1))), x_k - x_(k+1) chevron.r)/(2 dot 2 gamma_(k+1))})\
  &= C dot C' dot exp lr({- (norm(x_(k+1) - x_k - gamma_(k+1) f(x_k) + redText(2 gamma_(k+1) nabla log p_(k+1) (x_(k+1))))^2)/(2 dot 2 gamma_(k+1))})\  
  &approx C''exp lr({- (norm(x_(k+1) - x_k - gamma_(k+1) f(redText(x_(k+1))) + 2 gamma_(k+1) nabla log p_(k+1) (x_(k+1)))^2)/(2 dot 2 gamma_(k+1))}) #tab x_k approx x_(k+1)\
  &= cal(N) (x_k; x_(k+1) - gamma_(k+1)f(x_(k+1)) + nabla log p_(k+1)(x_(k+1)), 2 gamma_(k+1) mtxId))
$<reverse-mc-transition-prob>

由于 $nabla log p_(k+1)(x_k+1)$ 事先未知，可以用分数匹配的方法训练一个网络 $s_theta (x, k+1)$ 拟合之，具体原因如下：假设条件概率 $p_(k+1|0) (x_(k+1)|x_0)$ 已经给定，那么 $p_(k+1|k)(x_(k+1)|x_k)$ 就可以写为

$
p_(k+1)(x_(k+1)) = int p_0 (x_0) p_(k+1|0)(x_(k+1)|x_0) dd x_0
$

可以得到等式左边的对数梯度：

$
  nabla_(x_(k+1)) log p_(k+1) (x_(k+1)) &= nabla_(x_(k+1)) log int p_0 (x_0) p_(k+1|0)(x_(k+1)|x_0) dd x_0 \  
  &= display(nabla_(x_(k+1))  int p_0 (x_0) p_(k+1|0)(x_(k+1)|x_0) dd x_0)/display(p_(k+1) (x_(k+1)))\  
  &= display( int p_0 (x_0) nabla_(x_(k+1)) p_(k+1|0)(x_(k+1)|x_0) dd x_0)/display(p_(k+1) (x_(k+1)))\  
  &= display( int (p_0 (x_0) p_(k+1|0)(x_(k+1)|x_0))/display(p_(k+1) (x_(k+1))) dot nabla_(x_(k+1)) log p_(k+1|0)(x_(k+1)|x_0) dd x_0)\  
  &= int p_(0|k+1) (x_0|x_(k+1)) nabla_(x_(k+1)) log p_(k+1|0)(x_(k+1)|x_0) dd x_0) dd x_0 \  
  &= EE_(p_(0|k+1)) [nabla_(x_(k+1)) log p_(k+1|0)(x_(k+1)|x_0) ]
$

因此分数网络的学习目标为 $EE_(p_(0|k+1)) [nabla_(x_(k+1)) log p_(k+1|0)(x_(k+1)|x_0) ]$ 这样就自然地得到了下面的分数匹配的优化目标：

$
  limits("minimize")_theta sum_(k=1)^N EE_(p_(0, k)) lr([norm(
    s_theta(x_k, k) - nabla_(x_k) log p_(k|0) (x_k|x_0)
  )^2])
$

其中 $p_(0, k) = p_0 p_(k|0)$。如果无法计算 $p_(k|0)$，就将上述目标中的 $k|0$ 一项改为 $k|k-1$，和 SMLD 论文中的形式相同。最后，要从先验分布 $p_"prior"$ 生成样本，只需从中采样 $x_N ~ p_"prior"$，然后进行 Langevin 采样。

==== 连续情形

#tab 考虑将上一节中的离散情形求极限，发现离散情形为遵循下面 SDE 的随机过程 $(bX_t)_(t in [0, T])$ Euler-Maruyama 离散化形式：

$
  dd bX_t = f(bX_t) dd t + sqrt(2) dd bW, tab bX_0 ~ p_0 = p_"data".
$

其中 $bW(dot)$ 是和 $bX$ 维数相同的标准 Brown 运动，映射 $f: RR^d -> RR^d$ 为使得强解存在而恰当选取的函数。一般令其为线性函数，即 $f(bx) = -alpha bx$。在此种情况下，其逆向过程 $(bY_t)_(t in [0, T]) = (bX_(T-t))_(t in [0, T])$ 满足下面的逆向 SDE：

$
  dd bY_t = lr([-f(bY_t) + 2 nabla log p_(T-t) (bY_t)]) dd t + sqrt(2) dd bW_t
$

读者可以看见，其 Euler-Maruyama 离散化的情形正好对应#ref(<reverse-mc-transition-prob>)。值得注意的是，DDPM 和 SMLD 论文中对应的正向 SDE 都为 Ornstein-Uhlenbeck 过程。下面的定理给出了分数匹配模型应用于逆时 OU 过程预测 $X_0$ 数据分布 $cal(L)(X_0)$ 的误差界：

#theorem[
  假设存在某个 $M gt.slant 0$，是的对任意 $t in [0, T]$ 和 $x in RR^d$，都有

  $
    norm(s_theta (x, t) - nabla log p_t (x)) lt.slant M,
  $

  其中 $s_(theta^*) in scr(C) ([0, T] times RR^d, RR^d)$。并假设数据分布 $p_"data" in scr(C)^3 (RR^d, RR_(>0))$ 有界，且满足一些条件#footnote([见论文原文])时，有依照上述方法生成得到的分布 $cal(L)(X_0)$ 和数据分布 $p_"data"$ 之间的全变差范数距离被控制：

  $
    norm(cal(L)(X_0) - p_"data")_"TV" lt.slant C(alpha, M, T)
  $

  其中 $C(alpha, M ,T)$ 是一个依赖于 $alpha$，$M$ 和 $T$ 的常数。
]

换言之，该定理给出了这样的叙述：如果我们有一个不甚准确的分数模型 $s_theta$，那么我们依靠它生成得到的分布 $cal(L) (X_0)$ 相比于 $p_"data"$ 可以近到什么程度。该定理还揭示了一个权衡的问题。当 $alpha$ 比较大，即 SDE 中的偏移项较大时，需要更小的数值模拟步长，以及更小的混合总时间 $T$；$alpha$ 较小，即混合速度较慢时，需要更大的混合步长和更长的混合总时间以趋近先验分布 $p_"prior"$。

=== #schrodinger 桥

#tab 在生成的语境中，先验分布 $p_"prior"$ 和数据分布 $p_"data"$ 之间的 #schrodinger 桥指的是满足下面条件的路径测度 $pi^star in scr(P)_(N+1)$：

$
  pi^star = argmin lr({KL(pi|p): pi in scr(P)_(N+1), pi_0 = p_"data", pi_N = p_"prior"})
$

自然地，如果我们得到了 $pi^star$，自然就可以通过它构造出前文中提到的逆向 Markov 链的转移概率 $pi^star_(k|k+1)$。将视角缩小到 #schrodinger 桥的两头，能得到所谓*静态 #schrodinger 桥问题*。给定路径测度 $pi in scr(P)_(N+1)$，$pi_(|0, N)$ 指的是 $X_(1:N-1)$ 的条件分布。对于路径测度 $pi, p in scr(P)_(N+1)$，有下面的恒等关系

$
  KL(pi|p) &= int pi(x) log (pi(x))/(p(x)) dd x \ 
  &= int pi(x) log (pi_(0, N)(x_0, x_N))/(p_(0, N)(x_0, x_N)) dd x
   + int pi(x) log (pi_(|0, N)(x_(1:N)))/(p_(|0, N)(x_(1:N))) dd x \
  &= int pi_(|0, N)(x_(1:N)) blueText(int pi_(0, N)(x_0, x_N) log (pi_(0, N)(x_0, x_N))/(p_(0, N)(x_0, x_N)) dd x_(0, N)) dd x_(1:N) 
   \ & tab + int pi_(0, N)(x_0, x_N) redText(int pi_(|0, N)(x_(1:N)) log (pi_(|0, N)(x_(1:N)))/(p_(|0, N)(x_(1:N))) dd x_(1:N)) dd x_(0, N) \
  &= EE_(pi_(|0, N))underbrace([KL(pi_(0, N)|p_(0, N))], "常数") + EE_(pi_(0, N))underbrace([KL(pi_(|0, N)|p_(|0, N))], "随机变量")\
  &= KL(pi_(0, N)|p_(0, N)) + EE_(pi_(0, N))[KL(pi_(|0, N)|p_(|0, N))]
$

注意，$KL(pi|p)$ 是 #schrodinger 桥定义中的优化目标。根据上面的结果可以拆成两部分，第一部分 $KL(pi_(0, N)|p_(0, N))$ 是边缘分布的匹配项，第二部分 $EE_(pi_(0, N))[KL(pi_(|0, N)|p_(|0, N))]$ 是中间路径的匹配项。匹配两端的边缘分布的版本就是*静态 #schrodinger 桥*，下式给出了该问题的解：
$
  pi^("s", star) = argmin lr({KL(pi^"s"|p_(0, N)): pi in scr(P)_(2), pi^"s"_0 = p_"data", pi^"s"_N = p_"prior"})
$

得到该解后，完整的 #schrodinger 桥问题的解可以写成

$
  pi^star = pi^("s", star) times.o p_(|0, N)
$

其中 $p_(|0, N)$ 可以是任给的一个分布，例如可以是高斯转移核 $p_(k+1|k) = cal(N)(x_k, sigma_(k+1)^2)$，这样就得到了 #schrodinger 桥的解：若参考路径测度对应的是标准 Brown 运动，则将上述构造代入上述对 $KL(pi|p)$ 的分解中，第二项路径分布匹配项为 $0$，因此 $pi^star$ 确实是 #schrodinger 桥。

如果恰当的重写静态 #schrodinger 桥定义中的优化目标，我们可以看到它与最优传输的联系：

$
  KL(pi^"s"|p_(0, N)) &= int pi^"s" (x_(0, N)) log (pi^"s" (x_(0, N)))/(p_(0, N) (x_(0, N))) dd x_(0, N)  \
  &= int pi^"s" (x_(0, N)) log pi^"s" (x_(0, N)) dd x_(0, N) - int pi^"s" (x_(0, N)) log p_(N|0) (x_(N)|x_0) dd x_(0, N) \ 
    & tab - int pi^"s" (x_(0, N)) log p_0 (x_0) dd x_(0, N) \ 
  &= H(pi^"s") - EE_(pi^"s")[log p_(N|0) (x_N|x_0)] + const
$

因此静态 #schrodinger 桥也可以写成下面带有熵正则化的最优传输问题：

$
  pi^("s", star) = argmin lr({- EE_(pi^"s")[log p_(N|0) (x_N|x_0)] + H(pi^"s"): 
    pi in scr(P)_(2), pi^"s"_0 = p_"data", pi^"s"_N = p_"prior"})
$<static-schrodinger-bridge-1>

若取 $p_(k+1|k)$ 为高斯核，均值为 $0$，方差为 $sigma_k$，则 $p_(N|0)(x_N|x_0) = cal(N) (x_N;x_0, sigma^2)$，其中 $sigma^2 = sum_(k=1)^N sigma_k^2$。因此#ref(<static-schrodinger-bridge-1>)也可以写为

$
  pi^("s", star) = argmin lr({ EE_(pi^"s")[norm(x_0 - x_N)^2] - 2 sigma^2 H(pi^"s"): 
    pi in scr(P)_(2), pi^"s"_0 = p_"data", pi^"s"_N = p_"prior"})
$<static-schrodinger-bridge-2>

=== IPF (Iterative Proportional Fitting) 算法

#tab 遗憾的是，大多数非平凡情况下的 SB 都没有闭式解。但我们可以使用 IPF 算法迭代求解。取 $n in NN$，和初始路径测度 $pi^(0) = p$，然后循环执行下面的迭代操作：

$
  pi^(2n+1) &= argmin {KL(pi|pi^(2n)): pi in scr(P)_(N+1), pi_N = p_"prior"}\
  pi^(2n+2) &= argmin {KL(pi|pi^(2n+1)): pi in scr(P)_(N+1), pi_0 = p_"data"}\
$

这相当于将 $pi^0$ 重复地在 KL 散度的意义下分别向限制了一个端点边缘分布的空间中投影。不过这样的形式依然难以计算。作者在引入改进前，首先提到了下面的命题

#proposition[
  假设 $KL(p_"data" tensor p_"prior"|p_(0, N)) < infinity$，则对任意 $n in NN$，$pi^(2n)$ 和 $pi^(2n+1)$ 分别在 Lebesgue 测度 $p^n$ 和 $q^n$ 下总有正密度，且对任意 $x_(0:N) in cal(X)$，我们有 $p^(0) (x_(0:N)) = p(x_(0:N))$ 以及
  $
    q^n (x_(0:N)) = p_"prior" (x_N) prod_(k=0)^(N-1) p^(n)_(k|k+1) (x_k|x_(k+1)) \
    p^(n+1) (x_(0:N)) = p_"data" (x_0) prod_(k=0)^(N-1) q^(n)_(k+1|k) (x_(k+1)|x_(k)) \
  $
]

在实际操作中，我们使用 Bayes 公式相互转化 $p^n_(k|k+1)$ 和 $q^n_(k+1|k)$。我们最初使用加噪过程得到的联合分布 $p$ 作为初始值 $p^0$。在第 $2n$ 步时，有 $pi^(2n) = p^n$。从 $p^n$ 得到的逆向过程 $p^n_(k|k+1)$ 结合上数据分布 $p_"data"$ 就得到了逆向过程 $pi^(2n+1) = q^n$。最后 $q^n$ 对应的正向过程 $q^n_(k+1|k)$ 又定义了新的 $pi^(2n+2)=p^(n+1)$。根据前文对逆向过程分布的推导，我们可以得到前向转移分布 $p^n_(k+1|k) (x_(k+1)|x_k) = normal (x_(k+1); x_k + gamma_(k+1)f_k^n (x_k), 2 gamma_(k+1) mtxId))$ 对应的逆向转移分布 
$pbk(n,k) approx normal (x_k; x_(k+1) + gamma_(k+1) b_(k+1)^n (x_(k+1), 2 gamma_(k+1) mtxId)
$
其中 $b_(k+1)^n (x_(k+1)) = - f^b_k (x_(k+1)) + 2 nabla log p_(k+1)^n (x_(k+1))$。我们还可以用同样的方法估计 $pfw(n+1,k)$ 为 
$
  pfw(n+1,k) &approx normal (x_(k+1); x_k + gamma_(k+1)f_k^(n+1) (x_k), 2 gamma_(k+1) mtxId) \
  &= normal (x_(k+1); x_k + gamma_(k+1) [-b_(k+1)^n (x_k) + 2 nabla log q_k^n (x_k) ] , 2 gamma_(k+1) mtxId)
$

于是可以得到 $f_k^(n+1) (x_k) = f_k^n (x_k) - 2 nabla log p_(k+1)^n (x_k) + 2 nabla log q_k^n (x_k)$，我们可以通过训练分数网络达到估计 $f_k^(n+1)$ 和 $b_k^(n+1)$ 的目的。下面的命题给出了对这二者迭代估计的可行性。

#proposition[
  假设任意 $n in NN$, $k in {0, ..., N-1}$ 都有 
  $
  qbk(n, k) = normal (x_k; B_(k+1)^n (x_(k+1)), 2 gamma_(k+1) mtxId) \
  pfw(n,k) = normal (x_(k+1); F_k^n (x_k), 2 gamma_(k+1) mtxId)
  $
  其中 $B_(k+1)^n (x) = x + gamma_(k+1) b_(k+1)^n (x)$，$F_k^n (x) = x + gamma_(k+1) f_k^n (x)$；则对任意满足前述条件的 $n, k$，都有 
  $
    B_(k+1)^n &= argmin_(B in L^2 (RR^d, RR^d)) EE_(p^n_(k,k+1)) lr([norm(B(X_(k+1)) - [X_(k+1) + F_k^n (X_k) - F_k^n (X_(k+1))])^2])\
    F_(k)^(n+1) &= argmin_(B in L^2 (RR^d, RR^d)) EE_(p^n_(k,k+1)) lr([norm(F(X_(k)) - [X_(k+1) + B_(k+1)^n (X_(k+1)) - B_(k+1)^n (X_k)])^2])\
  $
]

我们可以分别用两个神经网络来学习 $B_k^n$ 和 $F_k^n$，然后迭代执行上面的两个优化步骤至收敛，就得到了切实可行的 #schrodinger 桥生成模型。

#figure(
  image("/assets/image-1.png", width: 40%)
)

文章在引入可行的 IPF 迭代算法后接着讨论了这个离散情形迭代算法的收敛性。文中指出在一定的前提下，迭代得到的路径测度序列 $(pi^n)_(n in NN)$ 是良定的、相邻之间的 KL 散度递减，其边缘分布与 $data$ 和 $prior$ 之间的 KL 散度之和为 $display(o(1"/"n))$。接着作者指出在同一假设条件下 SB 问题解的存在性和 IPF 算法的收敛性。文章最后考虑了连续情形的 IPF 算法并指出离散 IPF 为连续 IPF 之离散化这一关系。

=== 实验
==== Gauss 分布之间的 #schrodinger 桥

#tab 取 $prior = normal (-alpha, mtxId), data = normal (alpha, mtxId)$，其中 $alpha = 0.1 times bold(1)$。这个问题的存在解析形式的静态 SB。取数据维数为 $d = 5, 50$，并在其上训练 DSB，结果如下。小网络 (small) 可以在低维数据的情况下求得正确结果，但在高维情况下需要增加网络大小和训练轮数。

#figure(
  image("/assets/image-2.png", width: 70%),
  caption: [DSB 模型在高斯分布实验上收敛]
)

==== 二维分布

#figure(
  image("/assets/image-3.png", width: 50%), 
  caption: [DSB 模型二维数据集上的训练结果]
)

==== 图像生成
#figure(
  grid(columns: 1, image("/assets/image-4.png", width: 90%), image("/assets/image-6.png", width: 90%)), 
  caption: [DSB 模型二维数据集上的训练结果]
)

==== 数据插值
#figure(
  image("/assets/image-7.png", width: 70%), 
  caption: [DSB 模型二维数据集上的训练结果]
)

=== 讨论

#tab 本文首次将 #schrodinger 桥引入生成模型中，相比其他生成模型，SDB 可以以更少的步数达到较好的效果。此外，DSB 的解也是一个扩散过程，这意味着可以套用 Song 等人#cite(<DBLP:paper-yang_song-score_based_generative_modeling_sde>)文章中的 ODE 解法来实现更快地采样生成。尽管如此，如果将 DSB 的迭代轮数 $N$ 降得太低，则可能因 $p_N$ 与 $prior$ 不再接近导致生成效果变差。本文在理论推导上进行了很多探索，保证了分数模型用于生成的可行性，以及使用 IPF 迭代算法的收敛性。此外，DSB 还可以用于多边缘分布约束的 #schrodinger 桥问题、计算 Wasserstein barycenter、求解熵正则化的 Gromov-Wasserstein 问题的最小值解以及求解连续状态空间中的域适应问题。


#pagebreak()


// == Scalable Diffusion Models with Transformers

// #pagebreak()

= 学习进度
// == 机器学习理论
// === Markov Chain Monte Carlo (MCMC)


// === EM 算法 


// === 计算学习理论


// #pagebreak()

// == 随机过程

// #h(2em)本周学习了连续状态的 Markov 链。

// #pagebreak()

// == 随机微分方程
// #h(2em)本周开始学习 SDE 解的存在性和唯一性。

// #pagebreak()

== 实分析
=== 动机
#h(2em)第一个问题源于 Fourier 变换。

第二个问题是极限和积分的可交换性。

第三个问题是可求长曲线的问题。

第四个问题是

=== 方体

#tab 为了求 $RR^p$ 中某些集合的“大小”或者“体积”，我们需要后者分解为可以轻易求得体积的“基本构件”的“几乎无交”并。而我们采用方体为基本构件。

#definition[$RR^p$ 中的 (闭) 方体][
  $RR^p$ 中的方体是一个这样的集合 $R$:
  $
    R = [a_1, b_1] times [a_2, b_2] times dots.c times [a_p, b_p].
  $
  它的意思是
  $
    R = \{bx in RR^p: a_1 lt.slant x_1 lt.slant b_1, ..., a_p lt.slant x_p lt.slant b_p\}.
  $
  它的体积为 $display(|R| = product_(i=1)^p) (b_p - a_p)$.
]

相应地；开方体只需将定义中的闭区间变成开区间即可，且其体积与相同端点的闭方体相同。如果两个方体的*内部*不交，我们称它们*几乎无交*。对于 $RR^2$ 中的开集，我们能得到十分有趣的结论：$RR^d (d gt.slant 1)$ 中的任意开集都可以写成可数个几乎无交的闭方体的并：

#theorem[
  $RR^p$ 中的任意开集 $cal(O)$ 都可以写成可数个几乎无交个闭方体的并。
]


证明它的方法并不复杂。首先画一个边长为 $1$ 的网格，就得到了若干边长为 $1$ 的方体。然后做下面的操作。(1) 如果某方体被集合 $cal(O)$ 包含，那么我们接受该方体；(2) 如果某方体与集合 $cal(O)$ 不交，我们拒绝之；(3) 如果该方体和集合 $cal(O)$ 的边界交集非空，我们暂且接受。接下来将每个暂且接受的方体，划分为大小相同、边长相等的四个小方体，然后继续一直做上面的接受-拒绝测试，然后对于暂且接受的方体一直划分下去。由于所有边长的方体可以和 $ZZ^p times ZZ$ 形成一一对应，因此全体方体的集合是可数的，接收得到的方体的集合也是可数的。


#figure(
  image("image.png", width: 60%), 
  caption: [将开集 $cal(O)$ 分解为可数个几乎无交的方体]
)

=== Cantor 集

#definition[Cantor 集][
  定义这样的一列集合 ${C_n}_(n=1)^(infinity)$，其中 $C_0 = [0, 1]$。$C_1 = display([0, 1/3] union [2/3, 1])$，相当于将 $C_0$ 中的闭区间每个切成三份，弃去中间的一份。然后一直这样做下去，得到 $C_2$，$C_3$ 等等。Cantor 集 $cal(C)$ 定义为这些集合的交：
  $
    cal(C) = inter.big_(i=1)^infinity C_i
  $
]

#figure(
  image("/assets/image.png", width: 50%), 
  caption: [Cantor 集的构造]
)

它有一些有趣的性质，例如它的“长度”为零，但它却是不可数集。

// === 外测度


#pagebreak() 

// == 动力系统基础
// #h(2em)

// #pagebreak()

// = 问题记录


// #pagebreak()

= 下周计划
*论文阅读*
+ 生成模型
  - 薛定谔桥
  - DDIM

*项目进度*
+ 使用神经网络学习生命游戏的演化动力学
  - 考虑另外两种方法的实现
  - 更新在线 Overleaf 文档
+ 耦合约瑟夫森结
  - 将 MATLAB 模拟代码全部迁移至 Python 
  - 考虑简单的 Neural SDE 方法解带参 OU 过程的参数

*理论学习*
+ 随机过程课程
  - 复习 Poisson 过程和 Markov 过程
+ 随机微分方程
  - 第五章完成

#pagebreak()

= 附录



#pagebreak()

#set text(lang: "en")

#bibliography("refs.bib",   // 你的 BibTeX 文件
              title: "参考文献",
              style: "ieee", 
              full: false
)