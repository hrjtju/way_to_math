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
#let KL = $D_("KL")$
#let argmin = $op("arg min", limits: #true)$
#let argmax = $op("arg max", limits: #true)$


#let ito = $"It"hat("o")$
#let schrodinger = "Schrödinger"

// Theorem environments
#let theorem = thmbox("theorem", "定理", fill: rgb("#eeffee"))
#let corollary = thmplain(
  "corollary",
  "推论",
  base: "theorem",
  titlefmt: strong
)
#let definition = thmbox("definition", "定义", inset: (x: 1.2em, top: 1em))
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

#tab #lorem(250)

#pagebreak()

= 项目进展
== 使用神经网络学习生命游戏的演化动力学
#pagebreak()
= 文献阅读


== Score-based Generative modeling through SDE 补遗

*#link("http://arxiv.org/abs/2011.13456")[ICLR 2021] | Yang Song et al. *

=== 概率流 ODE 和 Fokker-Plank 方程



#pagebreak()


== Diffusion Schrödinger Bridge with Application to Score-Based Generative Modeling

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
    [$cal(C)$], [$C([0, T], RR^d)$], [从 $[0, T]$ 到 $RR^d$ 的连续函数全体],
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
$

==== 连续情形

=== #schrodinger 桥

=== IPF (Iterative Proportional Fitting) 算法

==== IPF 算法的收敛性

==== IPF 的连续情形

=== 实验和讨论


#pagebreak()


== Scalable Diffusion Models with Transformers

#pagebreak()

= 学习进度
== 机器学习理论
=== Markov Chain Monte Carlo (MCMC)


=== EM 算法 


=== 计算学习理论


#pagebreak()

== 随机过程

#h(2em)本周学习了连续状态的 Markov 链。

#pagebreak()

== 随机微分方程
#h(2em)本周开始学习 SDE 解的存在性和唯一性。

#pagebreak()

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

=== 外测度


#pagebreak() 

== 动力系统基础
#h(2em)

#pagebreak()

= 问题记录


#pagebreak()

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