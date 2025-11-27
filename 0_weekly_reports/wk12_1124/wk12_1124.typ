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
#let ito = $"It"hat("o")$
#let be = $bold(epsilon)$
#let prod = $product$
#let int = $integral$
#let KL = $D_("KL")$
#let argmin = $op("arg min", limits: #true)$
#let argmax = $op("arg max", limits: #true)$

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
#let wk_report_name = "2025年11月24日至11月30日周报"
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


= 项目进展
== 使用神经网络学习生命游戏的演化动力学
=== 训练数据可视化和裁剪

#h(2em) 本周实现了可视化训练得到轨道的图形化界面程序。发现对于部分规则（例如 `B3678/S34678`）容易演化至平凡的不动点，因此可以控制在生成轨道的过程中，如果发现系统落入一阶不动点，那么就终止模拟。这样做可以提高得到数据集的质量，然而这样操作会使得数据集读取过程变得稍复杂。

=== 优化规则提取器
==== 遍历输入特征的映射方法


==== 统计存活邻居细胞数量的方法（黑盒方法）

#h(2em) 一个暂行的方法是做下面和经典的生命游戏完全相同的假定：一是下一状态只依赖于这个细胞本身和它周边的八个细胞，二是细胞周边的八个邻居细胞对细胞下一状态的权重相同。这样我们可以用神经网络作为演化器，对若干随机输入批量演化下一状态（随机的输入可以是形状为 `[N, 1, w, h]` 的张量），然后统计状态转移 $(bx_(t), cal(N)(bx_t)) -> bx_(t+1)$。这会得到一系列的直方图。此时可以使用例如 FixMatch 这样的半监督的方法，对于高置信度的转移规则，将其确定为神经网络所学到的系统演化规则。

==== 从神经网络权重出发的方法（白盒方法）

#h(2em) 从神经网络权重出发的主要思路为将一个复杂规则，例如 `B3678/S34678` 提取得到一系列的子规则，类似于空间的基。假设我们得到了可以构建所有规则的一个“基规则集”，并在该集合中的每条规则上训练一个和耦合规则一样结构的神经网络，对比训练完成后网络的权重，我们期望参与构成耦合规则的基规则网络的权重能在耦合规则网络的权重中有所体现。检查的方法目前想到的有两个。一是定性的，直接对权重进行可视化，然后比较可视化所见权重：基规则中权重模式是否出现在耦合规则网络的权重中；二是定量的，构造参数空间中的某种内积，然后做内积，得到耦合网络权重在各基规则网络权重方向的“分量”。也许后者可以通过类似度量学习一样，学习一个内积网络出来。

目前该规则中拆分或得到基规则是个难题。如果直接按照naive的想法，切成 `B1/S`, ..., `B8/S`, `B/S1`, ..., `B/S8`，模拟生成时对于大部分的规则，将极其容易跌至平凡不动点，可用于训练的数据极少。

#pagebreak()
= 文献阅读


== Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling

*#link("http://arxiv.org/abs/2106.01357")[NIPS 2021] | Valentin De Bortoli et al.*

#pagebreak()

== Score-based generative modeling through SDE #cite(<DBLP:paper-yang_song-score_based_generative_modeling_sde>)

*#link("http://arxiv.org/abs/2011.13456")[ICLR 2021] | Yang Song et al. *

==== 逆向 SDE 的推导

==== 概率流 ODE 的推导 

#pagebreak()

== Scalable Diffusion Models with Transformers

#pagebreak()

= 学习进度
// == 机器学习理论
// === Markov Chain Monte Carlo (MCMC)


== 随机过程

#h(2em)本周继续系统学习 Markov 链，了解了常返性的一些性质。特别地，在有限状态的时齐 Markov 链中，相互联通的节点常返性相同。

// #pagebreak()

== 随机微分方程
#h(2em)本周开始学习 SDE 解的存在性和唯一性。

// // #pagebreak()

== 量子力学
#h(2em) 了解了量子力学的一些基本概念，例如波函数、薛定谔方程、无限深势阱、算符。

// // #pagebreak()

// == Josephson 结


// // #pagebreak()

// == Riemann–Stieltjes 积分

= 问题记录
== SDE 数值解的问题

#h(2em) 我尝试使用 Python 求解下面的 SDE：
$
  
$



// #pagebreak()

= 下周计划
*论文阅读*
+ 生成模型
  - 薛定谔桥
  - DDIM
// + 几何深度学习
//   - General $E(2)$ - Equivariant Steerable CNNs

*项目进度*
+ 使用神经网络学习生命游戏的演化动力学
  - 调试完成代码，并考虑等变约束
+ 耦合约瑟夫森结
  - 将 MATLAB 模拟代码全部迁移至 Python 
  - 考虑简单的 Neural SDE 方法解带参 OU 过程的参数

*理论学习*
+ 随机过程课程
  - 复习 Poisson 过程和 Markov 过程
+ 随机微分方程
  - 第五章 



#pagebreak()

#set text(lang: "en")

#bibliography("refs.bib",   // 你的 BibTeX 文件
              title: "参考文献",
              style: "ieee", 
              full: false
)

#pagebreak()

#set text(lang: "zh", font: ("New Computer Modern", "Kai", "KaiTi"))

// = 附录


