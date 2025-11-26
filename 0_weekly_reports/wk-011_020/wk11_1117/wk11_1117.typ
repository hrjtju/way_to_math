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
#let wk_report_name = "2025年11月10日至11月16日周报"
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

#tab 综合上周提出的总框架图（#ref(<fig:life_framework>)），本周开始逐个考虑并使用代码实现其中的各关键部分。

#figure(
  image("framework.png"),
  caption: [方法框架总览]
)<fig:life_framework>

== 数据供应代理

#tab 可以用现成数据集代替，但修改可见的演化轨道条数。在初始化时，令可见演化轨道条数为 $1$，每执行完毕一个主动学习循环，且需要继续循环时，可见演化轨道条数 $+1$，从而已收集好的新的一条轨道可以加入训练数据中。另外，还可以考虑将新的轨道赋予更大的权重，可以在 `__getitem__` 方法中添加一个输出，表示其权重，相应地将维护一个可见轨道条数改为维护一个可见轨道的有序列表，依据列表中轨道的先后赋予依次从低到高的权重。

== 等变神经网络模块

#tab 神经网络训练模块有现成的代码，但是需要解决框架图中的施加等变约束的部分。注意刚开始的时候是只有平移不变性的，然后再加上其他的群等变约束，这意味着如果采用群等变卷积网络（`e2cnn`）包的话需要仔细研究其源码实现。一个暂行的替代方案是用普通的 CNN，然后加上不变性损失。例如对于晶体群 $p 4$，可以令 $3 times 3$ 卷积核的上下左右和四个对角的值分别相等，可以采取强制相等，或者旋转损失的方式添加等变性约束。另外同时可以研究 `e2cnn` 的底层实现，来研究如何在不断扩充等变群元时最大程度保留上一次训练好的网络权重，避免需要再人为蒸馏一遍。

== 指示数据集、规则提取器和生成器

#tab 先说指示数据集。虽然它生成起来非常简单：
```python
w, h = 3, 3
ls = [[0, 1] for _ in range(w*h)]
indicator_dataset = None
for idx, i in enumerate(product(*ls)):
    arr = torch.tensor(list(i)).reshape(1, 3, 3)
    if indicator_dataset is None:
        indicator_dataset = arr
    else:
        indicator_dataset = torch.concatenate([indicator_dataset, arr])
```
但在 `w` 和 `h` 变大时，如果要遍历所有的 $0$-$1$ 组合，其组合数将指数级增长（$2^(w h)$），带给生成和验证过程以很大的困难。另外，拿到指示数据集 `indicator_dataset` 和对应的网络预测值后，需要思考的是如何将其转化为确定性规则的问题。

在先前的想法中是选取网络的预测置信度高的转变对 $(x_t, N(x_t)) -> x_(t+1)$，但需要考虑两个问题，一是 `pyseagull` 的模拟器中是通过卷积来实现规则的，我们需要想办法将得到的转变对转化成卷积核的形式。换言之，我们需要将得到的这一堆转变对转换为显式的规则形式，我目前还没有什么好的想法。

#pagebreak()
= 文献阅读
== [续] Score-based generative modeling through SDE #cite(<DBLP:paper-yang_song-score_based_generative_modeling_sde>)

*#link("http://arxiv.org/abs/2011.13456")[ICLR 2021] | Yang Song et al. *

// === 补遗



// ==== 逆向 SDE 的推导

// ==== 概率流 ODE 的推导 


== Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data

*#link("https://arxiv.org/abs/2402.14989v6")[ICLR 2024] | YongKyung Oh et al.*


== Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling

*#link("http://arxiv.org/abs/2106.01357")[NIPS 2021] | Valentin De Bortoli et al.*


#pagebreak()

= 学习进度
== 机器学习理论
=== Markov Chain Monte Carlo (MCMC)


== 随机过程

#h(2em)本周继续系统学习 Markov 链。

#pagebreak()

== 随机微分方程


// #pagebreak()

== 量子力学


// #pagebreak()

== Josephson 结


// #pagebreak()

== Riemann–Stieltjes 积分

// = 问题解决记录
// == Typst 相关
// === 自定义图表标题位置和内容



#pagebreak()

= 下周计划
*论文阅读*
+ 生成模型
  - 薛定谔桥（精读）
  - Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data（精读）
  - DDIM（泛读）
// + 几何深度学习
//   - General $E(2)$ - Equivariant Steerable CNNs

*项目进度*
+ 使用神经网络学习生命游戏的演化动力学
  - 实现数据供应代理模块和等变神经网络模块
+ 耦合约瑟夫森结
  - 将 MATLAB 模拟代码全部迁移至 Python 
  - 考虑简单的 Neural SDE 方法解带参 OU 过程的参数

*理论学习*
+ 随机过程课程
  - 复习 Poisson 过程
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

= 附录


