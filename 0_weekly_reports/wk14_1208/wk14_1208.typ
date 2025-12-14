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
#let wk_report_name = "2025年12月8日至12月14日周报"
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

#tab 

#pagebreak()

= 项目进展
== 使用神经网络学习生命游戏的演化动力学

#tab 

#pagebreak()
= 文献阅读

// == Score-based Generative modeling through SDE 补遗

// *#link("http://arxiv.org/abs/2011.13456")[ICLR 2021] | Yang Song et al. *

// === 概率流 ODE 和 Fokker-Plank 方程



// #pagebreak()


== Demoising Diffusion Implicit Models #cite(<DBLP:paper-DDIM>)

*#link("https://arxiv.org/abs/2010.02502")[ICLR 2021] | Jiaming Song et al.*




#pagebreak()

== Reconciling modern machine learning practice  and the bias-variance trade-off #cite(<Belkin_2019_double_descent>)

*#link("http://arxiv.org/abs/1812.11118")[Arxiv] | Mikhail Belkin et al.*




#pagebreak()

== Scalable Diffusion Models with Transformers

#pagebreak()

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

== 动力系统基础
=== 分岔


=== 圆上的流


=== 线性系统


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