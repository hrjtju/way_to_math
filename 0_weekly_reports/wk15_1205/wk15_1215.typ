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
#let wk_report_name = "2025年12月15日至12月21日周报"
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

== Score-based Generative modeling through SDE 补遗

*#link("http://arxiv.org/abs/2011.13456")[ICLR 2021] | Yang Song et al. *

=== 概率流 ODE 和 Fokker-Plank 方程

#pagebreak()


== Scalable Diffusion Models with Transformers

#pagebreak()

= 动力系统与数学物理研讨会
#tab 本周一至周五，我赴湾大松山湖校区参加 2025 年动力系统与数学物理研讨会。由于我的数学和物理背景薄弱，只能"寻章摘句"，拿到一些自以为重要的概念加一学习。


== KAM 理论

== Anderson 局域化


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