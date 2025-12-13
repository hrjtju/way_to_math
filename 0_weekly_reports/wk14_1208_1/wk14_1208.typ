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
#let KL = $D_("KL")$
#let argmin = $op("arg min", limits: #true)$
#let argmax = $op("arg max", limits: #true)$

#let ito = $"It"hat("o")$
#let schrodinger = "Schrödinger"

// Theorem environments
#let theorem = thmbox("theorem", "定理", fill: rgb("#eeffee"), base_level: 1)
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


== Denoising Diffusion Implicit Models

*#link("http://arxiv.org/abs/2010.02502")[ICLR 2021] | Jiaming Song et al.*

在 DDPM 论文中，加噪过程为一个 Markov 过程，有状态转移分布 $p(bx_(t)|bx_(t-1))$ 和对应的反向转移分布 $q(bx_(t-1)|bx_t)$。这代表*加噪过程中进行的步数和模型参与的去噪步骤数是相同的*。另外，在假设状态转移分布为 Gauss 分布时，到达加噪过程终点时的分布 $bx_N$ 的分布理论上趋于标准正态分布 $N(0, I)$，当 $N$ 较小时，$bx_N$ 的分布与标准正态分布不再近似，而去噪过程又是从标准正态分布 $N(0, I)$ 开始的，这就造成*分布不对应*。

本文为解决这一问题引入了成为隐式去噪扩散模型（denoising diffusion implicit models）。对于第一个问题，DDIM 引入了一个增广 Markov 加噪和去噪过程，使得生成过程中可以跳步，同时也一定程度缓解了第二个问题：在加噪过程中可以引入任意大的加噪步数，使得 $bx_N$ 的分布趋于标准正态分布。除此之外，DDIM 还将噪声项趋于零，这表示加噪和去噪动力学从 SDE 变成了 ODE，从而获取更高的求解速度和更优的数值性能。

=== 增广 Markov 前向过程

#tab DDIM 引入的去噪过程的转移分布为 $p(bx_(t-1)|bx_t, bx_0)$，并令其为一个简单的 Gauss 分布 $q_sigma (bx_(t-1)|bx_t, bx_0) = N(bold(mu)(bx_t, bx_0), sigma_t^2 mtxId)$。由重参数化技巧，$bx_(t-1)$ 可以写为
$
  bx_(t-1) &~ bold(mu)(bx_t, bx_0) + sigma_t^2 bold(epsilon), tab bold(epsilon) ~ N(bold(0), mtxId)\
  &~ a bx_t + b bx_0 +  sigma_t^2 bold(epsilon)
$
其中 $a$ 和 $b$ 是实数。DDIM 引入的来自 DDPM 的假设为 $bx_t ~ N(sqrt(alpha_t)bx_0, (1-alpha_t) mtxId)$，因此用重参数化的语言来写，就是
$
  sqrt(alpha_(t-1)) bx_0 + sqrt(1-alpha_(t-1)) epsilon &~ a(sqrt(alpha_(t)) bx_0 + sqrt(1-alpha_t) epsilon') + b bx_0 + sigma_t epsilon''\
  N(sqrt(alpha_(t-1)) bx_0, (1-alpha_(t-1)) mtxId) &= N(a sqrt(alpha_t)bx_0 + b bx_0, (a^2(1- alpha_t) + sigma_t^2) mtxId)
$
分布参数对应相等，就得到
$
  a &= sqrt((1 - alpha_(t-1) - sigma_t^2)/(1 - alpha_(t)))\ 
  b &= sqrt(alpha_(t-1)) - sqrt(alpha_t) dot.c sqrt((1 - alpha_(t-1) - sigma_t^2)/(1 - alpha_(t)))
$
这样就得到逆向过程转移概率应该有的形式：
$
  q(bx_(t-1)|bx_t, bx_0) = N(sqrt(alpha_(t-1)) bx_0 + sqrt(1 - alpha_(t-1) - sigma_t^2) dot.c (bx_t - sqrt(alpha_t)bx_0)/sqrt(1-alpha_t), sigma_t^2 mtxId)
$
对应的加噪过程可以由 Bayes 公式给出
$
  q_sigma (bx_t | bx_(t-1), bx_0) = (q_sigma (bx_(t-1)|bx_t, bx_0)q_sigma (bx_t|bx_0))/(q_sigma (bx_(t-1)|bx_0))
$
然而事实上，由于上面推导中的 $t-1$ 时间步和 $t$ 时间步并没有联系，我们可以将两个时间步视作 $s, t$，满足 $0 < s < t$，这样一来*跳步的去噪过程*可以写成
$
  q(bx_(s)|bx_t, bx_0) = N(sqrt(alpha_(s)) bx_0 + sqrt(1 - alpha_(s) - sigma_(s,t)^2) dot.c (bx_t - sqrt(alpha_t)bx_0)/sqrt(1-alpha_t), sigma_(s,t)^2 mtxId)
$
此时 $sigma_(s,t)$ 需要做相应的调整。或者将 $t$ 和 $t-1$ 的下标视作序列 ${t}_(t=0)^N$ 的一个子列 ${t_k}_(k=1)^N$，相应地方差的下标就变成 $sigma_(t_k)$。

然而需要注意的是，对 $q_sigma (bx_(1:T)|bx_0)$ 的拆解依然使用了 Markov 性：
$
  q_sigma (bx_(1:T)|bx_0) = product_(k=1)^T q_sigma (bx_k|bx_(1:k-1), bx_0) = product_(k=1)^T q_sigma (bx_(k)|bx_(k-1), bx_0)
$
因此与其说 DDIM 使用了非 Markov 链的加噪/去噪模型，不如说它使用的是一个相比原版*增广 (augmented) 的 Markov 链*，其中状态 $hat(x)_t$ 可理解为 $(bx_t, bx_0)$。

=== 兼容 DDPM

#tab 注意 DDPM 中的目标可以写成
$
  L_gamma (epsilon_theta) = sum_(t=1)^T gamma_t EE_(*) lr(norm(epsilon_theta^((t))(sqrt(alpha_t) bx_0 + sqrt(1-alpha_t) epsilon_t) - epsilon_t)_2^2)
$
其目标所对齐或模拟的是 $q_sigma (bx_0|bx_t)$。我们假设通过 DDPM 训练目标训练好的模型 $epsilon_theta^(t)$ 给出的是 $bx_0$ 的一个估计。就可以利用该估计计算逆向转移概率。令
$
  f_theta^((t))(bx_t) = (bx_t - sqrt(1-alpha_t) epsilon_theta^((t))(bx_t))/sqrt(alpha_t)
$
则逆向转移概率可以写成 $p_theta(bx_(t-1)|bx_t) = q_sigma (bx_(t-1)|bx_t, f_theta^((t))(bx_t))$。自然地，我们需要对齐模型估计的去噪序列和真实的序列：
$
  J_sigma (epsilon_theta) = EE_(bx_(0:T) ~ q_sigma (bx_(0:T))) lr([log q_sigma (bx_(1:T)|bx_0) - log p_theta (x_(0:T))])
$
可以证明，该目标与 DDPM 的目标等价。

在生成时，给定 $bx_t$，可以根据下式预测 $bx_(s)$
$
  bx_s = sqrt(alpha_s) underbrace([(x_t - sqrt(1-alpha_t) sigma_theta^((t))(bx_t))/sqrt(alpha_t)], bx_0 " 的预测值") + underbrace(sqrt(1 - alpha_s - sigma_(s,t)^2) epsilon_theta^((t))(bx_t), "指向 "x_(t-1)" 的调节项") + sigma_(s,t) epsilon_(s,t)
$
如果对任意 $s,t$，都令 $sigma_(s,t) -> 0$，则去噪动力学的随机性逐渐消失，该离散动力学变成了下面的 ODE 的 Euler 离散化版本
$
  dd macron(bx) = epsilon_theta^((t)) [macron(bx)/sqrt(sigma^2 + 1)] dd sigma,
$
这还对应着 Song 等人 #ref(<DBLP:paper-yang_song-score_based_generative_modeling_sde>)论文中的 VE-SDE 的概率流 ODE 的形式。由于其没有随机性，可以应用更快速的 ODE 数值解法。

为更好理解论文的理论，本文需要参照 DDPM 和 DDIM 的代码。

#pagebreak()

== Reconciling Modern Machine Learning Practice and the Bias-variance Trade-off

*#link("http://arxiv.org/abs/1812.11118")[Arxiv] | Mikhail Belkin et al.*



#pagebreak()

// == Scalable Diffusion Models with Transformers

// #pagebreak()

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
    R = \{bx in RR^p: a_1 lt.slant bx_1 lt.slant b_1, ..., a_p lt.slant bx_p lt.slant b_p\}.
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