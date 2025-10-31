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

// Snippets
#let const = "constant"
#let bs = $bold(s)$
#let bx = $bold(x)$
#let by = $bold(y)$
#let bz = $bold(z)$
#let fg = $frak(g)$
#let fh = $frak(h)$
#let fu = $frak(u)$
#let fp = $frak(p)$
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

#set par(
  first-line-indent: 2em,
  justify: true,
)

#show figure.where(
  kind: table
): set figure.caption(position: top)

#set math.equation(numbering: "(1)")
#set math.mat(delim: ("[", "]"), align: center)
#set heading(numbering: "1.")
#set math.cases(gap: 0.5em)


// metadata
#let wk_report_name = "2025年10月20日至10月26日周报"
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

=== 下一步

#tab 目前将目标从提取显式的规则改为从已知部分规则和黑箱规则代理产生的少量数据，结合带有等变性的神经网络训练作为动力学仿真器。规则发现的具体流程是，从一些已知规则出发，首先根据一条有偏的轨道从一个规则空间中选取一系列的假设规则，在神经网络学习这条轨道后，再通过神经网络作为演化模拟器的演化统计结果，和原来的统计数据对比，以确定某些候选规则存在或不存在，从而逐步确定系统演化的真正动力学。

#pagebreak()
= 文献阅读

// == Denoising Diffusion Probabilistic Models #cite(<DBLP:paper-DDPM>)
// Jonathan Ho, Ajay Jain and Pieter Abbeel | https://arxiv.org/abs/2006.11239

// 本周把 DDPM 的剩余部分补完。

// === 补遗


// === 实验结果


// === 总结和讨论

// #pagebreak()

== Score-based generative modeling through SDE #cite(<DBLP:paper-yang_song-score_based_generative_modeling_sde>)

Yang Song et al. | https://arxiv.org/abs/2011.13456

本文从 SDE 的视角统一了先前的 Langevin 退火的 score matching 方法 SMLD#cite(<DBLP:paper-SMLD>)和原版扩散模型 DDPM#cite(<DBLP:paper-DDPM>)，从它们的共同点出发，先将加噪动力学连续化变成 SDE，再利用已有的逆向SDE 解析解，得到与加噪 SDE 相反演化方向的逆向 SDE；最后从 SMLD 和 DDPM 逆向求解的数值算法提出了新的预测-校正方法和概率流 ODE 方法。可以说本文所提出的是随机采样的改进，神经网络扮演的角色仅仅为 $nabla log p(bx)$ 的逼近器，它不是本文的重点。

=== Score matching: SMLD 和 DDPM

#tab Score-based 生成模型的核心思想是通过学习 $bs_theta (bx, t)$，让其逼近 $nabla log p(bx)$，然后再根据 Langevin 采样，生成来自未知分布 $p(bx)$ 的样本。在 SMLD 中，由于数据分布未知而无法计算的目标项 $nabla log p(bx)$ 是通过加噪解决的，即考虑高斯核 $q_sigma (hat(bx)|bx) ~ cal(N)(bold(0), sigma bold(I))$，加噪后变量对应的目标函数可以写成
$
  J(bold(theta)) 
  &= 1/2 EE_(bx ~ p(bx), hat(bx) ~ q_sigma (hat(bx)|bx)) [ norm(bs_theta (hat(bx), sigma) - nabla_(hat(bx)) log q_sigma (hat(bx)|bx) )_2^2]  \ 
  &= 1/2 EE_(bx ~ p(bx), hat(bx) ~ q_sigma (hat(bx)|bx)) [ norm(bs_theta (hat(bx), sigma) + sigma/epsilon )_2^2] & hat(bx) = bx + bold(epsilon), bold(epsilon) ~ cal(N)(bold(0), sigma bold(I))
$
可以看出，采用加噪技巧的 SMLD 中，分数函数实际上在预测噪声，这就与 DDPM 建立了联系。即使 DDPM 论文中并未显式地做出 score matching，但我们也可以认为 DDPM 是在训练一个分数函数。 

现在来看二者的去噪动力学。SMLD 使用 Langevin 采样的方法，对于某个固定的噪声水平 $sigma_i$，有
$
  bx^(m)_i = bx^(m)_(i-1) + alpha_i bs_theta (bx^(m)_(i-1), sigma_i) + sqrt(2 alpha_i) bold(z)^(m)_i, wide bold(z)^(m)_i ~ cal(N)(bold(0), bold(I)), m = 1, ..., M
$

其中 $i = 1, ..., N$。DDPM 的版本和 SMLD 相似，它通过两个 Gauss 分布之间 KL 散度的解析解，可以求出逆向条件分布 $q(bx_(i-1)|bx_i, bx_0)$ 的均值 $tilde(bold(mu))_t$ 和方差 $tilde(beta)_t$，最后得到去噪公式

$
  bx_(i-1) = 1/sqrt(1 - beta_i) (bx_i - beta_i bs_theta (bx_i, sigma_i)) + sqrt(beta_i) bold(z)_i, wide bold(z)_i ~ cal(N)(bold(0), bold(I))
$

=== Score matching 的 SDE 视角
==== 加噪过程

#tab 我们考虑将 SMLD 和 DDPM 的加噪动力学连续化为 $dd bx = bold(f)(bx, t) dd t + g(t) dd bold(w)$ 的形式，其中 $bold(w)$ 是 $n$ 维 Brown 运动，$bold(f)(bx, t)$ 是漂移项，$g(t)$ 是扩散项。对于 SMLD，有下面的改写：
$
  bx_i &= bx_(i-1) + sqrt(sigma_i^2 - sigma_(i-1)^2) bold(z)_i \
  bx(t + Delta t) - bx(t) &= sqrt(sigma(t+Delta t)^2 - sigma(t)^2)z(t) \
  bx(t + Delta t) - bx(t) &= sqrt( (dd [sigma(t)]^2)/(dd t) ) sqrt(Delta t) z(t) wide & "用" (dd [sigma(t)]^2)/(dd t) "对" sigma_i^2 - sigma_(i-1)^2 "做一阶近似" \
  dd bx &= sqrt( (dd [sigma(t)]^2)/(dd t) ) dd bold(w) & dd bold(w) ~ (dd t)^(1/2)bold(epsilon)
$
类似地，对 DDPM 的加噪过程，也有类似的改写：
$
  bx_i &= sqrt(1 - beta_i) bx_(i-1) + sqrt(beta_i) bold(z)_(i-1) \ 
$
$
  bx(t + Delta t) &= sqrt(1 - beta(t + Delta t) Delta t) bx(t) + sqrt(beta(t + Delta t) Delta t) bold(z)(t) wide quad & #block(text([令 $beta(i"/"N)=beta_i$ \ 其中 $1"/"N$ 是噪声步长]))\
  &approx (1 - 1/2 beta(t + Delta t) Delta t) bx(t) + sqrt(beta(t + Delta t)) (Delta t)^(1/2) bold(z)(t) & "Taylor 展开到一阶"\
  &approx bx(t) - 1/2 beta(t) bx(t) Delta t + sqrt(beta(t)) (Delta t)^(1/2) bold(z)(t) \
  dd bx &= -1/2 beta(t) bx dd t + sqrt(beta(t)) dd bold(w) & dd bold(w) ~ (dd t)^(1/2) bold(epsilon)
$
如果 DDPM 中的噪声表不是线性变化的，那将得到一个形式类似，但 $bold(f)$ 和 $g$ 不同的 SDE。上文中，SMLD 的加噪 SDE 被称为是 *variance exploding (VE)* 的，而 DDPM 的加噪 SDE 被称为是 *variance preserving (VP)* 的。作者还构造出一种 *sub-VP SDE*，加噪 SDE 为
$
  dd bx = -1/2 beta(t) bx dd t + underbrace(sqrt(beta(t) (1 - e^(- 2 int_0^t beta(s) dd s))), sqrt(macron(beta)(t))) dd bold(w).
$
其方差被 VP-SDE 的方差控制。以上三种 SDE 的一维特殊情形的方差推导可见附录，对特殊情形的讨论可以很容易看书这三种 SDE 叫做 VE，VP 以及 sub-VP SDE 的原因。

=== 求解逆向 SDE

Anderson 给出了上节中一般形式之加噪 SDE 的逆向 SDE，其形式为
$
  dd bx = [ - bold(f)(bx, t) + g^2(t) nabla_(bx) log p_t (bx) ] dd t + g(t) dd bold(macron(w))  
$
其中 $macron(bold(w))$ 是一个倒流的 Brown 运动，$dd t$ 是倒流的无穷小时间间隔。由于 $bold(f)$ 和 $g$ 已知，而 $nabla_(bx) log p_t (bx)$ 由神经网络估计。在 SMLD 中其总目标函数是

$
  bold(theta)^* = argmin_(bold(theta)) sum_(i=1)^N sigma_i^2 EE_(bx ~ p(bx), hat(bx) ~ q_sigma (hat(bx)|bx)) [ norm(bs_theta (hat(bx), sigma) - nabla_(hat(bx)) log q_sigma (hat(bx)|bx) )_2^2]
$

将其拓展至连续的情形，目标函数就变为

$
  bold(theta)^* = argmin_(bold(theta)) EE_t [lambda(t) EE_(bx(0), bx(t)|bx(0)) [ norm(bs_theta (bx(t), t) - nabla_(bx(t)) log p_(0, t) (bx(t)|bx(0)) )_2^2]]
$
其中 $lambda(t)$ 是正的权重函数，$t$ 在 $[0, T]$ 上均匀采样。可以取 $lambda prop EE [ norm(nabla_(hat(bx)) log p_(0, t) (bx(t)|bx(0)) )_2^2]$，使得每个 $t$ 下的损失项对总损失的平均贡献相同。剩余的问题来自于如何求解 $p_(0, t) (bx(t)|bx(0))$，或者说转移核。*如果 $bold(f)(dot, t)$ 是仿射函数，转移核是 Gauss 分布的密度函数，其参数均值和方差拥有闭式解；而对于一般的 SDE，需要求解 Kolmogorov 前向方程才能得到 $p_(0, t) (bx(t)|bx(0))$。*但如果不使用加噪技巧，而是使用*分片分数匹配*以计算目标函数，就可以规避计算转移核的困难（见附录）。

剩下的问题就变成了如何用这个逆向 SDE 采样。

==== 一般的 SDE 求解器
==== 预测-校正方法
==== 概率流 ODE
=== 可控生成
=== 实验结果和讨论
=== 依然存在的疑惑

- Variance Exploding SDE 和 Variance Preserving SDE 以及 sub-VP SDE 为什么叫这些名字，换言之，如何证明它们分别是 variance exploding 和 variance preserving 的？
- 概率流 ODE 是否只是一个求解逆向 SDE 的方法？它和文中提出的 SDE 求解方法之间有什么不同？

#pagebreak()

== Mean Flows for One-step Generative Modeling #cite(<DBLP:paper-mean_flow>)

Zhengyang Geng et al. | https://doi.org/10.48550/arXiv.2505.13447

#pagebreak()
// == Sliced Score Matching: A Scalable Approach to Density and Score Estimation #cite(<DBLP:paper-yang_song-sliced_score_matching>)

// - Yang Song, Sahaj Garg, Jiaxin Shi, Stefano Ermon 
// - https://arxiv.org/abs/1905.07088


// #pagebreak()

// == General E(2)-Equivariant Steerable CNNs #cite(<DBLP:paper-e2cnn>)

// - Maurice Weiler and Gabriele Cesa
// - https://arxiv.org/abs/1911.08251


// #pagebreak()

= 学习进度
// == 机器学习理论
// === Markov Chain Monte Carlo (MCMC)


== 随机过程

#h(2em)本周稍稍总结功率谱密度后的故事。


接着我跳过 Poisson 过程，开始看 Markov 链的部分。

== 随机微分方程

#h(2em)本周没有推进。

#pagebreak()
// = 问题解决记录

= 下周计划
*论文阅读*
+ 生成模型
  - DDPM 完结
  - 薛定谔桥（精读）
  - DDIM（泛读）
+ 几何深度学习
  - General $E(2)$ - Equivariant Steerable CNNs

*项目进度*
+ 使用神经网络学习生命游戏的演化动力学
  - 尝试对进一步简化后的模型权重进行解释，并用模型作为系统的演化模拟器，统计验证所学到的规则是否正确
  - 阅读等变CNN的代码
  - 尝试开始做单轨道训练样本的生成、训练和演化动力学统计

*理论学习*
+ 随机过程课程
  - 学习完毕 Markov 过程
+ 随机微分方程
  - 完成第四章 随机积分
  - 第五章 随机微分方程 开头

#pagebreak()
#bibliography("refs.bib",   // 你的 BibTeX 文件
              title: "参考文献",
              style: "ieee", 
              full: false
)

#pagebreak()
= 附录


== 一维特殊情形的 VE-SDE、VP-SDE 和 sub-VP SDE 方差推导

可以这样直观理解：假设 $x$ 都是标量随机过程，对 SMLD 的加噪过程，有
$
  "Var"[x(t)]
  &="Var"[x(t)-x(0)]\
  &="Var" [int_0^t sqrt(2 sigma sigma') dd w ] = int_0^t 2 sigma sigma' dd tau & #ito "等式"\
  &= sigma^2(t) - sigma^2(0) & "换元积分法" 
$
对 DDPM 的加噪过程，有一阶矩：
$
  dd EE[x(t)] 
  &= EE[dd x(t)] = -1/2 beta(t) EE[x(t)] dd t\
  (dd EE[x(t)])/(EE[x(t)]) &= -1/2 beta(t) dd t & "分离变量"\
  log(|EE[x(t)]|) &= -1/2 int_0^t beta(t) dd t + C ==> |EE[x(t)]| = C' e^(-1/2 int_0^t beta(t) dd t) & "积分"\
  EE[x(t)] &= e^(-1/2 int_0^t beta(t) dd t) EE[x(0)] & "假设"EE[x(0)] gt.slant 0\
$
二阶矩：
$
  dd EE[x^2(t)] 
  &= EE[dd x^2(t)] = EE[2x(t) dd x(t) + beta(t) dd t] wide & #ito "乘积法则"\
  &= EE[x(t) (2 sqrt(beta(t)) dd w) - beta(t) x(t) dd t) + beta(t) dd t]\
  &= EE[ 2 x(t) sqrt(beta(t)) dd w] - beta(t) EE[x^2(t)] dd t + beta(t) dd t\
  EE[int_0^t dd x^2(t)] 
  &= underbrace(EE[ 2 int_0^t x(t) sqrt(beta(t)) dd w], 0) - int_0^t beta(tau) EE[x^2(tau)] dd tau + int_0^t beta(tau) dd tau  & "期望的线性性"\
  dd EE[x^2(t)] &= - beta(t) EE[x^2(t)] dd t + beta(t) dd t & "换回随机微分"\
  (dd EE[x^2(t)])/(1- EE[x^2(t)]) &= beta(t) dd t & "分离变量"\
  - log(|1 - EE[x^2(t)]|) &= int_0^t beta(t) dd t + C ==> |1-EE[x^2(t)]| = C' e^(- int_0^t beta(t) dd t) & "积分"\
  EE[x^2(t)] &= 1 - e^(- int_0^t beta(t) dd t) (1 - EE[x^2(0)]) & "假设"1-EE[x^2(t)] gt.slant 0\
$
因此方差为
$
  "Var"[x(t)] 
  &= EE[x^2(t)] - (EE[x(t)])^2 \
  &= 1 - e^(- int_0^t beta(t) dd t) (1 - EE[x^2(0)]) - e^(- int_0^t beta(t) dd t) (EE[x(0)])^2 \
  &= 1 + e^(- int_0^t beta(t) dd t) ("Var"[x(0)] - 1) 
$
因此当 $"Var"[x(0)] = 1$ 时，有 $x(t)$ 的方差恒为 1。

对于 sub-VP SDE，套用上面的结果，并记 $B(t) = int_0^t beta(s) dd s$，容易得到
$
  EE[x(t)] 
  &= e^(-1/2 int_0^t beta(t) dd t) EE[x(0)]\
  EE[x^2(t)]
  &= e^(-int_0^t beta(s) dd s) [int_0^t macron(beta)(tau) e^(int_0^tau beta(s) dd s) dd tau + EE[x^2(0)]] \
  &= e^(-int_0^t beta(s) dd s) [int_0^t beta(tau) (1 - e^(- 2 int_0^tau beta(s) dd s)) e^(int_0^tau beta(s) dd s) dd tau + EE[x^2(0)]] \
  &= e^(-B(t))EE[x^2(0)]] +  e^(-B(t))[int_0^t B'(tau) (1 - e^(- 2 B(tau))) e^(B(tau)) dd tau ]\
  &= 1 + e^(-2B(t)) + e^(-B(t))(EE[x^2(0)]] - 2)\ 
$ 
$
  "Var"[x(t)]
  &= EE[x^2(t)] - (EE[x(t)])^2 \
  &= 1 + e^(-2B(t)) + e^(-B(t))(EE[x^2(0)]] - 2) - e^(- B(t)) (EE[x(0)])^2 \
  &= 1 + e^(- B(t)) ("Var"[x(0)] - 2) + e^(-2 B(t)) \
  &= 1 + e^(- int_0^t beta(s) dd s) ("Var"[x(0)] - 2) + e^(-2 int_0^t beta(s) dd s)\
$
一些简单的推导，可以得出 sub-VP SDE 的方差小于等于 VP-SDE 的方差。

== 逆向 SDE 推导




