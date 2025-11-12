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

#set text(lang: "zh", font: ("New Computer Modern", "KaiTi"))

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
#let wk_report_name = "2025年11月10日至11月6日周报"
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


#pagebreak()
= 文献阅读
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

本节介绍逆向 SDE 的解析解和数值解法。Anderson 给出了上节中一般形式之加噪 SDE 的逆向 SDE，其形式为
$
  dd bx = [ - bold(f)(bx, t) + g^2(t) nabla_(bx) log p_t (bx) ] dd t + g(t) dd bold(macron(w))  
$<eq:inverse-sde-cont>
其中 $macron(bold(w))$ 是一个倒流的 Brown 运动，$dd t$ 是倒流的无穷小时间间隔。由于 $bold(f)$ 和 $g$ 已知，而 $nabla_(bx) log p_t (bx)$ 由神经网络估计。在 SMLD 中其总目标函数是

$
  bold(theta)^* = argmin_(bold(theta)) sum_(i=1)^N sigma_i^2 EE_(bx ~ p(bx), hat(bx) ~ q_sigma (hat(bx)|bx)) [ norm(bs_theta (hat(bx), sigma) - nabla_(hat(bx)) log q_sigma (hat(bx)|bx) )_2^2]
$

将其拓展至连续的情形，目标函数就变为

$
  bold(theta)^* = argmin_(bold(theta)) EE_t [lambda(t) EE_(bx(0), bx(t)|bx(0)) [ norm(bs_theta (bx(t), t) - nabla_(bx(t)) log p_(0, t) (bx(t)|bx(0)) )_2^2]]
$
其中 $lambda(t)$ 是正的权重函数，$t$ 在 $[0, T]$ 上均匀采样。可以取 $lambda prop EE [ norm(nabla_(hat(bx)) log p_(0, t) (bx(t)|bx(0)) )_2^2]$，使得每个 $t$ 下的损失项对总损失的平均贡献相同。剩余的问题来自于如何求解 $p_(0, t) (bx(t)|bx(0))$，或者说转移核。*如果 $bold(f)(dot, t)$ 是仿射函数，转移核是 Gauss 分布的密度函数，其参数均值和方差拥有闭式解；而对于一般的 SDE，需要求解 Kolmogorov 前向方程才能得到 $p_(0, t) (bx(t)|bx(0))$。*但如果不使用加噪技巧，而是使用*分片分数匹配*以计算目标函数，就可以规避计算转移核的困难（见附录）。

==== 一般的 SDE 求解器

#tab 对于SDE的模拟，有若干数值方法，例如 *Euler-Maruyama 法*和*随机 Runge-Kutta 法*。DDPM 中提出的*祖先采样*也是和前二者类似的对逆向 VP-SDE 的一种离散化。

===== 祖先采样 (Ancestral Sampling)

#tab 我们可以得到 SMLD 版本的祖先采样。首先 SMLD 的 "加噪过程" 可以写为 $display(bx_i = bx_(i-1) + sqrt(sigma_i^2 - sigma_(i-1)^2) bz_(i-1))$，即 $p(bx_i | bx_(i-1)) ~ cal(N)(bx_(i-1), (sigma_i^2 - sigma_(i-1)^2) mtxId)$。接着我们考虑加噪 Markov 链 $x_0 -> x_1 -> dots.c -> x_N$，可以依照 DDPM 论文中的类似推导（#ref(<appendix:SMLD-ancestral>)）得到 $q(bx_(t-1)|bx_t, bx_0)$:
$
  q(bx_(t-1)|bx_t, bx_0) &= q(bx_t|bx_(t-1), bx_0) dot (q(bx_(t-1) | bx_0))/(q(bx_t | bx_0)) \
  &= cal(N)( sigma_(t-1)^2/(sigma_t^2) bx_t 
    + (1 - sigma_(t-1)^2/(sigma_(t)^2)) bx_0, ((sigma_t^2 - sigma_(t-1)^2)sigma_(t-1)^2)/sigma_t^2)
$
接着将参数分布 $p_(bold(theta))(bx_(i-1)|bx_i)$ 参数化为 $cal(N)(bx_(i-1); bold(mu)_(bold(theta))(bx_i, i), tau_i^2 mtxId)$，不难得到 (#ref(<appendix:SMLD-ancestral>)）DDPM 中的 $L_(t-1)$ 损失这一项可以写为
$
  L_(t-1) = EE_(bx_0, bold(z)) [1/(2 tau_i^2) norm(bx_i (bx_0, bz) - (sigma_i^2 - sigma_(i-1)^2)/(sigma_i) bz - bold(mu)_bold(theta)(bx_i (bx_0, bz), i))_2^2] + C
$
其中 $C$ 是一个无关常数。接着就可以像 DDPM 那样得出采样生成的迭代规则：
$
  bx_(i-1) = underbrace(bx_i + (sigma_i^2 - sigma_(i-1)^2) bold(s)_(bold(theta)^*) (bx_i, i), bold(mu)_bold(theta)(bx_i, i)) + sqrt(((sigma_t^2 - sigma_(t-1)^2)sigma_(t-1)^2)/sigma_t^2) bz_i
$

===== 逆向扩散采样器 (Reverse Diffusion Sampler)
对一般 SDE 而言，要推导出它的 Ancestral 采样并不容易。对于此问题，作者根据逆向 SDE 提出了一种简单有效的离散化方法。考虑一般形式的 SDE $dd bx = bf(bx, t) dd t + bG(t) dd bw$，并固定离散化的时间点序列，可以得到前向 SDE 的离散版本:
$
  bx_(i+1) = bx_i + bf_i (bx_i) + bG_i bz_i, wide i=0, ..., N-1
$
根据#ref(<eq:inverse-sde-cont>)中的逆向 SDE 结果，立刻可以得到下面的采样迭代法，作者称之为*逆向扩散采样器*：
$
  bx_i = bx_(i+1) - bf_(i+1)(bx_(i+1)) + bG_(i+1)bG_(i+1)^top bs_(bold(theta)^*) (bx_(i+1), i+1) + bG_(i+1) bz_(i+1)
$
考虑 DDPM 的情形并对噪声序列进行适当标号，有：$bx_(i+1) = sqrt(1 - beta_(i)) bx_(i) + sqrt(beta_(i)) bold(z_i)$, 得到 $bf_i (bx_i) = [1- sqrt(1-beta_(i))]bx_i$，$bG_i = sqrt(beta_(i))$；其对应的逆向扩散采样为
$
  bx_i = [2 - sqrt(1-beta_(i+1))]bx_(i+1) + beta_(i+1) bs_(bold(theta)^*) (bx_(i+1), i+1) + sqrt(beta_(i+1)) bz_(i+1)
$
而 DDPM 中的 Ancestral 采样的形式为
$
  bx_(i) &= 1/sqrt(1 - beta_(i+1)) (bx_(i+1) - beta_(i+1) bs_theta (bx_(i+1), i+1)) + sqrt(beta_(i+1)) bold(z)_(i+1)
$
当 $beta_i -> 0$ 时，有 $display(1/sqrt(1 - beta_(i+1)) ~ 1 + 1/2 beta_(i+1) = 2 - [1 - 1/2 beta_(i+1)]) ~ 2 - sqrt(1 - beta_(i+1))$，$beta_(i+1)^2 -> 0$，因此此时逆向扩散采样和 DDPM 的 Ancestral 采样等价。


==== 概率流 ODE

#tab 最后介绍本文提出的一个逆向 SDE 采样的另一种方法。对任意扩散过程，存在一个确定的 ODE，它描述轨道和扩散过程中随时间变化的边缘分布 $p_t (bx)$ 相同：
$
  dd bx = [bf(bx, t) + 1/2 g(t)^2 nabla_bx log p_t (bx)] dd t.
$<eq:neuralODE>
作者将其称为概率流 ODE，其推导和一般形式的推导见#ref(<appendix:preb-ode>)，#ref(<fig:overview-sdeode>)给出了一个统合的视角。考虑预测 $nabla log p(bx)$ 的分数模型，它也可视作是 #highlight[Neural ODE] 的一个例子。

#figure(
  image("image-8.png"),
  caption: [综观： 正向 SDE、逆向 SDE 个概率流 ODE],
  placement: top
)<fig:overview-sdeode>

在训练好分数模型 $bold(s)_(bold(theta))$ 后，将其替换掉#ref(<eq:neuralODE>)中的 $nabla log p(bx)$ 一项，得到
$
  dd bx = [bf(bx, t) + 1/2 g(t)^2 bold(s)_(bold(theta))(bx, t)] dd t = tilde(bf)_bold(theta)(bx, t) dd t.
$
我们可以依照下式得到 $p_0(bx)$：
$
  log p_0 (bx_0) = log p_T (bx_T) + int_0^T"div"[tilde(bf)_bold(theta)(bx, t)] dd t.
$
而散度项可以由 $EE_(p(bold(epsilon)))[bold(epsilon)^top nabla^2 tilde(bf)_bold(theta)(bx, t) bold(epsilon)]$ 估计，其中 $nabla^2$ 表示 $tilde(bold(f))$ 的第一项的 Jacobian 矩阵，$bold(epsilon)$ 服从标准多维正态分布 $cal(N)(bold(0), bold(I))$。采样时，我们从 $bx_T$ 出发，然后使用数值方法解上述 ODE，得到对应的 $bx_0$。

该方法的另一个优势是我们可以利用 #highlight[NeuralODE] 的一些技巧处理数据。例如我们可以输入图像 $bx_0$ 然后让其沿着概率流 ODE 得到其在隐空间中的表示 $bx_T$。我们能在隐空间做的事情就很多了，例如#highlight[图像编辑、插值、温度缩放]等等。此外，在理想条件下，由于前向 SDE 没有可训练的参数，该框架还可以达到输入数据在隐空间中的表示由输入数据本身唯一确定。

利用这一优势并搭配更好的神经网络模型和 sub-VP SDE，我们可以得到相比于经典算法（如 DDPM）更好的结果。

=== 预测-校正算法

#tab 本节介绍作者提出的综合求解的新方法，由于上述的一般解法和概率ODE同属于数值解法，而本节中的预测-校正算法将上述的各种数值解法统合在一个框架中，故本节位置与原文不同，我将其单列出来，以便于读者分辨不同模块的功能。

生成模型中的SDE和传统的SDE不同，生成模型的SDE中有新的信息 $bold(s)_bold(theta)$，这使得我们可以借助一些 Markov 链 Monte Carlo（MCMC）的方法。具体而言，每次执行上一节中的一般求解器迭代预测之后，可以利用所学到的分数模型的预测结果对其进行校正，这就是所谓的*预测-校正算法*。DDPM和SMLD的采样算法都可以视作预测-校正算法的特殊情形：前者的预测项为恒等映射，校正项为 Langevin MCMC 采样；后者的预测项为祖先采样，校正项为恒等映射。抽象地讲，预测项可以是任意求解逆向 SDE 的数值算法，校正项可以是任意基于分数的 MCMC 算法。整个预测-校正算法的轮廓如#ref(<alg:predict-correct>)所示。在不同架构下的测试结果如#ref(<tab:predict-correct>)所示。可见预测-校正算法拥有更好的FID分数。

#show: style-algorithm
#algorithm-figure(
    "Predictor-Corrector Algorithm",
    vstroke: .5pt + luma(200),
    {
      import algorithmic: *
      Procedure("Predictor-Corrector", none,
        [Some preparations],
        For([$i = N-1$ to $0$],
          [$bx_i <-$ Predictor$(bx_(i+1))$],
          For([$j = 1$ to $M$], [$bx_i <-$ Corretor$(bx_i)$])
        ),
        Return([$bx_0$])
      )
    }
)<alg:predict-correct>

#figure(
  table(
    stroke: none,
    image("image-7.png"),
  ),
  caption: [预测-校正算法 (PC) 和仅预测 (P)、仅校正算法 (C) 的性能比较]
)<tab:predict-correct>


=== 可控生成
=== 实验结果和讨论

#pagebreak()



// == Sliced Score Matching: A Scalable Approach to Density and Score Estimation #cite(<DBLP:paper-yang_song-sliced_score_matching>)

// - Yang Song, Sahaj Garg, Jiaxin Shi, Stefano Ermon 
// - https://arxiv.org/abs/1905.07088


// #pagebreak()

// == General E(2)-Equivariant Steerable CNNs #cite(<DBLP:paper-e2cnn>)

// - Maurice Weiler and Gabriele Cesa
// - https://arxiv.org/abs/1911.08251

// = 学习进度
// == 机器学习理论
// === Markov Chain Monte Carlo (MCMC)


== 随机过程

#h(2em)本周继续系统学习 Markov 链。

#pagebreak()

== 随机微分方程
=== #ito 链式法则和乘积法则的多维情形

=== 典型的随机微分方程
==== 股市模型

==== Brown 桥

==== Langevin 方程

==== Ornstein–Uhlenbeck (OU) 过程

==== 随机谐振子



=== 解的存在性和唯一性

=== 随机微分方程的数值模拟

#tab 考虑一个自守的 SDE 
$
  dd X = F(X) dd t + G(X) dd bold(W)
$
下面给出模拟它的三种方法。

==== Euler-Maruyama 法

Euler-Maruyama 法主要思想是 Brown 运动 $W(dot)$ 的增量独立性，即 $W(t + Delta t) - W(t) ~ cal(N)(0, 1)$，因此有
$
  X(t + Delta t) approx X(t) + F(X) Delta t + sqrt(Delta t) dot G(X) Z
$
其中 $Z ~ cal(N)(0, 1)$

==== Milstein 法

Milstein 法相比于 Euler-Maruyama 法更细，来源于对 $display(int_t^(t+Delta t) G(X)dd W)$ 更精确的估计。其形式为
$
  X(t + Delta t) approx& X(t) + F(X) Delta t + sqrt(Delta t) dot G(X) Z \ &+ 1/2 G(X) partial/(partial X)G(X)[Z^2t - (Delta t)^2]
$

其推导我还需要再看一下。

==== 随机 Runge-Kutta 法


#pagebreak()

== 量子力学初步


#pagebreak()

== Josephson 结


#pagebreak()

= 问题解决记录
== Typst 相关
=== 自定义图表标题位置和内容



#pagebreak()

= 下周计划
*论文阅读*
+ 生成模型
  - Score Matching 和 SDE（完结）
  - 薛定谔桥（精读）
  - DDIM（泛读）
// + 几何深度学习
//   - General $E(2)$ - Equivariant Steerable CNNs

*项目进度*
+ 使用神经网络学习生命游戏的演化动力学
  - 阅读等变CNN的代码
  - 尝试开始做单轨道训练样本的生成、训练和演化动力学统计
+ 耦合约瑟夫森结
  - 了解约瑟夫森结的基本知识
  - 浏览 matlab 数值模拟代码

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



== SMLD 的祖先采样<appendix:SMLD-ancestral>

令 $sigma_0^2 = 0$，执行和 DDPM 类似的操作，有
$
& quad q(bx_(t-1) | bx_t, bx_0) \
&= q(bx_t | bx_(t-1), bx_0) q(bx_(t-1) | bx_0) / q(bx_t | bx_0) \
&= q(bx_t | bx_(t-1)) q(bx_(t-1) | bx_0) / q(bx_t | bx_0) \
&prop exp lr(\{ -1/2 [ 
  norm(bx_t - bx_(t-1))^2/(sigma_t^2 - sigma_(t-1)^2) 
  + norm(bx_(t-1) - bx_(0))^2/(sigma_(t-1)^2) 
  - norm(bx_t - bx_(0))^2/(sigma_t^2) 
] \}) \
&= exp lr(\{ -1/2 [ 
  (
    norm(bx_t)^2 
    - 2 chevron.l bx_t, #text(red)[$bx_(t-1)$] chevron.r  
    + norm(#text(red)[$bx_(t-1)$])^2
  )/(sigma_t^2 - sigma_(t-1)^2) 
  + (
    norm(#text(red)[$bx_(t-1)$])^2 
    - 2 chevron.l #text(red)[$bx_(t-1)$], bx_0 chevron.r  
    + norm(bx_0)^2
  )/(sigma_(t-1)^2)  \
  & wide wide wide - (
    norm(bx_t)^2 
    - 2 chevron.l bx_t, bx_0 chevron.r  
    + norm(bx_0)^2
  )/(sigma_t^2) 
] \}) \
&= exp lr(\{ -1/2 [ 
  sigma_t^2/((sigma_t^2 - sigma_(t-1)^2)sigma_(t-1)^2) #text(red)[$norm(bx_(t-1))^2$] 
  - lr(
    2 lr(chevron.l sigma_(t-1)^2/(sigma_t^2) bx_t 
    + (1 - sigma_(t-1)^2/(sigma_(t)^2)) bx_0, 
    #text(red)[$bx_(t-1)$] chevron.r) ) 
  + const 
] \}) \
&prop cal(N)( sigma_(t-1)^2/(sigma_t^2) bx_t 
    + (1 - sigma_(t-1)^2/(sigma_(t)^2)) bx_0, ((sigma_t^2 - sigma_(t-1)^2)sigma_(t-1)^2)/sigma_t^2) \
$

== 概率流 ODE 推导<appendix:preb-ode>



