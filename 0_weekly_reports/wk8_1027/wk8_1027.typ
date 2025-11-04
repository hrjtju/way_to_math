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
#let bf = $bold(f)$
#let bF = $bold(F)$
#let bg = $bold(g)$
#let bG = $bold(G)$
#let bx = $bold(x)$
#let bX = $bold(X)$
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

#set math.equation(numbering: "(1)")
#set math.mat(delim: ("[", "]"), align: center)
#set heading(numbering: "1.")
#set math.cases(gap: 0.5em)


// metadata
#let wk_report_name = "2025年10月27日至11月2日周报"
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

#tab 本周将 `CNN-small` 的层数从三层降低至两层，在 $8$ epoch 内可以成功收敛，其权重待分析。

=== 下一步

#tab 目前将目标从提取显式的规则改为从已知部分规则和黑箱规则代理产生的少量数据，结合带有等变性的神经网络训练作为动力学仿真器。规则发现的具体流程是，从一些已知规则出发，首先根据一条有偏的轨道从一个规则空间中选取一系列的假设规则，在神经网络学习这条轨道后，再通过神经网络作为演化模拟器的演化统计结果，和原来的统计数据对比，以确定某些候选规则存在或不存在，从而逐步确定系统演化的真正动力学。

#pagebreak()
= 文献阅读

== Denoising Diffusion Probabilistic Models #cite(<DBLP:paper-DDPM>)
Jonathan Ho, Ajay Jain and Pieter Abbeel | https://arxiv.org/abs/2006.11239

本周把 DDPM 的剩余部分补完。

=== 实验
==== 预测目标和目标函数匹配程度

#tab 实验发现，预测 $tilde(bold(mu))$ 需要匹配未化简形式的加权期望损失；而预测噪声 $bold(epsilon)$ 需要匹配简化的损失函数。这两个搭配效果几乎一样好。

==== 去噪过程的协方差矩阵

#tab 如果将去噪过程的协方差矩阵变成科学系的对角形式，训练过程会变得不稳定，采样得到的图像质量也会下降。

=== DDPM 的信息论视角
==== 编码效率和失真度

#tab 模型在 CIFAR 上训练得到的对应于每一数据维度的信息比特数相差 $0.03$，这说明模型没有在训练集上过拟合。此外，再回到目标函数，已知有这样的分解：

$
  EE_q [underbrace(KL[q(bx_T|bx_0)\|p(bx_T)], L_T) + sum_(t=2)^T underbrace(KL[q(bx_(t-1)|bx_t, bx_0)\|pt(bx_(t-1)|bx_t)], L_(t-1)) underbrace( - log pt (bx_(0)|bx_1), L_0)]
$

注意 $L_1 + dots.c + L_(T-1)$ 中的每一项都是 KL 散度，分别描述神经网络参数化的 $p_(bold(theta))$ 和分布 $q$ 的区别，它是使用 $p_(bold(theta))$ 描述 $q$ 需要的额外编码开销（见附录），因此可将其看做生成过程中的*编码效率 (rate)*，这体现在相对于该体系中认为的最优逆向过程 $q$ 而言所需的额外编码量；而最后一项 $L_0$ 是一个负对数似然，它将 $bx_1$ 和 $bx_0$ 对齐比较，我们可将其看做从 $bx_T$ 经过降噪过程一路走来的结果相比于 $bx_0$ 的*失真度 (distortion)*。

==== 分步有损压缩的信道模型

#tab 我们可以将 DDPM 的去噪过程看作是发送端向接受端发送信息。接收端在接收前只知道 $p$，发送端可以同步逐步使用
$p_theta (bx_t|bx_(t+1))$ 编码服从分布 $q(bx_t|bx_(t+1), bx_0)$ 的数据 $bx_t$，然后接收端用 $p_theta$ 对其进行解码。如果发送端仅发送 $x_T$，接收端仅依靠此信息和 $p_theta$ 估计 $hat(bx)_0$，这样会产生较高的失真度。

#grid(
  columns: (55%, 44%),
  column-gutter: 2%,
  align: (horizon, horizon),
  text(size: 10pt)[
    #show: style-algorithm
    #algorithm-figure(
        [Sending $bx_0$],
        vstroke: .5pt + luma(200),
        {
          import algorithmic: *
          Procedure([Sending], [$bx_0$],
            [Sendin $bx_T ~ q(bx_T|bx_0)$ using $p(bx_T)$],
            For(
              [$t = T-1, ..., 2, 1$ do],
              [Send $bx_t ~ q(bx_t|bx_(t+1), bx_0)$ using $p_theta (bx_t|bx_(t+1))$]
            ),
            [Send $bx_0$ using $p_theta (bx_0|bx_1)$]
          )
        }
    )
  ],
  text(size: 10pt)[#show: style-algorithm
  #algorithm-figure(
      "Recieving",
      vstroke: .5pt + luma(200),
      {
        import algorithmic: *
        Procedure("Recieving", none,
          [Receive $bx_T$ using $p(bx_T)$],
            For(
              [$t = T-1, ..., 1, 0$ do],
              [Receive $bx_t$ using $p_theta (bx_t|bx_(t+1))$]
            ),
            Return([$bx_0$])
        )
      }
  )]
)

每个时刻 $t$，接收端都会计算估计得到的 $hat(bx)_0$ 和真实值的均方损失作为失真率，而记录自传输开始至该时刻接收器获得的所有每维度比特数为码率：即 $H(bx_t) + KL[q(bx_(t-1)|bx_t, bx_0)\|pt(bx_(t-1)|bx_t)]$，画出的码率-失真率曲线可以看出，大量的信息被分配至肉眼难以看见的细节中：

#figure(
  image("image.png"),
  caption: [信道模型中随迭代次数接收器对 $bx_0$ 预测的失真率和码率的关系]
)

==== 分步生成

作者还用各部得到的 $bx_t$ 直接预测 $bx_0$，得到的结果由下图所示。可见随着 $t$ 的减小，预测得到的 $hat(bx)_0$ 逐步先显现总体特征，再逐步丰富局部细节。

#figure(
  image("image-1.png"),
  caption: [CIFAR10 数据集上随 $t$ 减小从 $bx_t$ 预测得到的 $hat(bx)_0$ 结果，从左至右 $t$ 逐渐降低]
)

如果使模型都从某个共用的 $bx_t$ 出发执行降噪过程，越大的 $t$ 出发降噪得到的结果差异越大；越小的 $t$ 出发得到的降噪结果越相似。

#figure(
  image("image-2.png"),
  caption: [CelebA-HQ 数据集从同一个 $bx_t$（每个子图的右下角）出发降噪得到的三个结果]
)

=== DDPM 和自回归生成模型的比较

#tab 最后讨论 DDPM 的降噪过程和序贯生成的自回归生成模型（例如 ChatGPT 的模式）的相似性和区别。我们考虑这样的场景（暂时抛开加噪过程是加 Gauss 过程这件事）：生成图像的尺寸是 $N times N$，所谓的“加噪过程”和“降噪过程”的步数为 $N^2$，$q(bx_t | bx_(t-1))$ 是一个离散的 dirac-delta 点质量，它做的事情是将图像从左至右，从上至下的第 $t$ 个像素变成空白；相应地，$q(bx_(t-1))$ 做的事情就是将第 $t$ 个像素恢复成原来的颜色。这样经过一整条“加噪过程”后，图像就变成了完全空白；再经过理想的“降噪过程”后图像又被逐像素涂色成了原来的样子。

这是我们假设神经网络建模的 $p_theta$，足够强大——它可以学习到上面假设中 $q(bx_(t-1)|bx_t)$ 的分布，那么这俨然成了自回归模型。然而实践过程中，*DDPM 选择的使用 Gauss 噪声加噪、去噪的过程，可以看作是更为广义的自回归生成模型，它按照图像中的某种空间语义token按照从整体至局部的顺序逐渐生成图像*。相比于添加更强归纳偏置的一般自回归生成模型——它们都假设后生成的区域依赖于先生成的区域，而这在图像生成上没有道理！——DDPM拥有更高的灵活性和更少的归纳偏置，这也解释了其优异的生成能力。

=== 隐空间插值

#tab 最后，作者将不同图像加噪过程中某个 $t$（例如 $t = 500$）得到的 $bx_t$ 和 $hat(bx)_t$ 做凸组合，再做去噪过程，得到了原图 $bx_0$ 和 $hat(bx)_0$ 之间的顺滑渐变。

#figure(
  image("image-3.png"),
  caption: [CelebA-HQ 数据集中的插值实验]
)

#pagebreak()

// == Score-based generative modeling through SDE #cite(<DBLP:paper-yang_song-score_based_generative_modeling_sde>)

// Yang Song et al. | https://arxiv.org/abs/2011.13456

// 本文从 SDE 的视角统一了先前的 Langevin 退火的 score matching 方法 SMLD#cite(<DBLP:paper-SMLD>)和原版扩散模型 DDPM#cite(<DBLP:paper-DDPM>)，从它们的共同点出发，先将加噪动力学连续化变成 SDE，再利用已有的逆向SDE 解析解，得到与加噪 SDE 相反演化方向的逆向 SDE；最后从 SMLD 和 DDPM 逆向求解的数值算法提出了新的预测-校正方法和概率流 ODE 方法。可以说本文所提出的是随机采样的改进，神经网络扮演的角色仅仅为 $nabla log p(bx)$ 的逼近器，它不是本文的重点。

// === Score matching: SMLD 和 DDPM

// #tab Score-based 生成模型的核心思想是通过学习 $bs_theta (bx, t)$，让其逼近 $nabla log p(bx)$，然后再根据 Langevin 采样，生成来自未知分布 $p(bx)$ 的样本。在 SMLD 中，由于数据分布未知而无法计算的目标项 $nabla log p(bx)$ 是通过加噪解决的，即考虑高斯核 $q_sigma (hat(bx)|bx) ~ cal(N)(bold(0), sigma bold(I))$，加噪后变量对应的目标函数可以写成
// $
//   J(bold(theta)) 
//   &= 1/2 EE_(bx ~ p(bx), hat(bx) ~ q_sigma (hat(bx)|bx)) [ norm(bs_theta (hat(bx), sigma) - nabla_(hat(bx)) log q_sigma (hat(bx)|bx) )_2^2]  \ 
//   &= 1/2 EE_(bx ~ p(bx), hat(bx) ~ q_sigma (hat(bx)|bx)) [ norm(bs_theta (hat(bx), sigma) + sigma/epsilon )_2^2] & hat(bx) = bx + bold(epsilon), bold(epsilon) ~ cal(N)(bold(0), sigma bold(I))
// $
// 可以看出，采用加噪技巧的 SMLD 中，分数函数实际上在预测噪声，这就与 DDPM 建立了联系。即使 DDPM 论文中并未显式地做出 score matching，但我们也可以认为 DDPM 是在训练一个分数函数。 

// 现在来看二者的去噪动力学。SMLD 使用 Langevin 采样的方法，对于某个固定的噪声水平 $sigma_i$，有
// $
//   bx^(m)_i = bx^(m)_(i-1) + alpha_i bs_theta (bx^(m)_(i-1), sigma_i) + sqrt(2 alpha_i) bold(z)^(m)_i, wide bold(z)^(m)_i ~ cal(N)(bold(0), bold(I)), m = 1, ..., M
// $

// 其中 $i = 1, ..., N$。DDPM 的版本和 SMLD 相似，它通过两个 Gauss 分布之间 KL 散度的解析解，可以求出逆向条件分布 $q(bx_(i-1)|bx_i, bx_0)$ 的均值 $tilde(bold(mu))_t$ 和方差 $tilde(beta)_t$，最后得到去噪公式

// $
//   bx_(i-1) = 1/sqrt(1 - beta_i) (bx_i - beta_i bs_theta (bx_i, sigma_i)) + sqrt(beta_i) bold(z)_i, wide bold(z)_i ~ cal(N)(bold(0), bold(I))
// $

// === Score matching 的 SDE 视角
// ==== 加噪过程

// #tab 我们考虑将 SMLD 和 DDPM 的加噪动力学连续化为 $dd bx = bold(f)(bx, t) dd t + g(t) dd bold(w)$ 的形式，其中 $bold(w)$ 是 $n$ 维 Brown 运动，$bold(f)(bx, t)$ 是漂移项，$g(t)$ 是扩散项。对于 SMLD，有下面的改写：
// $
//   bx_i &= bx_(i-1) + sqrt(sigma_i^2 - sigma_(i-1)^2) bold(z)_i \
//   bx(t + Delta t) - bx(t) &= sqrt(sigma(t+Delta t)^2 - sigma(t)^2)z(t) \
//   bx(t + Delta t) - bx(t) &= sqrt( (dd [sigma(t)]^2)/(dd t) ) sqrt(Delta t) z(t) wide & "用" (dd [sigma(t)]^2)/(dd t) "对" sigma_i^2 - sigma_(i-1)^2 "做一阶近似" \
//   dd bx &= sqrt( (dd [sigma(t)]^2)/(dd t) ) dd bold(w) & dd bold(w) ~ (dd t)^(1/2)bold(epsilon)
// $
// 类似地，对 DDPM 的加噪过程，也有类似的改写：
// $
//   bx_i &= sqrt(1 - beta_i) bx_(i-1) + sqrt(beta_i) bold(z)_(i-1) \ 
// $
// $
//   bx(t + Delta t) &= sqrt(1 - beta(t + Delta t) Delta t) bx(t) + sqrt(beta(t + Delta t) Delta t) bold(z)(t) wide quad & #block(text([令 $beta(i"/"N)=beta_i$ \ 其中 $1"/"N$ 是噪声步长]))\
//   &approx (1 - 1/2 beta(t + Delta t) Delta t) bx(t) + sqrt(beta(t + Delta t)) (Delta t)^(1/2) bold(z)(t) & "Taylor 展开到一阶"\
//   &approx bx(t) - 1/2 beta(t) bx(t) Delta t + sqrt(beta(t)) (Delta t)^(1/2) bold(z)(t) \
//   dd bx &= -1/2 beta(t) bx dd t + sqrt(beta(t)) dd bold(w) & dd bold(w) ~ (dd t)^(1/2) bold(epsilon)
// $
// 如果 DDPM 中的噪声表不是线性变化的，那将得到一个形式类似，但 $bold(f)$ 和 $g$ 不同的 SDE。上文中，SMLD 的加噪 SDE 被称为是 *variance exploding (VE)* 的，而 DDPM 的加噪 SDE 被称为是 *variance preserving (VP)* 的。作者还构造出一种 *sub-VP SDE*，加噪 SDE 为
// $
//   dd bx = -1/2 beta(t) bx dd t + underbrace(sqrt(beta(t) (1 - e^(- 2 int_0^t beta(s) dd s))), sqrt(macron(beta)(t))) dd bold(w).
// $
// 其方差被 VP-SDE 的方差控制。以上三种 SDE 的一维特殊情形的方差推导可见附录，对特殊情形的讨论可以很容易看书这三种 SDE 叫做 VE，VP 以及 sub-VP SDE 的原因。

// === 求解逆向 SDE

// Anderson 给出了上节中一般形式之加噪 SDE 的逆向 SDE，其形式为
// $
//   dd bx = [ - bold(f)(bx, t) + g^2(t) nabla_(bx) log p_t (bx) ] dd t + g(t) dd bold(macron(w))  
// $
// 其中 $macron(bold(w))$ 是一个倒流的 Brown 运动，$dd t$ 是倒流的无穷小时间间隔。由于 $bold(f)$ 和 $g$ 已知，而 $nabla_(bx) log p_t (bx)$ 由神经网络估计。在 SMLD 中其总目标函数是

// $
//   bold(theta)^* = argmin_(bold(theta)) sum_(i=1)^N sigma_i^2 EE_(bx ~ p(bx), hat(bx) ~ q_sigma (hat(bx)|bx)) [ norm(bs_theta (hat(bx), sigma) - nabla_(hat(bx)) log q_sigma (hat(bx)|bx) )_2^2]
// $

// 将其拓展至连续的情形，目标函数就变为

// $
//   bold(theta)^* = argmin_(bold(theta)) EE_t [lambda(t) EE_(bx(0), bx(t)|bx(0)) [ norm(bs_theta (bx(t), t) - nabla_(bx(t)) log p_(0, t) (bx(t)|bx(0)) )_2^2]]
// $
// 其中 $lambda(t)$ 是正的权重函数，$t$ 在 $[0, T]$ 上均匀采样。可以取 $lambda prop EE [ norm(nabla_(hat(bx)) log p_(0, t) (bx(t)|bx(0)) )_2^2]$，使得每个 $t$ 下的损失项对总损失的平均贡献相同。剩余的问题来自于如何求解 $p_(0, t) (bx(t)|bx(0))$，或者说转移核。*如果 $bold(f)(dot, t)$ 是仿射函数，转移核是 Gauss 分布的密度函数，其参数均值和方差拥有闭式解；而对于一般的 SDE，需要求解 Kolmogorov 前向方程才能得到 $p_(0, t) (bx(t)|bx(0))$。*但如果不使用加噪技巧，而是使用*分片分数匹配*以计算目标函数，就可以规避计算转移核的困难（见附录）。

// 剩下的问题就变成了如何用这个逆向 SDE 采样。

// ==== 一般的 SDE 求解器




// ==== 预测-校正方法
// ==== 概率流 ODE
// === 可控生成
// === 实验结果和讨论
// === 依然存在的疑惑

// - Variance Exploding SDE 和 Variance Preserving SDE 以及 sub-VP SDE 为什么叫这些名字，换言之，如何证明它们分别是 variance exploding 和 variance preserving 的？
// - 概率流 ODE 是否只是一个求解逆向 SDE 的方法？它和文中提出的 SDE 求解方法之间有什么不同？

// #pagebreak()

== Mean Flows for One-step Generative Modeling #cite(<DBLP:paper-mean_flow>)

Zhengyang Geng et al. | https://doi.org/10.48550/arXiv.2505.13447

=== 流匹配
==== 流轨迹

#tab *流轨迹*是指从先验分布 $p_"prior"$ 连续地到数据分布 $p_"data"$ 的“流动”轨迹：对于 $x ~ p_"data"(x)$ 和与之配对的某个 $epsilon ~ p_"prior"(epsilon)$，它们之间随时间 $t$ 相互演变的轨迹可以是 
$
  z_t = a_t x + b_t epsilon
$
其中 $a_t$ 和 $b_t$ 都是预先定义好的确定函数。常用的流轨迹是 $z_t = (1-t) x + t epsilon$，当时间 $t$ 从 $0$ 逐渐增加至 $1$ 时，数据点从 $x$ 逐渐流动至 $epsilon$，反之亦然。

对给定的流轨迹和 $p_"data"$ 中的数据点 $x$，定义时刻 $t$ 时的*条件流场* $v_t$ 为 $z_t$ 对时间的导数：
$
  v_t = a'_t x + b'_t epsilon
$
在上节的常用例子 $z_t = (1-t)x + t epsilon$ 中，流速为 $v_t = epsilon - x$。

#figure(
  image("image-4.png", width: 55%),
  caption: [条件流场（左）和边缘流场（右）]
)

=== 流匹配

条件流速依赖于 $x$ ，如果将 $x$ 平均掉，就得到*边缘流场*
$
  v(z_t, t) := EE_(p_t (v_t |z_t))[v_t]
$
它考虑的是指轨迹在 $t$ 时刻交汇于 $v_t$ 的不同情况速度的加权平均。考虑边缘流场的意义在于其普适性：由于其不依赖于 $x$ 的选取，我们可以直接从先验分布中采样一个 $epsilon$，然后求解下面的初值问题：
$
  dd/(dd t) z_t &= v(z_t, t), #h(2em) t in [0, 1]\
  z_1 &= epsilon
$
显然，要得到 $z_r$，只需要做积分：
$
  z_r = z_1 - int_r^1 v(z_r, tau) dd tau
$
然而边缘流场我们不知道，因此就造一个神经网络来学它：最小化下面的损失
$
  J_"FM" (theta) = EE_(t, p_t (z_t)) norm(v_theta (z_t, t) - v(z_t, t))^2.
$
不过由于 $v(z_t, t)$ 不依赖 $x$，我们在不知道 $p_"data"$ 的时候根本没法算它，所以实际情况下我们借用条件流场采样到的值，然后对起点 $x$、终点 $epsilon$ 和时间 $t$ 做平均，得到下面条件流场版本的边缘流场模型的目标：
$
  J_"CFM" (theta) = EE_(t, x, epsilon) norm(v_theta (z_t, t) - v(z_t | x))^2
$
二者已被证明是等价的。类似分数匹配 (score matching)，这里的网络尝试拟合 $v(z_t, t)$，因此谓之*流匹配*。有了神经网络的流模型后，采样就变成了求上面那个 ODE 的数值解。

=== 平均流

本文的主要贡献——流模型使用了一个更一般化的*平均流*。顾名思义，它是从 $tau$ 时刻起在 $t$ 时刻汇聚于 $z_t$ 之边缘（平均化了的）轨迹上流速的平均：
$
  u(z_t, r, t) := 1/(t-r) int_r^t v(z_tau, tau) dd tau
$
若固定 $t$，令 $r -> t$，可得 $u(z_t, r, t) -> v(z_t, t)$，因此它和原来的边缘流场的定义也是相容的。除此之外，由于位移可以写为平均速度乘时间，平均流还有另一层次的一致性：跨一大步 $[r, t]$ 相当于跨两小步：$[r, s], [s, t]$。根据平均速度定义不难验证
$
  (t - r)u(z_t, r, t) = (s - r)u(z_s, r, s) + (t - s)u(z_t, s, t).
$

#tab 为什么要引入平均流？上文中，我们规定 $0$ 时刻对应数据分布中的元素 $x$，$1$ 时刻对应先验分布中的元素 $epsilon$，如果令 $r = 0$，$t = 1$，就得到 
$
  u(epsilon, 0, 1) = 1 dot u(z_1, 0, 1) = int_0^1 v(z_tau, tau) dd tau = z_1 - z_0 = epsilon - z_0
$
因此可以进行*一步生成 (single Number of Function Evaluations, single-NFE) *：
$
  x = u(epsilon, 0, 1).
$

=== 平均流的一些性质

#tab 如果假设 $r$ 是一个与 $t$ 不相关的独立常数，将时间 $r$ 到 $t$ 的总位移对 $t$ 求导，得到
$
  dd/(dd t) (t - r)u(z_t, r, t) = dd/(dd t) int_r^t v(z_tau, tau) dd tau ==> u(z_t, r, t) = v(z_t, t) - (t - r) dd/(dd t) u(z_r, r, t).
$
对于最后一项，有
$
  dd/(dd t) u &= (dd z_t)/(dd t) partial_z u + (dd r)/(dd t) partial_r u + (dd t)/(dd t) partial_t u \
  &= v(z_t, t) partial_z u + partial_t u
$
可以看出这是 $u$ 的 Jacobian 和向量 $[v, 0, 1]^top$ 的乘积，它可以通过 `torch.func.jvp` 方便实现。有了这两个等式，联立后可以得到
$
  u(z_t, r, t) = v(z_t, t) - (t - r) [v(z_t, t) partial_z u + partial_t u].
$


=== 训练

我们发现直接拿着上面的式子当匹配的目标还是不行，因为等式右边还存在着 $u$。因此我们可以考虑*自助法 (bootstrapping)*，即在等式右边使用 $u_theta$ 作为自助法的目标：
$
  J(theta) =& EE[norm(u_theta(z_t, r, t) - "StopGrad"(u_"target"))_2^2]\ #v(2em)
  "in which"& u_"target" = v(z_t, t) - (t - r) [v(z_t, t) partial_z u_theta + partial_t u_theta]
$
为了以免让优化过程变得复杂，我们需要停止目标项中的梯度传播，即使用 $"StopGrad"(dot)$ 操作。特别地，如果 $r equiv t$，上面的平均流匹配就退化成了一般形式的流匹配。

=== 带引导的平均流

#tab 有趣的是，对于条件生成，本文对其的处理和前文中的流匹配极其相似。其核心方法是*无分类器引导 (Classifier-Free Guidance, CFG)*。给定类别 $c$，定义一个*实况场 (ground-truth field)* $v^"cfg"$ 为
$
  v^"cfg" (z_t, t | c) :=& omega v(z_t, t | c) + (1 - omega) v(z_t, t)\ #v(2em)
  "in which"& v(z_t, t | c) := EE_(p_t (v_t |z_t, c))[v_t]; \
  & v(z_t, t) := EE_c [v(z_t, t | c)]
$
它是类条件速度场和类边缘速度场的线性组合。如果引入 $v^"cfg"$ 对应的平均流 $u^"cfg"$，根据上面的结果，依然有
$
  u(z_t, r, t) = v(z_t, t) - (t - r) dd/(dd t) u(z_r, r, t)
$
此时稍加一些推导，$v^"cfg"$ 可重写为
$
  v^"cfg" (z_t, t | c) :=& omega v(z_t, t | c) + (1 - omega) u^"cfg" (z_t, t, t).
$

这样情境下训练的目标和先前的目标大同小异：
$
  J (theta) =& EE norm(u_theta^"cfg" (z_t, r, t | c) - "StopGrad"(u_"target"))^2\ #v(2em)
  "in which"& u_"target" = tilde(v) - (t - r) [tilde(v) partial_z u_theta^"cfg" + partial_t u_theta^"cfg"]\
  & tilde(v) = omega v_t + (1-omega) u_theta^"cfg"(z_t, t, t)
$
可见当 $omega=1$ 时，上述目标退化为先前的无类别生成的目标。在训练带有类别引导的目标时，我们在数据中以一定概率丢弃类别标签，以同时训练类条件和无类条件的版本。其生成流程也和先前一样简单：如果不指定类别 $c$，那么生成的结果是无类别标签的，如果需要生成某一类的结果，只需在参数中添加所需的类别 $c$。

=== 实验 

#tab 消融实验发现
+ 采用平均流而不是流匹配一般的配置的训练结果更好
+ `jvp` 模块中参与乘法的向量必须要是正确的
+ 对于平均流的函数形式，$u(z_t, t, t-r)$ 相比于 $u(z_t, t, r)$ 更好，但先前的推导需要修正
+ $t$，$r$ 自特定参数的对数正态分布中采样效果更好

#figure(
  image("image-5.png"),
  caption: [Mean Flows 模型在 ImageNet $256 times 256$ 上训练后一步生成的消融实验]
)

得益于其一步生成的快速和低成本，Mean Flow 模型可以以更低的计算代价达到个更好的效果：

#figure(
  image("image-6.png", width: 50%),
  caption: [不同体量的 Mean Flows 模型在 ImageNet $256 times 256$ 上与其他一步生成模型的比较]
)

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

#h(2em)本周开始系统学习 Markov 链。

== 随机微分方程
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

== 信息论
=== 熵

#tab 给定一个分布 $p(x)$，它的*熵*定义为
$
  H(p) := EE_p [log p(x)] = int p(x) log p(x) dd x
$
它在信息论中的意义分布 $p$ 下编码 $x$ 所需的最小平均比特数

=== 交叉熵

#tab 类似地，给定分布 $p(x)$ 和 $q(x)$，其互信息 $H(p, q)$ 定义为
$
  H(p, q) := EE_p [log q(x)] = int p(x) log q(x) dd x
$
它表示使用基于分布 $q$ 的最短编码方案编码服从分布 $p$ 的数据的平均比特数

=== KL 散度

#tab KL 散度一般用于描述分布之间的差异，给定分布 $p(x)$ 和 $q(x)$，有 
$
  KL[p || q] = EE_p [log p(x)/q(x)] = H(p, q) - H(p)
$
它表示使用基于分布 $q$ 的最短编码方案编码服从分布 $p$ 的数据，相比于使用 $p$ 的最短编码方式造成的平均额外的比特数开销。

// === 后验熵
// #tab 



#pagebreak()
// = 问题解决记录

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

// #pagebreak()

// = 附录
// == 一维特殊情形的 VE-SDE、VP-SDE 和 sub-VP SDE 方差推导

// 可以这样直观理解：假设 $x$ 都是标量随机过程，对 SMLD 的加噪过程，有
// $
//   "Var"[x(t)]
//   &="Var"[x(t)-x(0)]\
//   &="Var" [int_0^t sqrt(2 sigma sigma') dd w ] = int_0^t 2 sigma sigma' dd tau & #ito "等式"\
//   &= sigma^2(t) - sigma^2(0) & "换元积分法" 
// $
// 对 DDPM 的加噪过程，有一阶矩：
// $
//   dd EE[x(t)] 
//   &= EE[dd x(t)] = -1/2 beta(t) EE[x(t)] dd t\
//   (dd EE[x(t)])/(EE[x(t)]) &= -1/2 beta(t) dd t & "分离变量"\
//   log(|EE[x(t)]|) &= -1/2 int_0^t beta(t) dd t + C ==> |EE[x(t)]| = C' e^(-1/2 int_0^t beta(t) dd t) & "积分"\
//   EE[x(t)] &= e^(-1/2 int_0^t beta(t) dd t) EE[x(0)] & "假设"EE[x(0)] gt.slant 0\
// $
// 二阶矩：
// $
//   dd EE[x^2(t)] 
//   &= EE[dd x^2(t)] = EE[2x(t) dd x(t) + beta(t) dd t] wide & #ito "乘积法则"\
//   &= EE[x(t) (2 sqrt(beta(t)) dd w) - beta(t) x(t) dd t) + beta(t) dd t]\
//   &= EE[ 2 x(t) sqrt(beta(t)) dd w] - beta(t) EE[x^2(t)] dd t + beta(t) dd t\
//   EE[int_0^t dd x^2(t)] 
//   &= underbrace(EE[ 2 int_0^t x(t) sqrt(beta(t)) dd w], 0) - int_0^t beta(tau) EE[x^2(tau)] dd tau + int_0^t beta(tau) dd tau  & "期望的线性性"\
//   dd EE[x^2(t)] &= - beta(t) EE[x^2(t)] dd t + beta(t) dd t & "换回随机微分"\
//   (dd EE[x^2(t)])/(1- EE[x^2(t)]) &= beta(t) dd t & "分离变量"\
//   - log(|1 - EE[x^2(t)]|) &= int_0^t beta(t) dd t + C ==> |1-EE[x^2(t)]| = C' e^(- int_0^t beta(t) dd t) & "积分"\
//   EE[x^2(t)] &= 1 - e^(- int_0^t beta(t) dd t) (1 - EE[x^2(0)]) & "假设"1-EE[x^2(t)] gt.slant 0\
// $
// 因此方差为
// $
//   "Var"[x(t)] 
//   &= EE[x^2(t)] - (EE[x(t)])^2 \
//   &= 1 - e^(- int_0^t beta(t) dd t) (1 - EE[x^2(0)]) - e^(- int_0^t beta(t) dd t) (EE[x(0)])^2 \
//   &= 1 + e^(- int_0^t beta(t) dd t) ("Var"[x(0)] - 1) 
// $
// 因此当 $"Var"[x(0)] = 1$ 时，有 $x(t)$ 的方差恒为 1。

// 对于 sub-VP SDE，套用上面的结果，并记 $B(t) = int_0^t beta(s) dd s$，容易得到
// $
//   EE[x(t)] 
//   &= e^(-1/2 int_0^t beta(t) dd t) EE[x(0)]\
//   EE[x^2(t)]
//   &= e^(-int_0^t beta(s) dd s) [int_0^t macron(beta)(tau) e^(int_0^tau beta(s) dd s) dd tau + EE[x^2(0)]] \
//   &= e^(-int_0^t beta(s) dd s) [int_0^t beta(tau) (1 - e^(- 2 int_0^tau beta(s) dd s)) e^(int_0^tau beta(s) dd s) dd tau + EE[x^2(0)]] \
//   &= e^(-B(t))EE[x^2(0)]] +  e^(-B(t))[int_0^t B'(tau) (1 - e^(- 2 B(tau))) e^(B(tau)) dd tau ]\
//   &= 1 + e^(-2B(t)) + e^(-B(t))(EE[x^2(0)]] - 2)\ 
// $ 
// $
//   "Var"[x(t)]
//   &= EE[x^2(t)] - (EE[x(t)])^2 \
//   &= 1 + e^(-2B(t)) + e^(-B(t))(EE[x^2(0)]] - 2) - e^(- B(t)) (EE[x(0)])^2 \
//   &= 1 + e^(- B(t)) ("Var"[x(0)] - 2) + e^(-2 B(t)) \
//   &= 1 + e^(- int_0^t beta(s) dd s) ("Var"[x(0)] - 2) + e^(-2 int_0^t beta(s) dd s)\
// $
// 一些简单的推导，可以得出 sub-VP SDE 的方差小于等于 VP-SDE 的方差。

// == 逆向 SDE 推导




