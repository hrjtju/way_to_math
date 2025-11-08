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

== Denoising Diffusion Probabilistic Models
Jonathan Ho, Ajay Jain and Pieter Abbeel | https://arxiv.org/abs/2006.11239

原论文中的推导跳过了大量细节，通过这些细节和来龙去脉花了较长时间，留下实验、结论、讨论和附录部分下周完成。

=== 综观

#h(2em)不同于 VAE 等等的一个隐变量空间的生成模型，DDPM 尝试用多步编码/解码，或是加噪/去噪方式生成数据。它的每一步可以视为是一个去噪自编码器（Denoising Autoencoder），而其采样（生成）过程通过与加噪过程的 Markov 链“共用”，并通过预测加噪过程中的随机噪声实现加噪过程的 “反转”，并添上 Langevin 动力学的随机噪声实现对分布的采样。与正常的 Langevin 动力学不同的是，从第 $t$ 步到 $t-1$ 步，我们认为其在尝试适配一个分布 $p(bx_(t-1)|bx_t)$ 过程中，只做了一次 Langevin 动力学的迭代。

#figure(
  image("image.png", width: 80%),
  caption: "图 1. DDPM 的工作流程示意图"
)

=== 加噪过程的建模


#h(2em)首先对于加噪过程，我们有一列噪声强度 $\{beta_t\}_(t=0)^T$，然后按照 $q(bx_t|bx_(t-1)) ~ N(bx_(t-1) | beta_t bold(I))$ 来进行：
$
bx_t 
&= sqrt(1 - beta_t) bx_(t-1) + sqrt(beta_t) bold(epsilon)_t, #h(1em) & bold(epsilon)_t ~ N(bold(0), bold(I)) \
&= sqrt(1 - beta_t) [ sqrt(1 - beta_(t-1)) bx_(t-2) + sqrt(beta_(t-1)) bold(epsilon)_(t-1) ] + sqrt(beta_t) bold(epsilon)_t,  wide & bold(epsilon)_(t-1) ~ N(bold(0), bold(I)) \
&= sqrt((1 - beta_t)(1 - beta_(t-1))) bx_(t-2) + sqrt((1 - beta_t) beta_(t-1)) bold(epsilon)_(t-1) + sqrt(beta_t) bold(epsilon)_t \
&= sqrt((1 - beta_t)(1 - beta_(t-1))) bx_(t-2) + sqrt((1 - beta_t) beta_(t-1) - (1 - beta_t) + 1) macron(bold(epsilon))_t, & macron(bold(epsilon))_t ~ N(bold(0), bold(I))\
&= sqrt((1 - beta_t)(1 - beta_(t-1))) bx_(t-2) + sqrt(1 - (1 - beta_t)(1 - beta_(t-1))) macron(bold(epsilon))_t \
& quad dots.v\
&= sqrt(product_(s=1)^t (1 - beta_s)) bx_0 + sqrt(1 - product_(s=1)^t (1 - beta_s)) macron(bold(epsilon))_t = sqrt(macron(alpha)_t) bx_0 + sqrt(1 - macron(alpha)_t) macron(bold(epsilon))_t
$

这给出了加噪过程中每一步得到的 $bx_t$ 更加便捷的表示，对后续的推导有帮助。

=== 优化目标


#h(2em)由于我们的目标是从随机噪声 $bx_0$ 生成图像，即 $bx_T$，因此需要最大化的项是 $EE_(pt)[-log pt(bx_0)]$。因此我们需要继续用变分推断的技巧，逐步将其转化为可以计算得到的项。首先我们有下面的上界

$
&quad EE_(pt)[-log pt(bx_0)] \
&= EE_(pt)[-log pt(bx_0) int q(bx_(1:T)|bx_0) dd bx_(1:T)] = EE_(pt)[-int q(bx_(1:T)|bx_0)log pt(bx_0)  dd bx_(1:T)] \
&= - EE_(pt(bx_0),  q(bx_(1:T)|bx_0))[
  log (pt(bx_0) pt(bx_(1:T)|bx_0)) / (pt(bx_(1:T)|bx_0))
] = - EE_(pt(bx_0),  q(bx_(1:T)|bx_0))[
  log (#text(red)[$pt(bx_(0:T))$] q(bx_(1:T)|bx_0)) / (pt(bx_(1:T)|bx_0) #text(red)[$q(bx_(1:T)|bx_0)$])
]\
$ 
$
&= - EE_(pt(bx_0),  q(bx_(1:T)|bx_0))[
  log #text(red)[$pt(bx_(0:T))  / q(bx_(1:T)|bx_0)$]
] - EE_(pt(bx_0),  q(bx_(1:T)|bx_0))[
  log (q(bx_(1:T)|bx_0)) / (pt(bx_(1:T)|bx_0))
]\
&= - EE_(pt(bx_0),  q(bx_(1:T)|bx_0))[
  log #text(red)[$pt(bx_(0:T))  / q(bx_(1:T)|bx_0)$]
] - underbrace(EE_(pt(bx_0))[KL[q(bx_(1:T)|bx_0) \| pt(bx_(1:T)|bx_0)]], gt.slant 0) \
&lt.slant EE_q [
  log  q(bx_(1:T)|bx_0) /(pt(bx_(0:T))  )
] =: L
$

上式中的 $bx_(1:T)$ 不好处理，我们可以将打包的变量拆开，最后可以将其写成若干项 KL 散度之和，其中两项对应着 Markov 链头和尾，剩余的对应加噪过程的中间状态。

$
L 
&= EE_q [ log q(bx_(1:T)|bx_0) /(pt(bx_(0:T)) ) ] \
&= EE_q [ log (q(bx_T|bx_0) prod_(t=2)^T q(bx_(t-1)|bx_t, bx_0)) / (pt(bx_T) prod_(t=1)^T pt(bx_(t-1)|bx_t)) ] \
&= EE_q [ log (q(bx_T|bx_0) ) / (p(bx_T)) + sum_(t=2)^T (q(bx_(t-1)|bx_t, bx_0))/(pt(bx_(t-1)|bx_t)) + log 1/(pt (bx_(0)|bx_1))] \
&= EE_q [KL[q(bx_T|bx_0)\|p(bx_T)] + sum_(t=2)^T KL[q(bx_(t-1)|bx_t, bx_0)\|pt(bx_(t-1)|bx_t)] - log pt (bx_(0)|bx_1)] \
&= EE_q [L_T + sum_(t=2)^T L_(t-1) - L_0].
$

显然，得到的结果符合我们的预期，我们需要训练一个带参分布 $p_theta (bx_(t-1) | bx_t, bx_0)$，并让其与 $q(bx_(t-1) | bx_t, bx_0)$ 对齐。后者是一个 Gauss 分布，使用 Bayes 公式不难得到它的分布参数

$
& quad q(bx_(t-1) | bx_t, bx_0) \
&= q(bx_t | bx_(t-1), bx_0) q(bx_(t-1) | bx_0) / q(bx_t | bx_0) \
&= q(bx_t | bx_(t-1)) q(bx_(t-1) | bx_0) / q(bx_t | bx_0) \
&prop exp lr(\{ -1/2 [ 
  norm(bx_t - sqrt(alpha_t) bx_(t-1))^2/(beta_t) 
  + norm(bx_(t-1) - sqrt(macron(alpha)_(t-1)) bx_(0))^2/(1 - macron(alpha)_(t-1)) 
  - norm(bx_t - sqrt(macron(alpha)_(t)) bx_(0))^2/(1 - macron(alpha)_(t)) 
] \}) \
&= exp lr(\{ -1/2 [ 
  (
    norm(bx_t)^2 
    - sqrt(alpha_t) angle.l bx_t, #text(red)[$bx_(t-1)$] angle.r  
    + alpha_t norm(#text(red)[$bx_(t-1)$])^2
  )/(beta_t) 
  + (
    norm(#text(red)[$bx_(t-1)$])^2 
    - sqrt(macron(alpha)_(t-1)) angle.l #text(red)[$bx_(t-1)$], bx_0 angle.r  
    + macron(alpha)_(t-1) norm(bx_0)^2
  )/(1 - macron(alpha)_(t-1))  \
  & wide wide wide - (
    norm(bx_t)^2 
    - sqrt(macron(alpha)_(t)) angle.l bx_t, bx_0 angle.r  
    + macron(alpha)_(t) norm(bx_0)^2
  )/(1 - macron(alpha)_(t)) 
] \}) \
&= exp lr(\{ -1/2 [ 
  (alpha_t/beta_t + 1/(1 - macron(alpha)_(t-1))) #text(red)[$norm(bx_(t-1))^2$] 
  - lr(
    angle.l sqrt(alpha_t)/beta_t bx_t 
    + sqrt(macron(alpha)_(t-1))/(1 - macron(alpha)_(t-1)) bx_0, 
    #text(red)[$bx_(t-1)$] angle.r ) 
  + const 
] \}) \
&prop exp lr(\{  
  norm(#text(red)[$bx_(t-1)$] - tilde(bold(mu))_t)^2 / (2 tilde(beta)_t) 
\}) \
$
其中均值 $tilde(bold(mu))_t$ 和方差 $tilde(beta)_t$ 为
$
tilde(beta)_t &= 1 / (display(alpha_t/beta_t + 1/(1 - macron(alpha)_(t-1)))) = (1 - macron(alpha)_(t-1)) / (1 - macron(alpha_t)) dot beta_t \
tilde(bold(mu))_t &= display(sqrt(alpha_t)/beta_t bx_t 
    + sqrt(macron(alpha)_(t-1))/(1 - macron(alpha)_(t-1)) bx_0) / display(alpha_t/beta_t + 1/(1 - macron(alpha)_(t-1))) = (sqrt(alpha_t)/beta_t bx_t 
    + sqrt(macron(alpha)_(t-1))/(1 - macron(alpha)_(t-1)) bx_0) (1 - macron(alpha)_(t-1)) / (1 - macron(alpha_t)) dot beta_t &= (sqrt(alpha_t)(1- macron(alpha)_(t-1)))/ (1 - macron(alpha_t))bx_t + (sqrt(macron(alpha)_(t-1)) beta_t) / (1 - macron(alpha_t))bx_0
$

由于需要匹配一个Gauss分布，带参分布 $q(bx_(t-1) | bx_t, bx_0)$ 也需要是一个Gauss分布 $N(bold(mu)_(theta, t), Sigma_(theta, t))$，不过在此我们将协方差矩阵简化为对角，即 $bold(mu)_(theta, t) = beta_(theta, t) bold(I)$。此时我们可以求中间过程的 KL 散度，由于其参数是两个 Gauss 分布，我们有现成的结论：

$
L_(t-1) &= KL[q(bx_(t-1)|bx_t, bx_0)\|pt(bx_(t-1)|bx_t)] \
&= 1 / 2 [
  log (|tilde(beta)_t bold(I)|)/(|beta_(theta, t) bold(I)|) - n + tr(tilde(beta)_t^(-1) bold(I) beta_(theta, t) bold(I)) + (tilde(bold(mu))_t - bold(mu)_(theta, t))^top beta_(theta, t)^(-1) bold(I) (tilde(bold(mu))_t - bold(mu)_(theta, t))
] \
&= 1 / (2 sigma_(t)) [
  norm(tilde(bold(mu))_t - bold(mu)_(theta, t))^2
]  + const, wide wide  "令" beta_(theta, t) "为只与时间相关的" sigma_t\
&= 1 / (2 sigma_(t)) [
  norm(
    [(sqrt(alpha_t)(1- macron(alpha)_(t-1)))/ (1 - macron(alpha_t))bx_t 
    + (sqrt(macron(alpha)_(t-1)) beta_t) / (1 - macron(alpha_t)) dot 1/sqrt(macron(alpha)_(t)) (bx_t - sqrt(1 - macron(alpha)_(t)) bold(epsilon))] - bold(mu)_(theta, t))^2
]  + const\
&= 1 / (2 sigma_(t)) [
  norm(
    [(alpha_t (1- macron(alpha)_(t-1)) + beta_t)/ ((1 - macron(alpha_t))sqrt(alpha_(t)))bx_t 
    + ( beta_(t))/(sqrt(alpha_t) sqrt(1 - macron(alpha)_(t)))  bold(epsilon)] - bold(mu)_(theta, t))^2
]  + const\
&= 1 / (2 sigma_(t)) [
  norm(
    [(cancel(1- macron(alpha)_(t)))/ (cancel((1 - macron(alpha_t)))sqrt(alpha_(t)))bx_t 
    - ( beta_(t))/(sqrt(alpha_t) sqrt(1 - macron(alpha)_(t)))  bold(epsilon)] - bold(mu)_(theta, t))^2
]  + const, wide alpha_t = 1 - beta_t, macron(alpha)_t = prod_(s=0)^t alpha_t\
&= 1 / (2 sigma_(t)) [
  norm(
    1 / sqrt(alpha_(t)) [bx_t 
    - (1 - alpha_(t))/( sqrt(1 - macron(alpha)_(t)))  bold(epsilon)] - bold(mu)_(theta, t))^2
]  + const\
$ 

进一步地，我们可以将预测均值 $display(bold(mu)_(theta, t))$ 建模为与 $tilde(bold(mu))_t$ 相同的形式，即

$
display(bold(mu)_(theta) (bx_t, t) = 1 / sqrt(alpha_(t)) [bx_t- (1 - alpha_(t))/( sqrt(1 - macron(alpha)_(t)))  bold(epsilon)_(theta)  (bx_t, t)])
$

因此 $L_(t-1)$ 还可以进一步简化

$
L_(t-1) 
&= 1 / (2 sigma_(t)) [
  norm(
    1 / sqrt(alpha_(t)) [bx_t 
    - (1 - alpha_(t))/( sqrt(1 - macron(alpha)_(t)))  bold(epsilon)] - bold(mu)_(theta, t))^2
]  + const\
&= 1 / (2 sigma_(t)) [
  norm(
    1 / sqrt(alpha_(t)) [cancel(bx_t) 
    - (1 - alpha_(t))/( sqrt(1 - macron(alpha)_(t)))  bold(epsilon)] 
    - 1 / sqrt(alpha_(t)) [cancel(bx_t) - (1 - alpha_(t))/( sqrt(1 - macron(alpha)_(t)))  bold(epsilon)_(theta)  (bx_t, t)]
  )^2
]  + const\
&= ((1 - alpha_(t))^2) / (2 sigma_(t)alpha_t (1 - macron(alpha)_(t))) 
  norm(
    bold(epsilon) - bold(epsilon)_(theta)  (bx_t, t)
  , size: #(150%))^2 + const\
$

*推了半天之后我们发现我们的优化目标也只是预测噪声而已（而这并不是新鲜事）*，我们就可以很顺利地得到 DDPM 的训练和采样算法。我们也可以使用更加简单的目标函数，只需要将 $L_(t-1)$ 前面的系数扔掉，同时考虑服从离散均匀分布的 $t$ 即可。在下面的算法中，我们实际上也是随机从均匀分布中采样 $t$ 然后训练模型。

=== 训练和采样算法

#show: style-algorithm
#algorithm-figure(
    "Training",
    vstroke: .5pt + luma(200),
    {
      import algorithmic: *
      Procedure("Training", none,
        While("not converged",
        $bx_0 ~ q(bx_0)$,
        $t ~ "Uniform"(\{1, ..., T\})$,
        $bold(epsilon) ~ N(bold(0), bold(I))$, 
        Comment[Perform gradient update],
        $theta <- theta - eta nabla_theta norm(bold(epsilon) - bold(epsilon)_(theta) (sqrt(macron(alpha)_t) bx_0 + sqrt(1 - macron(alpha)_t) bold(epsilon)_(theta) (bx_0, t)))^2$
      )
      )
    }
)
#algorithm-figure(
    "Sampling",
    vstroke: .5pt + luma(200),
    {
      import algorithmic: *
      Procedure("Sampling", none,
        $bx_T ~ N(bold(0), bold(I))$,
        For(
          $t = T, ..., 1$,
          IfElseChain(
            $t > 1$,
            $bold(z) ~ N(bold(0), bold(I))$,
          ),
          Else(
            $bold(z) <- 0$
          ),
          $display(bold(x)_(t+1) <- 1 / sqrt(alpha_(t)) [bx_t - (1 - alpha_(t))/( sqrt(1 - macron(alpha)_(t)))  bold(epsilon)_theta (bx_t, t)] + sigma_t bold(z))$
        ),
        Return($bx_0$)
      )
    }
)

可以将其与下面的模拟退火 Langevin 动力学采样进行对比

#algorithm-figure(
  "Annealed Langevin Dynamics",
  vstroke: .5pt + luma(200),
  {
    import algorithmic: *
    Procedure(
      "Annealed Langevin Dynamics",
      ($\{sigma_i\}_(i=1)^n$, $epsilon$, $T$),
      {
        Comment[Initialize the search range]
        Assign[$macron(bold(x))_0$][$bold(v)$]
        For(
          $t <- 1, ..., L$,

          Comment[Set step size $alpha_i$],
          $alpha_i <- epsilon dot sigma_i^2 "/" sigma_L^2$,
          For(
            $t <- 1, ..., T$,
            "Draw " + $bold(z)_t ~ N(0, I)$, 
            $display(macron(bold(x))_t <- macron(bold(x))_(t-1) + alpha_i "/" 2 dot s_(theta)(macron(bold(x))_(t-1), sigma_i) + sqrt(alpha_i)) bold(z)_t$
          ),
          $macron(bold(x))_0 <- macron(bold(x))_T$ 
        )
      },
      Return($macron(bold(x))_T$)
    )
  }
)

不难发现 DDPM 中的 Sampling 算法将 Annealed Langevin Dynamics 算法中的内层的循环减少到了 $1$。

=== 离散值图像生成的最后一步


#h(2em)由于需要生成的对象是计算机中离散编码（如八位）的图像，对应最后一步的 $p_theta (bx_0|bx_1)$ 需要得到的是离散分布。论文使用了一个简便的技巧，假设图像的像素值在离散化过程中从 $\{0, 1, ..., 255\}$ 线性地归一化到 $[-1, 1]$ 内，即 $0$ 被映射到 $-1$，$255$ 被映射到 $1$。在计算 $bx_0$ 的第 $i$ 个像素值是 $bx_0^((i))$ 时，我们可以简单地将 $RR$ 切分成 $256$ 块，其中 $1 ~ 254$ 分别对应着 $display([-1 - 1/255, 1 + 1/255])$ 中宽度为 $display(2/255)$ 的小区间 $[delta_- (x), delta_+ (x)]$，剩下的一头一尾就分别对应剩下的两个无限长度的区间。具体而言，
$
delta_+ (x) = cases(
  infinity &"if" x = 1, 
  display(x + 1/255) quad &"if" x < 1), wide
delta_- (x) = cases(
  -infinity &"if" x = -1, 
  display(x - 1/255) quad &"if" x > -1)
$
假设 $bx_0$ 的分布中各分量相互独立，就有
$
p_theta (bx_0) = product_(i=1)^n p_theta (bx_0^((i))) = prod_(i=1)^n int_(delta_-(bx_0^((i))))^(delta_+(bx_0^((i)))) N(x | bold(mu)_(theta)^((i)) (bx_1, 1), sigma_1 bold(I)) dd x
$

这样也不会破坏分布的归一化性质。

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
