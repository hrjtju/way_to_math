#import "@preview/algorithmic:1.0.5"
#import algorithmic: (
  style-algorithm, 
  algorithm-figure
)

#import "@preview/numbly:0.1.0": numbly

#show strong: set text(blue)

#let wk_report_name = "2025年9月8日至9月14日周报"
#let name_affiliation = "何瑞杰 | 中山大学 & 大湾区大学"

#let const = "constant"
#let bx = $bold(x)$
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

#set page(
  paper: "a4",
  numbering: "1",
  header: wk_report_name + " | " + name_affiliation,
)

#set par(
  first-line-indent: 2em,
  justify: true,
)

#set heading(numbering: "1.")

#align(
  center, 
  text(17pt)[#wk_report_name\ ] 
        + text(12pt)[\ 何瑞杰\ 中山大学, 大湾区大学]
)

= 项目进展
== 使用神经网络学习生命游戏的演化动力学

== 微型抗癌机器人在血液中的动力学



#pagebreak()
= 文献阅读
== Denoising Diffusion Probabilistic Models
Jonathan Ho, Ajay Jain and Pieter Abbeel | https://arxiv.org/abs/2006.11239

=== 综观

不同于 VAE 等等的一个隐变量空间的生成模型，DDPM 尝试用多步编码/解码，或是加噪/去噪方式生成数据。它的每一步可以视为是一个去噪自编码器（Denoising Autoencoder），而其采样（生成）过程通过与加噪过程的 Markov 链“共用”，并通过预测加噪过程中的随机噪声实现加噪过程的 “反转”，并添上 Langevin 动力学的随机噪声实现对分布的采样。与正常的 Langevin 动力学不同的是，从第 $t$ 步到 $t-1$ 步，我们认为其在尝试适配一个分布 $p(bx_(t-1)|bx_t)$ 过程中，只做了一次 Langevin 动力学的迭代。

#figure(
  image("image.png", width: 80%),
  caption: "图 1. DDPM 的工作流程示意图"
)

=== 加噪过程的建模

首先对于加噪过程，我们有一列噪声强度 $\{beta_t\}_(t=0)^T$，然后按照 $q(bx_t|bx_(t-1)) ~ N(bx_(t-1) | beta_t bold(I))$ 来进行：
$
bx_t 
&= sqrt(1 - beta_t) bx_(t-1) + sqrt(beta_t) bold(epsilon)_t, #h(1em) & bold(epsilon)_t ~ N(0, I) \
&= sqrt(1 - beta_t) [ sqrt(1 - beta_(t-1)) bx_(t-2) + sqrt(beta_(t-1)) bold(epsilon)_(t-1) ] + sqrt(beta_t) bold(epsilon)_t,  wide & bold(epsilon)_(t-1) ~ N(0, I) \
&= sqrt((1 - beta_t)(1 - beta_(t-1))) bx_(t-2) + sqrt((1 - beta_t) beta_(t-1)) bold(epsilon)_(t-1) + sqrt(beta_t) bold(epsilon)_t \
&= sqrt((1 - beta_t)(1 - beta_(t-1))) bx_(t-2) + sqrt((1 - beta_t) beta_(t-1) - (1 - beta_t) + 1) macron(bold(epsilon))_t \
&= sqrt((1 - beta_t)(1 - beta_(t-1))) bx_(t-2) + sqrt(1 - (1 - beta_t)(1 - beta_(t-1))) macron(bold(epsilon))_t \
& quad dots.v\
&= sqrt(product_(s=1)^t (1 - beta_s)) bx_0 + sqrt(1 - product_(s=1)^t (1 - beta_s)) macron(bold(epsilon))_t = sqrt(macron(alpha)_t) bx_0 + sqrt(1 - macron(alpha)_t) macron(bold(epsilon))_t
$

这给出了加噪过程中每一步得到的 $bx_t$ 更加便捷的表示，对后续的推导有帮助。

=== 优化目标

由于我们的目标是从随机噪声 $bx_0$ 生成图像，即 $bx_T$，因此需要最大化的项是 $EE_(pt)[-log pt(bx_0)]$。因此我们需要继续用变分推断的技巧，逐步将其转化为可以计算得到的项。首先我们有下面的上界

$
&quad EE_(pt)[-log pt(bx_0)] \
&= EE_(pt)[-log pt(bx_0) int q(bx_(1:T)|bx_0) dd bx_(1:T)] = EE_(pt)[-int q(bx_(1:T)|bx_0)log pt(bx_0)  dd bx_(1:T)] \
&= - EE_(pt(bx_0),  q(bx_(1:T)|bx_0))[
  log (pt(bx_0) pt(bx_(1:T)|bx_0)) / (pt(bx_(1:T)|bx_0))
] = - EE_(pt(bx_0),  q(bx_(1:T)|bx_0))[
  log (#text(red)[$pt(bx_(0:T))$] q(bx_(1:T)|bx_0)) / (pt(bx_(1:T)|bx_0) #text(red)[$q(bx_(1:T)|bx_0)$])
]\
&= - EE_(pt(bx_0),  q(bx_(1:T)|bx_0))[
  log #text(red)[$pt(bx_(0:T))  / q(bx_(1:T)|bx_0)$]
] - EE_(pt(bx_0),  q(bx_(1:T)|bx_0))[
  log (q(bx_(1:T)|bx_0)) / (pt(bx_(1:T)|bx_0))
]\
$
$
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

*推了半天之后我们发现我们的优化目标也只是预测噪声而已（而这并不是新鲜事）*，我们就可以很顺利地得到 DDPM 的训练和采样算法。我们也可以使用更加简单的目标函数，只需要将 $L_(t-1)$ 前面的系数扔掉即可。

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

#h(-2em)参考资料
+ https://arxiv.org/abs/1907.05600
+ https://arxiv.org/abs/2006.11239
+ https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice

#pagebreak()
== Sliced Score Matching: A Scalable Approach to Density and Score Estimation
Yang Song, Sahaj Garg, Jiaxin Shi, Stefano Ermon | https://arxiv.org/abs/1905.07088


#pagebreak()
= 学习进度

== 随机过程

#h(2em)首先复习回顾了有关于尾部概率的诸不等式。

== 随机微分方程

#h(2em)本周学习了 #ito 积分的链式法则、和乘积法则。

== Gray-Scott 系统

#h(2em)

参考资料

#pagebreak()
= 问题解决记录

== Typst 相关


== 推导相关

#pagebreak()
= 下周计划
*论文阅读*
- 

*项目进度*
- 

*理论学习*
- 
