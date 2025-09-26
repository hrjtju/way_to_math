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

#let wk_report_name = "2025年9月15日至9月21日周报"
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

#let redText(t) = {text(red)[$t$]}
#let blueText(t) = {text(blue)[$t$]}
#let greenText(t) = {text(green)[$t$]}
#let orangeText(t) = {text(orange)[$t$]}

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

#h(2em)本周由于时间规划原因没有做代码方面的更新。

找到了互联网上的一个模拟生命游戏的程序 Golly，其文档不仅包含原始版本的生命游戏，还包含极其丰富的变种，具体见
https://golly.sourceforge.io/Help/algos.html。可能可以通过下载该程序或对应的python包，以改进生成数据的代码，使得更换规则更加容易。

杨老师提到了 Gray-Scott 系统和交通流模型（https://www.thp.uni-koeln.de/~as/Mypage/traffic.html），前者我在上周自行寻找材料做了简单的阅读（见上周周报），且没想到如何用神经网络学习。交通流模型阅读后发现其属于一维元胞自动机，但目测情况下难以使用卷积网络学习其规则。

下周计划内容详见最后一节。

// == 微型抗癌机器人在血液中的动力学

#pagebreak()
= 文献阅读
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

=== 实验结果

#highlight[下周补齐]

=== 总结和讨论

#highlight[下周补齐]


参考资料
+ https://arxiv.org/abs/1907.05600
+ https://arxiv.org/abs/2006.11239
+ https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice

// #pagebreak()
// == Sliced Score Matching: A Scalable Approach to Density and Score Estimation
// Yang Song, Sahaj Garg, Jiaxin Shi, Stefano Ermon | https://arxiv.org/abs/1905.07088

#pagebreak()
= 学习进度

== 随机过程

#h(2em)首先复习回顾了有关于尾部概率的诸不等式：Markov 不等式、Chebyshev 不等式、Chernoff 界。我发现有一个更一般的情形可以轻松涵盖这三个不等式，以及我在 SDE 那本小册子中看到的 Chebyshev 不等式的版本：

#theorem("拓展 Chebyshev 不等式")[
考虑随机变量 $X: Omega -> RR$，和连续增函数 $g: RR_(gt.slant 0) -> RR_(gt.slant 0)$，如果 $g(a) > 0$ 且 $bb(E)(g(|X|))$ 存在，有 
$
EE(|X| gt.slant a) lt.slant EE(g(|X|)) / g(a)
$
]
- 取 $X = Y - bb(E)[Y]$，$g(x) = x^2$，得到 Chebyshev 不等式： $display(bb(E)(|Y-bb(E)[Y]|gt.slant a) lt.slant ("Var"(Y)) / (a^(2)))$
- 取 $g(x) = x$，得到 Markov 不等式： $display( bb(E)(|X| gt.slant a) lt.slant (bb(E)(|X|)) / (a))$
- 取 $g(x) = e^(lambda x)$，其中 $lambda gt.slant 0$，得到 Chernoff 界： $display( bb(E)(|X| gt.slant a) lt.slant (bb(E)(exp\{ lambda|X| \})) / (exp\{ a lambda  \}))$
- 取 $g(x) = x^(p)$，其中 $p gt.slant 1$，得到下面另一形式的 Chebyshev 不等式： $display( bb(E)(|X| gt.slant lambda) lt.slant (bb(E)(|X|^(p))) / (lambda^(p)))$


== 随机微分方程

#h(2em)本周学习了 #ito 积分的链式法则、和乘积法则。具体而言，我们研究的是具有下列基本形式之随机微分的随机过程：
$
dd X = F dd t + G dd W <==> X(r) = X(s) + int_s^r F dd t + int_s^r G dd W, quad 0 lt.slant s lt.slant r lt.slant T
$
其中 $F in LL^1 (0, T)$，$G in LL^2 (0, T)$。链式法则是说对于函数 $u: RR times [0, T] -> RR$，如果 $u_x, u_t, u_(x x)$ 都存在且连续，则有
#mitex(`
\begin{align} \mathrm{d}Y = \mathrm{d} u(X(t), t) &= u_{t} \mathrm{d}t + u_{x}\mathrm{d}X + \frac{1}{2}u_{x x}G^{2}\mathrm{d}t\\&=\left( u_{t} + u_{x} F + \frac{1}{2}u_{x x} G^{2} \right)\mathrm{d} t + u_{x} G \mathrm{d} W\end{align}
`)
或者说
#mitex(`
\begin{align}
Y(r) - Y(s) &= u(X(r), r) - u(X(s), s)\\&= \int_{s}^{r} u_{t} + u_{x} F + \frac{1}{2}u_{x x}G^{2} \, \mathrm{d}t + \int_{r}^{s} u_{x} G \, \mathrm{d}W \quad \text{a.s.}  \end{align}
`)

乘积法则说的是两个有上述基本形式之随机微分的过程
#mitex(`
\begin{align} \mathrm{d} X_{1} &= F_{1} \mathrm{d}t + G_{1}\mathrm{d}W\\ \mathrm{d} X_{2} &= F_{1} \mathrm{d}t + G_{1}\mathrm{d}W\\ \end{align}
`)
其乘积有下面的随机微分：
$
dd(X_1 X_2) = X_2 dd X_1 + X_1 dd X_2 + G_1 G_2 dd t.
$
#text[$" "$]

两个定理的证明具有一定的相似性。对于乘积法则，首先证明与时间无关的 $F$ 和 $G$ 情形下在 $[0, t]$（其中 $t lt.slant T$）满足条件，然后再证在任意区间 $[s, t]$ 上成立，这是第一步；接着可以证明对任意的阶梯过程 $F in LL^1(0, T)$ 和 $G in LL^2(0, T)$ 都成立，这是第二步；最后利用 $LL^1(0, T)$ 和 $LL^2(0, T)$ 的完备性，利用近似的思想推广到任意的 $F in LL^1(0, T)$ 和 $G in LL^2(0, T)$，这是第三步。

链式法则的证明与此类似。对于函数 $u$，首先考虑最简单的形式，即多项式 $u in FF[x]$，然后再推广到任意的关于 $x$ 和 $t$ 的多项式的乘积形式。由于 $FF[x, t]$ 上的任意元素都可以写成 $display(sum_(i=1)^n p_i (x) q_i (t))$ 的形式，其中 $p_i in FF[x]$，$q_i in FF[t]$，因此可以推广到任意的 $u in FF[x, t]$。最后让一列多项式逼近任意满足定理条件的 $u$ 即可。

= 问题解决记录

== Typst 相关
=== 直接套用 LaTeX 公式
可以通过导入 `mitex` 包实现直接将 LaTeX 或 Markdown 文档中的行内和行间公式嵌入 typst 文档：

#grid(
  columns: (48%, 48%),
  align: center,
  [
    ```typst
    #import "@preview/mitex:0.2.4": *
    #mitex(`
    \begin{align} \mathrm{d} X_{1} &= F_{1} \mathrm{d}t + G_{1}\mathrm{d}W\\ \mathrm{d} X_{2} &= F_{1} \mathrm{d}t + G_{1}\mathrm{d}W\\ \end{align}
    `)
    ```
  ],
  mitex(`
  \begin{align} \mathrm{d} X_{1} &= F_{1} \mathrm{d}t + G_{1}\mathrm{d}W\\ \mathrm{d} X_{2} &= F_{1} \mathrm{d}t + G_{1}\mathrm{d}W\\ \end{align}
  `)
)

=== 一些更灵活的 typst 函数定义

在撰写一些细节推导时，不可避免地需要使用不同颜色指示推导中有不同功能的项。例如

#grid(
  columns: (48%, 48%),
  align: center,
  [
    ```typst
    $y = #text(red)[$k$]x$
    ```
  ],
  $y = #text(red)[$k$]x$
)

在推导过程很复杂时显然不适宜，我们可以定义下面的函数：

```typst
#let redText(text) = {text(red)[$text$]}
```

这样就可以更方便地使用该函数
#grid(
  columns: (48%, 48%),
  align: center,
  [
    ```typst
    $y = redText(k)x$
    ```
  ],
  $y = redText(k)x$
)
类似还可以设计其他常用颜色的函数。

#pagebreak()
= 下周计划
*论文阅读*
+ 生成模型
  - 完成 Diffusion 剩下的部分
  - Sliced Score Matching: A Scalable Approach to Density and Score Estimation
  - Score-Based Generative Modeling through Stochastic Differential Equations

*项目进度*
+ 使用神经网络学习生命游戏的演化动力学
  - 解释神经网络
    - 寻找更多神经网络权重的解释方法
  - 其他的生命游戏规则
    - 探索并整理 Golly 文档中提到的可简单实现的若干种方法（五种左右）
    - 调整代码实现，使之可以各方面适配多数据集的情形（例如训练的输入输出、logging 的规格，以及输出文件的文件名和文件夹规格，需要包含所使用的数据集）
  - 其他模型
    - 考虑交通流模型和一维元胞自动机使用神经网络学习规则的方法
    - 考虑如何通过神经网络模型预测 Gray-Scott 的动力学

*理论学习*
+ 随机过程课程
  - 随机过程完成第一章和第二次作业
+ 随机微分方程
  - Evans 完成至第五章第一节或第二节
