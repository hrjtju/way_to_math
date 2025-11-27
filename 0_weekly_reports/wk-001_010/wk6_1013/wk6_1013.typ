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
#let wk_report_name = "2025年10月13日至10月19日周报"
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
=== 对模型和训练的改动
#linebreak()
+ 将所有 $3 times 3$ 的卷积核大小增加到 $5$
+ 依照 e2cnn #cite(<github_page-e2cnn>) 构建 `tiny` 版本的 `p4cnn`
+ 在训练代码中增加了 early stopping 机制。

=== 在不同规则的演化系统上的实验结果

#tab 实验结果如下。表中每格上侧的数表示最后的训练集损失，下侧的数表示验证集正确率。

#figure(
  table(
    columns: 7,
    align: (horizon+center, right, right, right, right, right, right, right, right),
    stroke: none,
    table.hline(),
    table.header([演化规则 / 模型名], [CNN-tiny], [CNN-small], [MCNN], [P4CNN-tiny], [P4CNN-small], [P4MCNN]),
    table.hline(stroke: 0.5pt),
    [`B36/S23`], [0.7002\ 86.79%], [0.0045\ 100.00%], [0.0055\ 99.97%], [0.3631\ 94.74%], [0.0037\ 100%], [*0.0036\ 100.00%*], 
    [`B36/S23`],  [0.3451\ 90.46%], [*0.0167\ 99.89%*], [0.0262\ 98.94%], [0.4740\ 92.32%], [0.0274\ 99.77%], [0.0148\ 99.86%], 
    [#highlight[`B3678/S34678`]],  [0.2035\ 92.36%], [0.0133\ 99.96%], [0.0660\ 98.19%], [0.4534\ 93.40%], [0.0159\ 99.96%], [*0.0097\ 99.98%*], 
    [#highlight[`B35678/S5678`]],  [0.0165\ 99.24%], [0.0216\ 98.73%], [0.0955\ 99.52%], [0.0086\ 99.32%], [0.0058\ 99.65%], [*0.0041\ 99.77%*], 
    [`B2/S`],  [0.0231\ 99.74%], [*0.0023\ 100.00%*], [0.0024\ 100.00%], [0.6136\ 88.79%], [0.0022\ 100.00%], [0.0034\ 100.00%], 
    [`B345/S5`],   [0.1710\ 96.25%], [0.0065\ 100.00%], [0.0039\ 100.00%], [*0.0028\ 100.00%*], [*0.0028\ 100.00%*], [0.0119\ 99.92%], 
    [`B13/S012V`],  [0.2489\ 92.30%], [0.0066\ 100.00%], [0.0045\ 99.99%], [0.1243\ 99.04%], [0.0016\ 100.00%], [*0.0010\ 100.00%*], 
    [`B2/S013V`], [0.5533\ 77.24%], [0.0046\ 100.00%], [0.0025\ 100.00%], [0.7082\ 84.71%], [0.0091\ 100.00%], [*0.0015\ 100.00%*], 
    table.hline(),
  )
)

不同模型的大小、参数量和计算量对比如下：

#figure(
  table(
    columns: 5,
    align: (horizon+center, right, right, right),
    stroke: none,
    table.hline(),
    table.header([模型名], [FLOPs (M)], [参数量], [估计大小 (MB)], [最低正确率 $(%)$]),
    table.hline(stroke: 0.5pt),
    [P4CNN-tiny], [$1.28$], [$94$], [11.21], [84.71],
    [CNN-tiny], [$5.32$], [$133$], [2.24], [77.24],
    [MCNN], [$40.48$], [$1,004$], [7.36], [98.19],
    [*P4MCNN*], [*$6.4$*], [*$1,276$*], [---], [*99.77*],
    [CNN-small], [$98.5$], [$2,450$], [---], [98.73%],
    [P4CNN-small], [$10.24$], [$3,202$], [---], [99.65],
    table.hline(),
  )
)

下面是各网络在较为困难的四个规则数据下的训练轨迹对比图，其中对于密集的数据点采用的是滑动窗口的中位数作为平滑后的值。

#figure(
  image("training_results_2-B3678_S34678.png", width: 90%),
  caption: [各网络在`B3678_S34678`规则下生成的生命游戏数据上的训练和测试结果]
)

#figure(
  image("training_results_3-B35678_S5678.png", width: 90%),
  caption: [各网络在`B35678/S5678`规则下生成的生命游戏数据上的训练和测试结果]
)

#figure(
  image("training_results_6-B13_S012V.png", width: 90%),
  caption: [各网络在`B13/S012V`规则下生成的生命游戏数据上的训练和测试结果]
)

#figure(
  image("training_results_7-B2_S013V.png", width: 90%),
  caption: [各网络在`B2/S013V`规则下生成的生命游戏数据上的训练和测试结果]
)


#pagebreak()
// == 微型抗癌机器人在血液中的动力学
// === 项目目的

// 微型抗癌机器人是通过癌症细胞散发出的化学吸引物 (chemoattractant) 趋化性驱动 (chemotaxis-driven) 运动，与癌细胞进行配体-受体结合后定向释放药物，达到治疗的目的。本项目研究理想状况下的微型抗癌机器人集群在血液中的动力学。

// === 建模

// 目前项目对血液中的化学吸引物、游离的微型机器人和与癌细胞结合的微型机器人分布进行建模。设 $t$ 时刻，位于血液中 $bx$ 位置的化学吸引物浓度为 $c(bx, t)$，化学吸引物正常的消耗或讲解速率为 $k$， ，则有
// $
// (partial c)/(partial t) = D_c nabla^2 c - k c + S_(Omega_t)(bx)
// $
// 其中 
// - $D_c$ 为化学吸引物在血液中的扩散系数
// - $k$ 为化学吸引物正常的消耗或讲解速率
// - $Omega_t$ 为癌细胞所在区域，$S_(Omega_t)(bx)$ 为癌细胞区域中 $bx$ 位置向血液中释放化学吸引物的速度
// 类似地，设 $rho(bx, t)$ 为游离机器人血液中的分布密度，$b(bx, t)$ 为非游离的机器人的分布密度，有
// $
// (partial rho)/(partial t) 
//   &= D_rho nabla^2 rho - nabla dot (chi rho nabla c) - k_b rho delta_(Omega_t) + k_u b \
// (partial b)/(partial t) 
//   &= k_b rho delta_(Omega_t) - k_u b 
// $
// 其中
// - $D_rho$ 为游离机器人在血液中的扩散系数
// - $k_b$ 为游离机器人绑定癌细胞的速率
// - $k_u$ 为绑定癌细胞的机器人释放药物后和癌细胞解绑的速率
// - $chi$ 为机器人逆浓度梯度制导的成功率

// === 局限性

// - 没有考虑机器人密度增大后的互相碰撞问题
// - 没有考虑血流对化学吸引物扩散和机器人运动的影响
// - 没有考虑血液中的其他细胞对机器人的影响

// #pagebreak()
// = 文献阅读

// == Denoising Diffusion Probabilistic Models #cite(<DBLP:paper-DDPM>)
// Jonathan Ho, Ajay Jain and Pieter Abbeel | https://arxiv.org/abs/2006.11239

// 本周把 DDPM 的剩余部分补完。

// === 补遗


// === 实验结果


// === 总结和讨论



// 参考资料
// + https://arxiv.org/abs/1907.05600
// + https://arxiv.org/abs/2006.11239
// + https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice

// #pagebreak()
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

#h(2em)本周学习到了 Poisson 过程，以及一般宽平稳随机过程相关函数的诸性质。对于实值随机过程 $X(dot.c)$，（宽）平稳指的是其相关函数 $R_X (s, t) = EE[X(s)X(t)]$ 仅依赖于 $s$ 和 $t$ 之间的时间差，因此也常记为 $R_X (tau)$。对于宽平稳的随机过程，其相关函数 $R_X (tau)$ 一定是偶函数，且从相关函数事实上是两个随机变量 $X(s)$ 和 $X(t)$ 的内积的角度看由 Cauchy-Schwartz 不等式可以得到
$
|R_X (tau)| = |R_X (s, t)| = |lr(angle.l X(s), X(t) angle.r)| lt.slant sqrt(lr(angle.l X(s), X(s) angle.r^2)lr(angle.l X(t), X(t) angle.r^2)) = R_X (0).
$
另外，如果有一宽平稳过程的相关函数满足存在某个 $T>0$，使得 $R_X (0) = R_X (T)$，一定有对任意 $tau$，$R_X (tau) = R_X (tau + T)$ 几乎处处成立。考虑 $EE[ |X(t)-X(t+T)|^2 ]$，有
$
EE[ |X(t)-X(t+T)|^2 ] &= EE[X^2(t) - 2X(t)X(t+T) + X^2(t+T)]\
&= 2 R_X (0) - 2 R_X (T) = 0. 
$

#tab 相关函数都是正定函数，一个函数是正定函数，是指任取正整数 $n$，任取定义域中一列点 $x_1, ..., x_n$，矩阵 $R = (f(x_i - x_j))_(i,j)$ 是正定的。该定义难以作为判断依据，但所幸有下面的定理

#theorem(
  [Bochner],
  [函数 $f$ 是正定的，当且仅当其 Fourier 变换非负。]
)

#proof(
  [
    先证必要性。首先回忆 Fourier 变换和 Fourier 逆变换：
    $
    F(omega) &= int_(-infinity)^infinity f(x)e^(-i omega x) dd x #h(2em)& "Fourier transform"\
    f(x) &= int_(-infinity)^infinity F(omega)e^(i omega x) dd omega& "Inverse Fourier transform"\
    $
    注意 $F(omega) gt.slant 0$，如果 $e^(i omega x)$ 是正定的，由积分的线性性，然后去极限，求和变成积分，容易推出 $f(x)$ 是正定的。现任取 $n$，任取 $x_1, ..., x_n$，考察矩阵 $R = (e^(j omega (x_j - x_k)))_(j,k)$。任选 $z in CC^n$ 计算二次型 $z^dagger R z$:
    $
    z^dagger R z &= sum_(j,k=1)^n macron(z_j) e^(i omega (x_j - x_k)) z_k
    = sum_(j,k=1)^n overline(z_j e^(- i omega x_j)) z_k e^(- i omega x_k)\
    &= overline((sum_(j) z_j e^(- i omega x_j)))(sum_(k) z_k e^(- i omega x_k)) gt.slant 0.
    $
    再来看充分性。已知 $f$ 是正定的，选择区间 $[-T"/"2, T"/"2]$ 的一个均匀的划分 $K = \{t=x_1, x_2, ..., x_(n-1), x_n = s\}$，按上述规律组成的矩阵 $R$ 是正定的。现取 $z = [e^(-i omega x_1), ..., e^(-i omega x_n)]^top$。假设 $f$ 的 Fourier 变换总存在，它在区间 $[-T"/"2, T"/"2]$ 上的积分也总存在，根据 $f$ 的正定条件得到下面的二次型的值一定非负，经过先加细区间划分，然后令 $T -> infinity$ 的方法，可以得到极限为 $f$ 的 Fourier 变换。由极限的性质和二次型的非负性可使充分性得到证明。 
    $
    z^dagger R z &= sum_(j,k=1)^n f(x_k - x_j) e^(j omega (x_j - x_k)) \
    &prop sum_(j,k=1)^n f(x_k - x_j) e^(j omega (x_j - x_k)) (Delta x)^2 #h(2em)& "不变号"\
    &-> integral_(-T/2)^(T/2) integral_(-T/2)^(T/2) f(t-s) e^(- j omega (t -s)) dd t dd s & |K| --> 0, "极限唯一性"\
    &= 1/2 integral_(-T)^(T) integral_(-T+|tau|)^(T-|tau|) f(tau) e^(- j omega tau) dd sigma dd tau & tau = t-s, sigma = t+s \
    &prop 1/T integral_(-T)^(T) 2(T-|tau|) f(tau) e^(- j omega tau) dd tau\
    &-> "P.V." int_(-infinity)^infinity f(tau) e^(-j omega tau) dd x = int_(-infinity)^infinity f(tau) e^(-j omega tau) dd x #h(2em) & T --> infinity, "极限唯一性"
    $
  ]
)

#tab 这个事实还可以是另一方向的观察结果。考虑平稳随机过程 $X(dot)$ 在对称 $[-T"/"2, T"/"2]$ 上的短时 Fourier 变换，考虑下面的极限：
$
L 
=& lim_(T -> infinity) 1/T EE [ lr(| int_(-T/2)^(T/2) X(t)e^(-i omega t) dd t|)^2 ] >= 0\
=& lim_(T -> infinity) 1/T EE [ overline((int_(-T/2)^(T/2) X(t)e^(-i omega t) dd t))(int_(-T/2)^(T/2) X(s)e^(-i omega s) dd s) ] \
=& lim_(T -> infinity) 1/T  int_(-T/2)^(T/2) int_(-T/2)^(T/2) EE [X(s)X(t)]  e^(-i omega (t- s)) dd t dd s \
=& lim_(T -> infinity) 1/T int_(-T/2)^(T/2) int_(-T/2)^(T/2) R_X (s, t) e^(-i omega (t- s)) dd t dd s \
=& int_(-infinity)^(infinity) R_X (tau) e^(- i omega tau) dd tau = S_X (omega) & "上面算过了"
$
该极限自然地证明了上面的定理，除此之外，$S_X (omega)$ 称为随机过程 $X(dot)$ 的*功率谱密度 (power spectrum density)*。

== 随机微分方程

#h(2em)本周没有推进。

#pagebreak()
// = 问题解决记录
// == uv 相关
// uv 是基于 Rust 的新一代 Python 包管理器，具有可迁移性强、快速、简单的特点。

// === Pytorch CUDA 版本的配置


// == Typst 相关

// === 数学公式自动编号



// == Python 相关

// #pagebreak()

= 下周计划
*论文阅读*
+ 生成模型
  - DDPM 收尾
  - Sliced Score Matching: A Scalable Approach to Density and Score Estimation 
+ 动力学
  - 暂无
+ 其他
  - General $E(2)$ - Equivariant Steerable CNNs

*项目进度*
+ 使用神经网络学习生命游戏的演化动力学
  - 尝试对模型权重进行解释，并用模型作为系统的演化模拟器，统计验证所学到的规则是否正确
  - 重新生成部分规则下的生命游戏演化数据
+ 微型抗癌机器人在血液中的动力学
  - 开始学习 PDE 的数值解方法

*理论学习*
+ 随机过程课程
  - 完成 Poisson 过程的学习
  - 预习 Markov 过程
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
== 剩余规则数据下的各网络训练结果

#figure(
  image("training_results_0-B3_S23.png", width: 90%),
  caption: [各网络在`B3/S23`规则下生成的生命游戏数据上的训练和测试结果]
)

#figure(
  image("training_results_1-B36_S23.png", width: 90%),
  caption: [各网络在`B36/S23`规则下生成的生命游戏数据上的训练和测试结果]
)

#figure(
  image("training_results_4-B2_S.png", width: 90%),
  caption: [各网络在`B2/S`规则下生成的生命游戏数据上的训练和测试结果]
)

#figure(
  image("training_results_5-B345_S5.png", width: 90%),
  caption: [各网络在`B345/S5`规则下生成的生命游戏数据上的训练和测试结果]
)
