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
==== 增大卷积核大小
#tab 我将 `SimpleCNNTiny` 和 `SimpleCNNSmall` 的卷积核大小增加到 $5$。并依照 e2cnn #cite(<github_page-e2cnn>) 构建 `tiny` 版本的 `p4cnn`。

=== 在不同规则的演化系统上的实验结果

#tab 实验正在运行，目前的结果是群等变 CNN (small) 可以很好的学习上述的所有规则（测试集正确率接近 $100%$）。而使用并行多尺度 CNN 的训练结果在区间不连续，例如 `B3678/S34678` 这样的规则下学习较为困难。另外我注意到对于不同的规则，不能使用一套超参数生成数据，由于规则不同，使用一套参数在某些规则下会产生大量的重复数据，这会影响网络的训练。

具体的结果分析会在下周的周报中呈现，其中包括所有情况的训练曲线，以及对神经网络作为演化模拟器和训练权重的分析。可遇见的困难是，对于群等变 CNN，我应该如何提取并解释它的权重。


#pagebreak()
== 微型抗癌机器人在血液中的动力学
=== 项目目的

微型抗癌机器人是通过癌症细胞散发出的化学吸引物 (chemoattractant) 趋化性驱动 (chemotaxis-driven) 运动，与癌细胞进行配体-受体结合后定向释放药物，达到治疗的目的。本项目研究理想状况下的微型抗癌机器人集群在血液中的动力学。

=== 建模

目前项目对血液中的化学吸引物、游离的微型机器人和与癌细胞结合的微型机器人分布进行建模。设 $t$ 时刻，位于血液中 $bx$ 位置的化学吸引物浓度为 $c(bx, t)$，化学吸引物正常的消耗或讲解速率为 $k$， ，则有
$
(partial c)/(partial t) = D_c nabla^2 c - k c + S_(Omega_t)(bx)
$
其中 
- $D_c$ 为化学吸引物在血液中的扩散系数
- $k$ 为化学吸引物正常的消耗或讲解速率
- $Omega_t$ 为癌细胞所在区域，$S_(Omega_t)(bx)$ 为癌细胞区域中 $bx$ 位置向血液中释放化学吸引物的速度
类似地，设 $rho(bx, t)$ 为游离机器人血液中的分布密度，$b(bx, t)$ 为非游离的机器人的分布密度，有
$
(partial rho)/(partial t) 
  &= D_rho nabla^2 rho - nabla dot (chi rho nabla c) - k_b rho delta_(Omega_t) + k_u b \
(partial b)/(partial t) 
  &= k_b rho delta_(Omega_t) - k_u b 
$
其中
- $D_rho$ 为游离机器人在血液中的扩散系数
- $k_b$ 为游离机器人绑定癌细胞的速率
- $k_u$ 为绑定癌细胞的机器人释放药物后和癌细胞解绑的速率
- $chi$ 为机器人逆浓度梯度制导的成功率

=== 局限性

- 没有考虑机器人密度增大后的互相碰撞问题
- 没有考虑血流对化学吸引物扩散和机器人运动的影响
- 没有考虑血液中的其他细胞对机器人的影响

#pagebreak()
= 文献阅读

== Denoising Diffusion Probabilistic Models #cite(<DBLP:paper-DDPM>)
Jonathan Ho, Ajay Jain and Pieter Abbeel | https://arxiv.org/abs/2006.11239

本周把 DDPM 的剩余部分补完。

=== 补遗


=== 实验结果


=== 总结和讨论



参考资料
+ https://arxiv.org/abs/1907.05600
+ https://arxiv.org/abs/2006.11239
+ https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice

#pagebreak()
== Sliced Score Matching: A Scalable Approach to Density and Score Estimation #cite(<DBLP:paper-yang_song-sliced_score_matching>)

- Yang Song, Sahaj Garg, Jiaxin Shi, Stefano Ermon 
- https://arxiv.org/abs/1905.07088


#pagebreak()

== General E(2)-Equivariant Steerable CNNs #cite(<DBLP:paper-e2cnn>)

- Maurice Weiler and Gabriele Cesa
- https://arxiv.org/abs/1911.08251

#pagebreak()

= 学习进度
== 机器学习理论
=== Markov Chain Monte Carlo (MCMC)


== 随机过程

#h(2em)本周学习到了 Poisson 过程，以及一般宽平稳随机过程相关函数的诸性质。


== 随机微分方程

#h(2em)本周没有推进。

#pagebreak()
= 问题解决记录
== uv 相关
uv 是基于 Rust 的新一代 Python 包管理器，具有可迁移性强、快速、简单的特点。

=== Pytorch CUDA 版本的配置


== Typst 相关

=== 数学公式自动编号



== Python 相关

#pagebreak()
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
  - 收集实验数据，若有必要，实现更小的 $p 4$-等变 CNN 模型并测试
  - 尝试对模型权重进行解释，并用模型作为系统的演化模拟器，统计验证所学到的规则是否正确
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

