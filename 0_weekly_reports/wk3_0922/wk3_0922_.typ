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

// metadata
#let wk_report_name = "2025年9月22日至9月28日周报"
#let name_affiliation = "何瑞杰 | 中山大学 & 大湾区大学"

// Snippets
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
#let tab = {h(2em)}

#show strong: set text(blue)
#show: thmrules.with(qed-symbol: $square$)
#set page(
  paper: "a4",
  numbering: "1",
  header: wk_report_name + " | " + name_affiliation,
)
#set par(
  first-line-indent: 2em,
  justify: true,
)

#show figure.where(
  kind: table
): set figure.caption(position: top)

#set math.equation(numbering: "(1)")
#set heading(numbering: "1.")
#set math.cases(gap: 0.5em)
#align(
  center, 
  text(17pt)[#wk_report_name\ ] 
        + text(12pt)[\ 何瑞杰\ 中山大学, 大湾区大学]
)

// #show figure.caption: it => [
//   图
//   #context it.counter.display(it.numbering) : 
//   #it.body
// ]

= 项目进展
== 使用神经网络学习生命游戏的演化动力学

本周受台风影响和其他因素，我的精神状态欠佳，正在努力调整中。本周计划中缺失的部分将尽量在下周补齐。

=== 对模型和训练的改动
==== 增大卷积核大小

增加到 $5$ 或者 $7$，并期待训练好的模型第一层权重在九宫格以外的接近于零。

==== 缩小并行模型的参数量


==== 权重稀疏化

在优化器中添加 `weight_decay` 项，并注意一下能不能设置成 `L1`

=== 不变性
==== 对卷积权重的硬约束

==== 权重旋转不变性损失


==== 群等变 CNN


=== 生命游戏规则的变种

本周取 Golly 文档中若干邻域大小为 $3$ 的九种其他规则进行实验，每种规则的具体信息列表如下

#align(center,
  table(
    columns: 4,
    align: (center, center, center, left),
    stroke: none,
    table.hline(),
    table.header([规则名称], [邻域大小], [邻域类型], [特性描述]),
    table.hline(stroke: 0.5pt),
    [`B36/S23`],      [3], [Moore],       [和 Conway 的原版生命游戏相似，但有自我复制结构],
    [`B3678/S34678`], [3], [Moore],       [活细胞群中的死细胞的行为与死细胞群中的活细胞的行为相同],
    [`B35678/S5678`], [3], [Moore],       [有不可预测行为的菱形斑点],
    [`B2/S`],         [3], [Moore],       [活细胞每代都会死亡，但该系统常常爆发],
    [`B234/S`],       [3], [Moore],       [单个的 $2 times 2$ 会演化为一个波斯地毯],
    [`B345/S5`],      [3], [Moore],       [周期极长的振荡器可以自然地出现],
    [`B13/S012V`],    [3], [Von Neumann], [],
    [`B2/S013V`],     [3], [Von Neumann], [],
    table.hline(),
  )
)


=== 数据生成

#h(2em)经过进一步研究发现，Golly 虽然支持大量规则，但无法作为包导入 Python 中使用，只限于其程序之内。一番搜索后我找到了 `pyseagull`，并对所需的关键部分进行了检查。其运行模式十分简单，下面是一段官网给出的模拟代码：

```python
import seagull as sg
from seagull.lifeforms import Pulsar

# Initialize board
board = sg.Board(size=(19,60))

# Add three Pulsar lifeforms in various locations
board.add(Pulsar(), loc=(1,1))
board.add(Pulsar(), loc=(1,22))
board.add(Pulsar(), loc=(1,42))

# Simulate board
sim = sg.Simulator(board)
sim.run(sg.rules.conway_classic, iters=1000)
```
相比于先前代码中的逻辑，它要简单得多。即使运行过程中被包装成一个函数，我们依然可以通过 `sg.Simulator` 中的 `get_history()` 方法得到这一次模拟的所有历史数据的 `ndarray`，其形状为 `[iters+1, w, h]`，其中 `iters` 为迭代轮数，`w` 和 `h` 为网格尺寸大小。

另外，`pyseagull` 还支持自定义的简单规则。生命游戏的简单规则可以写为 `B[...]/S[...]` 的字符串格式。最经典的生命游戏的规则为 `B3/S23`，意为死细胞邻居存活数为 $3$ 时，下一时刻复活；活细胞邻居存活数为 $2$ 或 $3$ 时下一时刻继续存活，其他情况下一时刻细胞死亡。自定义函数签名如下

```python
seagull.rules.life_rule(X: ndarray, rulestring: str)
```
该函数在 `pyseagull` 的源码实现中通过正则表达式提取 `rulestring` 中的规则信息。由于其灵活性，在需要时我们可以将其拓展为其他更加复杂的规则，例如改变邻域的形状、将邻域的贡献从各向同性改为各向异性。

我正使用该库重写数据生成部分的代码，正在调试中。

=== 对模型权重的解释
==== 训练完毕的神经网络作为生命游戏模拟器


==== 通过 CNN 权重的直接解释方案


#align(bottom, 
  [
    参考资料
    + https://pyseagull.readthedocs.io/
    +
  ]
) 

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


#pagebreak()
= 文献阅读

// == Denoising Diffusion Probabilistic Models
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
// == Sliced Score Matching: A Scalable Approach to Density and Score Estimation
// - Yang Song, Sahaj Garg, Jiaxin Shi, Stefano Ermon 
// - https://arxiv.org/abs/1905.07088


// #pagebreak()
== The Yeast Cell-cycle Network is Robustly Designed
- Fangting Li, Tao Long, Ying Lu, Qi Ouyang, and Chao Tang
- https://pnas.org/doi/full/10.1073/pnas.0305937101

#h(2em)本文来自杨武岳老师的推荐阅读，通过对酵母菌细胞周期调控网络构建简化 Bool 网络，然后研究该网络的演化特征、吸引子、主要演化路径和对外界扰动的稳定性。

=== 调控酵母菌细胞周期的核心因素

#h(2em)酵母菌细胞周期或增殖可以大致划分为四个时期：
+ *$G_1$*：细胞生长，当细胞大小足够、外部营养充足时开始进入分裂过程
+ *$S$*：DNA、染色体复制
+ *$G_2$*：S 期和 M 期之间的间隔
+ *$M$*：染色体分离、细胞分裂
即使酵母菌和细胞周期相关的基因数量众多，但其中的核心角色十分有限，它们可以分为四类
+ 与激酶 (kinase) 结合的细胞周期蛋白 (cyclins) ：Cln1,-2,-3、Clb1,-2,-5,-6
+ 上述复合体的抑制 (inhibitors)、降解 (degraders) 和竞争蛋白 (competitors) ：Sic1、Cdh1、Cdc20、Cdc14
+ 转录因子 (transcription factor) ：SBF、MBF、Mcm1/SFF、Swi5
+ 检查点 (checkpoints)：细胞大小、DNA的复制和损伤、纺锤体组装 (spindle assembly)
下面是一些激活和抑制的例子：
- 细胞生长到一定大小，且环境营养充足时，激活*细胞周期蛋白(cyclin) Cln3 和激酶(kinase) Cdc28 的复合体* Cln3/Cdc28
- Cln3/Cdc28 通过磷酸化激活*转录因子(transcription factor)* SBF 和 MBF
- SBF 和 MBF 分别激活*细胞周期蛋白* Cln1,-2 和 Clb5,-6
- Sic1 蛋白可与 Cln3/Cdc28 复合体结合，抑制后者的功能
- Clb1，-2 可以使 Swi5 磷酸化，抑制后者进入细胞核
- Cdh1 可以降解 Clb1，-2
本文为了简化模型，仅仅考虑细胞大小的一个检查点，并在一些节点上添加自降解回路。具体如 @yeast-cycle 所示。

#figure(
  image("image-1.png", width: 90%),
  caption: [
    (A) 发芽酵母的细胞周期调节网络 
    (B) 仅有细胞大小这一个检查点的简化网络
  ],
) <yeast-cycle>

=== 网络动力学建模和演化实验
==== 动力学模型

#h(2em)本文的设定中，上述网络的每个节点仅拥有两种状态。具体地，设第 $i$ 个节点在 $t$ 时刻的状态为 $S_i (t)$，那么有 $S_i (t) in \{0, 1\}$，并且有下面的动力学


$
S_i (t+1) = cases(
  1 & display(sum_(j in cal(N)(i)) a_(i,j)S_j (t) > 0),
  0 & display(sum_(j in cal(N)(i)) a_(i,j)S_j (t) < 0),
  S_i (t)#h(2em) & display(sum_(j in cal(N)(i)) a_(i,j)S_j (t) = 0)
) 
$ <eq-system-dynamics>

其中 $a_(i,j)$ 表示图中从节点 $i$ 指向节点 $j$ 的箭头之权重。在本文设定中，对于红色（抑制）箭头，对应的 $a_(i,j)$ 均被设置为 $a_r$，绿色箭头均被设置为 $a_g$，且有 $a_g = -a_r = 1$。对于带有自降解回路的节点，其自降解规则如下。假如某个带自降解回路的节点 $i$ 的外部输入（也就是前面的和式）在 $t+1$ 时刻到 $t + t_d$ 时刻都为零，则该节点的状态在 $t+t_d$ 时刻变为零，即 $S_i (t+t_d ) = 0$。 

==== 吸引子

#h(2em)作者让网络由所有 $2^11$ 种初始状态开始演化，发现*所有的起始状态都将演化到七个吸引子中的一个*，其中最多初始状态演化到达的那个吸引子对应的是占细胞周期时间最长的 G1 期。各吸引子的状态如下图所示。

#figure(
  image("image-2.png", width: 90%),
  caption: [
    每个吸引子容纳的起始状态大小及吸引子对应的各节点状态
  ],
  kind: table
) <attractors>

==== 生物演化路径

#h(2em)下一步作者通过启动检查点信号（细胞大小）手动启动“细胞”的生命周期，*发现网络的演化方向和真实世界中酵母菌的演化过程相同，即 $G_1  -> S -> G_2 -> M -> G_1$，如 @tab-main-trajectory 所示。
除此之外，其他不在 $G_1$ 期对应状态开始演化的情形，也会随着演化过程回归到这条路径，然后沿着这条路径演化至回到 $G_1$ 期，如 @fig-trajectory-visualization 所示。*

#figure(
  image("image-3.png", width: 100%),
  caption: [
    每个吸引子容纳的起始状态大小及吸引子对应的各节点状态
  ],
  kind: table
) <tab-main-trajectory>


#figure(
  image("image-4.png", width: 90%),
  caption: [
    最终流向吸引子 $G_1$（蓝色圆点）的 $1,764$ 个状态（绿色圆点）的状态转移图。橙色箭头表示从一个状态演化至另一个状态。箭头和圆点的粗细或大小表示经过它们的演化路径数目。蓝色箭头指示现实中酵母菌的系统演化路径。
  ],
)<fig-trajectory-visualization>

=== 和随机网络之比较
==== 吸引子和演化路径重合程度

#h(2em)作者也将酵母菌的调控网络与相同节点数量的随机网络进行了比较，后者的吸引子关于其容纳的初始状态多少呈现幂律分布。作者还定义了一个变量 $w_n$ 其中 $n$ 指示某个系统初始状态。该变量指示不同初始状态的演化路径之间的重叠程度，如 @fig-comparizon-1 所示。具体而言，先定义 $T_(n,i-1,i)$ 为系统所有初始状态演化路径中，经过当前初始状态在 $t-1$ 时刻演化至 $t$ 时刻经过的有向边 $A_(j,k)$ (这里的有向边是指如 @fig-trajectory-visualization 那样的系统状态迁移图的有向边，不是系统本身的图结构) 的演化路径条数，$L_n$ 定义为该初始状态下演化路径到达吸引子经过的状态数目 ( @fig-trajectory-visualization 中到达吸引子之前的跳数)。则 $w_n$ 定义为

$
w_n = 1/L_n sum_(i=1)^(L_n) T_(i-1,i)
$

通过图中可以看到，*随机网络产生像酵母菌调控网络那样的一个巨大吸引子的概率很低，且后者的 $w_n$ 值总体更高，这说明大部分的演化路径重叠度高、殊途同归。*

#figure(
  image("image-6.png", width: 90%),
  caption: [
    酵母菌调控网络和相同节点数的随机网络之比较。(A) 随机网络吸引子大小的分布 (B) 酵母菌网络和随机网络的 $w_n$ 值分布 (C) $w_n$ 值的计算规则。
  ],
)<fig-comparizon-1>

==== 网络稳定性

#tab 作者接下来对酵母菌网络和用于与之对比的随机网络做了扰动实验，观察扰动后系统演化中的最大吸引子容量的大小变化 $Delta B "/" B$。其中 $B$ 为扰动前的最大吸引子容量，$Delta B$ 为扰动后最大吸引子容量的变化量。本文作者采用的扰动方式 (对系统本身的网络，而不是状态演化图) 具体为：*删除一条边、增加一条本不存在的边 (可能是绿色或红色)、将一条边变成另一种颜色*。作者对这三种扰动造成的 $Delta B "/" B$ 大小以及三种方法的平均效应进行统计，结果如 @fig-perturbation-B 所示。

#figure(
  image("image-5.png", width: 70%),
  caption: [
    酵母菌调控网络和相同节点数的随机网络面临扰动时 $Delta B "/" B$ 的分布。(A) 随机删除 34 个连接 (B) 随机添加 174 条连接 (C) 随机改变 29 条连接的性质 (D) A-C 的平均
  ],
)<fig-perturbation-B>

原论文在得到结果后声称相比于随机网络，酵母菌网络在扰动下 $Delta B "/" B = 1$ 的频数显著下降，该数值对应着之前的吸引子在扰动后被完全抹去。原文作者根据上图结果还得出了“对大部分扰动而言，最大吸引子的相对容量变化很小”，很遗憾我并未从图中得出：如 (A)、(C)、(D) 中的酵母菌网络在扰动后存在相当一部分 $Delta B "/" B > 0.5$ 的情况，因此我认为这个推论有些差强人意。

作者对扰动后 $w$ 值的变化分布也进行了统计，得到的结果与 $Delta B "/" B$ 的分布相似。

=== 演化动力学的其它参数组合和其他检查点

#tab 作者尝试其他情况的 @eq-system-dynamics 中的参数，发现系统动力学对参数 $a_(i,j)$ 的选取不敏感。对于其他检查点的情况，作者进行了单一开放一个其他检查点的实验，实验结果和上文类似。

=== 总结和讨论 

#tab 本文发现酵母菌的调控网络具有一个超级吸引子，且生物意义上的标准系统演化过程对应着模拟实验中的吸引轨道。由于本文中对酵母菌系统进行了较大的简化，一个合理的猜测是考虑更多因素的系统依然会具有简化后系统的良好性质（即对应 $G_1$ 期的超级吸引子、吸引轨道和对扰动的稳定性）。这样的性质可以使的细胞周期系统也许可以和其他系统更好的配合，并使酵母菌在进化中占据有利地位。

#pagebreak()

// = 学习进度
// == 机器学习理论
// === Markov Chain Monte Carlo (MCMC)



// == 随机过程



// #h(2em)本周


// == 随机微分方程

// #h(2em)本周

// #pagebreak()
= 问题解决记录

== Typst 相关
=== `grid()` 中的多重对齐

多重对齐可以用 `+` 同时指定。例如想实现高度居中（`horizon`）和水平居中（`center`），则可以在对应函数的 `align` 参数中填入 `horizon+center`。下图中从左至右的对齐方式分别为 `center`、`left+horizon` 和 `bottom+center`。

#grid(
  columns: (33%, 33%, 33%),
  align: (center, left+horizon, bottom+center), 
  stroke: red,
  table(
    columns: 3,
    stroke: none,
    table.hline(),
    table.header([a], [b], [c]),
    table.hline(stroke: 0.5pt),
    [d], [e], [f],
    [g], [h], [i],
    [j], [k], [l],
    [m], [n], [o],
    table.hline(),
  ) ,
  table(
    columns: 3,
    stroke: none,
    table.hline(),
    table.header([a], [b], [c]),
    table.hline(stroke: 0.5pt),
    [d], [e], [f],
    [g], [h], [i],
    table.hline(),
  ),
  table(
    columns: 3,
    stroke: none,
    table.hline(),
    table.header([a], [b], [c]),
    table.hline(stroke: 0.5pt),
    [d], [e], [f],
    [g], [h], [i],
    [j], [k], [l],
    table.hline(),
  )
)

具体地，typst 文档中定义了下面的若干种对齐变量值：

- `start`：对齐至文本方向起点
- `end`：对齐至文本方向终点
- `left`：对齐至左侧边界
- `center`：对齐至水平方向的中央位置
- `right`：对齐至右侧边界
- `top`：对齐至上侧边界
- `hotizon`：对齐至水平中线
- `bottom`：对齐至下册边界

=== 三线表的绘制

#grid(
  columns: (48%, 48%),
  gutter: 4%,
  align: horizon + center, 
  [
    ```typst
    #table(
      columns: 3,
      stroke: none,
      table.hline(),
      table.header([a], [b], [c]),
      table.hline(stroke: 0.5pt),
      [d], [e], [f],
      [g], [h], [i],
      table.hline(),
    )
    ```
  ],
  table(
  columns: 3,
  stroke: none,
  table.hline(),
  table.header([a], [b], [c]),
  table.hline(stroke: 0.5pt),
  [d], [e], [f],
  [g], [h], [i],
  table.hline(),
)
)

参考资料
+ https://typst.app/docs/reference/model/table/#parameters-align

=== 数学公式自动编号

实现简单的数字编号，只需添加下面的代码至文件开头

```typst
#set math.equation(numbering: "(1)")
```


== Python 相关

== 深度学习相关


#pagebreak()
= 下周计划
*论文阅读*
+ 生成模型
  - Denoising Diffusion Probabilistic Models 完成
  - Sliced Score Matching: A Scalable Approach to Density and Score Estimation
+ 动力学
  - 暂无

*项目进度*
+ 使用神经网络学习生命游戏的演化动力学
  - 填补所有本周未完成的部分
+ 微型抗癌机器人在血液中的动力学
  - 尝试学习并撰写 PDE 的数值模拟代码

*理论学习*
+ 随机过程课程
  - 完成第一章（独立随机变量序列）和第二章（Poisson过程）
+ 随机微分方程
  - 完成第四章，并开启第五章
+ 机器学习理论
  - Markov Chain Monte Carlo
  
