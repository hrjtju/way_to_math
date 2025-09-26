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

#let wk_report_name = "2025年9月22日至9月28日周报"
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

// #show figure.caption: it => [
//   图
//   #context it.counter.display(it.numbering) : 
//   #it.body
// ]

= 项目进展
== 使用神经网络学习生命游戏的演化动力学
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


=== 建模


#pagebreak()
= 文献阅读

== Denoising Diffusion Probabilistic Models
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
== Sliced Score Matching: A Scalable Approach to Density and Score Estimation
Yang Song, Sahaj Garg, Jiaxin Shi, Stefano Ermon | https://arxiv.org/abs/1905.07088


#pagebreak()
== The Yeast Cell-cycle Network is Robustly Designed

=== 调控酵母菌细胞周期的核心因素

#h(2em)即使酵母菌和细胞周期相关的基因数量众多，但其中的核心角色十分有限：它们可以分为四类
+ 与激酶 (kinase) 结合的细胞周期蛋白 (cyclins) ：Cln1,-2,-3、Clb1,-2,-5,-6
+ 上述复合体的抑制 (inhibitors)、降解 (degraders) 和竞争蛋白 (competitors) ：Sic1、Cdh1、Cdc20、Cdc14
+ 转录因子 (transcription factor) ：SBF、MBF、Mcm1/SFF、Swi5
+ #highlight[检查点 (checkpoints)：细胞大小、DNA的复制和损伤、主轴组件]


@yeast-cycle 是 下面是一些激活和抑制的例子：

+ 细胞生长到一定大小，且环境营养充足时，激活*细胞周期蛋白(cyclin) Cln3 和激酶(kinase) Cdc28 的复合体* Cln3/Cdc28
+ Cln3/Cdc28 通过磷酸化激活*转录因子(transcription factor)* SBF 和 MBF
+ SBF 和 MBF 分别激活*细胞周期蛋白* Cln1,-2 和 Clb5,-6
+ Sic1 蛋白可与 Cln3/Cdc28 复合体结合，抑制后者的功能
+ Clb1，-2 可以使 Swi5 磷酸化，抑制后者进入细胞核
+ Cdh1 可以降解 Clb1，-2

#figure(
  image("image-1.png", width: 90%),
  caption: [
    (A) 发芽酵母的细胞周期调节网络 
    (B) 仅有细胞大小这一个检查点的简化网络
  ],
) <yeast-cycle>



#pagebreak()
= 学习进度
== 机器学习理论
=== Markov Chain Monte Carlo (MCMC)

== 随机过程

#h(2em)本周


== 随机微分方程

#h(2em)本周

#pagebreak()
= 问题解决记录

== Typst 相关
=== `grid()` 中的多重对齐


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





== Python 相关

== 深度学习相关


#pagebreak()
= 下周计划
*论文阅读*
+ 生成模型
  - 
+ 动力学
  - 

*项目进度*
+ 使用神经网络学习生命游戏的演化动力学
  - 
+ 微型抗癌机器人在血液中的动力学
  -

*理论学习*
+ 随机过程课程
  - 
+ 随机微分方程
  - 
