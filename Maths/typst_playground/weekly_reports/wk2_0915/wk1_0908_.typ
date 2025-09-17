#import "@preview/algorithmic:1.0.5"
#import algorithmic: style-algorithm, algorithm-figure

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
#let dd = "d"
#let ito = $"It"hat("o")$

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
  text(17pt)[#wk_report_name\ ] + text(12pt)[\ 何瑞杰\ 中山大学, 大湾区大学]
)

= 项目进展
== 使用神经网络学习生命游戏的演化动力学

== 微型血管机器人


#pagebreak()
= 文献阅读
== Denoising Diffusion Probabilistic Models
Jonathan He, Ajay Jain and Pieter Abbeel | https://arxiv.org/abs/2006.11239


#h(-2em)参考资料
+ 

#pagebreak()
== Sliced Score Matching: A Scalable Approach to Density and Score Estimation
Yang Song, Sahaj Garg, Jiaxin Shi, Stefano Ermon | https://arxiv.org/abs/1905.07088


#pagebreak()
= 学习进度

== 随机过程

#h(2em)首先复习回顾了有关于尾部概率的诸不等式。

== 随机微分方程

#h(2em)

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
