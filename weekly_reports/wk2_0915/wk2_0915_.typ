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
Jonathan He, Ajay Jain and Pieter Abbeel | https://arxiv.org/abs/2006.11239

有关加噪过程中的权重推导
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


负对数似然函数的变分上界

$
EE_(pt)[-log pt(bx_0)] 
&= EE_(pt)[-log pt(bx_0) int q(bx_(1:T)|bx_0) dd bx_(1:T)] \
&= EE_(pt)[-int q(bx_(1:T)|bx_0)log pt(bx_0)  dd bx_(1:T)] \
&= - EE_(pt(bx_0),  q(bx_(1:T)|bx_0))[
  log (pt(bx_0) pt(bx_(1:T)|bx_0)) / (pt(bx_(1:T)|bx_0))
]\
&= - EE_(pt(bx_0),  q(bx_(1:T)|bx_0))[
  log (#text(red)[$pt(bx_(0:T))$] q(bx_(1:T)|bx_0)) / (pt(bx_(1:T)|bx_0) #text(red)[$q(bx_(1:T)|bx_0)$])
]\
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

将变分上界写成三项 KL 散度之和

$
L 
&= EE_q [ log q(bx_(1:T)|bx_0) /(pt(bx_(0:T)) ) ] \
&= EE_q [ log (q(bx_T|bx_0) prod_(t=2)^T q(bx_(t-1)|bx_t, bx_0)) / (pt(bx_T) prod_(t=1)^T pt(bx_(t-1)|bx_t)) ] \
&= EE_q [ log (q(bx_T|bx_0) ) / (pt(bx_T)) + sum_(t=2)^T (q(bx_(t-1)|bx_t, bx_0))/(pt(bx_(t-1)|bx_t)) + log 1/(pt (bx_(0)|bx_1))] \
&= EE_q [KL[q(bx_T|bx_0)\|pt(bx_T)] + sum_(t=2)^T KL[q(bx_(t-1)|bx_t, bx_0)\|pt(bx_(t-1)|bx_t)] - log pt (bx_(0)|bx_1)] \
&= EE_q [L_T + sum_(t=2)^T L_(t-1) - L_0].
$

$q(bx_(t-1) | bx_t, bx_0)$ 的计算

$
q(bx_(t-1) | bx_t, bx_0)
&= q(bx_t | bx_(t-1), bx_0) q(bx_(t-1) | bx_0) / q(bx_t | bx_0) \
&= q(bx_t | bx_(t-1)) q(bx_(t-1) | bx_0) / q(bx_t | bx_0) \
&= N(bx_t; sqrt(1 - beta_t) bx_(t-1), beta_t I) N(bx_(t-1); sqrt(macron(alpha)_(t-1)) bx_0, (1 - macron(alpha)_(t-1)) I) / N(bx_t; sqrt(macron(alpha)_t) bx_0, (1 - macron(alpha)_t) I) \
&= N(bx_(t-1); mu_q(bx_t, bx_0), Sigma_q) \
& quad mu_q(bx_t, bx_0) = (sqrt(macron(alpha)_(t-1)) beta_t bx_0 + sqrt(1 - beta_t) (1 - macron(alpha)_(t-1)) bx_t) / (1 - macron(alpha)_t),  wide Sigma_q = (beta_t (1 - macron(alpha)_(t-1))) / (1 - macron(alpha)_t) I
$

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
