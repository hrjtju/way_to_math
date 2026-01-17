#import "@preview/algorithmic:1.0.7"
#import algorithmic: (
  style-algorithm, 
  algorithm-figure
)
#import "@preview/ctheorems:1.1.3": *
#import "@preview/mitex:0.2.4": *
#import "@preview/numbly:0.1.0": numbly
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.7": *

#show strong: set text(blue)
#show: thmrules.with(qed-symbol: $square$)
#show: codly-init.with()
#show link: underline

#codly(
  languages: codly-languages, 
  zebra-fill: none, 
  display-icon: false,
)

#let MM1 = $M"/"M"/"1$

#let wk_report_name = "中山大学 DCS5706《随机过程及应用》期末作业"
#let header_name = "中山大学 DCS5706《随机过程及应用》期末作业"
#let project_name = [#MM1 排队系统的控制研究]
#let name_no = "何瑞杰 25110801"

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

#set text(lang: "zh", font: "Kai")

#set page(
  paper: "a4",
  numbering: "1",
  header: header_name + " | " + name_no,
)

#set par(
  first-line-indent: 2em,
  justify: true,
)

#set heading(numbering: "1.")

#align(
  center, 
  text(12pt)[#wk_report_name\ ] + v(0.5em) 
        + text(17pt)[#project_name\ ]
        + text(12pt)[\ #name_no]
)

#align(center, [摘#h(2em)要])
#pad(
  left: 6em, right: 6em, 
  [
    
  ]
)

#outline(depth: 2)


= 问题描述

#h(2em) #MM1 排队系统广泛存在于生产生活中，它指的是一个先到先服务的单服务台的服务系统。顾客按照参数为 $lambda$ 的 Poisson 过程到达；服务台的服务时间服从参数为 $mu$（即服务速率）的指数分布，且和顾客的到达过程独立。

现考虑带有服务速率控制的 #MM1 排队系统，其服务速率 $mu(i) in (lambda, macron(mu)]$ 取决于系统中的顾客数目 $i$，该数目包括等待的顾客和正在服务的顾客。系统有两重成本：第一重为单位时间的服务成本 $q(mu)$，其满足 $q(0) = 0$；第二重为顾客等待成本 $c(i)$。对该 #MM1 系统的控制目标为对系统内不同顾客数量时采用不同服务速率，以期最小化单位总成本。

= 模型建立

#h(2em) 带有控制的 #MM1 排队系统可使用连续时间 Markov 决策过程建模，其各参数如下：

#figure(
  table(
    columns: 2,
    stroke: none,

    table.hline(),
    table.header([CTMDP 资料], [#MM1 系统中的元素]),
    table.hline(stroke: 0.5pt),
    [状态 $x(t)$], [系统中该时刻的顾客数目 $i$],
    [动作 $u(t)$], [系统该时刻的服务速率 $mu$], 
    [代价函数 $g(x(t), u(t))$], [单位时间总成本 $q(mu)+c(i)$], 
    [策略 $mu_k$], [系统的服务速度策略 $mu(i)$],
    table.hline()
  )
)

若转移速度对所有状态和动作均匀，则有

$
  J_pi (x_0) = EE lr([sum_(k=0)^infinity (nu/(beta + nu))^k g(x_k, mu_k (x_k))/(beta + nu)]) = EE lr([sum_(k=0)^infinity alpha^k dot.c tilde(g)(x_k, mu_k (x_k))]),
$

对应的 Bellman 方程为

$
  J(i) = 1/(beta + nu) min_(u in U(i)) lr([g(i,u) + nu sum_j p_(i,j)(u) J(j)])
$

考虑转移速度对所有状态和动作不均匀，但存在上界 $nu$，若对状态 $i$ 和动作 $u$，有转移速度 $nu_i (u)$，考虑下面拥有新的转移概率的均匀转移速度的 CTMDP：

$
  tilde(p)_(i,j) = cases(
    display((nu_i (u))/nu p_(i,j) (u)) #h(2em) & "if" i eq.not j, 
    display((nu_i (u))/nu p_(i,i) (u) + 1 - (nu_i (u))/nu) #h(2em) & "if" i = j, 
  )
$

因此新的 CTMDP 的 Bellman 方程为

$
  J(i) = 1/(beta + nu) min_(u in U(i)) lr([g(i,u) + (nu - nu_i (u)) J(i) + nu_i (u) sum_j p_(i,j)(u) J(j)])
$

在 #MM1 队列中，转移速率 $nu_i (mu)$ 在系统中无顾客（$i = 0$）时为 $lambda$，在有顾客时为 $lambda + mu$，则依照上述结果的转移速率上界为 $nu = lambda + macron(mu)$。由于该系统的状态只可能向相邻状态转移，且当系统中没有顾客时，规定 $mu(0) = 0$，因此可以得到其 Bellman 方程为

$
  J(i) = cases(
    display(1/(beta + nu)  lr([c(0) + (nu - lambda) J(0) + lambda J(1)]))  & i = 0,
    display(1/(beta + nu) min_(mu) lr([c(i)+q(mu) + (nu - lambda - mu) J(i) + lambda J(i+1) + mu J(i-1)])) #h(2em) & 1 lt.slant i lt M,
    display(1/(beta + nu) min_(mu) lr([c(M)+q(mu) + (nu - mu) J(M) + mu J(M-1)]))  & i = M,
  )
$

注意系统中转移概率 $p_(i,i+1)(u)$ 对应着新顾客进入系统，其值为 $display(lambda/(lambda + mu))$，而 $p_(i,i-1)(u)$ 对应着顾客服务完成离开系统，其值为 $display(mu/(lambda + mu))$。

= 最优策略计算

#h(2em) 本节介绍代价函数的取法和求解 Bellman 方程用到的算法。

== 代价函数

#h(2em) 本项目研究排队代价和服务代价分别为线性、二次函数、指数函数情况时的最优控制策略，共有九种组合。具体地，线性、二次代价和指数代价分别取
$
  f_"linear" (x) &= x,\
  f_"quad" (x) &= 1/2 x^2, \
  f_"exp" (x) &= e^(0.1x).
$

== 值迭代

#h(2em) 第一种求解方法是值迭代，其原理为直接应用 Bellman 方程的定义，并用其迭代让边界处的值逐渐传导到其他各个状态，直至收敛：
$
  J_(k+1) (i) = min_(u in U(i)) lr([g(i,u) + sum_(j=1)^n p_(i,j)(u)J_k (j)])
$
在 #MM1 系统中，值迭代算法可以写为

#show: style-algorithm
#algorithm-figure(
  "Value Iteration for Controlled M/M/1 Queue",
  vstroke: .5pt + luma(200),
  {
    import algorithmic: *
    Procedure(
      "Value-Iteration",
      ($c(i)$, $q(mu)$, $lambda$, $macron(mu)$, $beta$, $epsilon$),
      {
        Assign[$nu$][$lambda + macron(mu)$];
        Assign[$J_0(i)$][$0, forall i in {0, ..., N}$];
        Assign[$k$][$0$];
        While($"true"$, {
          Assign[$J_(k+1)(0)$][$display(1/(beta + nu) [c(0) + (nu - lambda)J_k(0) + lambda J_k(1)])$];
          For($i <- 1, ..., N-1$, {
            Assign[$J_(k+1)(i)$][ 
             $1/(beta + nu) min_(mu in (lambda, macron(mu)]) [c(i) + q(mu) + mu J_k(i-1) + (nu - lambda - mu)J_k(i) + lambda J_k(i+1)]$
            ]
          })
          Assign[$J_(k+1)(N)$][$J_(k+1)(N-1)$]
          If($max_i |J_(k+1)(i) - J_k(i)| < epsilon$, {
            Break
          })
          Assign[$k$][$k + 1$]
        })
        Assign[$mu^*(i)$][$arg min_(mu) [q(mu) - mu(J_(k+1)(i) - J_(k+1)(i-1))], forall i >= 1$]
        Return[$(J_(k+1), mu^*)$]
      }
    )
  }
)

== 策略迭代

#show: style-algorithm

#algorithm-figure(
  "Policy Iteration for Controlled M/M/1 Queue",
  vstroke: .5pt + luma(200),
  {
    import algorithmic: *
    Procedure(
      "Policy-Iteration",
      ($c(i)$, $q(mu)$, $lambda$, $macron(mu)$, $beta$),
      {
        Assign[$mu_0(i)$][$macron(mu), forall i in {1, ..., N}$]
        Assign[$k$][$0$]
        While($"true"$, {
          
          Line[ Solve linear system for$J_(mu_k) $: ]
          Line[$(beta + lambda)J(0) - lambda J(1) = c(0)$]
          For($i <- 1, ..., N-1$, {
            Line[$-mu_k(i)J(i-1) + (beta + lambda + mu_k(i))J(i) - lambda J(i+1) = c(i) + q(mu_k(i))$]
          })
          
          
          For($i <- 1, ..., N$, {
            Assign[$Delta J$][$J_(mu_k)(i) - J_(mu_k)(i-1)$]
            Assign[$mu_(k+1)(i)$][$arg min_(mu in (lambda, macron(mu)]) [q(mu) - mu dot Delta J]$]
          })
          
          
          If($mu_(k+1)(i) = mu_k(i), forall i$, {
            Break
          })
          Assign[$k$][$k + 1$]
        })
        Return[$(J_(mu_k), mu_k)$]
      }
    )
  }
)

= 仿真验证



= 模型参数对最优策略的影响


= 结论


= 代码附录

// #pagebreak()

// #set text(lang: "en")
// #bibliography("ref.bib",   // 你的 BibTeX 文件
//               title: "参考文献",
//               style: "ieee", 
//               full: true
// )
