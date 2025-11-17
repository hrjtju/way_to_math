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

#set text(lang: "zh", font: ("New Computer Modern", "Kai", "KaiTi"))

// Snippets
#let const = "constant"
#let bs = $bold(s)$
#let bf = $bold(f)$
#let bF = $bold(F)$
#let bg = $bold(g)$
#let bG = $bold(G)$
#let bx = $bold(x)$
#let bX = $bold(X)$
#let bw = $bold(w)$
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
#show link: underline

#set par(
  first-line-indent: 2em,
  justify: true,
)

#show figure.where(
  kind: table
): set figure.caption(position: top)

#set underline(offset: 2.5pt, stroke: 0.5pt)

#show figure.caption: it => [
  #underline[
    #it.supplement #context it.counter.display(it.numbering)
  ]
  #h(5pt)#it.body
]

#set math.equation(numbering: "(1)")
#set math.mat(delim: ("[", "]"), align: center)
#set heading(numbering: "1.")
#set math.cases(gap: 0.5em)

// metadata
#let wk_report_name = "2025年11月10日至11月16日周报"
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

#tab 综合上周提出的总框架图（#ref(<fig:life_framework>)），本周开始逐个考虑并使用代码实现其中的各关键部分。

#figure(
  image("framework.png"),
  caption: [方法框架总览]
)<fig:life_framework>

== 数据供应代理

#tab 可以用现成数据集代替，但修改可见的演化轨道条数。在初始化时，令可见演化轨道条数为 $1$，每执行完毕一个主动学习循环，且需要继续循环时，可见演化轨道条数 $+1$，从而已收集好的新的一条轨道可以加入训练数据中。另外，还可以考虑将新的轨道赋予更大的权重，可以在 `__getitem__` 方法中添加一个输出，表示其权重，相应地将维护一个可见轨道条数改为维护一个可见轨道的有序列表，依据列表中轨道的先后赋予依次从低到高的权重。

== 等变神经网络模块

#tab 神经网络训练模块有现成的代码，但是需要解决框架图中的施加等变约束的部分。注意刚开始的时候是只有平移不变性的，然后再加上其他的群等变约束，这意味着如果采用群等变卷积网络（`e2cnn`）包的话需要仔细研究其源码实现。一个暂行的替代方案是用普通的 CNN，然后加上不变性损失。例如对于晶体群 $p 4$，可以令 $3 times 3$ 卷积核的上下左右和四个对角的值分别相等，可以采取强制相等，或者旋转损失的方式添加等变性约束。另外同时可以研究 `e2cnn` 的底层实现，来研究如何在不断扩充等变群元时最大程度保留上一次训练好的网络权重，避免需要再人为蒸馏一遍。

== 指示数据集、规则提取器和生成器

#tab 先说指示数据集。虽然它生成起来非常简单：
```python
w, h = 3, 3
ls = [[0, 1] for _ in range(w*h)]
indicator_dataset = None
for idx, i in enumerate(product(*ls)):
    arr = torch.tensor(list(i)).reshape(1, 3, 3)
    if indicator_dataset is None:
        indicator_dataset = arr
    else:
        indicator_dataset = torch.concatenate([indicator_dataset, arr])
```
但在 `w` 和 `h` 变大时，如果要遍历所有的 $0$-$1$ 组合，其组合数将指数级增长（$2^(w h)$），带给生成和验证过程以很大的困难。另外，拿到指示数据集 `indicator_dataset` 和对应的网络预测值后，需要思考的是如何将其转化为确定性规则的问题。

在先前的想法中是选取网络的预测置信度高的转变对 $(x_t, N(x_t)) -> x_(t+1)$，但需要考虑两个问题，一是 `pyseagull` 的模拟器中是通过卷积来实现规则的，我们需要想办法将得到的转变对转化成卷积核的形式。换言之，我们需要将得到的这一堆转变对转换为显式的规则形式，我目前还没有什么好的想法。

#pagebreak()
= 文献阅读
== [续] Score-based generative modeling through SDE #cite(<DBLP:paper-yang_song-score_based_generative_modeling_sde>)

Yang Song et al. | https://arxiv.org/abs/2011.13456

// === 补遗



// ==== 逆向 SDE 的推导

// ==== 概率流 ODE 的推导 


=== 可控生成

在可控生成中，给定标签 $by$，我们需要从假设的位置分布 $p_0 (bx_0|by)$ 采样，即从对应于该类的先验分布 $p_N (bx_T|by)$ 开始采样。此时对应的对数梯度就变为
$
  nabla_bx p_t (bx|by) &= nabla_bx log p_t (bx) + nabla_bx log p_t (by|bx) - underbrace(nabla_bx p_t (by), =0)\ 
  &= nabla_bx log p_t (bx) + nabla_bx log p_t (by|bx) 
$

因此得到的逆向 SDE 的形式就是

$
  dd bx = { bf(bx, t) - g(t)^2 [ redText(nabla_bx log p_t (bx) + nabla_bx log p_t (by|bx)) ]} dd t + g(t) dd tilde(bw)
$

当 $by$ 表示标签时，我们需要额外训练一个神经网络，以预测加噪过程中的 $p_t (by|bx_t)$。

==== 图像修补
#tab 如果做图像修补任务，我们需要根据不全的图像点信息 $Omega(by)$ 预测完整的图像信息 $by$。如果记 $macron(Omega)$ 为图像的未知部分，并记 $bf_(macron(Omega))$ 和 $bG_(macron(Omega))$ 为相应函数在未知像素点 $macron(Omega)$ 的限制，图像修补问题就是从 $p(macron(Omega)(by)|Omega(bx_0) = by)$ 中采样。定义一个对 $bz(t) = macron(Omega)(bx(t))$ 的新的扩散过程：
$
  dd bz = bf_(macron(Omega)) (bz, t) dd t + bG_(macron(Omega)) (bz, t) dd bw
$
这对应的逆向条件 SDE 为
$
  dd bz = { bf_(macron(Omega))& (bz, t) - nabla dot [bG_(macron(Omega)) (bz, t) bG_(macron(Omega)) (bz, t) ^top] \ &- bG_(macron(Omega)) (bz, t) bG_(macron(Omega)) (bz, t) ^top nabla_bz log p_t (bz|Omega(z(0))=by)}dd t + bG_(macron(Omega)) (bz, t) dd macron(bw)
$
注意上式中的 $p_t (bz|Omega(z(0))=by)$ 无法计算，可以用下面的方法近似。记为 $Omega(z(0))=by$ 事件 $A$，有 
$
  p_t (bz|Omega(z(0))=by) 
  &= p_t (bz|A) = int p_t (bz|Omega(z(t)), A) p_t (Omega(bx(t))|A) dd Omega(bx(t))\
  &= EE_(p_t (Omega(bx(t))|A)) [p_t (bz|Omega(z(t)), A)] approx EE_(p_t (Omega(bx(t))|A)) [p_t (bz|Omega(z(t)))]
  &approx p_t (bz|hat(Omega)(z(t)))
$
其中 $hat(Omega)$ 是 $p_t (Omega(bx(t))|A)$ 对于梯度，可以有下面的估计
$
  nabla_bz log p_t (bz(t)|Omega(bz(0)=by)) &approx nabla_bz p_t (bz(t)|hat(Omega)(bx(t)))\
  &= nabla_bz p_t ([bz(t); hat(Omega)(bx(t))])
$
其中 $[bz(t); hat(Omega)(bx(t))]$ 表示一个图像向量 $bold(u)(t)$，满足 $Omega(bold(u)(t))=hat(Omega)(bx(t))$以及$macron(Omega)(bold(u)(t)) = bz(t)$，换句话说 $bold(u)(t)$ 就是由 $bz(t)$ 和 $hat(Omega)(bx(t))$ 两部分拼起来的一个完整的图像向量。

==== 图像着色

#tab 对于图像着色，可以先用一个正交矩阵将灰度图向三个颜色方向做正交投影，然后对每个通道分别做图像补全。

==== 一般的反问题

#tab 正问题为从 $bx$ 根据条件分布 $p(bx|by)$ 生成 $by$，相应地反问题需要从 $by$ 根据反向条件分布 $p(by|bx)$ 生成 $bx$。我们可以考虑一系列用 SDE 扰动得到的扩散过程 ${bx(t)}_(t=0)^T$，和一个训练用于估计 $nabla_bx log p_t (bx(t))$ 的分数模型 $bs_(bold(theta)^*) (bx(t), t)$。只要我们能估计 $nabla_bx log p_t (bx(t)|by)$，我们就可以用逆向 SDE 得到 $p_0 (bx(0)|y)$，也就是 $p(bx|by)$。首先有
$
  nabla_bx log p_t (bx(t)|by) = nabla_bx log int p_t (bx(t)|by(t), by) p(by(t)|by) dd by(t)
$
其中 $by(t)$ 是 $bx(t)$ 关于前向过程 $p(by(t)|bx(t))$ 生成的结果。

现在假定 $p(by(t)|by)$ 是可达的，且 $p_t (bx(t)|by(t), by) approx p_t (bx(t)|by(t))$（当 $t$ 较小时，$by(t)$ 和 $by$ 相差无几；而当 $t$ 较大时，$by$ 对 $bx(t)$ 的影响可以忽略不计。）根据这两个条件，可以得到
$
  nabla_bx log p_t (bx(t)|by) 
  &approx nabla_bx log int p_t (bx(t)|by(t)) p(by(t)|by)dd by\
  &approx nabla_bx log p_t (bx(t)|hat(by)(t)) \
  &approx nabla_bx log p_t (bx(t)) + nabla_bx log p_t (hat(by)(t)|bx(t)) \
  &approx bs_(bold(theta)^*) (bx(t), t) + nabla_bx log p_t (hat(by)(t)|bx(t))
$
其中 $hat(by)(t)$ 是从 $p(by(t)|by)$ 中采样得到的。得到梯度的形式后就可以将它放进条件逆向 SDE 中求解了。


#pagebreak()

// == Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data

// *YongKyung Oh, Dong-Young Lim, Sungil Kim | #link("https://arxiv.org/abs/2402.14989v6")[ICLR 2024]*

= 学习进度
// == 机器学习理论
// === Markov Chain Monte Carlo (MCMC)


// == 随机过程

// #h(2em)本周继续系统学习 Markov 链。

// #pagebreak()

== 随机微分方程

#tab 本周结束了 Ito 随机微分那一章。其中一度卡在了章末的注记法部分。以乘积公式为例，


接着我最终进入随机微分方程一章，并学习了若干种典型形式的随机微分方程，例如股市模型、Brown 桥、随机谐振子、Langevin 方程以及其更一般地形式，即 Ornstein–Uhlenbeck 过程。
// === #ito 链式法则和乘积法则的多维情形

// === 典型的随机微分方程
// ==== 股市模型

// ==== Brown 桥

// ==== Langevin 方程

// ==== Ornstein–Uhlenbeck (OU) 过程

// ==== 随机谐振子



// === 解的存在性和唯一性

// === 随机微分方程的数值模拟

// #tab 考虑一个自守的 SDE 
// $
//   dd X = F(X) dd t + G(X) dd bold(W)
// $
// 下面给出模拟它的三种方法。

// ==== Euler-Maruyama 法

// Euler-Maruyama 法主要思想是 Brown 运动 $W(dot)$ 的增量独立性，即 $W(t + Delta t) - W(t) ~ cal(N)(0, 1)$，因此有
// $
//   X(t + Delta t) approx X(t) + F(X) Delta t + sqrt(Delta t) dot G(X) Z
// $
// 其中 $Z ~ cal(N)(0, 1)$

// ==== Milstein 法

// Milstein 法相比于 Euler-Maruyama 法更细，来源于对 $display(int_t^(t+Delta t) G(X)dd W)$ 更精确的估计。其形式为
// $
//   X(t + Delta t) approx& X(t) + F(X) Delta t + sqrt(Delta t) dot G(X) Z \ &+ 1/2 G(X) partial/(partial X)G(X)[Z^2t - (Delta t)^2]
// $

// 其推导我还需要再看一下。

// ==== 随机 Runge-Kutta 法


// #pagebreak()

// == 量子力学初步


// #pagebreak()

// == Josephson 结


// #pagebreak()

// = 问题解决记录
// == Typst 相关
// === 自定义图表标题位置和内容



#pagebreak()

= 下周计划
*论文阅读*
+ 生成模型
  - 薛定谔桥（精读）
  - Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data（精读）
  - DDIM（泛读）
// + 几何深度学习
//   - General $E(2)$ - Equivariant Steerable CNNs

*项目进度*
+ 使用神经网络学习生命游戏的演化动力学
  - 实现数据供应代理模块和等变神经网络模块
+ 耦合约瑟夫森结
  - 将 MATLAB 模拟代码全部迁移至 Python 
  - 考虑简单的 Neural SDE 方法解带参 OU 过程的参数

*理论学习*
+ 随机过程课程
  - 复习 Poisson 过程
+ 随机微分方程
  - 第五章 



#pagebreak()

#set text(lang: "en")

#bibliography("refs.bib",   // 你的 BibTeX 文件
              title: "参考文献",
              style: "ieee", 
              full: false
)

// #pagebreak()

// #set text(lang: "zh", font: ("New Computer Modern", "Kai", "KaiTi"))

// = 附录
// == 一维特殊情形的 VE-SDE、VP-SDE 和 sub-VP SDE 方差推导

// 可以这样直观理解：假设 $x$ 都是标量随机过程，对 SMLD 的加噪过程，有
// $
//   "Var"[x(t)]
//   &="Var"[x(t)-x(0)]\
//   &="Var" [int_0^t sqrt(2 sigma sigma') dd w ] = int_0^t 2 sigma sigma' dd tau & #ito "等式"\
//   &= sigma^2(t) - sigma^2(0) & "换元积分法" 
// $
// 对 DDPM 的加噪过程，有一阶矩：
// $
//   dd EE[x(t)] 
//   &= EE[dd x(t)] = -1/2 beta(t) EE[x(t)] dd t\
//   (dd EE[x(t)])/(EE[x(t)]) &= -1/2 beta(t) dd t & "分离变量"\
//   log(|EE[x(t)]|) &= -1/2 int_0^t beta(t) dd t + C ==> |EE[x(t)]| = C' e^(-1/2 int_0^t beta(t) dd t) & "积分"\
//   EE[x(t)] &= e^(-1/2 int_0^t beta(t) dd t) EE[x(0)] & "假设"EE[x(0)] gt.slant 0\
// $
// 二阶矩：
// $
//   dd EE[x^2(t)] 
//   &= EE[dd x^2(t)] = EE[2x(t) dd x(t) + beta(t) dd t] wide & #ito "乘积法则"\
//   &= EE[x(t) (2 sqrt(beta(t)) dd w) - beta(t) x(t) dd t) + beta(t) dd t]\
//   &= EE[ 2 x(t) sqrt(beta(t)) dd w] - beta(t) EE[x^2(t)] dd t + beta(t) dd t\
//   EE[int_0^t dd x^2(t)] 
//   &= underbrace(EE[ 2 int_0^t x(t) sqrt(beta(t)) dd w], 0) - int_0^t beta(tau) EE[x^2(tau)] dd tau + int_0^t beta(tau) dd tau  & "期望的线性性"\
//   dd EE[x^2(t)] &= - beta(t) EE[x^2(t)] dd t + beta(t) dd t & "换回随机微分"\
//   (dd EE[x^2(t)])/(1- EE[x^2(t)]) &= beta(t) dd t & "分离变量"\
//   - log(|1 - EE[x^2(t)]|) &= int_0^t beta(t) dd t + C ==> |1-EE[x^2(t)]| = C' e^(- int_0^t beta(t) dd t) & "积分"\
//   EE[x^2(t)] &= 1 - e^(- int_0^t beta(t) dd t) (1 - EE[x^2(0)]) & "假设"1-EE[x^2(t)] gt.slant 0\
// $
// 因此方差为
// $
//   "Var"[x(t)] 
//   &= EE[x^2(t)] - (EE[x(t)])^2 \
//   &= 1 - e^(- int_0^t beta(t) dd t) (1 - EE[x^2(0)]) - e^(- int_0^t beta(t) dd t) (EE[x(0)])^2 \
//   &= 1 + e^(- int_0^t beta(t) dd t) ("Var"[x(0)] - 1) 
// $
// 因此当 $"Var"[x(0)] = 1$ 时，有 $x(t)$ 的方差恒为 1。

// 对于 sub-VP SDE，套用上面的结果，并记 $B(t) = int_0^t beta(s) dd s$，容易得到
// $
//   EE[x(t)] 
//   &= e^(-1/2 int_0^t beta(t) dd t) EE[x(0)]\
//   EE[x^2(t)]
//   &= e^(-int_0^t beta(s) dd s) [int_0^t macron(beta)(tau) e^(int_0^tau beta(s) dd s) dd tau + EE[x^2(0)]] \
//   &= e^(-int_0^t beta(s) dd s) [int_0^t beta(tau) (1 - e^(- 2 int_0^tau beta(s) dd s)) e^(int_0^tau beta(s) dd s) dd tau + EE[x^2(0)]] \
//   &= e^(-B(t))EE[x^2(0)]] +  e^(-B(t))[int_0^t B'(tau) (1 - e^(- 2 B(tau))) e^(B(tau)) dd tau ]\
//   &= 1 + e^(-2B(t)) + e^(-B(t))(EE[x^2(0)]] - 2)\ 
// $ 
// $
//   "Var"[x(t)]
//   &= EE[x^2(t)] - (EE[x(t)])^2 \
//   &= 1 + e^(-2B(t)) + e^(-B(t))(EE[x^2(0)]] - 2) - e^(- B(t)) (EE[x(0)])^2 \
//   &= 1 + e^(- B(t)) ("Var"[x(0)] - 2) + e^(-2 B(t)) \
//   &= 1 + e^(- int_0^t beta(s) dd s) ("Var"[x(0)] - 2) + e^(-2 int_0^t beta(s) dd s)\
// $
// 一些简单的推导，可以得出 sub-VP SDE 的方差小于等于 VP-SDE 的方差。

// == 逆向 SDE 推导



// == SMLD 的祖先采样<appendix:SMLD-ancestral>

// 令 $sigma_0^2 = 0$，执行和 DDPM 类似的操作，有
// $
// & quad q(bx_(t-1) | bx_t, bx_0) \
// &= q(bx_t | bx_(t-1), bx_0) q(bx_(t-1) | bx_0) / q(bx_t | bx_0) \
// &= q(bx_t | bx_(t-1)) q(bx_(t-1) | bx_0) / q(bx_t | bx_0) \
// &prop exp lr(\{ -1/2 [ 
//   norm(bx_t - bx_(t-1))^2/(sigma_t^2 - sigma_(t-1)^2) 
//   + norm(bx_(t-1) - bx_(0))^2/(sigma_(t-1)^2) 
//   - norm(bx_t - bx_(0))^2/(sigma_t^2) 
// ] \}) \
// &= exp lr(\{ -1/2 [ 
//   (
//     norm(bx_t)^2 
//     - 2 chevron.l bx_t, #text(red)[$bx_(t-1)$] chevron.r  
//     + norm(#text(red)[$bx_(t-1)$])^2
//   )/(sigma_t^2 - sigma_(t-1)^2) 
//   + (
//     norm(#text(red)[$bx_(t-1)$])^2 
//     - 2 chevron.l #text(red)[$bx_(t-1)$], bx_0 chevron.r  
//     + norm(bx_0)^2
//   )/(sigma_(t-1)^2)  \
//   & wide wide wide - (
//     norm(bx_t)^2 
//     - 2 chevron.l bx_t, bx_0 chevron.r  
//     + norm(bx_0)^2
//   )/(sigma_t^2) 
// ] \}) \
// &= exp lr(\{ -1/2 [ 
//   sigma_t^2/((sigma_t^2 - sigma_(t-1)^2)sigma_(t-1)^2) #text(red)[$norm(bx_(t-1))^2$] 
//   - lr(
//     2 lr(chevron.l sigma_(t-1)^2/(sigma_t^2) bx_t 
//     + (1 - sigma_(t-1)^2/(sigma_(t)^2)) bx_0, 
//     #text(red)[$bx_(t-1)$] chevron.r) ) 
//   + const 
// ] \}) \
// &prop cal(N)( sigma_(t-1)^2/(sigma_t^2) bx_t 
//     + (1 - sigma_(t-1)^2/(sigma_(t)^2)) bx_0, ((sigma_t^2 - sigma_(t-1)^2)sigma_(t-1)^2)/sigma_t^2) \
// $

// == 概率流 ODE 推导<appendix:preb-ode>



