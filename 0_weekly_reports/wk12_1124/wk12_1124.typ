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
#let wk_report_name = "2025年11月24日至11月30日周报"
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

#h(2em) 目前的实现尚未考虑改变等变约束，实现的是其他所有模块的功能。其大致是算法结构如下

```python
# read and process the command line arguments  
# initialize dataset and dataloader with only one visible trajectory. 
# initialize RuleSimulator
# initialize predictor model.

# while prediction loss and simulation loss is not converged
for round_id in range(1, 21):
    # train predictor on visible trajectories.
    # update rules in RuleSimulator from predictor.
    #randomly select one of the invisible trajectories to simulate.
    #evaluate predictor on one of the visible trajectories.
    # calculate simulation acc.

    if sim_acc > 0.99 and train_loss < 0.01 and evaluate_acc > 0.95:
        # break
    else:
        # randomly add one trajectory 
# save predictor model.
# save rules.
```


#pagebreak()
= 文献阅读

== Stable Neural SDE in Analyzing Irregular Time Series Data #ref(<Arxiv:paper-StableNeuralSDE>)

*#link("https://arxiv.org/abs/2402.14989v6")[ICLR 2024] | YongKyung Oh et al.*

本论文讨论了提出了一些保证解的唯一性、随机稳定性和数值稳定性的 Neural SDE 模式。

=== Neural ODE 和 Neural SDE 简介

#tab Neural ODE 将神经网络作为导数 $display((dd f)/(dd x))$ 的预测器。令 $h(x; theta_h)$ 将数据点投影至隐空间（对应地还得有一个 decoder），得到隐空间中的表示 $z$，隐空间中的 ODE $dd z = f(t, z(t); theta_f) dd t$ 的解就可以写为
$
  z(t) = z(0) + int_0^t f(tau, z(tau); theta_f) dd tau, #h(2em) z(0) = h(x; theta_h)
$
其中 $f$ 和 $h$ 都是神经网络。Neural CDE (neural controlled differential equation) 添加了一个控制信号 $X$，它常常是数据点的插值曲线。Neural CDE 将 Neural ODE 中的积分改为 Riemann-Stieltjes 积分：
$
  z(t) = z(0) + int_0^t f(tau, z(tau); theta_f) dd X(tau), #h(2em) z(0) = h(x; theta_h)
$
Neural SDE 的思想也很类似，它将偏移项 $f$ 和扩散项 $g$ 都由神经网络代替，其求解形式为
$
  z(t) = z(0) + int_0^t f (tau, z(tau); theta_f) dd tau + int_0^t g (tau, z(tau); theta_s) dd W(tau), #h(2em) z(0) = h(x; theta_h)
$
其中 $W(t)$ 是和 $z$ 维数相同的 Brown 运动。我们也可以仿照 Neural CDE 的形式，为 Neural SDE 引入一条控制轨道。令
$
  macron(z)(t) = zeta (t, z(t), X(t); theta_zeta)
$
然后将 $  z(t)$ 替换为 $macron(z)(t)$。

有了上述的积分形式，*Neural ODE 和 SDE 的求解可以交由数值算法完成。神经网络参与进来的点是估计偏移项和扩散项（若有），训练好的神经网络可以参与数值计算过程，当做原本的偏移项函数和扩散项函数来用。*引入随机项使得 Neural SDE 面临三大难题，分别是解的存在*唯一性、随机稳定性和数值稳定性*。如果某形式的 SDE 缺乏唯一强解，同一初值可能经过计算进入完全不同的轨道，使得训练变得困难。引入的随机项会使得得到的轨道不平稳，可能导致训练过程中的梯度爆炸。另外，随机项的引入会使得数值计算中的稳定性更加重要，需要设置更加数值稳定的 SDE 形式。

=== 良好性质的 Neural SDE 结构

作者提出了三个 Neural SDE 结构，分别是 *Langevin 型 SDE*、*线性噪声 SDE* 和 *几何 SDE*。其形式分别如下
$
  dd z(t) &= gamma (z(t); theta_gamma ) dd t + sigma (t ; theta_sigma ) dd W(t) #h(2em)  & "Langevin 型 SDE (LSDE)" \
  dd z(t) &= gamma (t, z(t); theta_gamma ) dd t + sigma (t ; theta_sigma ) z(t) dd W(t) #h(2em)  & "线性噪声 SDE (LNSDE)" \
  (dd z(t))/(z(t)) &= gamma (t, z(t); theta_gamma ) dd t + sigma (t ; theta_sigma ) dd W(t) #h(2em)  & "几何 SDE (GSDE)"
$
这三种 Neural SDE 结构理论上可以解决先前提到的三个问题，并在实验中表现良好。

// #pagebreak()

// == Score-based generative modeling through SDE #cite(<DBLP:paper-yang_song-score_based_generative_modeling_sde>)

// *#link("http://arxiv.org/abs/2011.13456")[ICLR 2021] | Yang Song et al. *

// ==== 逆向 SDE 的推导

// ==== 概率流 ODE 的推导 

// #pagebreak()


// == Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling

// *#link("http://arxiv.org/abs/2106.01357")[NIPS 2021] | Valentin De Bortoli et al.*


#pagebreak()

= 学习进度
// == 机器学习理论
// === Markov Chain Monte Carlo (MCMC)


== 随机过程

#h(2em)本周继续系统学习 Markov 链，了解了常返性的一些性质。特别地，在有限状态的时齐 Markov 链中，相互联通的节点常返性相同。

// #pagebreak()

== 随机微分方程
#h(2em)本周开始学习 SDE 解的存在性和唯一性。

// // #pagebreak()

== 量子力学
#h(2em) 了解了量子力学的一些基本概念，例如波函数、薛定谔方程、无限深势阱、算符。

// // #pagebreak()

// == Josephson 结


// // #pagebreak()

// == Riemann–Stieltjes 积分

= 问题记录
== SDE 数值解的问题

#h(2em) 我尝试使用 Python 求解下面的 SDE：
$
  
$



// #pagebreak()

= 下周计划
*论文阅读*
+ 生成模型
  - 薛定谔桥
  - DDIM
// + 几何深度学习
//   - General $E(2)$ - Equivariant Steerable CNNs

*项目进度*
+ 使用神经网络学习生命游戏的演化动力学
  - 调试完成代码，并考虑等变约束
+ 耦合约瑟夫森结
  - 将 MATLAB 模拟代码全部迁移至 Python 
  - 考虑简单的 Neural SDE 方法解带参 OU 过程的参数

*理论学习*
+ 随机过程课程
  - 复习 Poisson 过程和 Markov 过程
+ 随机微分方程
  - 第五章 



#pagebreak()

#set text(lang: "en")

#bibliography("refs.bib",   // 你的 BibTeX 文件
              title: "参考文献",
              style: "ieee", 
              full: false
)

#pagebreak()

#set text(lang: "zh", font: ("New Computer Modern", "Kai", "KaiTi"))

// = 附录


