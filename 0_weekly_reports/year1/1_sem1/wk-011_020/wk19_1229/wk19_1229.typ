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
#let bY = $bold(Y)$
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
#let be = $bold(epsilon)$
#let prod = $product$
#let int = $integral$

#let leq = $lt.slant$
#let geq = $gt.slant$
#let tensor = $times.o$

#let cdots = $dots.c$

#let KL = $D_("KL")$
#let argmin = $op("arg min", limits: #true)$
#let argmax = $op("arg max", limits: #true)$

#let normal = $cal(N)$
#let prior = $p_"prior"$
#let data = $p_"data"$
#let score = $s_theta$

#let ito = $"It"hat("o")$
#let schrodinger = "Schrödinger"

// quantum
#let bra(v) = $chevron.l#v|$
#let ket(v) = $|#v chevron.r$
#let braket(u,v) = $chevron.l#u|#v chevron.r$
#let braket2(u,A,v) = $chevron.l#u|#A|#v chevron.r$

// Theorem environments
#let theorem = thmbox("theorem", "定理", fill: rgb("#eeffee"), base_level: 1)
#let proposition = thmbox("proposition", "命题", fill: rgb("#e9f6ff"), base_level: 1)
#let corollary = thmplain(
  "corollary",
  "推论",
  base: "theorem",
  titlefmt: strong
)
#let definition = thmbox("definition", "定义", inset: (x: 1.2em, top: 1em), base_level: 1)
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

#let pfw(n, k) = {[$p^(#n)_(#k+1|#k) (x_(#k+1)|x_#k)$]}
#let pbk(n, k) = {[$p^(#n)_(#k|#k+1) (x_#k|x_(#k+1))$]}
#let qfw(n, k) = {[$q^(#n)_(#k+1|#k) (x_(#k+1)|x_#k)$]}
#let qbk(n, k) = {[$q^(#n)_(#k|#k+1) (x_#k|x_(#k+1))$]}

// metadata
#let wk_report_name = "2026年一月上旬工作汇报和寒假工作计划"
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

#outline(depth: 2)

#linebreak()
#grid(columns: (100%), align: center, text(size: 12pt)[速 览])

#h(2em) 一月上半旬由于期末考试后休息、未完成的两个课程作业和其他事宜，消耗了大部分时间，因此这一期间仅进行了一些知识学习和论文阅读工作。主要内容包括对实变函数的知识做了简单的概览，并阅读了和 Neural SDE 相关的论文一篇。一月下旬的工作计划附在文后。



= 文献阅读

// == Score-based Generative modeling through SDE 补遗

// *#link("http://arxiv.org/abs/2011.13456")[ICLR 2021] | Yang Song et al. *

// === 概率流 ODE 和 Fokker-Plank 方程



// #pagebreak()


== Neural Stochastic Differential Equations: Deep Latent Gaussian Models in the Diffusion Limit

*#link("http://arxiv.org/abs/1905.09883")[arxiv 2011.13456] | Belinda Tzen and Maxim Raginsky*

=== 隐 Gauss 模型

机器学习常常将高维的数据 $x$ 投射到某个维度更低的隐空间中，得到其“潜在表示” $z$。隐 Gauss 模型意为潜在表示服从某个 Gauss 分布。假设需要生成某个数据 $y$，隐 Gauss 模型尝试通过若干 Gauss 随机变量 $z_1, z_2, cdots, z_k$ 生成之，具体操作如下：
$
  x_0 &= z_0, \
  x_i &= x_(i-1) + b_i (x_(i-1)) + sigma_i z_i, quad i = 1, 2, cdots, k, \
  y &~ p_theta (dot.c|x_k). 
$<eq-latent-gaussian-model>
其中 $z_i ~ normal(0, I)$，$theta$ 是所有 $b_i$ 和 $sigma_i$ 中的参数。和很多生成模型一样，我们要最大化训练数据 $y$ 的对数似然 $log p_theta (y)$。本文中给出了一个相比于常规生成模型论文中长篇累牍的 ELBO 推导的一个更自然的视角，即 Gibbs 变分原理 (Gibbs variational principle)。 

#theorem[Gibbs 变分原理][
  对 $Omega = (RR^d)^(k+1)$ 上的任意测度 $mu$，和任意可测函数 $F: Omega -> RR$，有
  $
    -log EE_mu [exp(-F(z_1, ..., z_k))] = inf_(nu in scr(P)(Omega)) 
    lr( \{ KL(nu||mu) + EE_nu [F(z_1, ..., z_k)] \} ).
  $<eq-gibbs-variational-thm>
]<thm-gibbs-variational-principle>

在隐 Gauss 模型的语境下，令 $F(z_1, ..., z_k) = -log p (y|f_theta (z_1, ..., z_k))$，就得到
$
  -log p_theta (y) = inf_(nu in scr(Omega)) lr({KL(nu||mu) - int_Omega log p_theta (y|z)nu(dd(z)) }).
$<eq-gibbs-variational-inf>
其中 $mu$ 对应的是 $z = z_0, ..., z_k$ 的分布。等号右边取下确界的这一项称为*变分自由能* $sans(F)_(theta)$。#ref(<eq-gibbs-variational-inf>) 中等号右侧的积分无法计算，常用的技巧是使用一个带参后验 $nu_beta (dd""z|y) = q_beta (z|y)dd""z$ 进行近似，最后得到带参的变分自由能 $sans(F)_(theta, beta)$，其形式为
$
  sans(F)_(theta, beta) = KL(nu||mu) - int_Omega log p (y|f_theta (z_1, ..., z_k)) nu_beta (dd""z_1, ..., dd""z_k|y).
$
该形式可以通过对适当的参数化后的梯度进行 Monte-Carlo 得到对变分自由能梯度的估计。

=== Neural SDE

#h(2em) 我们发现 #ref(<eq-latent-gaussian-model>) 就是某个 SDE 的 Euler-Maruyama 离散化形式，如果我们转而考虑形如 $dd""X = F""dd""t + G""dd""W$ 这样的随机微分方程时，若将 $F$ 和 $G$ 使用神经网络替代，就得到了 Neural SDE。对比隐 Gauss 模型，SDE 语境下的隐变量在 $[0, 1]$ 上的 $d$ 维路径组成的空间 $bb(W)=scr(C)([0, 1], RR^d)$ 中，隐变量本身是 $[0, 1]$ 上的某个连续的 Brown 运动轨道。可以利用随机积分的定义得到循序可测映射 $f: bb(W) -> bb(W)$，其形式为
$
  [f_theta (W)]_t = int_0^t b ([f_theta (W)]_s, s; theta) dd""s + int_0^t sigma ([f_theta (W)]_s, s; theta) dd""W_s,.
$
上一节中，联合分布 $p_theta (y, z)$ 可以写为
$
  p_theta (y, z) = p_theta (y|z) p(z) =p_theta (y|z) phi.alt_d (z_1) dots.c phi.alt_d (z_k).
$
其测度元可以写为
$
  P_theta (dd""y, dd""z) = p_theta (y|z) dd""y  prod_(i=1)^k phi.alt_d (z_i) dd""z_i.
$<eq-finite-dim-measure>
现在将#ref(<eq-finite-dim-measure>) 中的隐变量替换为 $W$，并将 $display(prod_(i=1)^k phi.alt_d (z_i) dd""z_i)$ 替换为 Wiener 测度 $bold(mu)(dd""W)$，就得到了 Neural SDE 的联合测度元
$
  P_theta (dd""y, dd""W) = underbrace(p_theta (y|f_theta (W)), "条件密度") dot.c underbrace(bold(mu)(dd""W), "Wiener 测度") dot.c underbrace(dd""y, "Lebesgue 测度") .
$
因此 $y$ 的边缘密度为 $display(p_theta (y) = int_(bb(W)) p_theta (y|f_theta (W)) bold(mu)(dd""W)) = EE_bold(mu) [p_theta (y|f_theta (W))]$。对应于离散版本中用到的#ref(<thm-gibbs-variational-principle>)，$bb(W)$ 上也有类似的结论，即
$
  -log p_theta (y) = inf_(bold(nu) in scr(P)(bb(W))) lr( { KL(bold(nu)||bold(mu)) + EE_redText(bold(nu)) [F(W)] } ).
$<eq-gibbs-sde>
由 *Girsanov 定理*，每个 $bb(W)$ 上关于 Wiener 测度绝对连续的的测度 $bold(nu)$ 都可以对应于一个 Brown 运动 $W$ 加上一个漂移。可以得到对于满足 $dd""Z = u""dd""t + dd""W$ 的过程 $Z in bold(nu)$，有
$
  KL(bold(nu)||bold(mu)) = 1/2 EE_bold(mu) [int_0^1 norm(u(t))^2 dd""t], #h(2em) EE_redText(bold(nu)) [F(W)] = EE_blueText(bold(mu)) [F(W + int_0^circle.small.filled u(s) dd""s)].
$
因此 #ref(<eq-gibbs-sde>) 可以写成所谓变分的 Girsanov 表示：
$
  -log p_theta (y) = inf_(u) EE_bold(mu) [ 1/2 int_0^1 norm(u(t))^2 dd""t + F_theta (W + int_0^circle.small.filled u(s) dd""s) ].
$<eq-variational-girsanov>
其中$W + int_0^circle.small.filled u(s) dd""s$是由Brown运动轨道生成的新的随机过程$Z$；$F_theta (w) = - log p_theta (y|[f_theta (w)]_1)$。注意此时期望下标从 $bold(nu)$ 变成了 $bold(mu)$。

在使用 SDE 的无限层隐 Gauss 模型中，考虑对上面的下确界使用平均场近似。令 $u_s = tilde(b)(y, s; beta), s in [0, 1]$，其中 $tilde(b)$ 是确定的神经网络，并满足 $int_0^1 norm(tilde(b)(y, t; beta)) < infinity$，平均场的思想体现在我们采用确定的偏移项而不是随机的，这样使#ref(<eq-variational-girsanov>)的第一项不再需要计算期望，同时变为对所有 $beta$ 求下确界，得到负对数似然的上界：
$
  -log p_theta (y) lt.slant inf_beta {1/2 int_0^1 norm(tilde(b)(y, s; beta))^2 dd""t + EE_bold(mu) [F_theta (W + int_0^circle.small.filled tilde(b)(y, s; beta) dd""s) ]}
$
*在这里 $beta$ 的作用是作为 $tilde(b)$ 的参数构成所谓变分网络，用于估计 $p_theta (W|y)$，因此 $theta$ 可以看做解码器参数，$beta$ 可以看做编码器参数。*


=== 对 Neural SDE求梯度

#h(2em) 研究 Neural SDE 的变分自由能
$
  sans(F)_(theta, beta) =  1/2 int_0^1 norm(tilde(b)(y, s; beta))^2 dd""t + EE_bold(mu) [F_theta (redText(W) + blueText(int_0^circle.small.filled tilde(b)(y, s; beta) dd""s)) ].
$
等式右侧的第一项是确定函数的积分，可以使用通常的梯度求解方法；我们关注第二项。第二项 $F_theta$ 括号里面是下面的 #ito 过程
$
  X_t^(theta, beta) = redText(W_t + int_0^t b(X_t^(theta, beta), s; theta) dd""s) + blueText(int_0^t tilde(b)(y, s; beta) dd""s) + redText(int_0^t sigma (X_s^(theta, beta), s; theta) dd""W_s), t in [0, 1].
$<eq-ito-process>
多出来的蓝色的一项是因为先前使用了 Girsanov 定理。右边这一项就对应着 $-EE [log p (y|X_1^(theta, beta))]$。而要对其求导，我们希望内部的函数 $log p (y|X_1^(theta, beta))$ 的性质足够好，是的偏导数算子和期望算子可以交换：
$
  partial/(partial beta) EE [log p (y|X_1^(theta, beta))] &= EE[partial/(partial x) log p(y|X_1^(theta, beta)) partial/(partial beta) X_1^(theta, beta) ], \
  partial/(partial theta) EE [log p (y|X_1^(theta, beta))] &= EE[partial/(partial x) log p(y|X_1^(theta, beta)) partial/(partial theta) X_1^(theta, beta) ].
$

#h(2em) 需要解决的核心问题为如何计算 $display(partial/(partial circle.small.filled) X_1^(theta, beta))$。论文中给出了两种方法：第一种为求解后微分，第二种为先微分后求解。具体而言，因为 $X_1^(theta, beta)$ 是一个随机过程在 $t=1$ 处的取值，它是按照#ref(<eq-ito-process>)生成的，对其求导必须要经过这个随机积分。于是第一种方法考虑先将 $X_1^(theta, beta)$ 离散化，选取一个划分 $0=t_0 < t_1 < dots.c < t_N = 1$，然后采样独立同分布的 $Z_1, Z_2, ..., Z_N$，做 Euler-Maruyama 算法：
$
  hat(X)_(t_(i+1))^(theta, beta) = hat(X)_(t_(i))^(theta, beta) + (t_(i+1)-t_i)[b(hat(X)_(t_(i))^(theta, beta), t_i; theta) - tilde(b)(y, t_i, beta)] + sqrt(t_(i+1)-t_i) sigma (hat(X)_(t_(i))^(theta, beta), t_i; theta)Z_(i+1),
$
然后求得 $X_1^(theta, beta)$ 的数值解后调用自动求导工具进行求导，其时间复杂度大约为 $O[N(sans(T)(b) + sans(T)(tilde(b)) + sans(T)(sigma))]$。

另一种方法是尝试直接得出 $X_1^(theta, beta)$ 对 $theta$ 或 $beta$ 的 SDE 解析形式，这个 SDE 中仅包含通常意义上的导数计算，然后根据这些求导结果，进行 SDE 的数值计算。具体而言，有下面的定理

#theorem[Neural SDE 求导方程][
  设 $X_i^(theta, beta)$ 满足#ref(<eq-ito-process>)，且满足
  + $b(x, t; theta)$ 和 $sigma (x, t; theta)$ 在 $[0, 1]$ 上一致 Lipschitz 连续，且它们对 $x$ 和 $theta$ 的 Jacobian 矩阵在 $[0, 1]$ 上也一致 Lipschiz 连续；
  + $tilde(b)(y, t; beta)$ 在 $[0, 1]$ 上一致 Lipschitz 连续，且对 $beta$ 的 Jacobian 矩阵在 $[0, 1]$ 上也一致 Lipschiz 连续；
  则随机过程 $X = X^(theta, beta)$ 对 $theta$ 和 $beta$ 的路径导数为
  $
    (partial X_t)/(partial beta^((i))) &= int_0^t ((partial b_s)/(partial x) (partial X_s)/(partial beta^((i))) + (partial tilde(b)_s)/(partial beta^((i))))dd""s + sum_(l=1)^d int_0^t (partial sigma_(s, l))/(partial x) (partial X_s)/(partial beta^((i)))dd""W_s^((l)) & ("a")\
    (partial X_t)/(partial theta^((j))) &= int_0^t ((partial b_s)/(partial theta^((j))) + (partial b_s)/(partial x) (partial X_s)/(partial theta^((j)))) dd""s + sum_(l=1)^d int_0^t ((partial sigma_(s, l))/(partial theta^((j))) + (partial sigma_(s, l))/(partial x) (partial X_s)/(partial theta^((j))))dd""W_s^((l)) & ("b")\
  $
  其中 $theta in RR^n$，$beta in RR^k$，$b_s = b(X_s^(theta, beta), s; theta)$，$sigma_(s, l)$ 是 $sigma (X_s^(theta, beta), s; theta)$ 的第 $l$ 列，其他记号以此类推。
]<thm-neural-sde-gradient>

对于 (a)，我们要求 $b$ 对 $x$ 的导数、$tilde(b)$ 对 $beta$ 的导数和 $sigma$ 对 $x$ 的导数。解 SDE 时每次迭代求解这些导数的时间复杂度为 $O[k(d sans(T)(b) + min(d,k)sans(T)(tilde(b)) + d sans(T)(sigma))]$，对 (b)，需要求的是 $b$ 和 $sigma$ 分别对 $theta$ 和 $x$ 的导数，其时间复杂度为 $O[n(min(d, n) + d)(sans(T)(b) + sans(T)(sigma))]$。

#pagebreak()


// == Scalable Diffusion Models with Transformers

// #pagebreak()

// = 学习进度
// == 机器学习理论
// === Markov Chain Monte Carlo (MCMC)


// === EM 算法 


// === 计算学习理论


// #pagebreak()

// == 随机过程

// #h(2em)本周学习了连续状态的 Markov 链。

// #pagebreak()

// == 生成模型理论

// #tab 本周参阅了 MIT 的生成模型课程笔记，这是一本五十页的小册子，本周读完了大半部分内容。由于先前阅读了不少相关方面的论文，阅读起来没什么障碍，不过该讲义依然给予我了一些比较优雅的视角。

// #pagebreak()

// == 随机微分方程
// === 一维解的构造

// #tab 设 $b: RR -> RR$ 为 $scr(C)^1$ 函数且 $|b'| <= L$。考虑 SDE:
// $
//   dd X = b(X) dd t + dd W,  X(0) = x_0 in RR
// $

// 通过 Picard 迭代构造解 $X^((0))(t) equiv x_0$递推式为
// $
// X^((k+1))(t) := x_0 + integral_0^t b(X^((k))) dd s + W(t)
// $
// 并定义 
// $
// D^((k))(t) := max_(0 <= s <= t) |X^((k+1))(s) - X^((k))(s)|
// $
// 可以用归纳法证明 
// $
// D^((k))(t) <= C L^k/(k!) t^k
// $
// 由此可得$X^((k))$是 Cauchy 列，几乎必然一致收敛到 SDE 的解$X $。

// === 变量替换法

// #tab 对一般 SDE $dd X = b(X) dd t + sigma(X) dd W, quad X(0) = x_0$，设 $X = u(Y) $，其中 $Y$ 满足
// $dd Y = f(Y) dd t + dd W, quad Y(0) = y_0$。由 Itô 公式可得
// $ 
//   dd X = [u'(Y) f(Y) + 1/2 u''(Y)] dd t + u'(Y) dd W 
// $
// 因此需要满足：
// $ 
//   u'(Y) = sigma(u(Y)), #h(1em)u'(Y) f(Y) + 1/2 u''(Y) = b(u(Y)), #h(1em)u(y_0) = x_0 
// $
// 可以先解 ODE $u'(z) = sigma(u(z)), u(y_0) = x_0 $，然后定义：
// $ 
//   f(z) := [b(u(z)) - 1/2 u''(z)] / sigma(u(z)) 
// $
// 即为原 SDE 的解。

// === 存在唯一性定理

// #theorem[Gronwall 不等式][
//   设$f, phi$是 $[0, T]$ 上的非负连续函数，若
//   $ 
//     phi(t) <= C_0 + integral_0^t f phi dd s 
//   $
//   则
//   $ 
//     phi(t) <= C_0 exp(integral_0^t f dd s) 
//   $
// ]
// #proof[
//   令 $Phi = C_0 + integral_0^t f phi dd s $ ，则 $Phi' = f phi <= f Phi $。计算$[exp(-integral_0^t f dd s) Phi]' <= 0 $，故 $exp(-integral_0^t f dd s) Phi(t) <= C_0 $，得证。
// ]

// #theorem[存在唯一性定理][
//   设$b: RR^n x [0,T] -> RR^n$和$B: RR^n x [0,T] -> RR^(m x n)$满足一致 Lipschitz 条件：
//   $ |b(x,t) - b(hat(x),t)| <= L|x - hat(x)| $
//   $ |B(x,t) - B(hat(x),t)| <= L|x - hat(x)| $
//   及线性增长条件：
//   $ |b(x,t)| <= L(1+|x|) $
//   $ |B(x,t)| <= L(1+|x|) $

//   若$E|X_0|^2 < infinity $，则 SDE:
//   $ dd X = b(X,t) dd t + B(X,t) dd W, quad X(0) = X_0 $
//   存在唯一解$X in L_n^2(0,T) $，且在概率$1$意义下唯一。
// ]


// #pagebreak()

// == 实分析

// #tab 我们希望扩展长度、面积或是体积的概念，使之适用于一般的集合。形式上，我们希望存在一个这样的映射 $mu: cal(A) -> [0, infinity]$，其中 $cal(A) subset 2^Omega$，并满足下面的性质：
// + 非负性 (non-negativity)：对任意 $A in cal(A)$，有 $mu(A) >= 0$；
// + 空集的测度为零 (null empty set)：$mu(emptyset) = 0$；
// + 可数可加性 (countable additivity)：对任意可数个两两不交的集合 ${A_i}_(i in NN) subset cal(A)$，有 $
// mu(union.big_(i in NN) A_i) = sum_(i in NN) mu(A_i)
// $

// #pagebreak()

// == 量子力学

// #tab 在 1921 年和 1922 年，O.Stern 和 W.Gerlach 进行了下面的实验。他们将银放在一个留有一个小孔的加热炉中加热，然后在银原子逃逸的路径上设置一个非均匀磁场。根据经典力学的预测，银原子束在通过磁场后应该会发生扩散，因为每个原子的磁矩方向是随机的。然而实验结果显示，银原子束在通过磁场后分裂成了两个离散的部分。如果这个磁矩是由旋转角动量产生的，那么我们应该观察到一个连续的分布。因此这说明存在一个未知的内秉角动量，它在某一方向上只能取两个值。

// #figure(
//   image("/assets/image-12.png", width: 55%),
//   // placement: bottom,
//   caption: [Stern 和 Gerlach 的实验#footnote([本节内容参考自#cite(<Sakurai:ModernQM>)])]
// )

// 故事没有结束，接下来再看下面一个级联的 Stern-Gerlach 实验。上述实验对应下图中 (a) 子图的左边。当在 $hat(bold("z"))$ 方向施加非均匀磁场时，银原子束分裂成两束。现在遮挡其中一束，对另一束考虑下面三种处理
// + 再次通过一个 $hat(bold("z"))$ 方向的非均匀磁场，银原子束不再分裂；
// + 通过一个 $hat(bold("x"))$ 方向的非均匀磁场，银原子束分裂成两束；
// + 通过一个 $hat(bold("x"))$ 方向的非均匀磁场后，遮挡其中一束；再通过一个 $hat(bold("z"))$ 方向的非均匀磁场，银原子束再次分裂成两束。

// #figure(
//   image("/assets/image-13.png", width: 75%),
//   // placement: bottom,
//   caption: [级联 Stern-Gerlach 的实验]
// )

// 该实验表明，每次经过某一方向的磁场，相当于对银原子的内秉角动量进行一次*测量*，测量结果只能取两个值的其中一个。*如果已在某个方向上测量过，不经过其他操作再次测量时得到的结果不会变化*。然而若在进行了某一方向的一次测量后再经过另一方向的测量，最初的测量结果会被“抹去”，换言之，*不能同时确定两个方向上的测量结果*。

// 为研究如上的量子现象，需要引入对应的数学工具。量子力学研究复 Hilbert 空间 $scr(H)$、其上的线性算子以及其对偶对象（复对偶 Hilbert 空间 $scr(H)^*$ 和其上的线性算子）。在量子力学中，称复 Hilbert 空间中的元素为*态矢量* (state vector)，写作 $ket(a)$，因此也称右矢量 (ket)，其对偶对象 $bra(a)$ 为左矢量 (bra)。复 Hilbert 空间中两个右矢 $ket(a)$ 和 $ket(b)$ 的内积为 $braket(a,b)$，英文就叫 bra(c)ket。注意它可以写成 $bra(a)ket(b)$，而左边一项是右矢 $ket(a)$ 的对偶，这是由 Riesz 表示定理保证的。由内积公理，对任意右矢 $ket(a)$，有 $braket(a, a) geq 0$；对任意两个右矢 $ket(a)$ 和 $ket(b)$，内积有共轭对称性 $braket(a, b) = braket(b, a)^*$。若 $scr(H)$ 为至多可数维，那么存在一个正交归一基 (orthonormal basis) ${e_i}_(i in NN)$，可以表出任意右矢 $ket(x)$ 为
// $
//   ket(x) = sum_(i in NN) x_i ket(e_i)
// $
// 若将基矢 $ket(e_j)$ 之对偶作用于 $ket(x)$，得到 
// $
//   braket(e_j, x) = sum_(i in NN) x_i braket(e_j, e_i) = sum_(i in NN) x_i delta_(i,j) = x_j
// $ 
// 这样我们就能“形式上”地将右矢 $ket(x)$ 写作一个“列矢量” $x$ 的形式。需要注意的是，列矢量 $x$ 是 $scr(H)$ 中的抽象元素 $ket(x)$ 在基（或称*表象*）${e_i}_(i in NN)$ 下的表示，类似地不难证明其对偶 $bra(x)$ 可以写成 $x^(top *) = x^dagger$。

// 量子力学中称 $scr(H)$ 上的线性算子 $hat(A): scr(H) -> scr(H)$ 为*算符*。算符 $hat(A)$ 作用在右矢 $ket(x)$ 上写作 $hat(A)ket(x)$，得到另一个右矢，其对偶的情形为 $bra(x)hat(A)^dagger$，其中 $square^dagger$ 表示算子的 Hermite 共轭。若将一个右矢和一个左矢形式上地乘在一起，即 $ket(a)bra(b)$，它定义了一个算符。事实上上述结果也可以写成张量积的形式，即 *$ket(a)bra(b) = ket(a) times.o bra(b)$*。现在还是考虑一个基 ${e_i}_(i in NN)$。一方面，右矢 $ket(x)$ 可以写成基矢的线性组合，也可以被基矢作用得到在基矢“方向”上的“分量”或投影，有
// $
//   ket(x) = sum_(i in NN) ket(e_i) dot x_i = sum_(i in NN) ket(e_i) braket(e_j, x) = [sum_(i in NN) ket(e_i) bra(e_j)] ket(x) = 1 ket(x).
// $
// 于是我们成功构造了恒等算符 $1$。另一方面，算符 $hat(A)$ 作用在右矢 $ket(x)$ 得到新的右矢 $ket(y)$，用一个基矢 $ket(e_j)$ 作用在 $ket(y)$ 上得到
// $
//   braket(e_j, y) = braket2(e_j, hat(A), x) = bra(e_j)hat(A) dot 1 dot ket(x) = sum_(i in NN)bra(e_j)hat(A) ket(e_i) braket(e_j, x)
// $
// 若令 $A_(j,i) = braket2(e_j, A, e_i)$，则有 $y_j = braket(e_j, y) = sum_(i in NN) A_(j,i) x_i$。这表明*算符 $hat(A)$ 在基 ${e_i}_(i in NN)$ 下的矩阵表示为 $A = (A_(j,i))$*，相应地，其*对偶算符 $A^dagger$ 的矩阵表示为 $A^dagger = (A_(i,j)^*)$*，即为原算符矩阵的共轭转置。考虑算符 $hat(A)$ 同时作用在右矢 $ket(x)$ 和左矢 $bra(y)$ 上，有
// $
//   braket2(x, hat(A), y) = bra(x)[hat(A)ket(y)] = [bra(y)hat(A)^(dagger)ket(x)]^*
// $
// 如果 $hat(A) = hat(A)^dagger$，我们称其为 *Hermite 算符*，Hermite 算符的所有本征值（特征值）为实数，每个本征值对应的本征态（特征向量）都是正交的。

// #pagebreak() 

// == 动力系统基础
// #h(2em)

// #pagebreak()

// = 问题记录


// #pagebreak()

= 学习工作心得

#h(2em) 本学期为我在中大就读的第一学期，也是尝试使用 typst 撰写周报和课程报告的第一个学期。我的体验是比较顺畅，和 markdown 和 LaTeX 的知识兼容性高，上手迅速，编译速度快。唯一缺陷也许是功能目前可能受限（在我暂时用不到的地方）以及 typst 的 slides 功能（如 `touying` 模块）的文档不够详细完善，暂时无法满足较为精细的调整，不过整体观感问题不大。

论文阅读方面，本学期共读论文 $11$ 篇，主要涉及 SDE 和 ODE 建模的生成模型，低于课题组每周一篇论文的基本要求。这和我时长分心，难以集中精力、未能科学规划时间有关（包括本次报告迟交 $1$ 天）。这个缺点还反映在各个方面，例如课程作业、考试复习、科研项目的时间安排。不过仔细阅读这些论文中的例如 Yang Song 关于 Score Matching 和 SDE 的论文、DDPM、群等变卷积等相对入学前读过论文的理论深一些的论文，我都期望可以完整的把握其中的核心思想，再不济也需要掌握大概，这让我不能容忍在撰写周报的论文阅读部分时遇到的任何一个不理解或是理解模糊的点。虽然本学期我的论文阅读效率不够，但我能感到自己主动走出舒适区解决困难问题的能力正在提升。

项目方面，我主要参与了生命游戏规则学习的项目，这是个人项目，除去指导老师外没有其他组员协作，这同样考验个人的时间管理和项目组织能力。我常常若干周没有实质性的进展，面临从神经网络中提取显式规则的问题，我由于缺乏相关经验和检索不到相关的成熟方案而因此对其望而却步，一拖再拖。另外该项目也暴露了我在主线方面不确定的问题，例如在某次会议前我临时将学习的目标改为少量轨道上利用主动学习逐步抽取规则，但项目仍未交付完整的基础版本。

理论学习方面，本学期我选了一门随机过程课，了解了 Poisson 过程、离散事件有限状态 Markov 过程和更新过程的基本概念，另外自学了 Evans 所著的 SDE 小册子的基础知识，目前这本小册子的前五章已经学习完毕，这些关于随机积分和随机微分方程的知识对我理解用到它们的论文很有帮助。

#pagebreak()

= 一月上旬工作计划
== 论文阅读

#h(2em) 寒假我将阅读和 SDE 与生成模型相关的论文，同时逐步将重心转移至 Neural SDE、薛定谔桥、随机控制和与神经动力学相关的论文，并尝试寻找它们和脑机接口之间的联系。

== 项目进度

#h(2em) 寒假我将在年前完成最基本的生命游戏规则学习的完整可运行项目，并抽时间与指导老师约定线上会议时间，同时修改线上论文草稿。

== 理论学习

#h(2em) 寒假期间我将结束对 Evans 的 SDE 小册子的学习；并开始补足必要的严格数学基础知识，即实分析和泛函分析；除此之外，我将使用富裕时间开始学习量子力学和动力系统的基础知识。

// #pagebreak()

// = 附录



// #pagebreak()

// #set text(lang: "en")

// #bibliography("../refs.bib",   // 你的 BibTeX 文件
//               title: "参考文献",
//               style: "ieee", 
//               full: false
// )