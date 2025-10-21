#import "@preview/touying:0.6.1": *
#import themes.dewdrop: *
#import "@preview/algorithmic:1.0.5"
#import algorithmic: (
  style-algorithm, 
  algorithm-figure
)
#import "@preview/ctheorems:1.1.3": *
#import "@preview/mitex:0.2.4": *
#import "@preview/numbly:0.1.0": numbly

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

#set math.mat(delim: ("[", "]"), align: center)
#set heading(numbering: "1.")

#show: dewdrop-theme.with(
  aspect-ratio: "16-9",
  footer: self => (self.info.institution + " | " + self.info.author),
  navigation: "mini-slides",
  config-info(
    title: [Group Equivariant Convolutional Networks],
    subtitle: [机器学习课程汇报],
    author: [何瑞杰],
    date: datetime(year: 2025, month: 10, day: 29),
    institution: [中山大学 #sym.hyph.point 数学学院],
  ),
  mini-slides: (
    height: 2em,
    x: 2em,
    display-subsection: false
  )
)

#set heading(numbering: numbly("{1}.", default: "1.1"))

#title-slide()

#outline-slide()


- Title: Group Equivariant Convolutional Networks
- Authors: Taco S. Cohen and Max Welling 
- Conference: ICML 2016
- Link: https://arxiv.org/abs/1602.07576 

---

= 何为等变，等变何为

---

考虑线性空间 $V$ 和上面的一个变换群 $frak(G)$，我们称之为 $frak(G)$-空间。对于线性空间上的一个函数 $Phi$，如果它满足
$
  Phi(T_fg x) = T'_fg Phi(x)
$
其中 $T_fg$ 是指对 $V$ 中的向量做对应于群元 $fg$ 的变换。$T$ 和 $T'$ 不必相同，但必须要是 $frak(G)$ 中元素的线性表示，即满足对任意 $fg, fh in frak(G)$，有 $T(fg fh) = T(fg) T(fh)$。

#figure(
  image("image.png", width: 40%),
  caption: [人眼的旋转等变性——对等变模型的期望]
)

---

= 对称群

---

典型的对称群分别为 $p 4$ 和 $p 4 m$。前者是包含了 $ZZ^2$ 上的所有平移变换和以 $display(pi/2)$ 为单位的旋转变换，它有下面的表示：
$
fg(r, u, v) = mat(
  cos((r pi)/2), -sin((r pi)/2), u;
  sin((r pi)/2), #h(0.84em) cos((r pi)/2), v;
  0, 0, 1
)
$
其中 $r in \{0, 1, 2, 3\}$，$(u, v) in ZZ^2$。群元作用在向量上的结果就可以写为
$
  fg x tilde.eq mat(
  cos((r pi)/2), -sin((r pi)/2), u;
  sin((r pi)/2), #h(0.84em) cos((r pi)/2), v;
  0, 0, 1
) mat(
  u'; v'; 1
)
$
$p 4 m$ 也有类似的表示。


= 特征图的信号视角

---

对于形状为 `[c, w, h]` 的特征图 `F`，为推到方便起见，我们可以将其看作是有一个有界支撑集的函数：
$
F: ZZ^2 &-> RR^c \
(x, y) &|-> cases(mono(F[:, x, y])","#h(1.5em) & x in [0, w-1] "," y in [0, h-1], bold(0)"," &"otherwise")
$
这样就可以建立和 $RR$ 上函数之间卷积操作类似的表达式。对于一个信号 $F$，我们定义群元素 $fg$ 作用在其上的结果为
$
  L_fg f(x) = f(fg^(-1) x).
$
其中 $L_fg$ 是对应于群元 $fg$ 之变换 $T_fg$ 的一个实例化，并满足 $L_fg L_fh = L_(fg fh)$。它的直观理解是，假如 $L_fg$ 是一个向左的平移变换，则 $L_fg f(x)$ 就将信号（例如图片）向左平移 $c$，则平移后图片中给定位置的像素值就等于原图片中向右平移相同距离的像素值，也即 $f(x+c)$。

= 等变性 
== 通常卷积模块的等变性

首先回忆 CNN 中的卷积操作 $*$ 和相关操作 $star$，它们在 CNN 的前向传播和反向传播中成对出现。考虑特征图 $f: ZZ^2 -> RR^(K^((l)))$ 和一组中的某个卷积核 $psi^((i)) : ZZ^2 -> RR^(K^((l)))$，有
$
[f * psi](bx) &= sum_(by in ZZ^2) sum_(k=1)^(K^((l))) f_k (bold(y)) psi^((i))_k (bx - by)#h(2em) & "convolution"\ 
[f star psi](bx) &= sum_(by in ZZ^2) sum_(k=1)^(K^((l))) f_k (bold(y)) psi^((i))_k (by - bx) & "correlation"\ 
$

---

现在我们验证相关操作的平移等变性。考虑 $ZZ^2$ 上的平移群元 $t$ 对应的变换 $L_t$，有
$
[(L_t f) star psi](bx) 
&= sum_(by in ZZ^2) sum_(k=1)^(K^((l))) [L_t f_k] (bold(y)) psi^((i))_k (by - bx) \
&= sum_(by in ZZ^2) sum_(k=1)^(K^((l))) f_k (bold(y) - bold(t)) psi^((i))_k (by - bx) \ 
&= sum_(bold(z) in ZZ^2) sum_(k=1)^(K^((l))) f_k (bold(z)) psi_k^((i))(bold(z) - (bx - bold(t)))\
&= [f star psi](bx - bold(t)) = [L_t [f star psi]](bx). #h(1em) & bold(z) <- by - bold(t)
$

---

相关变换对旋转没有等变性，对于 $p 4$ 中的群元 $r$ 对应的旋转变换 $L_r$，有：
$
[(L_r f) star psi](bx) 
&= sum_(by in ZZ^2) sum_(k=1)^(K^((l))) [L_r f_k] (by) psi^((i))_k (by - bx) \
&= sum_(by in ZZ^2) sum_(k=1)^(K^((l))) f_k (r^(-1)by) psi^((i))_k (by - bx) \
&= sum_(bz in ZZ^2) sum_(k=1)^(K^((l))) f_k (bz) psi^((i))_k (r bz - bx)\
&= sum_(bz in ZZ^2) sum_(k=1)^(K^((l))) f_k (bz) psi^((i))_k (r (bz - r^(-1)bx)) #h(1em) & bz <- r^(-1)bold(y)\
$

$
&= sum_(bz in ZZ^2) sum_(k=1)^(K^((l))) f_k (bz) [L_(r^(-1)) psi^((i))_k] (bz - r^(-1)bx)\
&= [f star L_(r^(-1)) psi](r^(-1) bx) = L_r [f star L_(r^(-1)) psi] (bx)
$
可见通常的相关操作对旋转没有等变性。对于卷积，注意只需令 $phi(x) = psi(-x)$，相关操作就变成了卷积操作。因此卷积的等变性和相关操作相同。

== 群等变相关操作

为了让卷积操作对旋转、以至更加一般的操作具有等变性，作者提出了群相关操作。注意上文中的相关操作
$
[f star psi^((i))](bx) &= sum_(by in ZZ^2) sum_(k=1)^(K^((l))) f_k (bold(y)) psi^((i))_k (- bx + by )
$
注意 $bx$ 对应着 $ZZ^2$ 上所有平移操作构成的群中的一个群元素 $fg$，而 $- bx$ 对应着它的逆元 $fg^(-1)$。

---
我们可以自然地将原来相关操作写成包含群元素的形式，这就得到了第一层的群相关：
$
f^((1))(fg) = [f star psi^((i))](fg) &= sum_(by in ZZ^2) sum_(k=1)^(K^((l))) f^((0))_k (bold(y)) psi^((0, i))_k (fg^(-1) by)
$
其中 $fg in frak(G)$，$f$ 和 $psi^((0, i))$ 都是 $ZZ^2$ 到 $RR$ 的函数，但得到的新信号 $[f star psi^((i))]$ 的定义域为 $frak(G)$。因此接下来的各层中群相关操作的表达式需要做一些微调：
$
f^((l+1))(fg) = [f star psi^((i))](fg) &= sum_(fh in frak(G)) sum_(k=1)^(K^((l))) f^((l))_k (fh) psi^((l, i))_k (fg^(-1) fh)
$
其中 $l gt.slant 1$，$fg, fh in frak(G)$，$f_k^((l))$ 和 $psi_k^((l, i))$ 的定义域都是 $frak(G)$（这里假设每层的变换群相同）。

---
接下来证明它是关于群 $frak(G)$ 的元素等变的：
$
[[L_fu f] star psi](fg) 
&= sum_(fh in X) sum_(k=1)^(K) [L_fu f_k] (fh) psi^((i))_k (fg^(-1) fh) \ &
= sum_(fh in X) sum_(k=1)^(K) f_k (fu^(-1)fh) psi^((i))_k (fg^(-1) fh) \
&= sum_(fp in X) sum_(k=1)^(K) f_k (fp) psi^((i))_k (fg^(-1) fu fp)  \ &
= sum_(fp in X) sum_(k=1)^(K) f_k (fp) psi^((i))_k ((fu^(-1) fg)^(-1) fp) #h(2em) & fp <- fu^(-1)fh\
&= [f star psi](fu^(-1) fg) = [L_fu [f star psi]](g)
$

---
注意当 $frak(G)$ 不是交换群时，群卷积和群相关操作也不交换，但是有 $f star psi = [psi star f]^*$，其中 $square^*$ 是内卷积操作，即 $f^*(g) = f(g^(-1))$：
$
[f star psi](fg) 
&= sum_(fh in X) sum_(k=1)^(K) f_k (fh) psi_k (fg^(-1) fh) \ &
= sum_(fh in X) sum_(k=1)^(K) f_k (fh) psi_k (fg^(-1) fh) #h(3em) & fp <- fg^(-1) fh\
&= sum_(fp in X) sum_(k=1)^(K) psi_k (fp) f_k ((fg^(-1))^(-1) fp) \ &
= [psi star f](fg^(-1)) = [psi star f]^* (fg).
$

== 线性组合和单点函数的等变性

#tab 假设有两个关于群 $frak(G)$ 等变的信号 $f(dot)$ 和 $g(dot)$，显然其线性组合也是关于群 $frak(G)$ 等变的：
$
[L_fg [a f + b g]] (fu) = [a f + b g](fg^(-1) fu) = [a L_fg f + b L_fg g](fu)
$
其中 $fg in frak(G)$。对于单点非线性函数 $sigma: RR --> RR$，其作用在信号上的方式是简单的函数复合，即 $sigma compose f$ 容易证明它对群 $frak(G)$ 中的任一元素 $fg$ 的等变性：
$
L_(fg) [sigma compose f] (fu) = [sigma compose f] (fg^(-1) fu) = sigma[f(fg^(-1) fu)] = [sigma compose L_fg] (fu).
$

== 子群池化和陪集池化
#tab 回忆通常卷积网络的（最大值）池化操作，对于单通道的信号 $f: ZZ^2 -> RR$，它其实可以写为
$
["MaxPool"_1 f](bx) = max_(bold(u) in cal(N)(bx)) f(bold(u))
$
其中 $cal(N)(bx)$ 是 $bx$ 的一个邻域，实践中常常为以 $bx$ 为中心的一个正方形邻域，如九宫格。再在 $ZZ^2$ 的某个子集（例如 $(3ZZ)^2$）上采样，最后再映射回 $ZZ^2$。这就是池化中的两步操作：

+ 计算给定位置 $bx$ 的邻域最大值
+ 选取一部分邻域最大值作为输出

---

我们可以依据这两步构造包含群的版本。首先对于第一步，我们可以考虑一个在单位变换 $frak(e)$ 的一个包含于群 $frak(G)$ 的 “邻域” $frak(U) in frak(G)$，然后考虑 $frak(g U) := \{frak(g u): frak(u) in frak(U) \}$，这就得到了邻域最大值的群元素版本：
$
P f (fg) = max_(frak(k) in frak(g U)) f(frak(k))
$
这个操作是群等变的：
$
[L_fu P] f(fg)  = L_fu [P f](fg)
&= [P f](fu^(-1) fg) = max_(frak(k) in fu^(-1) fg frak(U)) f(frak(k))\
&= max_( fh in fg frak(U)) f(frak(u)^(-1) fh) = max_( fh in fg frak(U))[L_fu f](fh) #h(2em) & fh <- frak(u k) \
&= [P L_fu](fg).
$

= 群相关的高效实现

---
对于平面的变换群 $frak(G)$，如果每个群元 $fg in frak(G)$ 都可以写成一个平移变换 $frak(t) in ZZ^2$ 和一个稳定子，即保零点的旋转变换 $frak(s)$ 的复合，我们称它是*可分的*。例如群 $p 4$，它的元素 $frak(g)$ 总是可以写成 $frak(g) = frak(t s)$，其中 $frak(t)$ 是平移，$frak(s)$ 是以原点为旋转中心的旋转，根据前文中提到的群元对应的变换的同态性质，有
$
[f star psi^((i))](frak(g)) = [f star psi^((i))](frak(t s)) = sum_(fh in frak(X)) sum_(k) f_k (fh) L_frak(t) [L_frak(s) psi_k^((i)) (frak(h))]
$
注意平移变换 $L_frak(t)$ 对应的就是通常卷积中的平移变换。所以可以先算出 $L_frak(s) psi_k$，然后直接套用通常的卷积即可。其次，注意这里的 $psi_k$ 是群 $frak(G)$ 到 $RR$ 的映射，因此该映射可以直接使用一个数组 `F` 来表示 —— 我们不需要群元素的表示，而是只需要将它们编号，然后构造一个字典/数组来查找即可。
---
考虑 $ZZ^2$ 上可分的变换群 $frak(G)$，那么该群一共有 $S times n times n$ 个群元。假设上一层的通道数为 $K$，下一层通道数为 $K'$，那么对应于这一层卷积核的数组 `F` 的形状为 
$
overbrace(K' times underbrace(K times underbrace(S times n times n, "群元素编号，对应于" psi_k^((i))), "考虑上一层的所有通道，对应于" psi^((i))), "考虑下一层的所有通道，对应于完整的" psi).
$
接着我们需要将 $psi$ 和 $L_frak(s)$ 配对，由于 $frak(s)$ 是任选的稳定子，配对后相当于一个新的函数 $g(redText(frak(s)'), blueText(frak(s)), greenText(frak(t))) = h(frak(s)'^(-1)frak(s t))$，所以配对后的结果是 $K' times redText(S) times K times blueText(S) times greenText(n times n)$，我们将配对后的数组记为 `G`。
---
为了计算 $frak(s)'^(-1)frak(s t)$ 我们需要用到一个双射 $g(s, u, v)$，它接受 $S times n times n$ 中的一个三元组，这对应于群 $frak(G)$ 中一个元素的指标，输出该群元素对应的（可逆的）矩阵表示。因此 $frak(s t)$ 对应的指标是 $(s, t_x, t_y)$，而 $frak(s)'$ 由于是稳定子，没有复合平移变换，因此对应的指标是 $(s', 0, 0)$，因此可以得到 $frak(s)'^(-1)frak(s t)$ 对应的指标是
$
(macron(s), macron(t)_x, macron(t)_y) = g^(-1)[(g(s', 0, 0))^(-1)g(s, t_x, t_y)]
$
因此就有对应 $mono(G[i,s',j,s,t_x,t_y]=F[i,j,macron(s),macron(t)_x,macron(t)_y])$。这样得到的 `G` 就是对应于 $L_frak(s) psi$ 的映射字典。

接下来要执行的是通常的卷积操作，只需将 `G` 的形状 $K' times S' times K times S times n times n$ 变形为 $(K'S) times (K S) times n times n$，即输入通道数为 $K S$ 输出通道数为 $K' S$ 的卷积；特征图 `f` 也可以做类似的操作，其原本的形状为 $K times S times n times n$，将前两维合并，得到的形状是 $(K S) times n times n$，和改变形状后的卷积核对应。

= 结果和讨论

---

实验中用作参照的模型是全卷积网络和 ResNet，在做了随机旋转的 RotatedMNIST 和 CIFAR 系列数据集上都有相比于其他模型更好的效果；在群卷积的范畴之中（通常卷积可以视作 $ZZ^2$ 上平移变换群卷积这一特殊情形）拥有最多群元素 $p 4 m$ 在 CIFAR 系列数据集上的效果最好。

#grid(
  align: horizon+center,
  columns: (42%, 53%),
  column-gutter: 6%,

  figure(
    image("image-1.png", width: 90%),
    kind: table,
    caption: [本文模型（P4CNN）与各基线模型在 Rotated MNIST 上的训练误差]
  ),
  figure(
    image("image-2.png", width: 90%),
    kind: table,
    caption: [通常卷积网络 ( $ZZ^2$)、$p 4$ 和 $p 4 m$ 在 CIFAR10 和 CIFAR10+ 上的训练误差和模型参数量]
  )
)

---

从上面的结果可以看到，群等变模块可以替换通常卷积模型中的卷积、非线性层和池化层，并得到更好的效果。

群等变 CNN 的进一步的拓展改进包括将六边形网格中的变换群、$RR^3$ 中的变换群、连续（局部紧）群以及更大的有限群。不过对于后两个实现难度和计算成本较大。

= 代码实现和应用
== `e2cnn`


== 生命游戏

