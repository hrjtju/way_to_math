
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

== Generative Modeling by Estimating Gradients of the Data Distribution

Yang Song, Stefano Ermon | NeurIPS 2019 | https://arxiv.org/abs/1907.05600

=== Score matching
==== 生成模型和 Score matching 动机

#h(2em)生成模型的目的是获取所需要生成范畴中的对象（如图片）的隐藏分布。生成的过程就是从该分布中采样。假设有从一个未知分布 $pd(bx)$ 中采样得到的数据集 $\{bx_i in RR^D\}_(i=1)^n$。我们要尝试估计该分布。一个自然的假设是 

$
pd(bx) = frac(exp(-f_(theta)(bx)), Z(theta))
$

其中 $f_(theta): RR^D -> RR$ 是某个函数，$Z(theta)$ 是归一化因子。若不加其他考虑，直接处理 $pd(bx)$ 将会不可避免地遇到计算 $Z(theta)$ 的困难。因此 Score matching 的一个核心思路是转而去估计分布的 Score function，其“几何直观”的意义是指向概率密度增加的方向：

$
s_(theta)(bx) := nabla_(bx) log p_(theta)(bx) = -nabla_(bx) f_(theta)(bx) - underbrace(nabla_(bx) log Z(theta), =0) = -nabla_(bx) f_(theta)(bx).
$

==== 以方便计算为目的的 Score matching 目标函数变换

#h(2em)自然地，我们有 Score matching 的原始目标函数：

$
J(theta) := frac(1,2)EE_(bx ~ pd) [norm(s_(theta)(bx)-nabla_(bx)log pd(bx))].
$

但由于我们不知道原始数据的分布，因此我们无法求得 $nabla_(bx) log pd(bx)$，可以做下面的变换

$
&frac(1,2)EE_(pd) [norm(s_(theta)(bx)-nabla_(bx)log pd(bx))] \
=& frac(1,2)EE_(pd) [ norm(s_(theta)(bx)) ] 
  + cancel(EE_(pd) [norm(nabla_(bx) log pd(bx))]) 
  - EE_(pd) [ angle.l s_(theta)(bx), nabla_(bx) log pd(bx) angle.r] \ 
=& frac(1,2)EE_(pd) [ norm(s_(theta)(bx)) ] 
  + integral p(bx) angle.l s_(theta)(bx), nabla_(bx) log pd(bx) angle.r "d"x 
  + const\  
// 内积
=& frac(1,2)EE_(pd) [ norm(s_(theta)(bx)) ] 
  + integral angle.l s_(theta)(bx), #text(red)[$nabla_(bx) pd(bx)$] angle.r "d"x 
  + const\  
=& frac(1,2)EE_(pd) [ norm(s_(theta)(bx)) ] 
  + ( cancel(integral_(partial RR^D) s_theta(bx) pd(bx) dd x) + integral_(RR^D) pd(bx) nabla_(bx) s_(theta)(bx)) 
  + const \
=& frac(1,2)EE_(pd) [ norm(s_(theta)(bx)) ] 
  + integral p(bx) "div"(s_(theta)(bx)) dd x 
  + const\  
=& EE_(pd) [ frac(1,2) norm(s_(theta)(bx)) 
  + tr(nabla_(bx) s_(theta)(bx)) ]
$

==== 降低目标函数计算成本

#h(2em)上式中的 $tr(nabla_(bx) s_(theta)(bx))$ 计算成本还是太高。庆幸我们可以对原数据增加 Gaussian 噪声，从而将其经过一个条件分布 $q_(sigma)(macron(bx)|bx) ~ N(0, sigma^2 I)$ 得到加噪声后的数据 $macron(bx)$。经计算后我们能得到更加实际的目标函数。我们可以从扰动后的数据向量 $mx ~ q_(sigma)(mx|bx)$ 对应的损失函数开始推导：

$
J(theta) 
&= frac(1,2)EE_(mx ~ ps(mx)) [norm(s_(theta)(mx)-nabla_(mx)log ps(mx))_2^2]\
&= frac(1,2)EE_(mx ~ ps(mx)) [ norm(s_(theta)(mx))_2^2 ] 
  + cancel(EE_(mx ~ ps(mx)) [norm(nabla_(mx) log pd(mx))_2^2]) 
  - EE_(mx ~ ps(mx)) [ angle.l s_(theta)(mx), nabla_(mx) log ps(mx) angle.r] \ 
&= frac(1,2)EE_(mx ~ ps(mx)) [ norm(s_(theta)(mx))_2^2 ] 
  - integral ps(mx)  angle.l s_(theta)(mx), nabla_(mx) log ps(mx) angle.r dd mx 
  + const \ 
&= frac(1,2)EE_(mx ~ ps(mx)) [ norm(s_(theta)(mx))_2^2 ] 
  - integral angle.l s_(theta)(mx), nabla_(mx) ps(mx) angle.r dd mx 
  + const \ 
&= frac(1,2)EE_(mx ~ ps(mx)) [ norm(s_(theta)(mx))_2^2 ] 
  - integral lr(angle.l s_(theta)(mx), nabla_(mx) integral qs(mx|bx)pd(bx) dd bx angle.r) dd mx 
  + const \ 
&= frac(1,2)EE_(mx ~ ps(mx)) [ norm(s_(theta)(mx))_2^2 ] 
  - integral.double ps(mx) pd(bx) angle.l s_(theta)(mx), nabla_(mx) qs(mx|bx) angle.r dd bx dd mx 
  + const \ 
&= EE_(mx ~ ps(mx), bx ~ pd(bx)) [ 
    frac(1,2) norm(s_(theta)(mx))_2^2
    - angle.l s_(theta)(mx), nabla_(mx) qs(mx|bx) angle.r ]
  + const \ 
&= frac(1,2) EE_(mx ~ ps(mx), bx ~ pd(bx)) [ 
    norm(s_(theta)(mx))_2^2
    - 2 angle.l s_(theta)(mx), nabla_(mx) qs(mx|bx) angle.r 
    + underbrace(norm(nabla_(mx) qs(mx|bx))_2^2, "constant w.r.t." theta)
  ]
  + const \ 
&= frac(1,2)EE_(bx ~ pd, ) [ norm(s_(theta)(macron(bx))-nabla_(macron(bx))log q_(sigma)(macron(bx)|bx))_2^2 ]
$

注意此时 $nabla_(macron(bx))log q_(sigma)(macron(bx)|bx)$ 还可以写为 $display(- frac(1, sigma) dot epsilon)$，其中 $epsilon$ 是从 $N(0, I)$ 中采样得到的噪声，原目标函数就变成

$
J(theta) = frac(1,2)EE_(bx ~ pd, ) [ norm(s_(theta)(macron(bx)) + frac(1, sigma) dot epsilon)_2^2 ]
$

换句话说，score function 在这个意义下正在预测加入到训练数据中的噪声。在实际操作中，我们需要权衡 $epsilon$ 的取值，如果太小，该方法起不到明显效果；如果太大，加噪声后的分布和原分布的区别大，难以学习原分布的特征。

=== Langevin 动力学

#h(2em)假如我们已经训练好 $s_(theta)(bold(x))$，我们应该如何做"生成"这个动作呢？根据 score function 拟合概率的对数梯度 $nabla_(bx) log pd(bx)$，它指向概率密度上升的方向。我们可以按照自然地想法做梯度上升，这样迭代就可以得到很有可能属于该分布的点：

$
bold(x)_{t+1} <- bold(x)_t - epsilon s_(theta)(bold(x)_t)
$

但这一方法总会使得若干步后的采样结果收敛于原始分布之密度函数的若干极大值点，与生成模型的多样性目标不符。因此为解决这一问题，我们可以用 Langevin Dynamics 来采样。其与原来方法的区别在于在每一步中加入噪声，最后迭代得到的结果将会服从原始分布 $p_("data")(bold(x))$：

$
bold(x)_{t+1} <- bold(x)_t - frac(epsilon, 2) s_(theta)(bold(x)_t) + sqrt(epsilon) z_t
$

其中 $z_t ~ N(0, I)$。

=== Score-based 生成模型的问题

+ #highlight[一是流形假设造成的问题]。我们所处世界中的高维数据往往分布在一个低维流形上。但上文中提到的对数据分布密度函数在低维流形的环绕空间 $RR^D$ 中求梯度是没有意义的。
+ #highlight[二是低概率密度区域的估计问题]。如果原分布是一个混合分布，且两个 “峰” 中间存在一个低概率密度的区域，模型学习时将难以获取该区域的信息，最后训练的结果在该区域的表现将会很差。

如果我们使用 Gauss 分布对原分布做扰动，则得到的新分布的支持集将会是整个环绕空间，而不是流形。另一方面，恰当强度的扰动也会使得低概率密度区域的概率密度增加，从而更容易采样到该区域中的点，为模型训练提供更多的信息。

=== 方法

#h(2em)上文中提到，对数据做扰动时，大强度的扰动会使得训练变得简单，但扰动后的分布与原分布相差很大；小强度的扰动使得扰动后分布近似原分布，但会有诸如低概率密度区域训练点不足的问题。

文中提出一个整合两者有点的方法，即不考虑单个扰动强度 $sigma$，而是考虑一个序列 $\{sigma_i\}_(i=1)^n$ 其中 $sigma_n$ 是一个足够小的数 (例如 0.01），$sigma_1$ 是一个足够大的数 (例如 25）。我们训练一个条件 score function $s_(theta)(bold(x), sigma)$ 预测不同扰动强度下的噪声方向。此时目标函数变为 

$
ell(theta, sigma) &:= frac(1, 2) EE_(p_("data")(bold(x)),macron(bold(x))~N(bold(x),sigma^2I)) [norm(s_(theta)(macron(bold(x))) + frac(macron(bold(x)) -bold(x), sigma^2))_2^2] \
ell(theta, \{sigma_i\}_(i=1)^n) &:= frac(1,n)sum_(i=1)^n lambda(sigma_i)ell(theta, sigma_i)
$

其中 $lambda(sigma_i)$ 是权重函数，常取为 $lambda(sigma_i) = sigma_i^2$，以平衡不同扰动强度下 score function 的范数。在采样时，我们就做类似模拟退火的采样动作。首先选取最高的扰动强度 $sigma_1$ 然后在该噪声强度下迭代若干次，然后选取次高的扰动强度 $sigma_2$ 并在该噪声强度下迭代若干次，以此类推。这样就可以综合利用大扰动强度和小扰动强度的有点，从而更好的采样。在实际实验中，作者提出的方法在 CIFAR-10 数据集上取得了不错的效果。整个过程如下面的算法所示。

#show: style-algorithm
#algorithm-figure(
  "Annealed Langevin Dynamics",
  vstroke: .5pt + luma(200),
  {
    import algorithmic: *
    Procedure(
      "Annealed Langevin Dynamics",
      ($\{sigma_i\}_(i=1)^n$, $epsilon$, $T$),
      {
        Comment[Initialize the search range]
        Assign[$macron(bold(x))_0$][$bold(v)$]
        For(
          $t <- 1, ..., L$,

          Comment[Set step size $alpha_i$],
          $alpha_i <- epsilon dot sigma_i^2 "/" sigma_L^2$,
          For(
            $t <- 1, ..., T$,
            "Draw " + $bold(z)_t ~ N(0, I)$, 
            $display(macron(bold(x))_t <- macron(bold(x))_(t-1) + alpha_i "/" 2 dot s_(theta)(macron(bold(x))_(t-1), sigma_i) + sqrt(alpha_i)) bold(z)_t$
          ),
          $macron(bold(x))_0 <- macron(bold(x))_T$ 
        )
      },
      Return($macron(bold(x))_T$)
    )
  }
)

#h(-2em)参考资料
+ https://www.youtube.com/watch?v=B4oHJpEJBAA

= 附录

在推导 Score matching 中的优化目标变换时，在
$
frac(1,2)EE_(pd) [ norm(s_(theta)(bx)) ] 
  + integral angle.l s_(theta)(bx), #text(red)[$nabla_(bx) pd(bx)$] angle.r "d"x 
= EE_(pd) [ frac(1,2) norm(s_(theta)(bx)) 
  + tr(nabla_(bx) s_(theta)(bx)) ] + const
$
这一步遇到了问题。讲解视频中常常将 $x$ 考虑为一维向量，从而规避了这里的推导。一维版本推导中使用到分部积分公式，我推测多维版本也使用了分部积分公式。我参考 https://www.jhanmath.com/?p=142 中的内容将其中的缺失步骤补齐，并填补先前没有学好的散度相关知识。

对于函数 $f: RR^3 --> RR$，其对应的微分算子可以写成 $display(nabla = [partial/(partial x), partial/(partial y), partial/(partial z)])$，这与 $f$ 的梯度相容，因为 $display(nabla f = [(partial f)/(partial x), (partial f)/(partial y), (partial f)/(partial z)])$。散度是该微分算子和向量场 $bold(V) = [bold(V)_x, bold(V)_y, bold(V)_z]$ 的内积：
$
"div"(bold(V)) = angle.l nabla, bold(V) angle.r = (partial bold(V)_x)/(partial x) + (partial bold(V)_y)/(partial y) + (partial bold(V)_z)/(partial z) = tr(nabla bold(V)).
$
对于标量值函数 $u$，可以证明有下面的性质
$
angle.l nabla, u bold(V) angle.r 
&= (partial u bold(V)_x)/(partial x) 
  + (partial u bold(V)_y)/(partial y) 
  + (partial u bold(V)_z)/(partial z) \
&= [(partial u)/(partial x)  bold(V)_x + u (partial bold(V)_x)/(partial x)]
  + [(partial u)/(partial y)  bold(V)_y + u (partial bold(V)_y)/(partial y)]
  + [(partial u)/(partial z)  bold(V)_z + u (partial bold(V)_z)/(partial z)]\
&= [(partial u)/(partial x)  bold(V)_x
    + (partial u)/(partial y)  bold(V)_y
    + (partial u)/(partial z)  bold(V)_z]
  + u[(partial bold(V)_x)/(partial x)
    + (partial bold(V)_y)/(partial y)
    +(partial bold(V)_z)/(partial z)]
= angle.l nabla u, bold(V) angle.r + u angle.l nabla, bold(V) angle.r
$
通过该运算法则，就有向量场的分部积分公式。$Omega$ 是 $RR^n$ 中的一个有界开集，其边界 $Gamma = partial Omega$ 分段光滑，有
$
integral_(Omega) angle.l nabla, u bold(V) angle.r dd Omega
 = integral_(Gamma) u angle.l bold(V), hat(bold(n)) angle.r dd Gamma   
&= integral_(Omega) angle.l nabla u, bold(V) angle.r dd Omega
  + integral_(Omega) u angle.l nabla, bold(V) angle.r dd Omega.
$
在 Score matching 的推导中，由于积分区域是全空间，并默认概率密度函数在无穷远处的极限为零，应用分部积分公式即可。

#h(-2em)参考资料
+ https://www.jhanmath.com/?p=142