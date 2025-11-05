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

#let wk_report_name = "中山大学 MA7259《机器学习》期中作业"
#let header_name = "中山大学 MA7259《机器学习》期中作业"
#let project_name = "美国高校招生与毕业率统计数据的分析和预测"
#let name_no = "何瑞杰 25110801"

#set text(lang: "zh", font: "Kai")

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

#set math.equation(numbering: "(1)")
#set math.mat(delim: ("[", "]"), align: center)
#set heading(numbering: "1.")
#set math.cases(gap: 0.5em)

#set page(
  paper: "a4",
  numbering: "1",
  header: [#header_name | #name_no],
)

#set par(
  first-line-indent: 2em,
  justify: true,
)

#set heading(numbering: "1.")

#show figure.where(
  kind: table
): set figure.caption(position: top)


#v(5em)

#align(
  center, 
  text(12pt)[#wk_report_name\ ] + v(0.5em) 
        + text(17pt)[#project_name\ ]
        + text(12pt)[\ #name_no]
)

#v(5em)

#align(center, [摘#h(2em)要])
#pad(
  left: 6em, right: 6em, 
  [
    #lorem(100)
  ]
)
#pagebreak()

= 研究背景与目的

#h(2em)#lorem(100)

#lorem(100)

#lorem(100)

#lorem(100)
#pagebreak()

= 探索性数据分析

#h(2em)#lorem(100)

#figure(
  caption: [`college` 数据集的各数据域名称及含义],
  table(
  columns: 2,
  stroke: none, // 设置顶部和底部边框
  align: left,
  table.hline(),
  table.header([数据域名称],[含义]),
  table.hline(stroke: 0.5pt),
  
  // 表格数据行
  [`Private`], [私立或公立大学],
  [`Apps`], [收到的申请数量],
  [`Accept`], [录取的申请数量],
  [`Enroll`], [入学新生数量],
  [`Top10perc`], [高中排名前10%的新生百分比],
  [`Top25perc`], [高中排名前25%的新生百分比],
  [`F.Undergrad`], [全日制本科学生人数],
  [`P.Undergrad`], [非全日制本科学生人数],
  [`Outstate`], [外州学生学费],
  [`Room.Board`], [食宿费用],
  [`Books`], [预估书本费用],
  [`Personal`], [预估个人开销],
  [`PhD`], [拥有博士学位的教师比例],
  [`Terminal`], [拥有最高学位的教师比例],
  [`S.F.Ratio`], [师生比例],
  [`perc.alumni`], [捐赠校友比例],
  [`Expend`], [生均教学支出],
  [`Grad.Rate`], [毕业率],
  
  table.hline(),
)
)

#lorem(100)

#lorem(100)

#lorem(100)
#pagebreak()

= 方法与模型
== 数据预处理

#h(2em)#lorem(100)

#lorem(100)

== 线性回归



#h(2em)线性回归可以用直线拟合、向数据矩阵 $bold(X)$ 的列空间做正交投影、极大似然估计等视角进行理解和解释。记数据集为 $D = lr(\{ (bx_{n}, y_(n)) \})_(n=1)^(N)$，定义对标签的预测函数为下面的线性形式
$
y(bx, bold(w)) = w_(0) + w_(1)x_(1) + dots.c + w_(D)x_(D) = bold(w)^top mat(1; bx)
$
其中 $bx in bb(R)^(D)$，$bold(w) in RR^(D+1)$. 这是线性回归的基本形式。若将特征和对应的标签堆叠起来，即 
$
  bold(X) = mat(bx_1^top; dots.v; bx_N^top), wide bold(y) = mat(y_(1); dots.v; y_(N))
$
则线性回归的预测结果为 $hat(bold(y)) = bold(X omega)$，我们要求预测结果 $hat(bold(y))$ 距离真实目标 $bold(y)$ 越近越好，这就得到了线性回归的目标函数：
$
  J(bold(w)) = 1/(2) sum_(n=1)^(N) (y_(n) - hat(y)_(n))^2 = 1/2 norm(bold(hat(y)) - bold(y))_2^2 = 1/2 norm(bold(X omega) - bold(y))_2^2
$
这是一个拥有光滑凸目标函数的优化问题，若 $bold(X)^top bold(X)$ 可逆，令 $J(bold(omega))$ 的梯度为零，我们能得到其解析解
$
  bold(omega)^* = (bold(X)^top bold(X))^(-1)bold(X)^top bold(y).
$
实际操作中，我们会在目标 $J(bold(omega))$ 中加入正则项 $R(bold(omega))$，通过对权重向量进行软限制的方式防止过拟合。修改过后的目标函数为
$
  J(bold(omega)) = 1/2 norm(bold(X omega) - bold(y))_2^2 +  lambda R(bold(omega)),
$
其中 $lambda gt.slant 0$ 是需要人为给定的权重超参数。常用的正则项可以是权重向量的 L2 范数（这将得到岭回归）或 L1 范数（这将得到 LASSO 回归）。若选取的 L2 范数，问题总是存在整洁的解析解：
$
  bold(omega)^* = (bold(X)^top bold(X) + lambda I)^(-1)bold(X)^top bold(y).
$
从计算上看，这会使得括号中的矩阵可逆，此时该模型总是有解析解。

线性回归还有结合了核函数的变体形式。具体而言，考虑一列核函数 $lr(\{phi_i\})_(i=1)^M$ 作用于每个样本 $bx_(n)$，得到新的特征矩阵 
$
  bold(X) = mat(phi_1(bx_1); dots.v; phi_M (bx_N))
$
这在数据科学中被称为*特征工程*。在得到了新的特征 $bold(X)$ 后，接下来的线性回归操作和上面相同。

== CART 决策树

#h(2em)#lorem(100)

#lorem(100)

#lorem(100)

#pagebreak()

= 实验与结论

#h(2em)#lorem(100)

#lorem(100)

#lorem(100)

#pagebreak()

#bibliography("ref.bib",   // 你的 BibTeX 文件
              title: "参考文献",
              style: "ieee", 
              full: true
)

#pagebreak()

= 附录