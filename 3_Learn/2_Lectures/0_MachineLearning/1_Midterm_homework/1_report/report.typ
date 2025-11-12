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

#set text(lang: "zh", font: ("New Computer Modern Math", "KaiTi"))

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

下面对上述 `describe()` 的结果进行分析。

整体概述

这份数据描述了777所美国大学在1995年的各项指标。数据包含分类变量（如Private）和数值变量。describe()函数为数值变量提供了丰富的统计信息，让我们能够快速了解数据的中心趋势、离散程度和分布形态。

== 分变量详细分析

=== 学校基本属性

- Private
    - 分析： 这是一个分类变量，describe() 显示其唯一值（unique）为2（Yes/No），其中 top 值为 True（即"Yes"），频数（freq）为565。
    - 结论： 数据集中私立大学占绝大多数。565所私立大学 / 777所总数 ≈ 72.7% 的学校是私立的。

=== 招生情况

- Apps（申请数）， Accept（录取数）， Enroll（入学数）

    - 平均数： 平均每所大学收到3001份申请，录取2018人，最终有779人入学。

    - 统计行为：
        - 差异巨大： 三个变量的标准差（std）都非常大，几乎接近甚至超过其平均值（例如Apps的std=3870 > mean=3001）。这表明不同大学的招生规模存在天壤之别。最大值（Apps max=48,094）和最小值（Apps min=81）也印证了这一点。

        - 录取与入学率： 我们可以粗略计算：

            - 平均录取率 = Accept / Apps ≈ 2018 / 3001 ≈ 67.3%

            - 平均入学率/报到率 = Enroll / Accept ≈ 779 / 2018 ≈ 38.6%

        - 分布形态： 中位数（50%）远小于平均数（均值）。例如，Apps的中位数是1558，但均值是3001。这意味着有少量大学拥有极其庞大的申请量（极右偏分布），拉高了整体平均值。大部分大学的申请数集中在较低水平（一半的大学申请数少于1558）。

=== 生源质量

- Top10perc（高中前10%）， Top25perc（高中前25%）

    - 平均数： 平均而言，新生中有27.6%来自高中排名前10%的学生，55.8%来自前25%的学生。

    - 统计行为： 分布相对均匀（标准差小于均值），Top25perc的中位数（54%）和均值（55.8%）很接近，说明分布相对对称。而Top10perc的中位数（23%）低于均值（27.6%），表明存在一些顶尖生源高度集中的大学，使分布轻微右偏。

=== 学生与教师规模

- `F.Undergrad`（全日制本科）， `P.Undergrad`（非全日制本科）

    - 平均数： 平均全日制本科生为3699人，非全日制为855人。

    - 统计行为： 同样呈现出极大的差异（标准差很大）。全日制学生的规模分布极右偏（中位数1707 << 均值3699），说明少数大型大学主导了数据。非全日制学生的最大值（21,836）和标准差（1522）表明，不同大学在教学模式上差异显著。

- `S.F.Ratio`（师生比）

    - 平均数： 平均师生比为14.09（即平均每14名学生对应1名教师）。

    - 统计行为： 分布相对集中（标准差3.96），中位数（13.6）和均值（14.09）接近，大部分学校的师生比在11.5到16.5之间（25%-75%分位数）。

=== 费用与开支

- `Outstate`（外州学费）， `Room.Board`（食宿费）， `Books`（书本费）， `Personal`（个人开销）， `Expend`（生均支出）
  - 平均数： 外州学费平均为10,441，食宿费为4,358，书本费为549，个人开销为1,341，生均教学支出为\$9,660。
  - 统计行为：
      - 学费和支出差异显著： Outstate和Expend的标准差非常大，表明大学的收费水平和资源投入相差悬殊。Expend的最大值（56,233）是均值（9,660）的5倍多，再次印证了资源的高度不平等。

      - 固定费用相对稳定： Room.Board和Books的分布相对集中，说明这些基础生活成本在不同大学间差异较小。

      - 分布形态： Outstate和Expend的分布明显右偏（中位数 < 均值），说明有一小部分高学费、高支出的精英大学。

=== 师资力量

- PhD（拥有博士学位教师比例）， Terminal（拥有终极学位教师比例）

    - 平均数： 平均72.7%的教师拥有博士学位，79.7%拥有终极学位（通常指本领域的最高学位，如博士、艺术硕士MFA等）。

    - 统计行为： 分布较为集中，大部分大学的教师博士学位比例在62%到85%之间（25%-75%分位数），师资队伍整体素质较高且在不同大学间相对均衡。Terminal的比例普遍高于PhD，这是合理的。

=== 学校成果与声誉

- perc.alumni（捐赠校友比例）

    - 平均数： 平均 22.7% 的校友会捐款。

    - 统计行为： 分布较为分散，不同大学的校友捐赠文化和忠诚度差异很大。

- Grad.Rate（毕业率）

    - 平均数： 平均毕业率为 65.5%。

    - 统计行为：

        - 异常值： 最大值118%是一个明显的异常值，因为毕业率不可能超过100%。这可能是数据录入错误，或者计算方法特殊（例如包含了超期毕业的学生），需要进一步核查。

        - 分布： 剔除异常值影响，毕业率的分布相对正常，中位数（65%）与均值（65.5%）基本一致，表明分布大致对称。但毕业率本身在不同大学间差异不小（标准差17.2%）。

=== 总结

1.  数据构成： 数据集以私立大学为主（72.7%）。
2.  极度不均衡： 大学在规模（申请数、学生人数）和资源（学费、支出）上表现出极端的差异，存在明显的“头部效应”。大部分统计量的平均值都被少数大型/富裕的大学拉高，中位数通常能更好地代表“典型”大学的情况。
3.  招生漏斗： 从申请到录取再到入学，数量大幅减少，平均入学率仅为38.6%。
4.  潜在数据问题： Grad.Rate存在超过100%的异常值，需要在后续分析中处理。
5.  相对稳定的指标： 师生比（S.F.Ratio）、师资博士比例（PhD, Terminal）、基础生活成本（Room.Board, Books）等指标在不同大学间的分布相对集中。

#pagebreak()

= 方法与模型
== 数据预处理

#h(2em)#lorem(100)

#lorem(100)

== 线性回归



#h(2em) 线性回归可以用直线拟合、向数据矩阵 $bold(X)$ 的列空间做正交投影、极大似然估计等视角进行理解和解释。记数据集为 $D = lr(\{ (bx_{n}, y_(n)) \})_(n=1)^(N)$，定义对标签的预测函数为下面的线性形式
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

#set text(lang: "en")
#bibliography("ref.bib",   // 你的 BibTeX 文件
              title: "参考文献",
              style: "ieee", 
              full: true
)

#set text(lang: "zh")

#pagebreak()

= 附录