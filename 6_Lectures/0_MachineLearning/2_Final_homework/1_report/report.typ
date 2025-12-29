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

#let wk_report_name = "中山大学 MA7259《机器学习》期末作业"
#let header_name = "中山大学 MA7259《机器学习》期末作业"
#let project_name = "基于模型的肺部肿瘤分割研究"
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

#set page(
  paper: "a4",
  numbering: "1",
  header: header_name + " | " + project_name + " | " + name_no,
)

#set par(
  first-line-indent: 2em,
  justify: true,
)

#set heading(numbering: "1.")

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

点击链接查看和 Kimi 的对话 https://www.kimi.com/share/19b5f848-c632-8690-8000-000076b6a4b4
点击链接查看和 Kimi 的对话 https://www.kimi.com/share/19b5f850-a912-8161-8000-000063fea791

= 研究背景与目的

#h(2em)#lorem(100)

#lorem(100)

#lorem(100)

#lorem(100)
#pagebreak()

= 探索性数据分析

#h(2em)#lorem(100)

#lorem(100)

#lorem(100)

#lorem(100)
#pagebreak()

= 方法与模型

#h(2em)#lorem(100)
#lorem(100)

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

