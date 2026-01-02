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
#let project_name = "基于nnU-Net模型的小样本肺部肿瘤分割研究"
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
  header: header_name + " | " + name_no,
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

#outline()

#pagebreak()

点击链接查看和 Kimi 的对话 https://www.kimi.com/share/19b5f848-c632-8690-8000-000076b6a4b4
点击链接查看和 Kimi 的对话 https://www.kimi.com/share/19b5f850-a912-8161-8000-000063fea791

= 研究背景与目的

#h(2em)

#pagebreak()

= 数据集介绍

#h(2em) 本研究采用的数据为数据集是医学图像分割十项全能挑战赛  (Medical Segmentation Decathlon, MSD)#cite(<MSD-Dataset-antonelli2022medical>) 中的第 6 个子任务，即 MSD Lung Tumours 数据集#cite(<MSD-Paper-simpson2019large>)。其目标是从 CT 图像中分割出肺部肿瘤，MSD 选择该数据集的原因是“在大的背景中分割出小目标”。该数据集包含 96 例非小细胞肺癌患者的薄层CT扫描，官方划分为 64 例训练集和 32 例测试集，其中测试集可以通过官网提交分割结果进行测试。

非小细胞肺癌（NSCLC）作为肺癌中最常见的类型，占据了大约 85% 的肺癌病例，主要包含鳞状细胞癌、腺癌和大细胞癌等亚型。与小细胞肺癌相比，NSCLC 的生长速度通常较慢，治疗手段也更为多样化，通常取决于肿瘤的具体类型、发展阶段以及患者整体的健康状况。在 NSCLC 的诊断和治疗过程中，CT扫描扮演着至关重要的角色。它能提供关于肿瘤大小、形状和位置的详细信息，帮助医生确定病变的精确阶段，并指导手术和放疗计划。此外，CT 图像对于监测肿瘤对治疗的响应和检测复发或转移至关重要



#pagebreak()

= 探索性数据分析

#h(2em)

#pagebreak()

= 方法与模型

#h(2em)

#pagebreak()

= 实验与结论

#h(2em)

#pagebreak()

= 计算机程序代码说明

#pagebreak()

#bibliography("ref.bib",   // 你的 BibTeX 文件
              title: "参考文献",
              style: "ieee", 
              full: true
)

