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
    title: [Small-Data Discovery of Complex Dynamics#linebreak() via Equivariant Neural Networks],
    author: [Ruijie HE$#h(0.1em)^("a,b")$, Liu HONG$#h(0.1em)^("a")$, Wuyue YANG$#h(0.1em)^("c")$],
    date: datetime(year: 2025, month: 10, day: 29),
    institution: [a. Sun Yat-sen University, b. Great Bay University, c. BIMSA],
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

