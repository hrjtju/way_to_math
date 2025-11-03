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
  justify: true,
)

#show figure.where(
  kind: table
): set figure.caption(position: top)

#set math.mat(delim: ("[", "]"), align: center)
#set heading(numbering: "1.")

#show: dewdrop-theme.with(
  aspect-ratio: "16-9",
  footer: self => (self.info.author),
  navigation: "mini-slides",
  config-info(
    title: [Small-Data Discovery of Complex Dynamics#linebreak() via Equivariant Neural Networks],
    author: [Ruijie HE$#h(0.1em)^("a,b")$, Liu HONG$#h(0.1em)^("a")$, Wuyue YANG$#h(0.1em)^("c")$],
    date: datetime(year: 2025, month: 10, day: 29),
    institution: [a. Sun Yat-sen University, b. Great Bay University, c. BIMSA],
  ),
  mini-slides: (
    height: 1.8em,
    x: 2em,
    display-subsection: true,
    linebreaks: false 
  )
)

#let in-outline = state("in-outline", false)
#let mini-slides- = state("mini-slides", false)
#show outline: it => {
  in-outline.update(true)
  it
  in-outline.update(false)
}

#let flex-caption(long, short) = context if mini-slides-.get() { short } else { long }


#set heading(numbering: numbly("{1}.", default: "1.1"))

#title-slide()

#outline-slide()

= Cellular Automata

---

#v(1em)
Cellular automata is a discrete dynamical system that evolves according to a set of rules. All rules are local and deterministic and applied simultaneously to all cells in the grid.

Generally, let $X$ denote the states of all cells in the grid, each cell has a state $q$ in set $Q$ of all states, $f$ denote the rule function, which acts on the state of a cell $X_t(bold(v))$ and its neighborhood $N_t (bold(v))$ to produce the next state $X_(t+1)(bold(v))$:

$
  X_(t+1)(bold(v)) = f(X_t (bold(v)), N_t (bold(v))).
$

For some specific cases, we can easily realize the transition on $1$ or $2$ dimensional cellular automata with the aid of convolution operation with carefully picked kernel $phi$ and post-processing function $g$:

$
  X_(t+1) = g(X_(t) * phi).
$

---

== 1D-CA

If we consider $1$-dimensional cellular automata, and let the neighborhood of a cell $bold(v)$ be the set of cells that are adjacent to $bold(v)$, we can uniquely determine the transition dynamics by exhausting the outcome of all possible combinations of the cell state $X_t(bold(v))$ and that of its neighborhood $N_t (bold(v))$:

#figure(
  image("image-3.png", width: 50%), 
  caption: [1D-Cellular Automata Transition Rules],
)

---

== Classification of 1D-CA

#v(2em)
Apparently, there are only $256$ possible sets of transition rules for aforementioned 1D-Cellular Automata, we can number them from $1$ to $256$.Stephen Wolfram classified 1D-Cellular Automata into $4$ classes based on their behavior:

+ *Stable*: System evolves quickly into a stable state.
+ *Oscillators*: System evolves between two or more stable states.
+ *Complex*: System non-periodic behavior, but not enough to be classified as chaos.
+ *Chaos*: System evolves into a highly sensitive, non-periodic behavior.

We can stack the evolution of system states in chronologically order to form a 2D image, to better investigate the behavior of the system.

---


#figure(
  image("image-5.png", width: 54%), 
  caption: [Evolution trajectory of 1D-Cellular Automata with Different Types of Rules],
)

---

== Conway's Game of Life

#v(2em)
Conway's game of life is a typical 2D-CA with $3 times 3$ sized neighborhood. The system evolves according to the following rules:

#v(2em)
#figure(
  grid(
    columns: 2,
    column-gutter: 2em,
    rows: 4,
    row-gutter: 1em,

    align: (right, left),
    
    [*Underpopulation*], [A cell dies if it has $< 2$ living neighbors.],
    [*Overpopulation*], [A cell dies if it has $> 3$ living neighbors.],
    [*Survival*], [A cell survives if it has $2$ or $3$ living neighbors.],
    [*Birth*], [A cell is born if it has exactly $3$ living neighbors.]
  )
)

---

=== Creatures

#figure(
  grid(
    columns: 2,
    image("image-6.png", width: 100%),
    image("Conways_game_of_life_breeder.png", width: 100%),
  ),
  caption: [
    Creatures in Conway's Game of Life.#linebreak()
    #text(size: 6pt, [Adopted from #link("https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life#/media/File:Conways_game_of_life_breeder.png")])
  ]
)

---
=== Turing Completeness

#figure(
  image("image-7.png", width: 65%),
  caption: [Turing Completeness of Conway's Game of Life Realized in Golly.#linebreak()#v(-1em)
    #text(size: 5pt, [By Andrew Trevorrow and Tomas Rokicki - Screenshot of Golly program, GPL, https://commons.wikimedia.org/w/index.php?curid=18644263])
  ],
)

---
== Variants of Life

Typical variants of Conway's Game of Life uses $3 times 3$ neighborhood, which is called *Moore* neighborhood. We use `BXX/SXX` to denote a rule with `B` being the birth condition and `S` being the survival condition.

#figure(
  table(
    columns: 3, 
    align: (center, center, left),
    rows: 7, 
    row-gutter: 0.5em,
    stroke: none,
    table.hline(),
    table.header([Rule Symbol], [Rule Name], [Description]),
    table.hline(stroke: 0.5pt),
    [`B3/S23`], text(size: 16pt)[Life], text(size: 14pt)[John Conway's rule is by far the best known and most explored CA.],
    [`B36/S23`], text(size: 16pt)[HighLife], text(size: 14pt)[Very similar to Conway's Life but with an interesting replicator.],
    [`B3678/S34678`], text(size: 16pt)[Day & Night], text(size: 14pt)[Dead cells in a sea of live cells behave the same as live cells in a sea of dead cells.],
    [`B35678/S5678`], text(size: 16pt)[Diamoeba], text(size: 14pt)[Creates diamond-shaped blobs with unpredictable behavior.],
    [`B2/S`], text(size: 16pt)[Seeds], text(size: 14pt)[Every living cell dies every generation, but most patterns still explode.],
    [`B234/S`], text(size: 16pt)[Serviettes or Persian Rug], text(size: 14pt)[A single 2x2 block turns into a set of Persian rugs.],
    [`B345/S5`], text(size: 16pt)[LongLife], text(size: 14pt)[Oscillators with extremely long periods can occur quite naturally.],
    table.hline()
  )
)

---

- *Von Neumann* neighborhood with $4$ adjacent cells, has rule symbol like `BXX/SXXV`.
- Likewise, in *hexagonal grid*, we can use $6$ adjacent cells to form a hexagonal neighborhood. 
- When the neighbors don't contribute equally, we get *non-totalistic rules* or *MAP rules*.
- Rules can be generalized to bigger sizes of neighborhoods.

#figure(
    grid(
    columns: (60%, 40%),
    column-gutter: -5%,
    image("image-8.png", width: 90%),
    image("image-9.png", width: 90%)
  )
)

---

= Equivariant Neural Networks

---

== Equivariance

Consider linear space $V$ with a transformation group $frak(G)$, which we call $frak(G)$-space, and a function $Phi: V -> V$. We say $Phi$ is *equivariant* if it satisfies

$
  Phi(T_fg x) = T'_fg Phi(x)
$
In which $T_fg$ is the transformation of $V$ corresponding to the group element $fg$ in $frak(G)$. $T$ and $T'$ are not necessarily the same, but must be linear representations of elements in $frak(G)$, i.e., for any $fg, fh in frak(G)$, we have $T(fg fh) = T(fg) T(fh)$.

#figure(
  image("image.png", width: 30%),
  caption: [Equivariance of the human eye]
)

---

Typical crystal group $p 4$ contains all translation and rotation on $ZZ^2$ with unit $display(pi/2)$. It has the following representation:
$
fg(r, u, v) = mat(
  cos((r pi)/2), -sin((r pi)/2), u;
  sin((r pi)/2), #h(0.84em) cos((r pi)/2), v;
  0, 0, 1
)
$
in which $r in \{0, 1, 2, 3\}$，$(u, v) in ZZ^2$. The result of group element $fg$ acting on vector $x$ can be written as
$
  fg x tilde.eq mat(
  cos((r pi)/2), -sin((r pi)/2), u;
  sin((r pi)/2), #h(0.84em) cos((r pi)/2), v;
  0, 0, 1
) mat(
  u'; v'; 1
)
$

---

We can rewrite feature map `F` with shape `[c, w, h]` as a function from $ZZ^2$ to $RR^c$:
$
F: ZZ^2 &-> RR^c \
(x, y) &|-> cases(mono(F[:, x, y])","#h(1.5em) & x in [0, w-1] "," y in [0, h-1], bold(0)"," &"otherwise")
$
For a signal $F$, we define group element $fg$ acting on $F$ as
$
  L_fg f(x) = f(fg^(-1) x),
$
in which $L_fg$ is a materialization of transform $T_fg$ corresponding to group element $fg$, such that $L_fg L_fh = L_(fg fh)$.

---

== Equivariance of Convolution

Consider ordinary spatial convolution $*$ and correlation $star$. Let input feature map to be $f: ZZ^2 -> RR^(K^((l)))$ and kernel to be $psi^((i)) : ZZ^2 -> RR^(K^((l)))$, we have 
$
[f * psi](bx) &= sum_(by in ZZ^2) sum_(k=1)^(K^((l))) f_k (bold(y)) psi^((i))_k (bx - by)#h(2em) & "convolution"\ 
[f star psi](bx) &= sum_(by in ZZ^2) sum_(k=1)^(K^((l))) f_k (bold(y)) psi^((i))_k (by - bx) & "correlation"\ 
$

It is not difficult to verify that they are equivariant w.r.t. translations but not equivariant to rotations.

---

To get a convolution or correlation layer which is equivariant w.r.t. both translation and rotation, or more broadly, to any finite group, consider the following construction by Cohen et al with the first layers be

$
f^((1))(fg) &= [f star psi^((i))](fg) &= sum_(by in ZZ^2) sum_(k=1)^(K^((l))) f^((0))_k (bold(y)) psi^((0, i))_k (fg^(-1) by),\
f^((l+1))(fg) &= [f star psi^((i))](fg) &= sum_(fh in frak(G)) sum_(k=1)^(K^((l))) f^((l))_k (fh) psi^((l, i))_k (fg^(-1) fh),
$

in which $l gt.slant 1$ and $fg, fh in frak(G)$; $f$ and $psi^((0, i))$ are maps from $ZZ^2$ to $RR$. The group correlation operator is equivariant w.r.t. group elements in $frak(G)$:

$
[[L_fu f] star psi](fg) = [L_fu [f star psi]](g).
$

---

== Other Necessary Equivariant Blocks and Operations

First we notice that linear combination and element-wise non-linearities preserves equivariance:

$
[L_fg [a f + b g]] (fu) = [a f + b g](fg^(-1) fu) = [a L_fg f + b L_fg g](fu)
$

$
L_(fg) [sigma compose f] (fu) = [sigma compose f] (fg^(-1) fu) = sigma[f(fg^(-1) fu)] = [sigma compose L_fg] (fu).
$

in which $fg in frak(G)$ and $sigma: RR --> RR$. 

Moreover, we can construct group version of pooling with subset $frak(U)$ acting as "neighborhood" with the following:

$
P f (fg) = max_(frak(k) in frak(g U)) f(frak(k))
$

which is equivariant.

---

= Rules Learning

---

We may use equivariant neural networks to learn dynamics of non-linear systems like cellular automata or other more sophisticated ones with translation, rotation and permutation symmetries or equivariances. 

For systems that is difficult to sample across the whole state space, we could use the extrapolation or generalization ability to learn on the evolution trajectories with limited state or motif diversity.

Moreover, we can let the well-trained network to be the system simulator, to predict future system states, which is difficult or impossible to model using explicit code.

---

#focus-slide(text(size: 40pt)[Thanks for Listening :)])

