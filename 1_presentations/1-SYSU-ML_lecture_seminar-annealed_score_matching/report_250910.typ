#import "@preview/touying:0.6.1": *
#import themes.dewdrop: *

#import "@preview/numbly:0.1.0": numbly

#show: dewdrop-theme.with(
  aspect-ratio: "16-9",
  footer: self => (self.info.institution + " | " + self.info.author),
  navigation: "mini-slides",
  config-info(
    title: [Denoising Diffusion Probabilistic Models],
    subtitle: [组会汇报 | 论文阅读报告],
    author: [何瑞杰],
    date: datetime(year: 2025, month: 9, day: 10),
    institution: [中山大学 #sym.hyph.point 数学学院],
  ),
  mini-slides: (
    height: 3em,
    x: 2em,
  )
)

#set heading(numbering: numbly("{1}.", default: "1.1"))

#title-slide()

#outline-slide()

= Background
== GANs
=== Model Setup

#image("image.png")

// the learning objective of GAN
$ min_G max_D V(D, G) = EE_(x ~ p_"data") [ log D(x) ] + EE_(z ~ p_z) [ log ( 1 - D(G(z)) ) ] $

== VAEs

== Flow-based Models

= Methods
== Forward Process

== Reverse Process

== Algorithm

== Implementation

= Experiments

= Speeding up and Distillation





// Add references


// References
// 1. https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

