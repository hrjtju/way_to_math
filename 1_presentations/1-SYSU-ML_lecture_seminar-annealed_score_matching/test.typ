#import "@preview/touying:0.6.1": *
#import themes.dewdrop: *

#import "@preview/numbly:0.1.0": numbly

#show: dewdrop-theme.with(
  aspect-ratio: "16-9",
  footer: self => (self.info.institution + " | " + self.info.author),
  navigation: "mini-slides",
  config-info(
    title: [[ICLR2021] Score-Based Generative Modeling 
    through SDE],
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

= SMLD & DDPM
== Denoising Score Matching w/ Langevin Dynamics (SMLD)


== Denoising Diffusion Probabilistic Models (DDPM)


= Score-Based Model & SDEs
== 



= VE, VP & Sub-VP SDEs


= SDEs in the Wild


= Prob. Flow ODE


= Sampling Methods


