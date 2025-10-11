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

// metadata
#let wk_report_name = "2025年9月29日至10月12日周报"
#let name_affiliation = "何瑞杰 | 中山大学 & 大湾区大学"

// Snippets
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
#let tab = {h(2em)}

#show strong: set text(blue)
#show: thmrules.with(qed-symbol: $square$)
#set page(
  paper: "a4",
  numbering: "1",
  header: wk_report_name + " | " + name_affiliation,
)
#set par(
  first-line-indent: 2em,
  justify: true,
)

#show figure.where(
  kind: table
): set figure.caption(position: top)

#set math.equation(numbering: "(1)")
#set heading(numbering: "1.")
#set math.cases(gap: 0.5em)
#align(
  center, 
  text(17pt)[#wk_report_name\ ] 
        + text(12pt)[\ 何瑞杰\ 中山大学, 大湾区大学]
)

// #show figure.caption: it => [
//   图
//   #context it.counter.display(it.numbering) : 
//   #it.body
// ]

= 项目进展
== 使用神经网络学习生命游戏的演化动力学
=== 生命游戏规则的变种

本周取 Golly 文档中若干邻域大小为 $3$ 的九种其他规则进行实验，每种规则的具体信息列表如下

#align(center,
  table(
    columns: 4,
    align: (center, center, center, left),
    stroke: none,
    table.hline(),
    table.header([规则名称], [邻域大小], [邻域类型], [特性描述]),
    table.hline(stroke: 0.5pt),
    [`B36/S23`],      [3], [Moore],       [和 Conway 的原版生命游戏相似，但有自我复制结构],
    [`B3678/S34678`], [3], [Moore],       [活细胞群中的死细胞的行为与死细胞群中的活细胞的行为相同],
    [`B35678/S5678`], [3], [Moore],       [有不可预测行为的菱形斑点],
    [`B2/S`],         [3], [Moore],       [活细胞每代都会死亡，但该系统常常爆发],
    [`B234/S`],       [3], [Moore],       [单个的 $2 times 2$ 会演化为一个波斯地毯],
    [`B345/S5`],      [3], [Moore],       [周期极长的振荡器可以自然地出现],
    [`B13/S012V`],    [3], [Von Neumann], [],
    [`B2/S013V`],     [3], [Von Neumann], [],
    table.hline(),
  )
)


=== 数据生成

#h(2em)经过进一步研究发现，Golly 虽然支持大量规则，但无法作为包导入 Python 中使用，只限于其程序之内。一番搜索后我找到了 `pyseagull`，并对所需的关键部分进行了检查。其运行模式十分简单，下面是一段官网给出的模拟代码：

```python
import seagull as sg
from seagull.lifeforms import Pulsar

# Initialize board
board = sg.Board(size=(19,60))

# Add three Pulsar lifeforms in various locations
board.add(Pulsar(), loc=(1,1))
board.add(Pulsar(), loc=(1,22))
board.add(Pulsar(), loc=(1,42))

# Simulate board
sim = sg.Simulator(board)
sim.run(sg.rules.conway_classic, iters=1000)
```
相比于先前代码中的逻辑，它要简单得多。即使运行过程中被包装成一个函数，我们依然可以通过 `sg.Simulator` 中的 `get_history()` 方法得到这一次模拟的所有历史数据的 `ndarray`，其形状为 `[iters+1, w, h]`，其中 `iters` 为迭代轮数，`w` 和 `h` 为网格尺寸大小。

另外，`pyseagull` 还支持自定义的简单规则。生命游戏的简单规则可以写为 `B[...]/S[...]` 的字符串格式。最经典的生命游戏的规则为 `B3/S23`，意为死细胞邻居存活数为 $3$ 时，下一时刻复活；活细胞邻居存活数为 $2$ 或 $3$ 时下一时刻继续存活，其他情况下一时刻细胞死亡。自定义函数签名如下

```python
seagull.rules.life_rule(X: ndarray, rulestring: str)
```
该函数在 `pyseagull` 的源码实现中通过正则表达式提取 `rulestring` 中的规则信息。由于其灵活性，在需要时我们可以将其拓展为其他更加复杂的规则，例如改变邻域的形状、将邻域的贡献从各向同性改为各向异性。



=== 对模型和训练的改动
==== 增大卷积核大小
我将 `SimpleCNNTiny` 和 `SimpleCNNSmall` 的卷积核大小增加到 $5$。

==== 缩小并行模型的参数量
我将 `MultiScale` 网络中并行层中输出的特征图的通道数都降为 $2$。具体而言，调整后网络的结构如下

```python
class MultiScale(nn.Module):
    __version__ = '0.2.0'
    def __init__(self):
        super(MultiScale, self).__init__()
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, stride=1, 
                      padding=1, padding_mode="circular"),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.1)
        )
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=5, stride=1,  
                      padding=2, padding_mode="circular"),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.1)
        )
        self.conv_3x3_dilated = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, stride=1,  
                      padding=2, dilation=2, padding_mode="circular"),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.1)
        )
        self.stem = nn.Sequential(
            nn.Conv2d(int(2*3), 4, kernel_size=3, stride=1,  
                      padding=1, padding_mode="circular"),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(2, 2, kernel_size=3, stride=1,  
                      padding=1, padding_mode="circular")
        )
    ...
```

==== 权重稀疏化

在损失函数中添加 L1 损失。具体计算方式如下

```python
l1_reg = 0
for name, param in model.named_parameters():
    if 'weight' in name:
        l1_reg = l1_reg + torch.linalg.vector_norm(param, ord=1, dim=None)
```
然后将 `l1_reg` 假如损失函数中，权重为 $10^(-5)$。

此方法来源于 https://stackoverflow.com/a/58533398

=== 群等变 CNN


=== 在不同规则的演化系统上的实验结果
==== 调整通常卷积网络参数和权重稀疏化后的结果
===== 训练曲线

===== 预测结果


==== 以上改动基础上将网络改为群等变CNN后的结果
===== 训练曲线

===== 预测结果


=== 对模型的可视化的解释
==== 训练完毕的神经网络作为生命游戏模拟器


==== 通过 CNN 权重的直接解释方案


#align(bottom, 
  [
    参考资料
    + https://pyseagull.readthedocs.io/
    +
  ]
) 

#pagebreak()
== 微型抗癌机器人在血液中的动力学
=== 项目目的

微型抗癌机器人是通过癌症细胞散发出的化学吸引物 (chemoattractant) 趋化性驱动 (chemotaxis-driven) 运动，与癌细胞进行配体-受体结合后定向释放药物，达到治疗的目的。本项目研究理想状况下的微型抗癌机器人集群在血液中的动力学。

=== 建模

目前项目对血液中的化学吸引物、游离的微型机器人和与癌细胞结合的微型机器人分布进行建模。设 $t$ 时刻，位于血液中 $bx$ 位置的化学吸引物浓度为 $c(bx, t)$，化学吸引物正常的消耗或讲解速率为 $k$， ，则有
$
(partial c)/(partial t) = D_c nabla^2 c - k c + S_(Omega_t)(bx)
$
其中 
- $D_c$ 为化学吸引物在血液中的扩散系数
- $k$ 为化学吸引物正常的消耗或讲解速率
- $Omega_t$ 为癌细胞所在区域，$S_(Omega_t)(bx)$ 为癌细胞区域中 $bx$ 位置向血液中释放化学吸引物的速度
类似地，设 $rho(bx, t)$ 为游离机器人血液中的分布密度，$b(bx, t)$ 为非游离的机器人的分布密度，有
$
(partial rho)/(partial t) 
  &= D_rho nabla^2 rho - nabla dot (chi rho nabla c) - k_b rho delta_(Omega_t) + k_u b \
(partial b)/(partial t) 
  &= k_b rho delta_(Omega_t) - k_u b 
$


#pagebreak()
= 文献阅读

== Denoising Diffusion Probabilistic Models
Jonathan Ho, Ajay Jain and Pieter Abbeel | https://arxiv.org/abs/2006.11239

本周把 DDPM 的剩余部分补完。

=== 补遗


=== 实验结果


=== 总结和讨论



参考资料
+ https://arxiv.org/abs/1907.05600
+ https://arxiv.org/abs/2006.11239
+ https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice

#pagebreak()
== Sliced Score Matching: A Scalable Approach to Density and Score Estimation
- Yang Song, Sahaj Garg, Jiaxin Shi, Stefano Ermon 
- https://arxiv.org/abs/1905.07088


#pagebreak()

#pagebreak()

= 学习进度
== 机器学习理论
=== Markov Chain Monte Carlo (MCMC)



== 随机过程



#h(2em)本周


== 随机微分方程

#h(2em)本周

#pagebreak()
= 问题解决记录
== uv 相关
uv 是基于 Rust 的新一代 Python 包管理器，具有可迁移性强、快速、简单的特点。

=== Pytorch CUDA 版本的配置
若不特意配置，在通过 uv 在环境中添加或下载 Pytorch 时自动下载的是 CPU 版本，无法享受硬件加速。假设系统中 CUDA 的版本是 11.8，可以在 `pyproject.toml` 中配置

// ```toml
// [[tool.uv.index]]
// name = "pytorch-cu118"
// url = "https://download.pytorch.org/whl/cu118"
// explicit = true
// [tool.uv.sources]
// torch = [
//   { index = "pytorch-cu118", marker = "sys_platform == 'win32'" }
// ]
// torchvision = [
//   { index = "pytorch-cu118", marker = "sys_platform == 'win32'" }
// ]
// ```
// 然后运行 `uv sync` 或执行下面的代码强制重新下载：
// ```bash
// uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
// ```

== Typst 相关

=== 数学公式自动编号



== Python 相关
=== 猴子补丁

在使用 `pyseagull` 这个包时，我对它的源码进行了一些改动。这导致这些改动在其他机器上搭建环境时不可迁移。解决此问题的方法是使用 Python 的*猴子补丁*。它利用 Python 的灵活性，直接覆盖包中的某些函数或类参数。以 `pyseagull` 为例，假如我需要修改它的 `life_rule` 函数，而 `life_rule` 又要用到 `_parse_rulestring` 和 `_count_neighbors`，就可以像下面这样写

```python
from seagull.rules import life_rule

def life_rule_monkey_patch(X: np.ndarray, rulestring: str) -> np.ndarray:
    """
    Monkey Patch for function `life_rule`. 
    Add support for Von Neumann Neighborhood.
    """
    ...

def _parse_rulestring_monkey_patch(r: str) -> Tuple[List[int], List[int]]:
    """
    Add support for Von Neumann Neighborhood.
    """
    ...

def _count_neighbors_monkey_patch(X: np.ndarray, von_neumann: bool) -> np.ndarray:
    """
    Add support for Von Neumann Neighborhood.
    """
    ...

# Apply monkey patch
life_rule = life_rule_monkey_patch
```

将这段代码放在前面，执行时就会对 `seagull` 的 `life_rule` 函数做直接的替换，以实现自定义的新功能。对于类变量，也是同理。

== 深度学习相关


#pagebreak()
= 下周计划
*论文阅读*
+ 生成模型
  - 
+ 动力学
  - 

*项目进度*
+ 使用神经网络学习生命游戏的演化动力学
  - 
+ 微型抗癌机器人在血液中的动力学
  -

*理论学习*
+ 随机过程课程
  - 
+ 随机微分方程
  - 
