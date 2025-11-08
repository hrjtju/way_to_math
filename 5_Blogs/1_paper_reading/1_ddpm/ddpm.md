# Denoising Diffusion Probabilistic Models

Jonathan Ho, Ajay Jain and Pieter Abbeel | https://arxiv.org/abs/2006.11239


## 综观

不同于 VAE 等等的一个隐变量空间的生成模型，DDPM 尝试用多步编码/解码，或是加噪/去噪方式生成数据。它的每一步可以视为是一个去噪自编码器（Denoising Autoencoder），而其采样（生成）过程通过与加噪过程的 Markov 链"共用"，并通过预测加噪过程中的随机噪声实现加噪过程的 "反转"，并添上 Langevin 动力学的随机噪声实现对分布的采样。与正常的 Langevin 动力学不同的是，从第 $t$ 步到 $t-1$ 步，我们认为其在尝试适配一个分布 $p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$ 过程中，只做了一次 Langevin 动力学的迭代。

!image.png

## 加噪过程的建模

首先对于加噪过程，我们有一列噪声强度 $\{\beta_t\}_{t=0}^T$，然后按照 $q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}) \sim N(\boldsymbol{x}_{t-1} | \beta_t \boldsymbol{I})$ 来进行：

$$
\begin{aligned}
\boldsymbol{x}_t 
&= \sqrt{1 - \beta_t} \boldsymbol{x}_{t-1} + \sqrt{\beta_t} \boldsymbol{\epsilon}_t, \quad & \boldsymbol{\epsilon}_t \sim N(\boldsymbol{0}, \boldsymbol{I}) \\
&= \sqrt{1 - \beta_t} [ \sqrt{1 - \beta_{t-1}} \boldsymbol{x}_{t-2} + \sqrt{\beta_{t-1}} \boldsymbol{\epsilon}_{t-1} ] + \sqrt{\beta_t} \boldsymbol{\epsilon}_t, \quad & \boldsymbol{\epsilon}_{t-1} \sim N(\boldsymbol{0}, \boldsymbol{I}) \\
&= \sqrt{(1 - \beta_t)(1 - \beta_{t-1})} \boldsymbol{x}_{t-2} + \sqrt{(1 - \beta_t) \beta_{t-1}} \boldsymbol{\epsilon}_{t-1} + \sqrt{\beta_t} \boldsymbol{\epsilon}_t \\
&= \sqrt{(1 - \beta_t)(1 - \beta_{t-1})} \boldsymbol{x}_{t-2} + \sqrt{(1 - \beta_t) \beta_{t-1} - (1 - \beta_t) + 1} \tilde{\boldsymbol{\epsilon}}_t, \quad & \tilde{\boldsymbol{\epsilon}}_t \sim N(\boldsymbol{0}, \boldsymbol{I}) \\
&= \sqrt{(1 - \beta_t)(1 - \beta_{t-1})} \boldsymbol{x}_{t-2} + \sqrt{1 - (1 - \beta_t)(1 - \beta_{t-1})} \tilde{\boldsymbol{\epsilon}}_t \\
& \quad \vdots \\
&= \sqrt{\prod_{s=1}^t (1 - \beta_s)} \boldsymbol{x}_0 + \sqrt{1 - \prod_{s=1}^t (1 - \beta_s)} \tilde{\boldsymbol{\epsilon}}_t = \sqrt{\bar{\alpha}_t} \boldsymbol{x}_0 + \sqrt{1 - \bar{\alpha}_t} \tilde{\boldsymbol{\epsilon}}_t
\end{aligned}
$$

这给出了加噪过程中每一步得到的 $\boldsymbol{x}_t$ 更加便捷的表示，对后续的推导有帮助。

## 优化目标

由于我们的目标是从随机噪声 $\boldsymbol{x}_0$ 生成图像，即 $\boldsymbol{x}_T$，因此需要最大化的项是 $\mathbb{E}_{p_t}[-\log p_t(\boldsymbol{x}_0)]$。因此我们需要继续用变分推断的技巧，逐步将其转化为可以计算得到的项。首先我们有下面的上界

$$
\begin{aligned}
& \quad \mathbb{E}_{p_t}[-\log p_t(\boldsymbol{x}_0)] \\
&= \mathbb{E}_{p_t}[-\log p_t(\boldsymbol{x}_0) \int q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0) \mathrm{d} \boldsymbol{x}_{1:T}] = \mathbb{E}_{p_t}[-\int q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0) \log p_t(\boldsymbol{x}_0) \mathrm{d} \boldsymbol{x}_{1:T}] \\
&= - \mathbb{E}_{p_t(\boldsymbol{x}_0), q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[
  \log \frac{p_t(\boldsymbol{x}_0) p_t(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}{p_t(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}
\right] = - \mathbb{E}_{p_t(\boldsymbol{x}_0), q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[
  \log \frac{{\color{red}{p_t(\boldsymbol{x}_{0:T})}} q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}{p_t(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0) {\color{red}{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}}}
\right] \\
&= - \mathbb{E}_{p_t(\boldsymbol{x}_0), q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[
  \log {\color{red}{\frac{p_t(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}}}
\right] - \mathbb{E}_{p_t(\boldsymbol{x}_0), q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[
  \log \frac{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}{p_t(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}
\right] \\
&= - \mathbb{E}_{p_t(\boldsymbol{x}_0), q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[
  \log {\color{red}{\frac{p_t(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}}}
\right] - \underbrace{\mathbb{E}_{p_t(\boldsymbol{x}_0)} \left[ D_{\text{KL}}[q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0) \| p_t(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)] \right]}_{> 0} \\
&< \mathbb{E}_q \left[
  \log \frac{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}{p_t(\boldsymbol{x}_{0:T})}
\right] =: L
\end{aligned}
$$

上式中的 $\boldsymbol{x}_{1:T}$ 不好处理，我们可以将打包的变量拆开，最后可以将其写成若干项 KL 散度之和，其中两项对应着 Markov 链头和尾，剩余的对应加噪过程的中间状态。

$$
\begin{aligned}
L 
&= \mathbb{E}_q \left[ \log \frac{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}{p_t(\boldsymbol{x}_{0:T})} \right] \\
&= \mathbb{E}_q \left[ \log \frac{q(\boldsymbol{x}_T|\boldsymbol{x}_0) \prod_{t=2}^T q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)}{p_t(\boldsymbol{x}_T) \prod_{t=1}^T p_t(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)} \right] \\
&= \mathbb{E}_q \left[ \log \frac{q(\boldsymbol{x}_T|\boldsymbol{x}_0)}{p(\boldsymbol{x}_T)} + \sum_{t=2}^T \frac{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)}{p_t(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)} + \log \frac{1}{p_t (\boldsymbol{x}_0|\boldsymbol{x}_1)} \right] \\
&= \mathbb{E}_q \left[ D_{\text{KL}}[q(\boldsymbol{x}_T|\boldsymbol{x}_0)\|p(\boldsymbol{x}_T)] + \sum_{t=2}^T D_{\text{KL}}[q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)\|p_t(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)] - \log p_t (\boldsymbol{x}_0|\boldsymbol{x}_1) \right] \\
&= \mathbb{E}_q \left[ L_T + \sum_{t=2}^T L_{t-1} - L_0 \right].
\end{aligned}
$$

显然，得到的结果符合我们的预期，我们需要训练一个带参分布 $p_\theta (\boldsymbol{x}_{t-1} | \boldsymbol{x}_t, \boldsymbol{x}_0)$，并让其与 $q(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t, \boldsymbol{x}_0)$ 对齐。后者是一个 Gauss 分布，使用 Bayes 公式不难得到它的分布参数

$$
\begin{aligned}
& \quad q(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t, \boldsymbol{x}_0) \\
&= \frac{q(\boldsymbol{x}_t | \boldsymbol{x}_{t-1}, \boldsymbol{x}_0) q(\boldsymbol{x}_{t-1} | \boldsymbol{x}_0)}{q(\boldsymbol{x}_t | \boldsymbol{x}_0)} \\
&= \frac{q(\boldsymbol{x}_t | \boldsymbol{x}_{t-1}) q(\boldsymbol{x}_{t-1} | \boldsymbol{x}_0)}{q(\boldsymbol{x}_t | \boldsymbol{x}_0)} \\
&\propto \exp \left\{ -\frac{1}{2} \left[ 
  \frac{\|\boldsymbol{x}_t - \sqrt{\alpha_t} \boldsymbol{x}_{t-1}\|^2}{\beta_t}   + \frac{\|\boldsymbol{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \boldsymbol{x}_{0}\|^2}{1 - \bar{\alpha}_{t-1}}   - \frac{\|\boldsymbol{x}_t - \sqrt{\bar{\alpha}_{t}} \boldsymbol{x}_{0}\|^2}{1 - \bar{\alpha}_{t}} 
\right] \right\} \\
&= \exp \left\{ -\frac{1}{2} \left[ 
  \frac{
    \|\boldsymbol{x}_t\|^2     - \sqrt{\alpha_t} \langle \boldsymbol{x}_t, {\color{red}\boldsymbol{x}_{t-1}} \rangle      + \alpha_t \|{\color{red}\boldsymbol{x}_{t-1}}\|^2
  }{\beta_t}   + \frac{
    \|{\color{red}\boldsymbol{x}_{t-1}}\|^2     - \sqrt{\bar{\alpha}_{t-1}} \langle {\color{red}\boldsymbol{x}_{t-1}}, \boldsymbol{x}_0 \rangle      + \bar{\alpha}_{t-1} \|\boldsymbol{x}_0\|^2
  }{1 - \bar{\alpha}_{t-1}} \right.\right. \\
& \quad \left.\left. - \frac{
    \|\boldsymbol{x}_t\|^2     - \sqrt{\bar{\alpha}_t} \langle \boldsymbol{x}_t, \boldsymbol{x}_0 \rangle      + \bar{\alpha}_t \|\boldsymbol{x}_0\|^2
  }{1 - \bar{\alpha}_t} 
\right] \right\} \\
&= \exp \left\{ -\frac{1}{2} \left[ 
  \left( \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} \right) {\color{red}\|\boldsymbol{x}_{t-1}\|^2}  - \left\langle \frac{\sqrt{\alpha_t}}{\beta_t} \boldsymbol{x}_t     + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \boldsymbol{x}_0,    {\color{red}\boldsymbol{x}_{t-1}} \right\rangle   + \text{const} 
\right] \right\} \\
&\propto \exp \left\{  
  \frac{\|\boldsymbol{x}_{t-1} - \tilde{\boldsymbol{\mu}}_t\|^2}{2 \tilde{\beta}_t} 
\right\}
\end{aligned}
$$

其中均值 $\tilde{\boldsymbol{\mu}}_t$ 和方差 $\tilde{\beta}_t$ 为

$$
\begin{aligned}
\tilde{\beta}_t &= 1 / \left( \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} \right) = \frac{(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)} \cdot \beta_t \\
\tilde{\boldsymbol{\mu}}_t &= \frac{\frac{\sqrt{\alpha_t}}{\beta_t} \boldsymbol{x}_t 
    + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \boldsymbol{x}_0}{\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}} = \left( \frac{\sqrt{\alpha_t}}{\beta_t} \boldsymbol{x}_t 
    + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \boldsymbol{x}_0 \right) \frac{(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)} \cdot \beta_t \\
&= \frac{\sqrt{\alpha_t}(1- \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)} \boldsymbol{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{(1 - \bar{\alpha}_t)} \boldsymbol{x}_0
\end{aligned}
$$

由于需要匹配一个 Gauss 分布，带参分布 $q(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t, \boldsymbol{x}_0)$ 也需要是一个 Gauss 分布 $N(\boldsymbol{\mu}_{\theta, t}, \Sigma_{\theta, t})$，不过在此我们将协方差矩阵简化为对角，即 $\boldsymbol{\mu}_{\theta, t} = \beta_{\theta, t} \boldsymbol{I}$。此时我们可以求中间过程的 KL 散度，由于其参数是两个 Gauss 分布，我们有现成的结论：

$$
\begin{aligned}
L_{t-1} &= D_{\text{KL}}[q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)\|p_t(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)] \\
&= \frac{1}{2} \left[
  \log \frac{|\tilde{\beta}_t \boldsymbol{I}|}{|\beta_{\theta, t} \boldsymbol{I}|} - n + \text{tr}(\tilde{\beta}_t^{-1} \boldsymbol{I} \beta_{\theta, t} \boldsymbol{I}) + (\tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_{\theta, t})^\top \beta_{\theta, t}^{-1} \boldsymbol{I} (\tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_{\theta, t})
\right] \\
&= \frac{1}{2 \sigma_t} \left[
  \|\tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_{\theta, t}\|^2
\right] + \text{const}, \quad \text{令} \beta_{\theta, t} \text{为只与时间相关的} \sigma_t \\
&= \frac{1}{2 \sigma_t} \left[
  \left\| 
    \frac{\sqrt{\alpha_t}(1- \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)} \boldsymbol{x}_t 
    + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{(1 - \bar{\alpha}_t)} \cdot \frac{1}{\sqrt{\bar{\alpha}_t}} (\boldsymbol{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}) - \boldsymbol{\mu}_{\theta, t}
  \right\|^2
\right] + \text{const} \\
&= \frac{1}{2 \sigma_t} \left[
  \left\| 
    \frac{(\alpha_t (1- \bar{\alpha}_{t-1}) + \beta_t)}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}} \boldsymbol{x}_t 
    + \frac{\beta_t}{\sqrt{\alpha_t} \sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} - \boldsymbol{\mu}_{\theta, t}
  \right\|^2
\right] + \text{const} \\
&= \frac{1}{2 \sigma_t} \left[
  \left\| 
    \frac{(1- \bar{\alpha}_t)}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}} \boldsymbol{x}_t 
    - \frac{\beta_t}{\sqrt{\alpha_t} \sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} - \boldsymbol{\mu}_{\theta, t}
  \right\|^2
\right] + \text{const}, \quad \alpha_t = 1 - \beta_t, \bar{\alpha}_t = \prod_{s=0}^t \alpha_t \\
&= \frac{1}{2 \sigma_t} \left[
  \left\| 
    \frac{1}{\sqrt{\alpha_t}} \left[ \boldsymbol{x}_t 
    - \frac{(1 - \alpha_t)}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right] - \boldsymbol{\mu}_{\theta, t}
  \right\|^2
\right] + \text{const} \\
\end{aligned}
$$ 

进一步地，我们可以将预测均值 $\boldsymbol{\mu}_{\theta, t}$ 建模为与 $\tilde{\boldsymbol{\mu}}_t$ 相同的形式，即

$$
\boldsymbol{\mu}_{\theta} (\boldsymbol{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left[ \boldsymbol{x}_t - \frac{(1 - \alpha_t)}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_{\theta} (\boldsymbol{x}_t, t) \right]
$$

因此 $L_{t-1}$ 还可以进一步简化

$$
\begin{aligned}
L_{t-1} 
&= \frac{1}{2 \sigma_t} \left[
  \left\| 
    \frac{1}{\sqrt{\alpha_t}} \left[ \boldsymbol{x}_t 
    - \frac{(1 - \alpha_t)}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right] - \boldsymbol{\mu}_{\theta, t}
  \right\|^2
\right] + \text{const} \\
&= \frac{1}{2 \sigma_t} \left[
  \left\| 
    \frac{1}{\sqrt{\alpha_t}} \left[ \boldsymbol{x}_t 
    - \frac{(1 - \alpha_t)}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right] 
    - \frac{1}{\sqrt{\alpha_t}} \left[ \boldsymbol{x}_t - \frac{(1 - \alpha_t)}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_{\theta} (\boldsymbol{x}_t, t) \right]
  \right\|^2
\right] + \text{const} \\
&= \frac{(1 - \alpha_t)^2}{2 \sigma_t \alpha_t (1 - \bar{\alpha}_t)} 
  \left\| 
    \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\theta} (\boldsymbol{x}_t, t)
  \right\|^2 + \text{const} \\
\end{aligned}
$$

*推了半天之后我们发现我们的优化目标也只是预测噪声而已（而这并不是新鲜事）*，我们就可以很顺利地得到 DDPM 的训练和采样算法。我们也可以使用更加简单的目标函数，只需要将 $L_{t-1}$ 前面的系数扔掉，同时考虑服从离散均匀分布的 $t$ 即可。在下面的算法中，我们实际上也是随机从均匀分布中采样 $t$ 然后训练模型。

## 训练和采样算法

**Algorithm: Training**

```pseudocode
Procedure Training
  While not converged:
    x₀ ~ q(x₀)
    t ~ Uniform({1, ..., T})
    ε ~ N(0, I)
    θ ← θ - η ∇θ ∥ε - εθ(√(ᾱₜ) x₀ + √(1 - ᾱₜ) εθ(x₀, t))∥²
End Procedure
```

**Algorithm: Sampling**

```pseudocode
Procedure Sampling
  x_T ~ N(0, I)
  For t = T, ..., 1:
    If t > 1:
      z ~ N(0, I)
    Else:
      z ← 0
    End If
    x_{t-1} ← 1/√(αₜ) [xₜ - (1 - αₜ)/√(1 - ᾱₜ) εθ(xₜ, t)] + σₜ z
  End For
  Return x₀
End Procedure
```

可以将其与下面的模拟退火 Langevin 动力学采样进行对比

**Algorithm: Annealed Langevin Dynamics**

```pseudocode
Procedure Annealed Langevin Dynamics({σᵢ}ᵢ₌₁ⁿ, ε, T)
  Initialize x̄₀ ← v
  For i = 1 to L:
    αᵢ ← ε · σᵢ² / σₗ²
    For t = 1 to T:
      Draw zₜ ~ N(0, I)
      x̄ₜ ← x̄ₜ₋₁ + (αᵢ/2) · sθ(x̄ₜ₋₁, σᵢ) + √(αᵢ) zₜ
    End For
    x̄₀ ← x̄_T
  End For
  Return x̄_T
End Procedure
```

不难发现 DDPM 中的 Sampling 算法将 Annealed Langevin Dynamics 算法中的内层的循环减少到了 $1$。

## 离散值图像生成的最后一步

由于需要生成的对象是计算机中离散编码（如八位）的图像，对应最后一步的 $p_\theta (\boldsymbol{x}_0|\boldsymbol{x}_1)$ 需要得到的是离散分布。论文使用了一个简便的技巧，假设图像的像素值在离散化过程中从 $\{0, 1, ..., 255\}$ 线性地归一化到 $[-1, 1]$ 内，即 $0$ 被映射到 $-1$，$255$ 被映射到 $1$。在计算 $\boldsymbol{x}_0$ 的第 $i$ 个像素值是 $\boldsymbol{x}_0^{(i)}$ 时，我们可以简单地将 $\mathbb{R}$ 切分成 $256$ 块，其中 $1 \sim 254$ 分别对应着 $[-1 - 1/255, 1 + 1/255]$ 中宽度为 $2/255$ 的小区间 $[\delta_- (x), \delta_+ (x)]$，剩下的一头一尾就分别对应剩下的两个无限长度的区间。具体而言，

$$
\delta_+ (x) = \begin{cases} 
  \infty & \text{if } x = 1 \\ 
  x + 1/255 & \text{if } x < 1 
\end{cases}, \quad
\delta_- (x) = \begin{cases} 
  -\infty & \text{if } x = -1 \\ 
  x - 1/255 & \text{if } x > -1 
\end{cases}
$$

假设 $\boldsymbol{x}_0$ 的分布中各分量相互独立，就有

$$
p_\theta (\boldsymbol{x}_0) = \prod_{i=1}^n p_\theta (\boldsymbol{x}_0^{(i)}) = \prod_{i=1}^n \int_{\delta_-(\boldsymbol{x}_0^{(i)})}^{\delta_+(\boldsymbol{x}_0^{(i)})} N(x | \boldsymbol{\mu}_{\theta}^{(i)} (\boldsymbol{x}_1, 1), \sigma_1 \boldsymbol{I}) \mathrm{d} x
$$

这样也不会破坏分布的归一化性质。

## 实验

### 预测目标和目标函数匹配程度

实验发现，预测 $\tilde{\boldsymbol{\mu}}$ 需要匹配未化简形式的加权期望损失；而预测噪声 $\boldsymbol{\epsilon}$ 需要匹配简化的损失函数。这两个搭配效果几乎一样好。

### 去噪过程的协方差矩阵

如果将去噪过程的协方差矩阵变成科学系的对角形式，训练过程会变得不稳定，采样得到的图像质量也会下降。

## DDPM 的信息论视角

### 编码效率和失真度

模型在 CIFAR 上训练得到的对应于每一数据维度的信息比特数相差 $0.03$，这说明模型没有在训练集上过拟合。此外，再回到目标函数，已知有这样的分解：

$$
\mathbb{E}_q \left[ \underbrace{D_{\text{KL}}[q(\boldsymbol{x}_T|\boldsymbol{x}_0)\|p(\boldsymbol{x}_T)]}_{L_T} + \sum_{t=2}^T \underbrace{D_{\text{KL}}[q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)\|p_t(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)]}_{L_{t-1}} + \underbrace{ - \log p_t (\boldsymbol{x}_0|\boldsymbol{x}_1)}_{L_0} \right]
$$

注意 $L_1 + \cdots + L_{T-1}$ 中的每一项都是 KL 散度，分别描述神经网络参数化的 $p_{\boldsymbol{\theta}}$ 和分布 $q$ 的区别，它是使用 $p_{\boldsymbol{\theta}}$ 描述 $q$ 需要的额外编码开销（见附录），因此可将其看做生成过程中的*编码效率 (rate)*，这体现在相对于该体系中认为的最优逆向过程 $q$ 而言所需的额外编码量；而最后一项 $L_0$ 是一个负对数似然，它将 $\boldsymbol{x}_1$ 和 $\boldsymbol{x}_0$ 对齐比较，我们可将其看做从 $\boldsymbol{x}_T$ 经过降噪过程一路走来的结果相比于 $\boldsymbol{x}_0$ 的*失真度 (distortion)*。

### 分步有损压缩的信道模型

我们可以将 DDPM 的去噪过程看作是发送端向接受端发送信息。接收端在接收前只知道 $p$，发送端可以同步逐步使用 $p_\theta (\boldsymbol{x}_t|\boldsymbol{x}_{t+1})$ 编码服从分布 $q(\boldsymbol{x}_t|\boldsymbol{x}_{t+1}, \boldsymbol{x}_0)$ 的数据 $\boldsymbol{x}_t$，然后接收端用 $p_\theta$ 对其进行解码。如果发送端仅发送 $\boldsymbol{x}_T$，接收端仅依靠此信息和 $p_\theta$ 估计 $\hat{\boldsymbol{x}}_0$，这样会产生较高的失真度。

<div style="display: flex; gap: 20px;">
  <div style="flex: 1;">
    <strong>Algorithm: Sending $\boldsymbol{x}_0$</strong>
    <pre>
Procedure Sending(x₀)
  Send x_T ~ q(x_T|x₀) using p(x_T)
  For t = T-1, ..., 2, 1:
    Send x_t ~ q(x_t|x_{t+1}, x₀) using pθ(x_t|x_{t+1})
  End For
  Send x₀ using pθ(x₀|x₁)
End Procedure
    </pre>
  </div>
  <div style="flex: 1;">
    <strong>Algorithm: Receiving</strong>
    <pre>
Procedure Receiving
  Receive x_T using p(x_T)
  For t = T-1, ..., 1, 0:
    Receive x_t using pθ(x_t|x_{t+1})
  End For
  Return x₀
End Procedure
    </pre>
  </div>
</div>

每个时刻 $t$，接收端都会计算估计得到的 $\hat{\boldsymbol{x}}_0$ 和真实值的均方损失作为失真率，而记录自传输开始至该时刻接收器获得的所有每维度比特数为码率：即 $H(\boldsymbol{x}_t) + D_{\text{KL}}[q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)\|p_t(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)]$，画出的码率-失真率曲线可以看出，大量的信息被分配至肉眼难以看见的细节中：

!image.png

### 分步生成

作者还用各部得到的 $\boldsymbol{x}_t$ 直接预测 $\boldsymbol{x}_0$，得到的结果由下图所示。可见随着 $t$ 的减小，预测得到的 $\hat{\boldsymbol{x}}_0$ 逐步先显现总体特征，再逐步丰富局部细节。

!image-1.png

如果使模型都从某个共用的 $\boldsymbol{x}_t$ 出发执行降噪过程，越大的 $t$ 出发降噪得到的结果差异越大；越小的 $t$ 出发得到的降噪结果越相似。

!image-2.png

## DDPM 和自回归生成模型的比较

最后讨论 DDPM 的降噪过程和序贯生成的自回归生成模型（例如 ChatGPT 的模式）的相似性和区别。我们考虑这样的场景（暂时抛开加噪过程是加 Gauss 过程这件事）：生成图像的尺寸是 $N \times N$，所谓的"加噪过程"和"降噪过程"的步数为 $N^2$，$q(\boldsymbol{x}_t | \boldsymbol{x}_{t-1})$ 是一个离散的 dirac-delta 点质量，它做的事情是将图像从左至右，从上至下的第 $t$ 个像素变成空白；相应地，$q(\boldsymbol{x}_{t-1})$ 做的事情就是将第 $t$ 个像素恢复成原来的颜色。这样经过一整条"加噪过程"后，图像就变成了完全空白；再经过理想的"降噪过程"后图像又被逐像素涂色成了原来的样子。

这是我们假设神经网络建模的 $p_\theta$，足够强大——它可以学习到上面假设中 $q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$ 的分布，那么这俨然成了自回归模型。然而实践过程中，*DDPM 选择的使用 Gauss 噪声加噪、去噪的过程，可以看作是更为广义的自回归生成模型，它按照图像中的某种空间语义 token 按照从整体至局部的顺序逐渐生成图像*。相比于添加更强归纳偏置的一般自回归生成模型——它们都假设后生成的区域依赖于先生成的区域，而这在图像生成上没有道理！——DDPM 拥有更高的灵活性和更少的归纳偏置，这也解释了其优异的生成能力。

## 隐空间插值

最后，作者将不同图像加噪过程中某个 $t$（例如 $t = 500$）得到的 $\boldsymbol{x}_t$ 和 $\hat{\boldsymbol{x}}_t$ 做凸组合，再做去噪过程，得到了原图 $\boldsymbol{x}_0$ 和 $\hat{\boldsymbol{x}}_0$ 之间的顺滑渐变。

!image-3.png