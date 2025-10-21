
可以从动力学角度看复杂性阶梯：每一级别对应不同级别的非线性动力系统。动力系统也和大模型的进化有密切联系。

# 0 何谓非线性动力学

所谓动力学（动力系统）指的是下面的迭代方程，其中 $X_{t}$ 、$X_{t+1}$ 是系统的状态，它可以使标量、向量、矩阵、甚至更复杂的对象，例如分布函数。$f$ 是一个映射，它将系统在 $t$ 时刻的状态
$$
X_{t+1} = f(X_{t})
$$
可以看到这个系统的演化是 Markov 的，即下一状态只取决于当前状态。所谓 **非线性**，是指迭代映射 $f$ 是非线性的。这个迭代方程包罗万象，宇宙是一个动态系统——$X$ 是宇宙的状态，$f$ 是所有已知或未知的物理法则。

所谓线性是指一个函数 $f$，它在输入线性组合处的取值相当于对输入分别取值后的线性组合，即 $f(ax + by) = af(x) + bf(y)$。直观来看，简单的线性函数的图像是一条直线或一个平面。而非线性函数不满足上面的性质，它的图像可以是一条曲线。下一节将会给出丰富的例子。

<图 1：线性和非线性函数>

## 0.1 非线性动力学的例子
### 0.1.1 三体运动

三体运动是著名科幻小说《三体》的主要背景，它由 Henri Poincare（庞加莱）提出。最早的时候人们研究人类所居住的太阳系（九大行星和太阳，当然现在是八大）是否稳定，结果发现这个问题太过复杂。将问题逐渐简化后变成的三体问题，却成为了一个很大的障碍：三体系统的轨迹异常混乱。

三体问题说的是宇宙空间中的三个天体，受到相互之间引力而各自运动。如果用 $x_{i}$ 表示第 $i$ 个天体的位置，$f_{j,i}$ 表示天体 $j$ 对天体 $i$ 的引力，并假设三个天体质量相同，我们由牛顿第三定律和万有引力定律得到下面的方程：
$$
m \displaystyle \underbrace{ \frac{ \mathrm{d}^{2}x_{i} }{ \mathrm{d}t^{2} } }_{ 加速度 } = \sum\limits_{j=1}^{3} f_{j,i} = \sum\limits_{j=1}^{3} \frac{Gm^{2}(x_{j} - x_{i})}{|x_{j} - x_{i}|^{3}}
$$
![[Pasted image 20251021101041.png]]

### 0.1.2 Boids 模型

现在简单的非线性系统可在计算机中得到较好的模拟，且与日常生活关系紧密。Boids 模型最初由计算机图形学家 Boids 提出，多体系统如果遵循三条简单的规则：分离、对齐、聚集，就可以逼真地模拟动物群体的行为。后续提出的简化版本 Vicsek 模型将其写成了微分方程的形式。

![[Pasted image 20251021101335.png]]


### 0.1.3 生命游戏

一个不是以微分方程形式出现的系统是在上世纪的美国风靡一时的生命游戏，它的提出者是 Conway（康威）。它描绘了一个网格世界，每个网格代表一个细胞，后者有存活和死亡两种状态。该细胞在下一时刻的状态取决于它周围八个邻居的状态。具体而言，演化的规则如下：

1. [孤独] 若存活细胞旁边的存活细胞少于两个，则死亡
2. [稳定] 若存活细胞旁边的存活细胞数量为 2 或 3，则继续存活
3. [拥挤] 若存活细胞旁边的存活细胞多于三个，则死亡
4. [繁殖] 若已死亡细胞旁边恰有三个活细胞，则复活

![[Pasted image 20251021101445.png]]

如果假设某个细胞的状态为 $c$，它的八个邻居的存活数为 $n_{c}$，那么它的下一个状态 $f(c)$ 可以写为
$$
f(c, n) = \begin{cases}
1, & c = 0, n_{c} = 3\\
1, & c = 1, n_{c} \in \{ 2, 3 \}\\
0, & \text{otherwise}
\end{cases}
$$
![[Pasted image 20251021101607.png]]
现在考虑整个网格世界，所有细胞的状态可以堆叠成一个向量 $\boldsymbol{c}$，对应于每个分量（细胞）的当前状态存活邻居个数是 $\boldsymbol{c}$ 的函数 $\boldsymbol{n}_{\boldsymbol{c}}$。所以生命游戏的演化规律也可以写成
$$
\boldsymbol{c}_{t+1} = f(\boldsymbol{c}_{t})
$$

### 0.1.4 Navier-Stokes 方程

物理学中也有许多非线性动力系统。流体力学中的经典模型 Navier-Stokes 方程是一个偏微分方程，它描述了流体的行为。
$$
\begin{align}
\nabla \cdot \boldsymbol{V} &= 0\\
 \underbrace{\rho  \frac{D \boldsymbol{V}}{D t} }_{ 全微分 } &{\color{gray} \;= \rho\left[ \displaystyle \frac{ \partial \boldsymbol{V} }{ \partial t } + (\boldsymbol{V} \cdot \nabla)\boldsymbol{V} \right]  } = \underbrace{ -\nabla p }_{ 压强梯度 } + \rho \boldsymbol{g} + \underbrace{ \mu \nabla^{2} \boldsymbol{V} }_{ 扩散 }
\end{align}
$$
我们依然可以将它与最初的模型 $X_{t+1} = f(X_{t})$ 建立联系：将流体存在的空间切成网格，将每个网格视为最小单元，做受力分析能得到它下一个时刻的状态仅和自身以及邻居单元有关。我们将所有网格的状态集合成一个总状态 $X$，就能得到类似 $X_{t+1} = f(X_{t})$ 的方程了。此时我们将网格加细，整个系统就会趋于一个无穷维的非线性动力系统。

### 0.1.5 Brown 运动

Brown 运动最初被植物学家 Brown（布朗）提出，用于描述花粉在水面的独立不规则运动，后被 Einstein（爱因斯坦）、Wiener（维纳）等人进一步研究并严格化。它是一个随机过程，一般用 $W(\cdot)$ 或 $B(\cdot)$ 表示。它满足下面四条性质：

1. $W(0) = 0 \quad \text{a.s.}$
2. （增量独立性）对任意 $t_{1} \leqslant t_{2} \leqslant t_{3} \leqslant t_{4}$，有 $W(t_{4}) - W(t_{3}) \perp W(t_{2}) - W(t_{1})$
3. （Gauss 分布）对任意 $s \leqslant t$，有 $W(t) - W(s) \sim \mathcal{N}(0, \sigma^{2}(t - s))$

Brown 运动作为一个基本组件广泛出现在随机微分方程（stochastic differential equations, SDE）中，后者的一个典型的形式是
$$
\frac{\mathrm{d}X}{\mathrm{d}t} = f(X, t) + \mathrm{d}W
$$
其中 $f(X, t)$ 称为漂移 (drift) 项，后者称为扩散 (diffusion) 项。如果不仅考虑遵守上述 SDE 的单个“粒子”，而是考虑无穷多的集中在原点处的粒子，它们的分布随时间的变化可以由下面的常微分方程刻画
$$
\displaystyle \frac{ \partial ^{2}p(\boldsymbol{x}, t) }{ \partial t^{2} } = -\nabla \cdot [p(x, t) \cdot v] + \nabla^{2}[p(x, t) \cdot D]
$$
上式中的二阶偏导可以以一些方法转换成一阶，再将其离散化后也变成 $X_{t+1} = f(X_{t})$ 的经典模式。


### 0.1.6 Schrödinger  方程

$$
\begin{align}
i \hbar \displaystyle \frac{ \partial   }{ \partial t }  \psi(x, t) &= - \frac{\hbar^{2}}{2m} \displaystyle \frac{ \partial   }{ \partial x^{2} } \psi(x, t) + V(x) \psi(x, t)\\
\psi(x, t) &= [A\sin(kx) + B\cos(kx)]e^{-\mathrm{i}wt}
\end{align}
$$

$$
\begin{align}
\mathrm{i} \hbar \displaystyle \frac{ \mathrm{d}  }{ \mathrm{d}t } \rho &= [H, \rho]\\
P(x_{i}) &= \text{tr}(\Pi_{i} \rho)
\end{align}
$$

### 0.1.7 游戏系统

塞尔达传说、我的世界、吃豆人



## 0.2 不为非线性动力学的例子
### 0.2.1 Ising 模型



### 0.2.2 场论



# 1 有序和无序


<诺奖、迟滞效应>

![[Pasted image 20251020220634.png]]

## 1.1 Logistic 映射

$$
X_{t+1} = rX_{t}(1-X_{t})
$$

## 1.2 元胞自动机


## 1.3 混沌边缘


# 2 多重尺度

## 2.1 涌现的因果力


## 2.2 因果等价原理


## 2.3 向下因果


# 3 自指


## 3.1 泛函动力学


## 3.2 复杂性阶梯



# 4 关于非线性动力学的讨论



