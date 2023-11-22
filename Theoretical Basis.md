## 1. 理论基础

### 1.1 基本方程组

#### 1.1.1 动量守恒

$$ 
\frac{\mathrm{D} \vec{u}}{\mathrm{D} t}+\vec{f}\times \vec{u}=-g\nabla \eta $$
>$$\frac{\partial u}{\partial t}=fv-g \frac{\partial h}{\partial x}\cdots Equation 1$$
>
>$$\frac{\partial v}{\partial t}=-fu-g \frac{\partial h}{\partial y}\cdots Equation 2$$


#### 1.1.2 质量守恒

$$
\frac{\mathrm{D} h}{\mathrm{D} t}+h \nabla \cdot \vec{u}=0$$
>$$\frac{\partial h}{\partial t}=-H(\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y})\cdots Equation 3$$

### 1.2 代码实现

#### 1.2.1 空间偏导

我们使用一阶中心差分逼近来计算空间偏导。

一阶中心差分常基于泰勒展开，相较于基础的向前、向后差分来说，其使用函数在该点两侧的值来近似导数，精度更高

对函数 `f(x)`，计算点 x 的导数。一阶中心差分方法可以表示为：

$$f'(x) ≈ \frac{f(x+h) - f(x-h)}{(2\cdot h)}$$

其中，h 是一个小的正数，表示 x 的增量。这个公式的右侧就是一阶中心差分的定义。

在实际的数值计算中，h 的选择很重要。如果 h 太大，那么近似的误差可能会很大。如果 h 太小，那么由于计算机的有限精度，可能会出现数值不稳定的问题。通常，h 的选择需要根据具体的问题和计算精度要求来确定。

#### 1.2.2 时间步长

使用三步 `Adams-Bashforth` 方法处理时间步长。

三步 `Adams-Bashforth` 方法是一种常用的显式时间步进方法，用于求解常微分方程或偏微分方程。使用前三个时间步的信息来预测下一个时间步的解

首先需要初始化前三个时间步的解。通常通过使用其他方法（如前向欧拉方法或龙格-库塔方法）完成；

然后，对于每个新的时间步，使用以下公式来预测新的解：

$$y_{n+3}=y_{n+2}+\frac{h}{12}(23f_{n+2}-16f_{n+1}+5f_n)$$

> $$u_{new} = u_{old} + \frac{dt}{12} \cdot (23\cdot f(t_{old}, u_{old}) - 16\cdot f(t_{old}-dt, u_{old}-dt) + 5\cdot f(t_{old}-2\cdot dt, u_{old}-2\cdot dt))$$

其中，u_new 是新的解，u_old 是旧的解，dt 是时间步长，f(t, u) 是你要解的方程。
