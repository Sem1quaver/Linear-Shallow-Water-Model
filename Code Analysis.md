## 2. 代码分析

### 2.1 基础部分

```Python
import time
# import time library
import numpy as np
# import numpy library as np
import matplotlib.pyplot as plt
# import matplotlib.pyplot library as plt
```

引用所需的头文件

```Python
experiment = '2d'
# switch between 1d and 2d
plot_interval = 20
# plot every 20 time steps
```

可以改变代码计算对象为1d或2d模型，

```Python
# set up the figure and axes
# Domain size
nx = 128
ny = 129
# Number of grid points in x and y

Lx = 2.0e7
# Zonal length
Ly = 1.0e7
# Meridional length
H  = 100.0
# Average depth
```

定义了二维流体模型的参数和网格设置。

`nx` 和 `ny` 分别表示在 x 方向和 y 方向上的网格点数量。其中 x 方向有128个网格点，y 方向有 129 个网格点；

`H` 为流体的平均深度，单位是米

```Python
boundary_condition = 'periodic'
# switch between 'periodic' and 'walls'

if experiment == '1d':
    boundary_condition = 'walls'
# if 1d, then use walls
```

边界条件分为 周期性边界条件 `periodic` 与墙壁边界条件 `walls` ，如果使用 1d 模拟则为墙壁边界

```Python
#Coriolis and Gravity
f0 = 1.0e-5 *1.
# [s^-1] f = f0 + beta y _from beta plane approximation
beta = 0
# [m^-1 s^-1] f = f0 + beta y _from beta plane approximation
g = 1.0
# [m s^-2] _acceleration due to gravity

# Diffusion and Friction
nu = 5.0e4
# [m^2 s^-1] viscosity
r = 1.0e-4
# [s^-1] bottom drag

# Time-stepping
dt = 1000.0
```

设置模型的物理常数，`g` 为重力加速度，单位是米每秒平方；`nu` 为扩散系数，单位是平方米每秒；`r` 为顶部和底部的瑞利阻尼。

本模型采用了 $\beta$-平面近似 计算科里奥利力，在这种近似中，科里奥利参数被表示为 $f = f_0 + \beta y$ ，模拟小范围或中等范围的流动时，科里奥利参数的空间变化通常可以被忽略，如果考虑全球范围的潮汐，则需要更精细的模拟

`f0` 是基础科里奥利参数，`beta` 是科里奥利参数随纬度变化的系数。


```Python
# Grid size
# Setup the Arakawa-C Grid:
# +-- v --+
# |       |    * (nx, ny)   h points at grid centres
# u   h   u    * (nx+1, ny) u points on vertical edges  (u[0] and u[nx] are boundary values)
# |       |    * (nx, ny+1) v points on horizontal edges
# +-- v --+
#
# Variables preceeded with underscore  (_u, _v, _h) include the boundary values,
# variables without (u, v, h) are a view onto only the values defined
# within the domain
_u = np.zeros((nx+3, ny+2))
# u points on vertical edges
_v = np.zeros((nx+2, ny+3))
# v points on horizontal edges
_h = np.zeros((nx+2, ny+2))
# h points at grid centres

u = _u[1:-1, 1:-1]
# (nx+1, ny)
v = _v[1:-1, 1:-1]
# (nx, ny+1)
h = _h[1:-1, 1:-1]
# (nx, ny)

# Combine unequal large two-dimensional arrays into three-dimensional matrices and fill the spaces with zeros
state = np.array([u, v, h])
```

设置 `Arakawa-C` 网格，

`_u`， `_v`，` _h` 是包含边界值的速度和高度数组；

`u`， `v`， `h`是不包含边界值的速度和高度数组，它们是`_u`， `_v`，`_h`的视图，只包含在模型域内的值；

`state`是一个包含`u`, `v`, `h`的数组，用于存储模型的状态

`_u`的大小为 (nx+3, ny+2)，表示在垂直边缘上的速度；`_v`的大小为 (nx+2, ny+3)，表示在水平边缘上的速度；`_h`的大小为 (nx+2, ny+2)，表示在网格中心的高度

在 `Arakawa-C` 网格中，u点位于垂直边缘，v点位于水平边缘，h点位于网格的中心。因此，u点的数量在x方向上比h点多1，v点的数量在y方向上比h点多1，

且为了便于处理边界条件，每个物理量在边界上添加了一个额外的网格点，因此`_u` 的大小为 (nx+3, ny+2)，`_v`的大小为 (nx+2, ny+3)，`_h` 的大小为 (nx+2, ny+2)

```Python
dx = Lx / nx            # [m]
dy = Ly / ny            # [m]
```

计算 x 方向和 y 方向上的网格间距 `dx` 、`dy`

```Python
# positions of the value points in [m]
ux = (-Lx/2 + np.arange(nx+1)*dx)[:, np.newaxis]
vx = (-Lx/2 + dx/2.0 + np.arange(nx)*dx)[:, np.newaxis]

vy = (-Ly/2 + np.arange(ny+1)*dy)[np.newaxis, :]
uy = (-Ly/2 + dy/2.0 + np.arange(ny)*dy)[np.newaxis, :]

hx = vx
hy = uy
```

`np.arange(nx+1)` 生成一个从 0 到 nx 的整数数组；

`np.arange(nx+1)*dx` 将上述整数数组中的每个元素乘以 dx，得到从 0 到 dx$\cdot$nx 的数组，每个元素之间的间隔为 dx；

`-Lx/2 + np.arange(nx+1)*dx` 将上述数组中的每个元素减去 Lx/2，结果是一个从 -Lx/2到 -Lx/2 + dx$\cdot$nx 的数组，每个元素之间的间隔为 dx，即表示 u 点在 x 方向上的位置；

`(-Lx/2 + np.arange(nx+1)*dx)[:, np.newaxis]` 将上述一维数组新增加一行转换为二维数组。新数组的形状为 (nx+1, 1)

### 2.2 函数部分

#### 2.2.1 GRID FUNCTIONS

用于设置和计算网格参数的函数

- `update_boundaries()`

```Python
def update_boundaries():

    # 1. Periodic Boundaries
    #    - Flow cycles from left-right-left
    #    - u[0] == u[nx]
    if boundary_condition == 'periodic':
        _u[0, :] = _u[-3, :]
        _u[1, :] = _u[-2, :]
        _u[-1, :] = _u[2, :]
        _v[0, :] = _v[-2, :]
        _v[-1, :] = _v[1, :]
        _h[0, :] = _h[-2, :]
        _h[-1, :] = _h[1, :]


    # 2. Solid walls left and right
    #    - No zonal (u) flow through the left and right walls
    #    - Zero x-derivative in v and h
    if boundary_condition == 'walls':
        # No flow through the boundary at x=0
        _u[0, :] = 0
        _u[1, :] = 0
        _u[-1, :] = 0
        _u[-2, :] = 0

        # free-slip of other variables: zero-derivative
        _v[0, :] = _v[1, :]
        _v[-1, :] = _v[-2, :]
        _h[0, :] = _h[1, :]
        _h[-1, :] = _h[-2, :]

    # This applied for both boundary cases above
    for field in state:
        # Free-slip of all variables at the top and bottom
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]

    	# fix corners to be average of neighbours
        field[0, 0] =  0.5*(field[1, 0] + field[0, 1])
        field[-1, 0] = 0.5*(field[-2, 0] + field[-1, 1])
        field[0, -1] = 0.5*(field[1, -1] + field[0, -2])
        field[-1, -1] = 0.5*(field[-1, -2] + field[-2, -1])
```

这段代码定义了 `update_boundaries()` 函数，用于处理、更新模拟区域的边界条件，

在周期性边界条件 `periodic` 中，流体从左边流出，然后从右边流入，形成一个循环。函数通过将左边界的值设置为右边界的值来实现模拟区域的左边界和右边界连通，如同环形区域

`_u[-3, :]` 表示的是 `_u` 数组的倒数第三行的所有元素，即在x方向上的倒数第三个位置的u速度，

周期性边界条件时，左边界的值被设置为右边界的值。`_u[0, :]` = `_u[-3, :]` 将左边界的值设置为右边界的倒数第三个位置的值，

同理，`_u[1, :]` = `_u[-2, :]`将左边界的下一个位置的值设置为右边界的倒数第二个位置的值

在固体墙壁边界条件 `walls` 中，左右两侧没有流动（ u 速度为 0），并且 v 速度和 h 高度在 x 方向上的导数为0（即，v和 h 在左右两侧是常数）。函数通过将 u 速度的边界值设置为0，将 v 速度和 h 高度的边界值设置为相邻内部点的值来实现

无论哪种边界条件，`update_boundaries()` 函数都会对所有物理量在上下边界应用自由滑移条件，即，物理量在y方向上的导数为0。通过将上下边界的值设置为相邻内部点的值来实现。

`update_boundaries()` 函数还修复了四个角点的值，使其等于相邻两个点的平均值，避免在角点出现数值不稳定。

- `diffx(psi)`

```Python
def diffx(psi):
    # Compute the x-derivative of psi on a single grid
    # using central differences
    # using first-order central differences/finite differences
    global dx
    return (psi[1:, :] - psi[:-1, :]) / dx
```

- `diffy(psi)`

```Python
def diffy(psi):
    # Compute the y-derivative of psi on a single grid
    # using central differences
    # using second-order central differences/finite differences
    global dy
    return (psi[:, 1:] - psi[:, :-1]) / dy
```

`diffx(psi)` 函数使用了一阶中心差分方法近似计算了输入数组 `psi` 在x方向上的导数

$$\frac{\mathrm{d}}{\mathrm{d} x}(psi[i,j]) = (psi[i+1/2, j] - psi[i-1/2, j]) / dx$$

`psi[1:,:]` 表示 `psi`数组从第二行开始到最后一行的所有元素，`psi[:-1,:]` 表示 `psi` 数组从第一行开始到倒数第二行的所有元素。这两个数组的差就是 `psi` 在x方向上的差分，然后除以 dx 就得到了 `psi` 在x方向上的导数

这个函数返回的新数组的形状比输入数组 `psi` 的形状在 x 方向上小 1，因为差分操作会减少一个元素。这个新数组的每个元素都是 `psi` 在对应位置上的 x 方向的导数。

-  `diff2x(psi)`

```Python
def diff2x(psi):
    # Compute the x-second derivative of psi on a single grid
    # using central differences
    global dx
    return (psi[2:, :] - 2*psi[1:-1, :] + psi[:-2, :]) / dx**2
```

- `diff2y(psi)`

```Python
def diff2y(psi):
    # Compute the y-second derivative of psi on a single grid
    # using central differences
    global dy
    return (psi[:, 2:] - 2*psi[:, 1:-1] + psi[:, :-2]) / dy**2
```

`psi[:-2, :]` 表示 `psi` 数组从第一行开始到倒数第三行的所有元素，`psi[1:-1, :]` 表示 `psi` 数组从第二行开始到倒数第二行的所有元素，`psi[2:, :]` 表示 `psi` 数组从第三行开始到最后一行的所有元素。这三个数组的线性组合就是 `psi` 在x方向上的二阶差分，然后除以 $dx^2$ 就得到了 `psi` 在x方向上的二阶导数

这个函数返回的新数组比输入数组 `psi` 的形状在x方向上小 2，因为二阶差分操作会减少两个元素。这个新数组的每个元素都是 `psi` 在对应位置上的x方向的二阶导数

- `centre_average(phi)`


```Python
def centre_average(phi):
    # Compute the average of phi at the centre of the grid
    # using four-point averaging 
    return 0.25*(phi[1:, 1:] + phi[1:, :-1] + phi[:-1, 1:] + phi[:-1, :-1])
```

函数 `centre_average(phi)` 计算输入数组 `phi` 在网格点之间的中心的四点平均值

`phi[:-1,:-1]` 表示 `phi` 数组从第一行开始到倒数第二行，从第一列开始到倒数第二列的所有元素，`phi[:-1,1:]` 表示 `phi` 数组从第一行开始到倒数第二行，从第二列开始到最后一列的所有元素，`phi[1:, :-1]` 表示 `phi` 数组从第二行开始到最后一行，从第一列开始到倒数第二列的所有元素，`phi[1:,1:]` 表示 `phi` 数组从第二行开始到最后一行，从第二列开始到最后一列的所有元素。这四个数组的和就是 `phi` 在网格点之间的中心的四点之和，然后乘以 0.25 就得到了四点平均值

这个函数返回的新数组比输入数组 `phi` 的形状在每个方向上都小 1，因为四点平均值操作会减少一个元素。这个新数组的每个元素都是 `phi` 在对应位置的四点平均值

- `x_average(phi)`

```Python
def x_average(phi):
    # Compute the average of phi in the x-direction
    # If phi has shape (nx, ny), the result will have shape (nx-1, ny)
    return 0.5*(phi[1:, :] + phi[:-1, :])
```

- `y_average(phi)`

```Python
def y_average(phi):
    # Compute the average of phi in the y-direction
    # If phi has shape (nx, ny), the result will have shape (nx, ny-1)
    return 0.5*(phi[:, 1:] + phi[:, :-1])
```

定义函数 `x_average` 和 `y_average` ，分别用于计算输入数组 `phi` 在 x 方向和 y 方向上的平均值

`phi[1:, :]` 表示 `phi` 数组从第二行开始到最后一行的所有元素，`phi[:-1, :]` 表示 `phi` 数组从第一行开始到倒数第二行的所有元素。这两个数组的和就是 `phi` 在 x 方向上的两点之和，然后乘以 0.5 就得到了两点平均值

这个函数返回的新数组比输入数组 `phi` 的形状在对应方向上小小 1，因为两点平均值操作会减少一个元素。这个新数组的每个元素都是 `phi` 在对应位置的两点平均值

- `divergence(u, v)`

```Python
def divergence(u, v):
    # Compute the divergence of the vector field (u, v)
    # Return the horizontal divergence at h points
    return diffx(u) + diffy(v)
```

函数 `divergence(u, v)` 用于向量场的散度计算，其接受 `u`、`v` 两个参量，分别表示向量场在 x 方向和 y 方向上的分量

`diffx(u)`计算 u 在 x 方向上的导数，`diffy(v)` 计算 v 在 y 方向上的导数。这两个导数的和就是 (u, v) 的散度

这个函数返回一个新的数组，其每个元素都是 (u, v) 在对应位置的散度

- `del2(phi)`

```Python
def del2(phi):
    # Compute the Laplacian of phi
    # Return the Laplacian at h points
    return diff2x(phi)[:,1:-1] + diff2y(phi)[1:-1,:]
```

函数 `del2(phi)` 用于计算向量场 `phi` 的拉普拉斯算子，即：梯度的散度

`diff2x(phi)[:, 1:-1]` 计算 `phi` 在 x 方向上的二阶导数,并选取从第二列开始到倒数第二列的所有元素；

`diff2y(phi)[1:-1, :]` 计算 `phi` 在 y 方向上的二阶导数,并选取从第二行开始到倒数第二行的所有元素。

这两个二阶导数的和就是 `phi` 的拉普拉斯算子。

这个函数返回一个新的数组，其形状比输入数组 `phi` 的形状在每个方向上都小 2，因为二阶导数操作会减少两个元素。这个新数组的每个元素都是 `phi` 在对应位置的拉普拉斯算子。

- `uvatuv()`

```Python
def uvatuv():
    # Calculate the value of u at v and v at u
    global _u,_v
    ubar = centre_average(_u)[1:-1, :]
    # ubar = centre_average(_u)[1:-1, :]
    vbar = centre_average(_v)[:, 1:-1]
    # vbar = centre_average(_v)[:, 1:-1]
    return ubar, vbar
```

函数 `uvatuv()` 用于计算速度场 `_u` 和 `_v` 在网格点之间的中心的平均值

此函数无需输入参数，而是使用全局变量 `global _u, _v` ,声明 `_u` 和 `_v` 为全局变量，即此函数将使用在函数外部定义的 `_u` 和 `_v` 的值

首先调用 `centre_average(phi)` 函数计算 `_u` 在网格点之间的中心的平均值，再选取从第二行开始到倒数第二行的所有元素，最后将得到的结果赋值给 `ubar`

这个函数返回两个新的数组，其的形状比输入数组 `_u` 和 `_v` 的形状在对应方向上都小 1，因为中心平均值操作会减少一个元素。这两个新数组的每个元素都是 `_u` 和 `_v` 在对应位置的中心平均值。

- `uvath()`

```Python
def uvath():
    # Calculate the value of u at h and v at h
    global u, v
    ubar = x_average(u)
    # ubar = x_average(u)
    vbar = y_average(v)
    # vbar = y_average(v)
    return ubar, vbar
```

函数 `uvath()` 用于计算速度场 `u` 和 `v` 在 x 方向和 y 方向上的平均值

此函数无需输入参数，而是使用全局变量 `global u, v` ,声明 `u` 和 `v` 为全局变量，即此函数将使用在函数外部定义的 `u` 和 `v` 的值

调用 `x_average(phi)` 函数计算 `u` 在 x 方向上的平均值，得到的结果赋值给 `ubar`

> `uvatuv()` 与 `uvath()` 的对比：
> - 处理的变量对象不同：`uvatuv()` 函数处理全局变量速度场 `_u` 和 `_v` ；`uvath()` 函数处理全局变量速度场 `u` 和 `v`
> - 使用的计算函数不同：`uvatuv()` 函数使用了 `centre_average(phi)` 函数，用网格点四角计算中心值；而 `uvath()` 函数使用 `x_average(phi)` 函数，计算目标值沿特定方向的均值
> - 返回数组形状相同，这两个函数都返回两个新的数组，其形状比输入数组的形状在对应方向上都小1，因为平均值操作会减少一个元素

- `absmax(psi)`

```Python
def absmax(psi):
    # Compute the maximum absolute value of psi
    return np.max(np.abs(psi))
```

函数 `absmax(psi)` 用于计算输入数组 `psi` 的绝对值的最大值

函数首先调用 `np.abs` 函数计算 `psi` 的绝对值，然后调用 `np.max` 函数找出绝对值的最大值

这个函数返回一个标量，可用于找出数组中的最大偏差，或者确定数组的范围

#### 2.2.2 DYNAMICS

动力学角度计算、更新浅水系统状态的函数

- `forcing()`

```Python
def forcing():
    # Add some external forcing terms to the u, v and h equations.
    # This is where we can add wind stress, heat fluxes, etc.
    # This function should return a state array (du, dv, dh)
    # Which will be added to the RHS of the equations(1)(2)(3)
    global u, v, h
    du = np.zeros_like(u)
    dv = np.zeros_like(v)
    dh = np.zeros_like(h)
    # Set empty arrays for du, dv, dh, waiting to be filled
    # Calculate some forcing terms here
    return np.array([du, dv, dh])
```

函数 `forcing()` 用于向浅水系统中加入外力

函数首先使用 `du = np.zeros_like(u)` 创建一个与 `u` 具有相同形状和类型的新数组，并将其所有元素初始化为 0。这个新数组将用于存储 `u` 的强迫项

`return np.array([du, dv, dh])` 返回一个新的数组，它包含了 `u` , `v` 和 `h` 的强迫项

- `damping(var)`

```Python
sponge_ny = ny//7
sponge = np.exp(-np.linspace(0, 5, sponge_ny))
```

这段代码首先设置了数组 `sponge` ,用于模拟 sponge layer（海绵层）概念，并计算海绵层的衰减系数，再使用函数 `damping(var)` 模拟 Rayleigh 摩擦

`sponge_ny = ny//7` 使用整数除法，向下取整计算了变量 `ny` ，即网格的 y 方向大小的七分之一，并将结果存储在 `sponge_ny` 中，用于模拟海绵层的厚度

`sponge = np.exp(-np.linspace(0, 5, sponge_ny))`  首先使用 `np.linspace` 函数生成一个从 0 开始，到 5 结束的等差数列，其包含 `sponge_ny` 个元素。然后对这个数列的每个元素取负数，最后计算其指数，得到新的数组。这个新的数组被赋值给 `sponge`

最终得到的 `sponge` 数组的长度等于 `ny` 的七分之一（向下取整），数组的元素从 exp(0)（即1）开始，到 exp(-5) 结束，元素之间的值呈指数衰减，即为衰减系数

这个数组可以用于模拟一个在 y 方向上的"sponge layer"，在这个区域内，模拟的流体会经历额外的阻尼，以减少边界反射

```Python
def damping(var):
    # sponges are active at the top and bottom of the domain by applying Rayleigh friction
    # with exponential decay towards the centre of the domain
    global sponge, sponge_ny
    var_sponge = np.zeros_like(var)
    var_sponge[:, :sponge_ny] = sponge[np.newaxis, :]
    var_sponge[:, -sponge_ny:] = sponge[np.newaxis, ::-1]
    return var_sponge*var
```

`var_sponge = np.zeros_like(var)` 创建了了一个与待处理数组 `var` 具有相同形状和类型的新数组，并将其所有元素初始化为 0，用于存储阻尼项

`var_sponge[:, :sponge_ny] = sponge[np.newaxis, :]` 将 `sponge` 数组的值赋给 `var_sponge` 数组的前 `sponge_ny` 列

`var_sponge[:, -sponge_ny:] = sponge[np.newaxis, ::-1]` 将 `sponge` 数组的值逆序赋给 `var_sponge` 数组的后 `sponge_ny` 列

`return var_sponge*var` 返回 `var_sponge` 和 `var` 的元素乘积，这就是应用了阻尼后的 `var`

这个函数返回一个新的数组，它是输入数组 `var` 和阻尼项的乘积，可以模拟一个在 y 方向上的"sponge layer"，在此区域内，模拟的流体会经历额外的阻尼，以减少边界反射

- `rhs()`

```Python
def rhs():
    u_at_v, v_at_u = uvatuv()   # (nx, ny+1), (nx+1, ny)

    # the height equation
    h_rhs = -H*divergence() + nu*del2(_h) - r*damping(h)

    # the u equation
    dhdx = diffx(_h)[:, 1:-1]  # (nx+1, ny)
    u_rhs = (f0 + beta*uy)*v_at_u - g*dhdx + nu*del2(_u) - r*damping(u)

    # the v equation
    dhdy  = diffy(_h)[1:-1, :]   # (nx, ny+1)
    v_rhs = -(f0 + beta*vy)*u_at_v - g*dhdy + nu*del2(_v) - r*damping(v)

    return np.array([u_rhs, v_rhs, h_rhs]) + forcing()
```

函数 `rhs()` 用于计算基本方程组

`u_at_v, v_at_u = uvatuv()` 调用 `uvatuv` 函数，并将返回的两个值赋给 `u_at_v` 和 `v_at_u`。这两个值分别代表对 u 沿 x 做微分和对 v 沿 y 做微分，即得：
>$$u\_at\_v=\frac{\partial u}{\partial x}$$  
>$$v\_at\_u=\frac{\partial v}{\partial y}$$

`h_rhs = -H*divergence() + nu*del2(_h) - r*damping(h)` 计算了方程 3 的右侧，即得：
>$$h\_rhs=-H(\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y})+nu\cdot \nabla^2h-r\cdot h' $$

`dhdx = diffx(_h)[:, 1:-1]` 计算了 `_h` 在 x 方向的差分，并将结果的第 1 列到倒数第 2 列赋给 `dhdx` 

`dhdy = diffy(_h)[1:-1, :]` 计算了 `_h` 的 y 方向的差分，并将结果的第 1 行到倒数第 2 行赋给 `dhdy`，即得：

>$$dhdx=\frac{\partial h}{\partial x}$$
>$$dhdy=\frac{\partial h}{\partial y}$$

`u_rhs = (f0 + beta*uy)*v_at_u - g*dhdx + nu*del2(_u) - r*damping(u)` 计算方程 1 的右侧，即得：

>$$u\_rhs=(f_0+\beta \cdot u_y)\frac{\partial u}{\partial x}-g\cdot \frac{\partial h}{\partial x}+nu\cdot \nabla^2u-r\cdot u'$$

`v_rhs = -(f0 + beta*vy)*u_at_v - g*dhdy + nu*del2(_v) - r*damping(v)` 计算方程 2 的右侧，即得：

>$$v\_rhs=-(f_0+\beta \cdot v_y)\frac{\partial v}{\partial y}-g\cdot \frac{\partial h}{\partial y}+nu\cdot \nabla^2v-r\cdot v'$$

`return np.array([u_rhs, v_rhs, h_rhs]) + forcing()` 函数返回一个新的数组，它是 `u` , `v` 和 `h` 方程的右侧加上外部强迫项的结果

- `step()`

```Python
_ppdstate, _pdstate = 0,0
def step():
    global dt, t, tc, _ppdstate, _pdstate

    update_boundaries()

    dstate = rhs()

    # take adams-bashforth step in time
    if tc==0:
        # forward euler
        dt1 = dt
        dt2 = 0.0
        dt3 = 0.0
    elif tc==1:
        # AB2 at step 2
        dt1 = 1.5*dt
        dt2 = -0.5*dt
        dt3 = 0.0
    else:
        # AB3 from step 3 on
        dt1 = 23./12.*dt
        dt2 = -16./12.*dt
        dt3 = 5./12.*dt

    newstate = state + dt1*dstate + dt2*_pdstate + dt3*_ppdstate
    u[:], v[:], h[:] = newstate
    _ppdstate = _pdstate
    _pdstate = dstate

    t  += dt
    tc += 1
```

函数 `step()` 用于执行模型的一步时间积分

`_ppdstate, _pdstate = 0,0`  初始化了两个变量 `_ppdstate` 和 `_pdstate`，它们将用于存储模型的前两个时间步的状态

函数首先调用 `update_boundaries` 函数，更新模型的边界条件；接着调用 `rhs` 函数，计算模型的右侧，并将结果赋给 `dstate`

此函数根据时间步计数器 `tc` 的值，会选择不同的时间积分方案。如果 `tc` 为 0，使用`前向欧拉方法`；如果 `tc` 为 1，使用`二阶亚当斯-巴什福斯方法`；如果 `tc` 大于 1，使用`三阶亚当斯-巴什福斯方法`

>前向欧拉方法、二阶亚当斯-巴什福斯方法与三阶亚当斯-巴什福斯方法
> - 前向欧拉方法：显式的一阶差分方法，只需要当前步的信息。实现简单但精度较低，且对于某些问题可能不稳定。
>>$$ y(n+1) = y(n) + h\cdot f(t(n), y(n))$$
>其中，y(n)是当前步的解，h 是步长，f(t(n), y(n))是微分方程的右侧
> - 二阶亚当斯-巴什福斯方法：这是一种二阶精度的方法，需要当前步和前一步的信息。相比于前向欧拉方法，它的精度更高，但实现起来稍微复杂一些。
>>$$y(n+2) = y(n+1) + h/2 \cdot [3\cdot f(t(n+1), y(n+1)) - f(t(n), y(n))]$$
>其中，y(n+1)和 y(n)分别是当前步和前一步的解，h 是步长，f(t(n+1), y(n+1))和 f(t(n), y(n))分别是当前步和前一步的微分方程的右侧
> - 三阶亚当斯-巴什福斯方法：这是一种三阶精度的方法，需要当前步，前一步和前两步的信息。它的精度比二阶亚当斯-巴什福斯方法更高，但实现起来更复杂。
>>$$y(n+3) = y(n+2) + h/12 \cdot [23\cdot f(t(n+2), y(n+2)) - 16\cdot f(t(n+1), y(n+1)) + 5\cdot f(t(n), y(n))]$$
>其中，y(n+2)，y(n+1)和 y(n)分别是当前步，前一步和前两步的解，h 是步长，f(t(n+2), y(n+2))，f(t(n+1), y(n+1))和 f(t(n), y(n))分别是当前步，前一步和前两步的微分方程的右侧。

`newstate = state + dt1*dstate + dt2*_pdstate + dt3*_ppdstate` 用于计算新的状态

`u[:], v[:], h[:] = newstate` 将新的状态赋给 `u` 、`v` 和 `h`

`_ppdstate = _pdstate和_pdstate = dstate` 用于更新前两个时间步的状态

`t += dt` 和 `tc += 1` 用于更新当前时间和时间步计数器

这个函数的功能是执行模型的一步时间积分。

首先更新模型的边界条件，然后计算三个基本方程的右侧，然后根据时间步计数器的值选择不同的时间积分方案，再计算新的状态，紧接着更新模型的状态，之后更新前两个时间步的状态，最后更新当前时间和时间步计数器。

### 2.3 INITIAL CONDITIONS

设置模型初始条件

```Python
## INITIAL CONDITIONS
# Set the initial state of the model here by assigning to u[:], v[:] and h[:].
if experiment == '2d':
    # create a single disturbance in the domain:
    # a gaussian at position gx, gy, with radius gr
    gx =  2.0e6
    gy =  0.0
    gr =  2.0e5
    h0 = np.exp(-((hx - gx)**2 + (hy - gy)**2)/(2*gr**2))*H*0.01
    u0 = u * 0.0
    v0 = v * 0.0

if experiment == '1d':
    h0 = -np.tanh(100*hx/Lx)
    v0 = v * 0.0
    u0 = u * 0.0
    # no damping in y direction
    r = 0.0

# set the variable fields to the initial conditions
u[:] = u0
v[:] = v0
h[:] = h0
```

`gx = 2.0e6，gy = 0.0，gr = 2.0e5`，定义了一个高斯扰动的位置`(gx, gy)`和半径`gr`

> - 高斯扰动
> 通常指的是遵循正态分布的扰动。在这段代码中，高斯扰动是指在二维空间中，以某一点为中心，按照高斯分布形状产生的扰动，以点 (gx, gy) 为中心，半径为 gr，扰动的大小遵循高斯分布
>>$$h_0 = np.exp(-((hx - gx)^2 + (hy - gy)^2)/(2\cdot gr^2))\cdot H\cdot 0.01$$

`u0 = u * 0.0，v0 = v * 0.0` 设置速度场 `u` 和 `v` 的初始条件为0

### 2.4 PLOTTING

实时绘制模拟的状态

```Python
# Create several functions for displaying current state of the simulation
# Only one is used at a time - this is assigned to `plot`
plt.ion()                         # allow realtime updates to plots
fig = plt.figure(figsize=(8*Lx/Ly, 8))  # create a figure with correct aspect ratio

# create a set of color levels with a slightly larger neutral zone about 0
nc = 12
colorlevels = np.concatenate([np.linspace(-1, -.05, nc), np.linspace(.05, 1, nc)])
```

`plt.ion()` 开启 matplotlib 交互模式实时更新图像

`fig = plt.figure(figsize=(8*Lx/Ly, 8))` 创建新的图像窗口，大小为 (8*Lx/Ly, 8)

`colorlevels = np.concatenate([np.linspace(-1, -.05, nc), np.linspace(.05, 1, nc)])` 创建颜色级别数组，包含从 -1 到 -0.05 和从 0.05 到 1 的均匀分布值

- `plot_all(u,v,h)`

```Python
def plot_all(u,v,h):
    hmax = np.max(np.abs(h))
    plt.clf()
    plt.subplot(222)
    X, Y = np.meshgrid(ux, uy)
    plt.contourf(X/Lx, Y/Ly, u.T, cmap=plt.cm.RdBu, levels=colorlevels*absmax(u))
    #plt.colorbar()
    plt.title('u')

    plt.subplot(224)
    X, Y = np.meshgrid(vx, vy)
    plt.contourf(X/Lx, Y/Ly, v.T, cmap=plt.cm.RdBu, levels=colorlevels*absmax(v))
    #plt.colorbar()
    plt.title('v')

    plt.subplot(221)
    X, Y = np.meshgrid(hx, hy)
    plt.contourf(X/Lx, Y/Ly, h.T, cmap=plt.cm.RdBu, levels=colorlevels*absmax(h))
    #plt.colorbar()
    plt.title('h')

    plt.subplot(223)
    plt.plot(hx/Lx, h[:, ny//2])
    plt.xlim(-0.5, 0.5)
    plt.ylim(-absmax(h), absmax(h))
    plt.title('h along x=0')

    plt.pause(0.001)
    plt.draw()
    plt.savefig("D:/figures/temp{}.png".format(i))
```

函数 `plot_all(u,v,h)` 用于绘制 u，v 和 h 的等高线图，以及 h 在 x=0 处的剖面图

- `plot_fast(u,v,h)`

```Python
im = None
def plot_fast(u,v,h):
    # only plots an imshow of h, much faster than contour maps
    global im
    if im is None:
        im = plt.imshow(h.T, aspect=Ly/Lx, cmap=plt.cm.RdBu, interpolation='bicubic')
        im.set_clim(-absmax(h), absmax(h))
    else:
        im.set_array(h.T)
        im.set_clim(-absmax(h), absmax(h))
    plt.pause(0.001)
    plt.draw()
```

函数 `plot_fast(u,v,h)` 用于快速绘制 h 的等高线图像

- `plot_geo_adj(u, v, h)`

```Python
def plot_geo_adj(u, v, h):
        plt.clf()

        h0max = absmax(h0)
        plt.subplot(311)
        plt.plot(hx, h[:, ny//2], 'b', linewidth=2)
        plt.plot(hx, h0[:], 'r--', linewidth=1,)
        plt.ylabel('height')
        plt.ylim(-h0max*1.2, h0max*1.2)

        plt.subplot(312)
        plt.plot(vx, v[:, ny//2].T, linewidth=2)
        plt.plot(vx, v0[:, ny//2], 'r--', linewidth=1,)
        plt.ylabel('v velocity')
        plt.ylim(-h0max*.12, h0max*.12)

        plt.subplot(313)
        plt.plot(ux, u[:, ny//2], linewidth=2)
        plt.plot(ux, u0[:, ny//2], 'r--', linewidth=1,)
        plt.xlabel('x/L$_\mathsf{d}$',size=16)
        plt.ylabel('u velocity')
        plt.ylim(-h0max*.12, h0max*.12)

        plt.pause(0.001)
        plt.draw()
```

函数 `plot_geo_adj(u, v, h)` 用于绘制水体的水平速度、垂直速度和高度在 y=0 处的剖面图

### 2.5 RUN

```Python
## RUN
# Run the simulation and plot the state
c = time.perf_counter()
nsteps = 1000
for i in range(nsteps):
    step()
    if i % plot_interval == 0:
        plot(*state)
        print('[t={:7.2f} u: [{:.3f}, {:.3f}], v: [{:.3f}, {:.3f}], h: [{:.3f}, {:.2f}]'.format(
            t/86400,
            u.min(), u.max(),
            v.min(), v.max(),
            h.min(), h.max()))
        #print('fps: %r' % (tc / (time.clock()-c)))
```

运行代码，并保存模拟结果
