from __future__ import (print_function, division)
import time
# import time library
import numpy as np
# import numpy library as np
import matplotlib.pyplot as plt
# import matplotlib.pyplot library as plt

experiment = '2d'
# switch between 1d and 2d
plot_interval = 20
# plot every 20 time steps

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

boundary_condition = 'periodic'
# switch between 'periodic' and 'walls'

if experiment == '1d':
    boundary_condition = 'walls'
# if 1d, then use walls

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


dx = Lx / nx
# grid spacing in x [m]
dy = Ly / ny
# grid spacing in y [m]

# positions of the value points in
# the x and y directions
ux = (-Lx/2 + np.arange(nx+1)*dx)[:, np.newaxis]
# Add a new axis to ux
uy = (-Ly/2 + np.arange(ny+1)*dy)[np.newaxis, :]
# Add a new axis to uy

vx = (-Lx/2 + dx/2.0 + np.arange(nx)*dx)[:, np.newaxis]
# Add a new axis to vx
vy = (-Ly/2 + dy/2.0 + np.arange(ny)*dy)[np.newaxis, :]
# Add a new axis to vy

hx = vx 
hy = vy

# Initial conditions
t = 0.0
# [s] Time since start of simulation
tc = 0
# [1] Number of integration steps taken


## Grid Function
# These functions are used to set grids and compute the grid parameters

def update_boundaries():
    # Update the boundary values of u, v and h
    # according to the boundary conditions

    # 1. Periodic Boundary Conditions
    # - Flow cycles form left to right
    # - u[0] = u[nx]
    if boundary_condition == 'periodic':
        _u[0, :] = _u[-3, :]
        # u[0] = u[nx]
        _u[1, :] = _u[-2, :]
        # u[1] = u[nx+1]
        _u[-1, :] = _u[2, :]
        # u[nx] = u[0]
        _v[0, :] = _v[-2, :]
        # v[0] = v[nx+1]
        _v[-1, :] = _v[1, :]
        # v[nx] = v[1]
        _h[0, :] = _h[-2, :]
        # h[0] = h[nx+1]
        _h[-1, :] = _h[1, :]
        # h[nx] = h[1]

    # 2. Walls Boundary Conditions
    # - No zonal (u) flow through the left and right walls
    # - Zero x-derivative in v and h
    elif boundary_condition == 'walls':
        # No flow through the boundary at x=0
        _u[0, :] = 0
        # u[0] = 0
        _u[-1, :] = 0
        # u[nx] = 0
        _u[1, :] = 0
        # u[1] = 0
        _u[-2, :] = 0
        # u[nx+1] = 0
        
        # Zero x-derivative in v and h
        _v[0, :] = -_v[1, :]
        # v[0] = -v[1]
        _v[-1, :] = -_v[-2, :]
        # v[nx] = -v[nx+1]
        _h[0, :] = _h[1, :]
        # h[0] = h[1]
        _h[-1, :] = _h[-2, :]
        # h[nx] = h[nx+1]

    # Applied for both boundary conditions
    for field in state:
        # Free-slip boundary conditions
        field[:, 0] = field[:, 1]
        # field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]
        # field[:, ny+1] = field[:, ny]

        # Fix corners to be average of adjacent values
        field[0, 0] = 0.5*(field[0, 1] + field[1, 0])
        # field[0, 0] = 0.5*(field[0, 1] + field[1, 0])
        field[-1, 0] = 0.5*(field[-1, 1] + field[-2, 0])
        # field[nx+1, 0] = 0.5*(field[nx+1, 1] + field[nx, 0])
        field[0, -1] = 0.5*(field[0, -2] + field[1, -1])
        # field[0, ny+1] = 0.5*(field[0, ny] + field[1, ny+1])
        field[-1, -1] = 0.5*(field[-1, -2] + field[-2, -1])
        # field[nx+1, ny+1] = 0.5*(field[nx+1, ny] + field[nx, ny+1])

def diffx(psi):
    # Compute the x-derivative of psi on a single grid
    # using central differences
    # using first-order central differences/finite differences
    global dx
    return (psi[1:, :] - psi[:-1, :]) / dx

def diff2x(psi):
    # Compute the x-second derivative of psi on a single grid
    # using central differences
    global dx
    return (psi[2:, :] - 2*psi[1:-1, :] + psi[:-2, :]) / dx**2

def diffy(psi):
    # Compute the y-derivative of psi on a single grid
    # using central differences
    # using second-order central differences/finite differences
    global dy
    return (psi[:, 1:] - psi[:, :-1]) / dy

def diff2y(psi):
    # Compute the y-second derivative of psi on a single grid
    # using central differences
    global dy
    return (psi[:, 2:] - 2*psi[:, 1:-1] + psi[:, :-2]) / dy**2

def centre_average(phi):
    # Compute the average of phi at the centre of the grid
    # using four-point averaging 
    return 0.25*(phi[1:, 1:] + phi[1:, :-1] + phi[:-1, 1:] + phi[:-1, :-1])

def x_average(phi):
    # Compute the average of phi in the x-direction
    # If phi has shape (nx, ny), the result will have shape (nx-1, ny)
    return 0.5*(phi[1:, :] + phi[:-1, :])

def y_average(phi):
    # Compute the average of phi in the y-direction
    # If phi has shape (nx, ny), the result will have shape (nx, ny-1)
    return 0.5*(phi[:, 1:] + phi[:, :-1])

def divergence(u, v):
    # Compute the divergence of the vector field (u, v)
    # Return the horizontal divergence at h points
    return diffx(u) + diffy(v)

def del2(phi):
    # Compute the Laplacian of phi
    # Return the Laplacian at h points
    return diff2x(phi)[:,1:-1] + diff2y(phi)[1:-1,:]

def uvatuv():
    # Calculate the value of u at v and v at u
    global _u,_v
    ubar = centre_average(_u)[1:-1, :]
    # ubar = centre_average(_u)[1:-1, :]
    vbar = centre_average(_v)[:, 1:-1]
    # vbar = centre_average(_v)[:, 1:-1]
    return ubar, vbar

def uvath():
    # Calculate the value of u at h and v at h
    global u, v
    ubar = x_average(u)
    # ubar = x_average(u)
    vbar = y_average(v)
    # vbar = y_average(v)
    return ubar, vbar

def absmax(psi):
    # Compute the maximum absolute value of psi
    return np.max(np.abs(psi))



## Dynamics
# These functions compute the dynamics of the system

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

# Set the sponges
# Which are active at the top and bottom of the domain by apply Rayleigh friction
sponge_ny = ny//7
# sponge layer thickness
sponge = np.exp(-np.linspace(0, 5, sponge_ny))
def damping(var):
    # sponges are active at the top and bottom of the domain by applying Rayleigh friction
    # with exponential decay towards the centre of the domain
    global sponge, sponge_ny
    var_sponge = np.zeros_like(var)
    var_sponge[:, :sponge_ny] = sponge[np.newaxis, :]
    var_sponge[:, -sponge_ny:] = sponge[np.newaxis, ::-1]
    return var_sponge*var

import numpy as np

def rhs():
    """
    Calculate the right hand side of the u, v and h equations.

    Returns:
    np.array: An array containing the right hand side of the u, v and h equations.
    """
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

## PLOTTING
# Create several functions for displaying current state of the simulation
# Only one is used at a time - this is assigned to `plot`
plt.ion()                         # allow realtime updates to plots
fig = plt.figure(figsize=(8*Lx/Ly, 8))  # create a figure with correct aspect ratio

# create a set of color levels with a slightly larger neutral zone about 0
nc = 12
colorlevels = np.concatenate([np.linspace(-1, -.05, nc), np.linspace(.05, 1, nc)])

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
        plt.savefig("D:/figures/plot_geo_adj/temp{}.png".format(i))

plot = plot_geo_adj
if experiment == '1d':
    plot = plot_geo_adj


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