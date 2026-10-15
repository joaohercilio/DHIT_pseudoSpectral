"""3-D incompressible Taylor-Green pseudo-spectral solver"""

import mpi4py
mpi4py.rc.thread_level = "serialized"
from mpi4py import MPI
import numpy as np
from shenfun import Array, Function, FunctionSpace, TensorProductSpace, VectorSpace, ShenfunFile

N = 256

real     = np.float64   # np.float64
complex_ = np.complex128 # np.complex128

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# periodic fourier space, the last direction is real-to-complex
Vx = FunctionSpace(N, "F", domain=(0, 2*np.pi), dtype=complex_)
Vy = FunctionSpace(N, "F", domain=(0, 2*np.pi), dtype=complex_)
Vz = FunctionSpace(N, "F", domain=(0, 2*np.pi), dtype=real)
V = [Vx, Vy, Vz]

T = TensorProductSpace(comm, V, dtype=real, slab=True, collapse_fourier=True)
VT = VectorSpace(T)

# same-sized transform space that truncates the upper 1/3 of modes
Td = T.get_dealiased(padding_factor=1, dealias_direct=True)
VTd = VectorSpace(Td)

k = T.local_wavenumbers(scaled=True)
k = [ki.astype(real, copy=False) for ki in k] # convert to real number type
k2 = np.zeros(T.shape(True), dtype=real)
for ki in k:
    k2 += ki*ki
inv_k2 = np.zeros_like(k2)
np.divide(1, k2, out=inv_k2, where=k2 != 0)

# allocate 

# physical space 
u         = Array(VT)
u_dealias = Array(VTd)
omega     = Array(VTd)
nonlinear = Array(VTd)

# fourier space
u_hat     = Function(VT)
omega_hat = Function(VT)
rhs_hat   = Function(VT)
residual  = Function(VT)
pressure_hat = Function(T) # scalar

# taylor green 
# x = T.local_mesh(True)
#u[0] =  np.sin(x[0])*np.cos(x[1])*np.cos(x[2])
#u[1] = -np.cos(x[0])*np.sin(x[1])*np.cos(x[2])
#u[2] = 0

import vtk
from vtk.util import numpy_support
reader = vtk.vtkStructuredPointsReader()
reader.SetFileName("initial.vtk")
reader.Update()
data = reader.GetOutput().GetPointData()

sl = T.local_slice(False) # each rank reads the hole vtk

ux = numpy_support.vtk_to_numpy(data.GetArray("ux")).reshape((N, N, N), order="F")
uy = numpy_support.vtk_to_numpy(data.GetArray("uy")).reshape((N, N, N), order="F")
uz = numpy_support.vtk_to_numpy(data.GetArray("uz")).reshape((N, N, N), order="F")

u[0] = ux[sl]
u[1] = uy[sl] # the rank stores only it's own slice
u[2] = uz[sl]

cfl = 0.6
dx = 2*np.pi/N
nu = 0.0001339440322141572

VT.forward(u, u_hat)

def compute_dt():
    VT.backward(u_hat, u)
    umax = comm.allreduce(np.max(np.abs(u)), op=MPI.MAX)
    return cfl*dx/umax

def energy():
    VT.backward(u_hat, u)
    local = np.sum(u*u, dtype=np.float64)
    return 0.5*comm.allreduce(local)/N**3

shell = np.rint(np.sqrt(k2)).astype(int).ravel()
kz = k[2]
w = np.broadcast_to(np.where((kz == 0) | (kz == N//2), 1.0, 2.0), k2.shape).ravel()

def energySpectrum():
    e = 0.5*np.sum(np.abs(u_hat)**2, axis=0).ravel()*w
    return comm.allreduce(np.bincount(shell, weights=e, minlength=N))

def rhs():
    """  P(u x curl(u)) - nu*k^2*u  """

    omega_hat[0] = 1j*(k[1]*u_hat[2] - k[2]*u_hat[1])
    omega_hat[1] = 1j*(k[2]*u_hat[0] - k[0]*u_hat[2])
    omega_hat[2] = 1j*(k[0]*u_hat[1] - k[1]*u_hat[0])

    VTd.backward(u_hat, u_dealias)
    VTd.backward(omega_hat, omega)

    nonlinear[0] = u_dealias[1]*omega[2] - u_dealias[2]*omega[1]
    nonlinear[1] = u_dealias[2]*omega[0] - u_dealias[0]*omega[2]
    nonlinear[2] = u_dealias[0]*omega[1] - u_dealias[1]*omega[0]
    VTd.forward(nonlinear, rhs_hat)

    # helmholtz projection: P(a) = a - k*(k.a)/|k|^2
    pressure_hat[:] = (k[0]*rhs_hat[0] + k[1]*rhs_hat[1] + k[2]*rhs_hat[2])*inv_k2
    
    for i in range(3):
        rhs_hat[i] -= k[i]*pressure_hat
        rhs_hat[i] -= nu*k2*u_hat[i]
    return rhs_hat

A = (0.0, -5/9, -153/128)
B = (1/3, 15/16, 8/15)
t = 0.0
step = 0


# ek = energy()
# f = open("energy.dat", "w") if rank == 0 else None


t0 = 318.457776566172
t_total = 10*t0

kmax = N/3
dt_visc = 1/(nu*kmax**2)
dt = min(compute_dt(), dt_visc)

t_save = np.linspace(0, t_total, 200).tolist()
iout = 0
t_wall = MPI.Wtime()

fl = ShenfunFile("fields", VT, backend="hdf5", mode="w")

while True:

    # ===== save step file
    if t >= t_save[0] - 1e-12:
        VT.backward(u_hat, u)
        fl.write(iout, {"u": [u]}, as_scalar=False)
        if rank == 0:
            with open("times.dat", "a") as g:
                g.write(f"{iout}\t{t:.10f}\n")
        iout += 1
        t_save.pop(0)
        if not t_save:
            break

    # ===== compute new dt (every step) + NaN check
    if step % 5 == 0:
        dt = min(compute_dt(), dt_visc)
        if not np.isfinite(dt):
            if rank == 0:
                print(f"NaN/inf: step={step} t={t:.6f} (t/t0={t/t0:.4f})", flush=True)
            comm.Abort(1)

    dt_step = max(min(dt, t_save[0] - t), 1e-12)

    # ===== runge kutta integration
    for a, b in zip(A, B):
        residual *= a
        residual += dt_step*rhs()
        u_hat += b*residual

    t += dt_step
    step += 1

    # ===== progress
    if step % 100 == 0 and rank == 0:
        print(f"step={step} t={t:.4f} t/t0={t/t0:.4f} ({100*t/t_total:.1f}%) "
              f"dt={dt:.4e} snapshots={iout} wall={(MPI.Wtime()-t_wall)/3600:.2f}h",
              flush=True)

fl.close()