import numpy as np
import h5py
from mpi4py import MPI
from shenfun import Array, FunctionSpace, TensorProductSpace, VectorSpace, ShenfunFile

N = 256
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

real = np.float64
complex_ = np.complex128

Vx = FunctionSpace(N, "F", domain=(0, 2*np.pi), dtype=complex_)
Vy = FunctionSpace(N, "F", domain=(0, 2*np.pi), dtype=complex_)
Vz = FunctionSpace(N, "F", domain=(0, 2*np.pi), dtype=real)
T = TensorProductSpace(comm, [Vx, Vy, Vz], dtype=real, slab=True, collapse_fourier=True)
VT = VectorSpace(T)

u = Array(VT)
fl = ShenfunFile("fields", VT, backend="hdf5", mode="r")

t0 = 318.457776566172

times = np.loadtxt("times.dat", ndmin=2)
tdict = {int(i): t for i, t in times}

with h5py.File("fields.h5", "r") as h:
    timesteps = sorted(map(int, h["u/3D"].keys()))
timesteps = [s for s in timesteps if s in tdict][:-1]   # [:-1] só enquanto a corrida estiver ativa

out = open("statistics.dat", "w") if rank == 0 else None
if rank == 0:
    out.write("#t_star\tk\ts\n")

for step in timesteps:
    fl.read(u, "u", step=step)

    # mean kinetic energy
    local = 0.5*np.sum(u*u)
    K = comm.allreduce(local)/N**3
    if rank == 0:
        out.write(f"{tdict[step]/t0:.6f}\t{K:.10e}\n")
        out.flush()
    

fl.close()
if rank == 0:
    out.close()