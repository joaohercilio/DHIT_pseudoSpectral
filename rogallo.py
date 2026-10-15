import numpy as np
import scipy as sc
from scipy.fft import irfftn, rfftn
import matplotlib.pyplot as plt
seed = 1          
rng = np.random.default_rng(seed)

# RESOLUTION
N = 256
L = 2*np.pi

# CRITERIA 
eta_delta_initial = 1.5

# ======== INITIAL SPECTRUM PARAMETERS ======== 

sigma = 4
kp = 3
cs = 1/np.sqrt(3)
q = cs * 0.05

kmin = 2*np.pi/L
kmax = N*np.pi/L
k = np.linspace(kmin, kmax, 1000)

ktil = np.linspace(0, 10, 1000)
A = np.trapezoid(ktil**sigma * np.exp(-sigma * ktil**2 / 2),ktil)

E0 = lambda k:(q**2 / (2 * A)) * (1 / kp**(sigma+1)) * k**sigma * np.exp(-sigma * (k/kp)**2 / 2)

print("\nSPECTRUM PARAMETERS")
print("N =", N)
print("sigma =", sigma)
print("kp =", kp)
print("initial eta/delta =", eta_delta_initial)

# ======== INITIAL TURBULENT QUANTITIES ======== 

K0 = np.trapezoid(E0(k), k)                  # TURBULENT KINETIC ENERGY
Om0 = np.trapezoid(k**2 * E0(k), k)          # ENSTROPHY
L110 = 3*np.pi/4 * np.trapezoid(E0(k)/k, k) / K0 # INTEGRAL LENGHTSCALE

nu0 = np.sqrt(2*Om0)*(eta_delta_initial*L/N)**2 # VISCOSITY
nu_lbm0 = nu0/(L/N)
eps0 = 2*nu0*Om0                         # DISSIPATION
urms0 = np.sqrt(2 * K0 / 3)              # URMS
lamb0 = np.sqrt(15 * urms0**2 / (2 * Om0)) # TAYLOR MICROSCALE
ReL0 = urms0 * lamb0 / nu0               # TAYLOR REYNOLDS LAMBDA
eta0 = (nu0**3/eps0)**0.25               # KOLMOGOROV SCALE

print("\nINITIAL TURBULENT PARAMETERS")
print("K =", K0)
print("Om =", Om0)
print("eps =", eps0)
print("urms =", urms0)
print("lambda =", lamb0)
print("ReL =", ReL0)
print("nu  =", nu0)
print("nu_lbm =", nu_lbm0)
print("eta =", eta0)
print("tau =", 3*nu_lbm0 + 0.5)
print("L/L11 =", L/L110)
print("eta*kmax =", eta0*kmax)
print("eta/(L/N) =", eta0/(L/N))

# ======== INITIAL VELOCITY FIELD CALCULATION ======== 

uHat = np.zeros( (3,N,N,N//2+1) , dtype=np.complex128)

kArr = np.array( np.meshgrid(
    sc.fft.fftfreq(N)*N,
    sc.fft.fftfreq(N)*N,
    sc.fft.rfftfreq(N)*N ,
    indexing = 'ij'))

kMag = np.linalg.norm( kArr , axis = 0 )
M = np.sqrt(kArr[0]**2+kArr[1]**2)

tt1 = rng.uniform(0, 2*np.pi, size=kMag.shape)
tt2 = rng.uniform(0, 2*np.pi, size=kMag.shape)
phi = rng.uniform(0, 2*np.pi, size=kMag.shape)

nonzero = kMag != 0
mask = nonzero & (M != 0)
axis = nonzero & (M == 0)

alpha = np.zeros_like(kMag,dtype=np.complex128)
beta  = np.zeros_like(kMag,dtype=np.complex128)

alpha[nonzero] = np.sqrt(E0(kMag[nonzero]) / (2*np.pi*kMag[nonzero]**2)) * np.exp(1j * tt1[nonzero]) * np.cos(phi[nonzero])
beta[nonzero]  = np.sqrt(E0(kMag[nonzero]) / (2*np.pi*kMag[nonzero]**2)) * np.exp(1j * tt2[nonzero]) * np.sin(phi[nonzero])

uHat[0][mask] = (alpha[mask] * kMag[mask] * kArr[1][mask] + beta[mask] * kArr[0][mask] * kArr[2][mask]) / (kMag[mask] * M[mask])

uHat[1][mask] = (beta[mask] * kArr[1][mask] * kArr[2][mask] - alpha[mask] * kArr[0][mask] * kMag[mask]) / (kMag[mask] * M[mask])

uHat[2][mask] = -(beta[mask] * M[mask]) / kMag[mask]

uHat[0][axis] = alpha[axis]
uHat[1][axis] = beta[axis]
uHat[2][axis] = 0

neg = (-np.arange(N)) % N
for iz in (0, N//2):
    plane = uHat[:,:,:,iz]
    uHat[:,:,:,iz] = (plane + np.conj(plane[:,neg][:,:,neg])) / np.sqrt(2)

uHat[:,0,0,0] = 0

# ======== BINNING OF THE ENERGY SPECTRUM ======== 

E = np.linalg.norm(uHat, axis=0)**2
E[:,:,1:-1] *=  2

shell = np.floor(kMag + 0.5).astype(np.int16)
kCenters = np.arange(shell.max() + 1)
kBins = np.arange(-0.5, shell.max() + 1.5)

Ebinned = 0.5 * np.bincount(shell.ravel(), weights=E.ravel(), minlength=len(kCenters))
Etarget = np.zeros_like(kCenters, dtype=np.float64)
for i in range(1, len(kCenters)):
    kb = np.linspace(kBins[i], kBins[i+1], 101)
    Etarget[i] = np.trapezoid(E0(kb), kb)

scale = np.zeros_like(Etarget)
np.divide(Etarget, Ebinned, out=scale, where=Ebinned > 0)
scale = np.sqrt(scale)
for i in range(3):
    uHat[i] *= scale[shell]

u = irfftn(uHat, axes=(1,2,3), norm="forward")

E = np.linalg.norm(uHat, axis=0)**2
E[:,:,1:-1] *=  2
Ebinned = 0.5 * np.bincount(shell.ravel(), weights=E.ravel(), minlength=len(kCenters))

K = Ebinned.sum()                       # TURBULENT KINETIC ENERGY
Om = 0.5 * (E*kMag**2).sum()            # ENSTROPHY
L11 = 3*np.pi/4 * np.sum(Ebinned[1:]/kCenters[1:]) / K # INTEGRAL LENGHTSCALE

delta = L/N
nu = np.sqrt(2*Om)*(eta_delta_initial*delta)**2 # VISCOSITY
nu_lbm = nu/delta
eps = 2*nu*Om                           # DISSIPATION
urms = np.sqrt(2 * K / 3)               # URMS
lamb = np.sqrt(15 * urms**2 / (2 * Om)) # TAYLOR MICROSCALE
ReL = urms * lamb / nu                  # TAYLOR REYNOLDS LAMBDA
eta = (nu**3/eps)**0.25                 # KOLMOGOROV SCALE

print("\nINITIAL TURBULENT PARAMETERS (BINNING)")
print("K =", K)
print("Om =", Om)
print("eps =", eps)
print("urms =", urms)
print("lambda =", lamb)
print("ReL =", ReL)
print("nu  =", nu)
print("nu_lbm =", nu_lbm)
print("eta =", eta)
print("tau =", 3*nu_lbm + 0.5)

print("\nINITIAL RESOLUTION CRITERIA")
print("L/L11 =", L/L11)
print("eta*kmax  = ", eta*kmax)         # RESOLUTION CRITERIA
print("eta/(L/N) = ", eta/(L/N))        # RESOLUTION CRITERIA (LBM)
print("spectrum relative error = ", np.linalg.norm(Ebinned-Etarget)/np.linalg.norm(Etarget))

tau = 3*nu_lbm + 0.5
h = L/N
T = 30

t0 = K/eps
dt = delta/t0
steps = T / dt
print("\nTurnover time t0 = ", t0)
print("Dimensionless step size = ", dt)
print("T' = 30: ", int(np.round(steps)), " steps")
print("Mach max: ", np.max(np.linalg.norm(u, axis=0)) / cs)


'''

SOLVE THE POISSON EQUATION TO GENERATE INITIAL PRESSURE FIELD

'''

import mpi4py
mpi4py.rc.thread_level = "serialized"
from mpi4py import MPI
import numpy as np
from shenfun import Array, Function, FunctionSpace, TensorProductSpace, VectorSpace

comm = MPI.COMM_WORLD

real     = np.float32   # np.float64
complex_ = np.complex64 # np.complex128

V = [FunctionSpace(N, "F", domain=(0, 2*np.pi), dtype=real if i == 2 else complex_) for i in range(3)]
T = TensorProductSpace(comm, V, dtype=real, slab=False, collapse_fourier=True)
VT = VectorSpace(T)

Td = T.get_dealiased(padding_factor=1, dealias_direct=True)
VTd = VectorSpace(Td)

x = T.local_mesh(True)
k = T.local_wavenumbers(scaled=True)
k = [ki.astype(real, copy=False)

 for ki in k]
k2 = np.zeros(T.shape(True), dtype=real)
for ki in k:
    k2 += ki*ki
inv_k2 = np.zeros_like(k2)
np.divide(1, k2, out=inv_k2, where=k2 != 0)

uArray = Array(VT)
uDealias = Array(VTd)
uHatArray = Function(VT)

uArray[0] = u[0]
uArray[1] = u[1]
uArray[2] = u[2]

VT.forward(uArray, uHatArray)
del uArray
VTd.backward(uHatArray, uDealias)
pHat = Function(T)
pHat[:] = 0
tmp = Function(T)
for i in range(3):
    for j in range(3):
        uu = Array(Td)
        uu[:] = uDealias[i]*uDealias[j]
        Td.forward(uu, tmp)

        pHat += k[i]*k[j]*tmp

pHat *= inv_k2         
p = Array(T)
T.backward(pHat, p)

rho = 3.0 * p

'''
INITIAL STRAIN RATE TENSOR
'''

ikx = 1j * kArr[0]
iky = 1j * kArr[1]
ikz = 1j * kArr[2]

sxx = irfftn(ikx * uHat[0], s=(N, N, N), norm="forward")
syy = irfftn(iky * uHat[1], s=(N, N, N), norm="forward")
szz = irfftn(ikz * uHat[2], s=(N, N, N), norm="forward")
sxy = 0.5 * irfftn(iky * uHat[0] + ikx * uHat[1], s=(N, N, N), norm="forward")
sxz = 0.5 * irfftn(ikz * uHat[0] + ikx * uHat[2], s=(N, N, N), norm="forward")
syz = 0.5 * irfftn(ikz * uHat[1] + iky * uHat[2], s=(N, N, N), norm="forward")

# ======== CREATE VTK FILE (Velocity and Density) ======== 

import vtk
from vtk.util import numpy_support

img = vtk.vtkImageData()
img.SetDimensions(N, N, N)
img.SetSpacing(1.0, 1.0, 1.0)
img.SetOrigin(0.0, 0.0, 0.0)

fields = {
    "drho": 3.0 * p,
    "ux": u[0],
    "uy": u[1],
    "uz": u[2],
    "sxx": sxx,
    "sxy": sxy,
    "sxz": sxz,
    "syy": syy,
    "syz": syz,
    "szz": szz,
}

for name, field in fields.items():
    array = np.asarray(field, dtype=np.float32)
    vtk_array = numpy_support.numpy_to_vtk(
        array.ravel(order="F"),
        deep=True,
        array_type=vtk.VTK_FLOAT
    )
    vtk_array.SetName(name)
    img.GetPointData().AddArray(vtk_array)

writer = vtk.vtkStructuredPointsWriter()
writer.SetFileName("initial.vtk")
writer.SetInputData(img)
writer.SetFileTypeToBinary()
writer.Write()

'''

SPECTRUM PARAMETERS
N = 256
sigma = 4
kp = 3
initial eta/delta = 1.5

INITIAL TURBULENT PARAMETERS
K = 0.00041411764695867476
Om = 0.004685657673284074
eps = 1.229598012569254e-06
urms = 0.01661560806307681
lambda = 0.6647549695843122
ReL = 84.18122314393749
nu  = 0.00013120869025280886
nu_lbm = 0.005345922977369067
eta = 0.03681553890925539
tau = 0.5160377689321072
L/L11 = 7.638827949993383
eta*kmax = 4.71238898038469
eta/(L/N) = 1.5

INITIAL TURBULENT PARAMETERS (BINNING)
K = 0.00041657894541915087
Om = 0.0048830603668517355
eps = 1.3081135901625261e-06
urms = 0.016664912149366422
lambda = 0.6531119251952778
ReL = 81.25821417471477
nu  = 0.0001339440322141572
nu_lbm = 0.005457370835083055
eta = 0.03681553890925539
tau = 0.5163721125052492

INITIAL RESOLUTION CRITERIA
L/L11 = 7.413190226542916
eta*kmax  =  4.71238898038469
eta/(L/N) =  1.5
spectrum relative error =  4.662563724294498e-16

Turnover time t0 =  318.457776566172
Dimensionless step size =  7.707047656620924e-05
T' = 30:  389254  steps
Mach max:  0.14046375632867755

'''