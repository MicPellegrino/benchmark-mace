import numpy as np

from mace.calculators import mace_mp
from mace.calculators import MACECalculator

from ase import build
from ase.md import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.build import fcc111, bcc100, add_adsorbate
from ase.optimize import BFGS
from ase import Atoms
from ase.io.lammpsdata import write_lammps_data

from testmd import test_md

model_name = "mace01_mp0_fix2.model"

print("##### Testing MACE MP-0 small #####")
macemp = mace_mp(model="medium-mpa-0", device='cuda', enable_cueq=True, default_dtype="float64")

print("##### Testing",model_name,"#####")
# macemp = MACECalculator(model_paths="models-uob/GPU_Models/BCC_Foundation_Model/"+model_name, device='cuda', enable_cueq=True, default_dtype="float64")

# Building the molecular system

# FCC Al
# slab = fcc111('Al', size=[2, 4, 3], a=4.05, orthogonal=True)

# BCC Cr
slab = bcc100('Cr', size=[3, 4, 3], a=2.8849, orthogonal=True)

p = np.array(
    [[0.27802511, -0.07732213, 13.46649107],
     [0.91833251, -1.02565868, 13.41456626],
     [0.91865997, 0.87076761, 13.41228287],
     [1.85572027, 2.37336781, 13.56440907],
     [3.13987926, 2.3633134, 13.4327577],
     [1.77566079, 2.37150862, 14.66528237],
     [4.52240322, 2.35264513, 13.37435864],
     [5.16892729, 1.40357034, 13.42661052],
     [5.15567324, 3.30068395, 13.4305779],
     [6.10183518, -0.0738656, 13.27945071],
     [7.3856151, -0.07438536, 13.40814585],
     [6.01881192, -0.08627583, 12.1789428]])
c = np.array([[8.490373, 0., 0.],
              [0., 4.901919, 0.],
              [0., 0., 26.93236]])
water = Atoms('4(OH2)', positions=p, cell=c, pbc=[1, 1, 0])
water.rotate(90, 'z', center=(0, 0, 0))
water.set_cell(slab.cell, scale_atoms=False)
zmin = water.positions[:, 2].min()
zmax = slab.positions[:, 2].max()
water.positions += (4.0, 0, 0)
water.positions += (0, 1.0, 0)
water.positions += (0, 0, zmax - zmin + 1.0)
interface = slab + water
interface.center(vacuum=6, axis=2)

interface.calc = macemp

write_lammps_data("h2o-cr-ase-premin.data", interface)

print("Begin minimizing with BFGS...")
opt = BFGS(interface)
opt.run(fmax=0.00001,steps=1000)

write_lammps_data("h2o-cr-ase-postmin.data", interface, masses=True)

print("Beginning MD simulation with Langevin propagator...")
dyn = Langevin(interface, 0.25*units.fs, temperature_K=300, friction=0.1/units.fs, logfile='water-on-cr.log')
# dyn = NoseHooverChainNVT(interface, 0.25*units.fs, temperature_K=300, tdamp=1000*units.fs, tchain=1, logfile='water-on-cr.log')

time, temperature, energies, _, _ = test_md(interface, macemp, dyn, nsteps=20000, fname='water-on-cr.xyz', ndump=50, seed=1234, T=300)

write_lammps_data("h2o-cr-ase-postmd.data", interface, masses=True, velocities=True)

print("time =",time)
print("T =",temperature)
print("e =", energies)

np.savez('h2o-bcc-cr', t=time, T=temperature, Epa=energies)
