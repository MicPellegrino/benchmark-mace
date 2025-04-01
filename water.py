from mace.calculators import mace_mp
from mace.calculators import MACECalculator

from ase import build
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase import Atoms

from testmd import test_md

"""
MPA_small.model              
ob3_small_foundation.model    
OMAT_small_foundation.model
small_mp0_foundation.model
#####
MPA_medium_foundation.model  
ob3_medium_foundation.model  
OMAT_medium_foundation.model  
medium_mp0_foundation.model  
"""

model_name = "OMAT_medium_foundation.model"
print("##### Testing",model_name,"#####")

# macemp = mace_mp(model="medium-mpa-0", device='cuda', enable_cueq=True, default_dtype="float64")
macemp = MACECalculator(model_paths="models-uob/GPU_Models/BCC_Foundation_Model/"+model_name, device='cuda', enable_cueq=True, default_dtype="float64")

# atoms = build.molecule('H2O')
# atoms.cell = [5, 5, 5, 90, 90, 90]
# print(atoms.get_positions())

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
W = Atoms('4(OH2)', positions=p, cell=c, pbc=[1, 1, 0])
W.write('WL.traj')

atoms.calc = macemp

# Set up the Langevin dynamics engine for NVT ensemble.
dyn = Langevin(atoms, 0.5*units.fs, temperature_K=300, friction=0.01/units.fs, logfile='water.log')

time, temperature, energies, _, _ = test_md(atoms, macemp, dyn, nsteps=200, fname='water.xyz', ndump=10, seed=1234, T=300)

print("time =",time)
print("T =",temperature)
print("e =", energies)
