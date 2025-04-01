from mace.calculators import mace_mp
from mace.calculators import MACECalculator

from ase import build
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

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

atoms = build.molecule('H2O')
atoms.cell = [5, 5, 5, 90, 90, 90]
print(atoms.get_positions())

atoms.calc = macemp

# Set up the Langevin dynamics engine for NVT ensemble.
dyn = Langevin(atoms, 0.5*units.fs, temperature_K=300, friction=0.01/units.fs, logfile='water.log')

time, temperature, energies, _, _ = test_md(atoms, macemp, dyn, nsteps=200, fname='water.xyz', ndump=10, seed=1234, T=300)

print("time =",time)
print("T =",temperature)
print("e =", energies)
