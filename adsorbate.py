from mace.calculators import mace_mp
from mace.calculators import MACECalculator

from ase import build
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.build import fcc111, add_adsorbate

from testmd import test_md

model_name = "OMAT_medium_foundation.model"
print("##### Testing",model_name,"#####")

# macemp = mace_mp(model="medium-mpa-0", device='cuda', enable_cueq=True, default_dtype="float64")
macemp = MACECalculator(model_paths="models-uob/GPU_Models/BCC_Foundation_Model/"+model_name, device='cuda', enable_cueq=True, default_dtype="float64")

# Water molecule on a FCC Al slab
slab = fcc111('Al', size=(2,2,3))
add_adsorbate(slab, 'H', 1.5, 'ontop')

slab.calc = macemp

# Set up the Langevin dynamics engine for NVT ensemble.
dyn = Langevin(slab, 0.5*units.fs, temperature_K=300, friction=0.01/units.fs, logfile='water-on-al.log')

time, temperature, energies, _, _ = test_md(slab, macemp, dyn, nsteps=200, fname='water-on-al.xyz', ndump=10, seed=1234, T=300)

print("time =",time)
print("T =",temperature)
print("e =", energies)
