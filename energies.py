from testmd import test_md

import random
import numpy as np

from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.bussi import Bussi

from mace.calculators import mace_mp

# ----- PARAMETERS ----- #
T0 = 300
dt = 1.0
tdamp_langevin = 0.01/units.fs
tdamp_bussi = (1e3)*units.fs
nsteps = 100000
ndump = 100
sys_name = 'WMo'
box_size = '7'
# ----- ---------- ----- #

# ----- IMPORTING A MACE CALCULATOR FROM MP-0 MEDIUM ----- #
# calc_mace = mace_mp(model="medium-mpa-0", device='cuda', default_dtype="float64")
calc_mace = mace_mp(model="medium-mpa-0", device='cuda', enable_cueq=True, default_dtype="float64")
# calc_mace = mace_mp(model="medium-mpa-0", device='cuda', default_dtype="float32")
# ----- -------------------------------------------- ----- #

s_mean_vec = []
s_std_vec = []
spa_mean_vec = []
spa_std_vec = []
n_atom_vec = []

input_file = 'confs/'+sys_name+'_'+box_size+'.data'
logfile = 'ase-output/'+sys_name+'_'+box_size+'.log'
trjfile = 'ase-output/'+sys_name+'_'+box_size+'.xyz'

print("### --------------------------------------------------------------- ###")
print("### Simulating",input_file)

init_conf = read(input_file, '0', format='lammps-data')

# dyn = Langevin(init_conf, dt*units.fs, temperature_K=T0, friction=tdamp_langevin, logfile=logfile)
# dyn = Bussi(init_conf, dt*units.fs, temperature_K=T0, taut=tdamp_bussi, logfile=logfile)
dyn = NoseHooverChainNVT(init_conf, dt*units.fs, temperature_K=T0, tdamp=tdamp_bussi, tchain=1, logfile=logfile)

n_atoms = len(dyn.atoms)
print("###",n_atoms,"atoms")

# ----- ACTUAL MACE MD RUN ----- #
time, temperature, energies, _, _ = test_md(init_conf, calc_mace, dyn=dyn, nsteps=nsteps, ndump=ndump, fname=trjfile, T=T0)
# ----- ------------------ ----- #

np.savez(sys_name, t=time, T=temperature, Epa=energies)
