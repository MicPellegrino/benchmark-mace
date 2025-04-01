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
nsteps = 100
ndump = 10
nrep = 5
# box_size = ['3','4','5','6','7','8','9']
box_size = ['3','4','5','6','7','8']
sys_name = 'CoFeNi'
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

for bs in box_size :

    input_file = 'confs/'+sys_name+'_'+bs+'.data'
    logfile = 'ase-output/'+sys_name+'_'+bs+'.log'
    trjfile = 'ase-output/'+sys_name+'_'+bs+'.xyz'

    print("### --------------------------------------------------------------- ###")
    print("### Simulating",input_file)

    init_conf = read(input_file, '0', format='lammps-data')

    # dyn = Langevin(init_conf, dt*units.fs, temperature_K=T0, friction=tdamp_langevin, logfile=logfile)
    # dyn = Bussi(init_conf, dt*units.fs, temperature_K=T0, taut=tdamp_bussi, logfile=logfile)
    dyn = NoseHooverChainNVT(init_conf, dt*units.fs, temperature_K=T0, tdamp=tdamp_bussi, tchain=1, logfile=logfile)

    n_atoms = len(dyn.atoms)
    print("###",n_atoms,"atoms")

    s_mean = 0
    spa_mean = 0
    s_mean2 = 0
    spa_mean2 = 0
    for nr in range(nrep) :
        # ----- ACTUAL MACE MD RUN ----- #
        _, _, _, s, spa = test_md(init_conf, calc_mace, dyn=dyn, nsteps=nsteps, ndump=ndump, fname=trjfile, T=T0)
        # ----- ------------------ ----- #
        s_mean += s
        spa_mean += spa
        s_mean2 += s*s
        spa_mean2 += spa*spa
    s_mean /= nrep
    spa_mean /= nrep 
    s_mean2 /= nrep 
    spa_mean2 /= nrep
    s_std = np.sqrt(s_mean2-s_mean*s_mean)
    spa_std = np.sqrt(spa_mean2-spa_mean*spa_mean)
    print("### --------------------------------------------------------------- ###")
    print("### mean comp. speed:",s_mean,"ns/day")
    print("### std comp. speed:",s_std,"ns/day")
    print("### mean comp. speed*atoms:",spa_mean,"(ns/day)*atom")
    print("### std comp. speed*atoms:",spa_std,"(ns/day)*atom")
    s_mean_vec.append(s_mean)
    s_std_vec.append(s_std)
    spa_mean_vec.append(spa_mean)
    spa_std_vec.append(spa_std)
    n_atom_vec.append(n_atoms)

print("### --------------------------------------------------------------- ###")

s_mean_vec = np.array(s_mean_vec)
s_std_vec = np.array(s_std_vec)
spa_mean_vec = np.array(spa_mean_vec)
spa_std_vec = np.array(spa_std_vec)
n_atom_vec = np.array(n_atom_vec)
np.savez(sys_name, s_mean=s_mean_vec, s_std=s_std_vec, spa_mean=spa_mean_vec, spa_std=spa_std_vec, n_atom=n_atom_vec)
