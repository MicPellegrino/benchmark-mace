from ase.io import write
from ase import units
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution

import random
import time

# from ase.io import read, write
# from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
# from ase.md.langevin import Langevin
# from ase.md.nose_hoover_chain import NoseHooverChainNVT
# from ase.md.bussi import Bussi
# import numpy as np

FS2PS = 1e-3
FS2NS = 1e-6
DAY2MIN = 1440

def test_md(init_conf, 
            calc, 
            dyn,
            nsteps=100, 
            fname='md_test.xyz',
            ndump=1,
            seed=1234,
            T=300):
    
    random.seed(seed)
    init_conf.calc = calc
    
    # Remove COM and rotational velocity
    MaxwellBoltzmannDistribution(init_conf, temperature_K=T)
    Stationary(init_conf)
    ZeroRotation(init_conf)

    time_ps = []
    temperature = []
    energies = []

    n_atoms = len(dyn.atoms)

    def write_frame():
            dyn.atoms.write(fname, append=True)
            time_ps.append(FS2PS*dyn.get_time()/units.fs)
            temperature.append(dyn.atoms.get_temperature())
            energies.append(dyn.atoms.get_potential_energy()/n_atoms)

    dyn.attach(write_frame, interval=ndump)
    t0 = time.time()
    dyn.run(nsteps)
    t1 = time.time()
    cpu_run_min = (t1-t0)/60
    sim_speed = FS2NS*DAY2MIN*(nsteps*dyn.dt/units.fs)/cpu_run_min
    print("### --------------------------------------------------------------- ###")
    print("MD finished in {0:.2f} minutes!".format(cpu_run_min))
    print("Estimate:",sim_speed,"ns/day")
    print("Estimate:",sim_speed*n_atoms,"(ns/day)*atoms")
    
    return time_ps, temperature, energies, sim_speed, sim_speed*n_atoms