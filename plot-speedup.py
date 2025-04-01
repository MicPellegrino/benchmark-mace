import matplotlib.pyplot as plt 
import numpy as np 

data_cofeni_sp = np.load("vanilla-sp/CoFeNi.npz")
data_cofeni_dp = np.load("cuequivariance-dp/CoFeNi.npz")

data_wmo_sp = np.load("vanilla-sp/WMo.npz")
data_wmo_dp = np.load("cuequivariance-dp/WMo.npz")

data_cu_sp = np.load("vanilla-sp/Cu.npz")
data_cu_dp = np.load("cuequivariance-dp/Cu.npz")

# print(data_cofeni_sp)

# Single precision

fig, (ax1,ax2) = plt.subplots(1,2)

ax1.errorbar(data_cofeni_sp['n_atom'],data_cofeni_sp['s_mean'],yerr=data_cofeni_sp['s_std'], fmt='ko')
ax1.errorbar(data_wmo_sp['n_atom'],data_wmo_sp['s_mean'],yerr=data_wmo_sp['s_std'], fmt='rs')
ax1.errorbar(data_cu_sp['n_atom'],data_cu_sp['s_mean'],yerr=data_cu_sp['s_std'], fmt='bx')

ax2.errorbar(data_cofeni_sp['n_atom'],data_cofeni_sp['spa_mean'],yerr=data_cofeni_sp['spa_std'], fmt='ko')
ax2.errorbar(data_wmo_sp['n_atom'],data_wmo_sp['spa_mean'],yerr=data_wmo_sp['spa_std'], fmt='rs')
ax2.errorbar(data_cu_sp['n_atom'],data_cu_sp['spa_mean'],yerr=data_cu_sp['spa_std'], fmt='bx')

plt.show()

# Double precision

fig, (ax1,ax2) = plt.subplots(1,2)

ax1.errorbar(data_cofeni_dp['n_atom'],data_cofeni_dp['s_mean'],yerr=data_cofeni_dp['s_std'], fmt='ko')
ax1.errorbar(data_wmo_dp['n_atom'],data_wmo_dp['s_mean'],yerr=data_wmo_dp['s_std'], fmt='rs')
# ax1.errorbar(data_cu_dp['n_atom'],data_cu_dp['s_mean'],yerr=data_cu_dp['s_std'], fmt='bx')

ax2.errorbar(data_cofeni_dp['n_atom'],data_cofeni_dp['spa_mean'],yerr=data_cofeni_dp['spa_std'], fmt='ko')
ax2.errorbar(data_wmo_dp['n_atom'],data_wmo_dp['spa_mean'],yerr=data_wmo_dp['spa_std'], fmt='rs')
# ax2.errorbar(data_cu_dp['n_atom'],data_cu_dp['spa_mean'],yerr=data_cu_dp['spa_std'], fmt='bx')

plt.show()