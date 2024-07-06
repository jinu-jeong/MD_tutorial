#%%
from utils import *
import matplotlib.pyplot as plt

# %%
path = "../Tutorial1/dump_Prod.xyz"
# %%
xyz,atom_type,box = read_xyz_file(path, same_atom_type=True, same_box_size=True)
# %%
box = np.diag(box)
# %%
compute_RDF = RDF_computer(torch.tensor(box), torch.device('cpu'), 10.0)
r_list, RDF_OO = compute_RDF(torch.tensor(xyz)[::10,0::3,:3])

# %%
plt.plot(r_list, RDF_OO, label='O-O RDF')
plt.xticks(fontsize=18)
plt.xlabel('r [$\AA$]', fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('RDF', fontsize=18)
plt.legend(fontsize=18)
# %% Do it yourself
# 1. Compute RDF for H-H pairs
# 2. Compute RDF for O-H pairs (advanced)

#%% MSD
# maximum time delay: 1ps
# number of ensembles: 30
compute_MSD = MSD_computer(10000, 1)
MSD = compute_MSD(torch.tensor(xyz)[:,0::3,:3])
# %%
# subplot 1. MSD is composed of early ballistic motion and later diffusive motion, followd by a linear behavior.
plt.figure(figsize=(8,12))
plt.subplot(2,1,1)
plt.plot(MSD, label='O MSD')
plt.xticks(fontsize=18)
plt.xlabel('t [fs]', fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('MSD [$\AA^2$]', fontsize=18)
plt.legend(fontsize=18)

# subplot 2: log-log scale. The slope of the ballistic behavior is 2, referring to mean free path. That of linear behavior is 1, referring to diffusion with collision.
plt.subplot(2,1,2)
plt.plot(MSD, label='O MSD')
plt.xticks(fontsize=18)
plt.xlabel('t [fs]', fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('MSD [$\AA^2$]', fontsize=18)
plt.legend(fontsize=18)

plt.xscale('log')
plt.yscale('log')
# %%
compute_VACF = VACF_computer(1000)
vel_COM = []
for i in range(len(xyz)):
    vel_COM.append(xyz[i, 0::3, 3:6] * 16/18 + xyz[i, 1::3, 3:6] * 1/18 + xyz[i, 2::3, 3:6] * 1/18)
vel_COM = np.array(vel_COM)

VACF = compute_VACF(torch.tensor(vel_COM))
plt.plot(VACF, label='O MSD')
plt.xticks(fontsize=18)
plt.xlabel('t [fs]', fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('VACF [$\AA^2/fs^2$]', fontsize=18)
plt.legend(fontsize=18)
# %%

# %% MSD computed from MSD and VACF
print("Ground truth [m^2/s]: 2.3 e-9")
print("MSD [m^2/s]: ", MSD[-1].item() / len(MSD) / 2 * 1e-20 / 1e-15)
print("VACF [m^2/s]: ", torch.sum(VACF).item() * 1e-20 / 1e-15)
print("Note that the MD simulation isn't equilibrated enough to explore the entire ensemble. To obtain the correct diffusion coefficient, you need to run the simulation longer. Equilibrate at least 2ns before production simulation might be helpful.")
# %%
# save r_list, RDF_OO, MSD, VACF in torch
torch.save(r_list, 'r_list.pt')
torch.save(RDF_OO, 'RDF_OO.pt')
torch.save(MSD, 'MSD.pt')
torch.save(VACF, 'VACF.pt')

# %%
