#%%
import os
import numpy as np
import torch


#%%
def read_xyz_file(path, same_atom_type=True, same_box_size=True):

    if os.path.exists(path)==False:
        return [], []

    Atom_type = []
    XYZ = []
    BOX = []
    f = open(path, "r")
    while(1):
        atom_type = []
        xyz = []
        box = []
        line = f.readline()
        if len(line)==0:
            f.close()
            break
        line = f.readline()
        line = f.readline()
        line = f.readline().split(); Natom = int(line[0])
        line = f.readline()
        # Assuming the box starts at 0 for simplicity; adjust if your simulation does otherwise
        line = f.readline().split(); Lx = float(line[1]); box.append(Lx)
        line = f.readline().split(); Ly = float(line[1]); box.append(Ly)
        line = f.readline().split(); Lz = float(line[1]); box.append(Lz)
        line = f.readline()
        for _ in range(Natom):
            line = f.readline().split()
            atom_id, element, x,y,z, vx,vy,vz, fx,fy,fz = line
            atom_type.append(element)
            xyz.append([float(x), float(y), float(z), float(vx), float(vy), float(vz)])
        
        Atom_type.append(atom_type)
        BOX.append(box)
        XYZ.append(xyz)
    
    if same_atom_type==True:
        Atom_type = Atom_type[0]
    if same_box_size==True:
        BOX = BOX[0]
    XYZ = np.array(XYZ)
    return XYZ, Atom_type, BOX


#%%
class RDF_computer(torch.nn.Module):
    def __init__(self, cell, device, r_max):
        super(RDF_computer, self).__init__()
        self.cell = cell

        self.r_max = r_max
        self.dr = 0.1
        self.device = device
        self.r_list = torch.arange(
            0.5*self.dr, self.r_max - self.dr*2, self.dr, device=self.device)


    def forward(self, Traj):
        Hist = []
        for t, q in enumerate(Traj):
            nbr_list, offsets, pdist, unit_vector = generate_nbr_list(
                q, self.cell, cutoff=self.r_max)

            pdist_gaussian = gaussian_smearing(
                pdist-self.r_list, self.dr).sum(0)*self.dr
            if t == 0:
                Pdist_gaussian = pdist_gaussian
            else:
                Pdist_gaussian += pdist_gaussian

        Pdist_gaussian /= (t+1)

        v = 4 * np.pi / 3 * ((self.r_list+0.5*self.dr) **
                             3 - (self.r_list-0.5*self.dr)**3)
        natom = len(Traj[0])
        bulk_density = (natom-1)/(torch.det(self.cell))
        gr = Pdist_gaussian/v * (torch.det(self.cell))/(natom-1)/natom*2

        return self.r_list, gr
    
    
def gaussian_smearing(centered, sigma):
    return 1/(sigma*(2*np.pi)**0.5)*torch.exp(-0.5*(centered/sigma)**2)



def generate_nbr_list(coordinates, lattice_matrix, cutoff):

    lattice_matrix_diag = torch.diag(lattice_matrix).view(1, 1, -1)

    device = coordinates.device
    displacement = (
        coordinates[..., None, :, :] - coordinates[..., :, None, :])

    # Transform distance using lattice matrix inverse
    offsets = ((displacement+lattice_matrix_diag/2) // lattice_matrix_diag).detach()


    # Apply periodic boundary conditions
    displacement = displacement - offsets*(lattice_matrix_diag)

    # Compute squared distances and create mask for cutoff
    squared_displacement = torch.triu(displacement.pow(2).sum(-1))

    within_cutoff = (squared_displacement < cutoff **2) & (squared_displacement != 0)
    neighbor_indices = torch.nonzero(within_cutoff.to(torch.long), as_tuple=False)


    offsets = offsets[neighbor_indices[:, 0], neighbor_indices[:, 1], :]

    # Compute unit vectors and actual distances
    unit_vectors = displacement[neighbor_indices[:,0], neighbor_indices[:,1]]
    magnitudes = squared_displacement[neighbor_indices[:,0], neighbor_indices[:,1]].sqrt()
    
    unit_vectors = unit_vectors / magnitudes.view(-1, 1)

    actual_distances = magnitudes[:, None]

    return neighbor_indices.detach(), offsets, actual_distances, -unit_vectors


#%%
class MSD_computer(torch.nn.Module):
    def __init__(self, td_max, N_ensemble):
        super(MSD_computer, self).__init__()
        self.td_max = td_max
        self.N_ensemble = N_ensemble

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.view(len(x), -1, 3)

        t0_list = list(range(0, len(x)-self.td_max, self.N_ensemble))

        MSD = torch.zeros(self.td_max, device=x.device)

        for t0 in t0_list:
            MSD += (x[t0:t0+self.td_max] - x[t0]).pow(2).mean((1, 2))
        MSD /= len(t0_list)
        return MSD



class VACF_computer(torch.nn.Module):
    def __init__(self, td_max):
        super(VACF_computer, self).__init__()
        self.td_max = td_max

    def forward(self, vel):
        vacf = [(vel * vel.detach()).mean()[None]]
        # can be implemented in parrallel
        vacf += [ (vel[:-t] * vel[t:].detach()).mean()[None] for t in range(1, self.td_max)]

        vacf = torch.stack(vacf).reshape(-1)

        return vacf

