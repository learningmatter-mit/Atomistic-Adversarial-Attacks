import torch
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Atom indices for dihedral angles and branches to rotate for Alanine Dipeptide
# PDB file of alanine dipeptide in data/alanine_dipeptide.pdb
DIHED = {"phi": (2, 1, 6, 7),
         "psi": (6, 1, 2, 4)}
BRANCH = {"phi": [7, 8, 9, 18, 19, 20, 21],
          "psi": [3, 4, 5, 14, 15, 16, 17]}


def get_axis_vector(u1, u2):
    u = u2 - u1
    return u/((u**2).sum()**0.5)


def get_rotation_matrix(th, u, device=DEVICE):
    theta = th * np.pi/180
    R = torch.zeros(3, 3).to(device)
    R[0, 0] = torch.cos(theta)+u[0]**2*(1-torch.cos(theta))
    R[0, 1] = (u[0]*u[1]*(1-torch.cos(theta)))-(u[2]*torch.sin(theta))
    R[0, 2] = (u[0]*u[2]*(1-torch.cos(theta)))+(u[1]*torch.sin(theta))
    R[1, 0] = (u[0]*u[1]*(1-torch.cos(theta)))+(u[2]*torch.sin(theta))
    R[1, 1] = torch.cos(theta)+u[1]**2*(1-torch.cos(theta))
    R[1, 2] = (u[1]*u[2]*(1-torch.cos(theta)))-(u[0]*torch.sin(theta))
    R[2, 0] = (u[2]*u[0]*(1-torch.cos(theta)))-(u[1]*torch.sin(theta))
    R[2, 1] = (u[2]*u[1]*(1-torch.cos(theta)))+(u[0]*torch.sin(theta))
    R[2, 2] = torch.cos(theta)+u[2]**2*(1-torch.cos(theta))
    return R


def get_mask(n, mask_indices, device=DEVICE):
    mask = torch.zeros((n, n)).to(device)
    for i in mask_indices:
        mask[i, i] += 1
    return mask


def get_rotated_xyz(theta, u, center, xyz, mask_indices, device=DEVICE):
    n = len(xyz)
    mask = get_mask(n, mask_indices, device=device)

    xyz = xyz - center
    mask_xyz = mask@xyz

    R = get_rotation_matrix(theta, u, device=device)
    new_xyz = (R@mask_xyz.T).T
    new_xyz += (((torch.eye(n).to(device) - mask)@xyz) + center)

    return new_xyz


def set_dihedrals(nxyz, phi, psi, dihedral=DIHED, branch=BRANCH, device=DEVICE):
    xyz = nxyz[:, 1:]
    # set phi
    phi_xyz = get_rotated_xyz(
        theta=phi,
        u=get_axis_vector(xyz[dihedral['phi'][1]], xyz[dihedral['phi'][2]]),
        center=xyz[dihedral['phi'][2]],
        xyz=nxyz[:, 1:],
        mask_indices=branch['phi'],
        device=device,
    )
    # set psi
    psi_xyz = get_rotated_xyz(
        theta=psi,
        u=get_axis_vector(xyz[dihedral['psi'][1]], xyz[dihedral['psi'][2]]),
        center=xyz[dihedral['psi'][2]],
        xyz=nxyz[:, 1:],
        mask_indices=branch['psi'],
        device=device,
    )
    phi_t = phi_xyz - xyz
    psi_t = psi_xyz - xyz

    new_xyz = xyz + phi_t + psi_t
    new_nxyz = torch.cat([nxyz[:, 0][:, None], new_xyz], dim=1)
    return new_nxyz, torch.cat([phi.reshape(-1, 1), psi.reshape(-1, 1)], dim=0).reshape(-1, 2).detach()
