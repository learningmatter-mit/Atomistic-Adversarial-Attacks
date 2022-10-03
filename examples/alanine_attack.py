import os
import torch
import ase
import argparse
from torch.utils.data import DataLoader

from nff.data import Dataset, concatenate_dict, collate_dicts
from nff.utils.cuda import batch_to, batch_detach

from robust.loss import AdvLoss
from robust.dihedrals import set_dihedrals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments to perform adversarial attack on dihedral angles of alanine dipeptide. Models have to be trained before performing attacks."
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to trained models",
    )
    parser.add_argument(
        "generation",
        type=int,
        help="Number of active learning loop",
    )
    parser.add_argument(
        "num_attacks",
        type=int,
        help="Number of data points to perform attack on",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=1000,
        help="Number of epochs to perform attacks for",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate",
    )
    parser.add_argument(
        "--kT",
        type=float,
        default=20,
        help="Temperature at which the adversarial loss is set to",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    args = parser.parse_args()

    print(f"Loading trained model from {args.model_dir}")
    models = torch.load(
        os.path.join(args.model_dir, "best_model"), map_location=args.device
    )

    print("Loading dataset from previous generation")
    seed_dset = Dataset.from_file(
        os.path.join("dataset", f"gen{args.generation-1}.pth.tar")
    )

    # choose random seeds from the dataset
    randperm = torch.randperm(len(seed_dset))[: args.num_attacks]
    seed_configs = [seed_dset[i] for i in randperm]

    starting_points = []
    for config in seed_configs:
        mol = ase.Atoms(
            symbols=config["nxyz"][:, 0],
            positions=config["nxyz"][:, 1:],
        )
        phi = mol.get_dihedral(a1=7, a2=6, a3=1, a4=2)
        psi = mol.get_dihedral(a1=4, a2=2, a3=1, a4=6)
        starting_points.append([phi, psi])

    starting_points = torch.Tensor(starting_points).to(args.device)
    delta = torch.randn_like(starting_points, requires_grad=True, device=args.device)

    print("Start adversarial attack")

    opt = torch.optim.Adam([delta], lr=args.lr)
    loss_fun = AdvLoss(energies=seed_dset.props["energy"], temperature=args.kT)

    nbr_list = torch.combinations(torch.arange(22), r=2, with_replacement=False).to(
        args.device
    )

    for t in range(args.n_epochs):
        opt.zero_grad()

        inputs = ((starting_points + delta) % 360).to(args.device)

        dset = []
        for (inp, config) in zip(inputs, seed_configs):
            seed_nxyz = config["nxyz"].to(args.device)

            nxyz, phi_psi = set_dihedrals(seed_nxyz, inp[0], inp[1], device=args.device)
            dset.append(
                {
                    "nxyz": nxyz.detach(),
                    "phi_psi": phi_psi.reshape(-1, 2),
                    "energy": torch.Tensor([0]),
                    "energy_grad": torch.zeros(size=(len(nxyz), 3)),
                    "nbr_list": nbr_list,
                    "num_atoms": torch.Tensor([len(nxyz)]),
                }
            )
        dataloader = DataLoader(dset, batch_size=len(dset), collate_fn=collate_dicts)

        batch = next(iter(dataloader))

        results = []
        for i, model in enumerate(models):
            batch = batch_to(batch, args.device)
            results.append(model(batch))
            batch = batch_detach(batch)

        energy = torch.stack([r["energy"] for r in results], dim=-1)
        forces = -torch.stack([r["energy_grad"] for r in results], dim=-1)

        loss = loss_fun.loss_fn(e=energy, f=forces).sum()

        loss.backward()
        opt.step()
        if t % 10 == 0:
            print(t, loss.item())

    adv_path = os.path.join("dataset", f"gen{args.generation}_attacks.pth.tar")
    print("Save attacks to dataset {}".format(adv_path))
    advdset = Dataset(concatenate_dict(*dset))
    advdset.save(adv_path)

    print("Done")
