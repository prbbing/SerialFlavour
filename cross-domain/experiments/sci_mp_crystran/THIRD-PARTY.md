# Source and adaptation

CrystalTransformer reference: fduabinitio/ct-UAE, MIT, commit
`0141ff9e09277d2229c9d7a24c1bcc5eac9de78e`.
Unmodified reference files and full license are in `tests/reference/`.
`model.py` adapts the author's ST and MT@2p forward paths at reduced width/depth,
removes unused positional/coordinate-difference parameters, disables PyTorch's
nested-tensor acceleration, and exposes H/g/native/auxiliary outputs.
ST retains the author's two linear layers with no activation; MT retains two
Linear-ReLU-Linear heads (head1 Ef, head2 Eg). No author weights are loaded.
The MT-main-only control retains the MT model and disables Ef loss.

Paper: https://www.nature.com/articles/s41467-025-56481-x
Code: https://github.com/fduabinitio/ct-UAE/tree/0141ff9e09277d2229c9d7a24c1bcc5eac9de78e
Dataset metadata: https://github.com/hackingmaterials/matminer/blob/f89a530b76070fb613737b181381833330fb50c2/matminer/datasets/dataset_metadata.json
Dataset: MP 2018-10-18, 83,989 released rows (not the paper's MP/MP* snapshots).
Use of MP data requires attribution; source data are not distributed in this package.
Material citation: Jain et al., APL Materials 1, 011002 (2013), doi:10.1063/1.4812323.
MP attribution/license: https://doi.org/10.17188/1654373 (CC BY 4.0 data statement).