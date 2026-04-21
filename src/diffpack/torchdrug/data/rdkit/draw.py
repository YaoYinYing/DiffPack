from matplotlib import pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, Draw


def _infer_draw_size(ax):
    if ax is None:
        return (300, 300)
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    width = max(100, int(bbox.width * 100))
    height = max(100, int(bbox.height * 100))
    return (width, height)


def MolToMPL(mol, ax=None, kekulize=True, wedgeBonds=True, imageType=None, fitImage=False,
             options=None, **kwargs):
    """Generate a molecule drawing on a matplotlib axis using modern RDKit APIs."""
    if not mol:
        raise ValueError("Null molecule provided")

    if kekulize:
        mol = Chem.Mol(mol.ToBinary())
        Chem.Kekulize(mol)

    if not mol.GetNumConformers():
        AllChem.Compute2DCoords(mol)

    is_root = ax is None
    if ax is None:
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_axes([0, 0, 1, 1])
    else:
        fig = ax.figure

    ax.set_axis_off()
    image = Draw.MolToImage(
        mol,
        size=_infer_draw_size(ax),
        kekulize=False,
        wedgeBonds=wedgeBonds,
        fitImage=fitImage,
        options=options,
        **kwargs,
    )
    ax.imshow(image)
    return fig
