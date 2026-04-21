import pytest


def test_torchdrug_rdkit_draw_mol_to_mpl():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    plt = pytest.importorskip("matplotlib.pyplot")
    chem = pytest.importorskip("rdkit.Chem")

    from diffpack.torchdrug.data.rdkit.draw import MolToMPL

    mol = chem.MolFromSmiles("CCO")
    fig = MolToMPL(mol)
    assert fig is not None
    plt.close(fig)
