from rdkit import Chem
from .calculated_features import *


class AtomFeatureVector:
    """Constructs a atom feature vector
    Especially for hydrocarbons, and hetero-atoms in steam cracking"""

    def __init__(self, atom: Chem.rdchem.Atom, mol: Chem.rdchem.Mol, property: str = "solvation"):
        self.atom = atom
        self.mol = mol
        self.vector = []
        self.property = property

    def construct_vector(self):
        self.vector += oneHotVector(self, get_atomic_number(self.atom),
                                    [1, 6, 7, 8, 9, 16, 17, 35])
        self.vector += oneHotVector(self, self.atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
        self.vector += [float(self.atom.GetTotalNumHs())]
        self.vector += [1 if self.atom.GetExplicitValence() == 3 else 0]
        self.vector += oneHotVector(self, self.atom.GetHybridization(), [Chem.rdchem.HybridizationType.S,
                                                                         Chem.rdchem.HybridizationType.SP,
                                                                         Chem.rdchem.HybridizationType.SP2,
                                                                         Chem.rdchem.HybridizationType.SP3,
                                                                         Chem.rdchem.HybridizationType.SP3D,
                                                                         Chem.rdchem.HybridizationType.SP3D2])
        self.vector += [1 if self.atom.GetIsAromatic() else 0]
        self.vector += [self.atom.GetMass() * 0.01]
        self.vector += [get_in_ring_size(self.atom)]
        self.vector += [self.distance_to_rad()]

        # maybe add one to determine degree of branching
        # radical species?
        return self.vector

    def get_vector(self):
        return self.vector

    def get_atom_feature_dim(self):
        return len(self.vector)

    def distance_to_rad(self):
        id_ref = self.atom.GetIdx()
        id = self.atom.GetIdx()
        for atom_i in self.mol.GetAtoms():
            if atom_i.GetExplicitValence() == 3:
                id = atom_i.GetIdx()
                break
        return Chem.GetDistanceMatrix(self.mol)[id_ref, id]


class BondFeatureVector:
    """Constructs a bond feature vector"""

    def __init__(self, bond: Chem.rdchem.Bond, property: str = "solvation"):
        self.bond = bond
        self.vector = []
        self.property = property

    def construct_vector(self):
        if self.bond is None:
            self.vector = [1] + [0] * 13
        else:
            bt = self.bond.GetBondType()
            self.vector += oneHotVector(self, self.bond.GetBondType(), [0,
                                                                         Chem.rdchem.BondType.SINGLE,
                                                                         Chem.rdchem.BondType.DOUBLE,
                                                                         Chem.rdchem.BondType.TRIPLE,
                                                                         Chem.rdchem.BondType.AROMATIC])
            self.vector += [1 if self.bond.GetIsConjugated() else 0]
            self.vector += [1 if self.bond.IsInRing() else 0]
            self.vector += [get_if_bond_is_rotable(self.bond)]
            self.vector += [self.is_radical(self.bond.GetBeginAtom())+self.is_radical(self.bond.GetEndAtom())]
            if str(self.bond.GetStereo()) == "STEREOE":
                self.vector += [1]
            elif str(self.bond.GetStereo()) == "STEREOZ":
                self.vector += [2]
            else:
                self.vector += [0]
        return self.vector

    def get_vector(self):
        return self.vector

    def get_bond_feature_dim(self):
        return len(self.vector)

    def is_radical(self, atom):
        if atom.GetExplicitValence() == 3:
            return 1
        return 0

class MolFeatureVector:

    def __init__(self, mol: Chem.rdchem.Mol, property: str = "solvation"):
        self.mol = mol
        self.vector = []
        self.property = property
        self.substructures = ["[C;X4v4](-[C;v3])(-[C;v4])(-[C;v4])-[C;v4]-[C;X4v4](-[C;v4])(-[C;v4])(-[#1])",
                              "[C;X4v4](-[C;v4])(-[C;v4])(-[C;v4])-[C;v3]-[C;X4v4](-[C;v4])(-[C;v4])(-[#1])",
                              "[C;X4v4](-[C;v4])(-[C;v4])(-[C;v4])-[C;v4]-[C;X3v3](-[C;v4])(-[C;v4])",
                              "[C;X4v4](-[C;v4])(-[C;v4])(-[C;v4])-[C;v4]-[C;X4v4](-[C;v3])(-[C;v4])(-[#1])",
                              "[C;X4v4](-[C;v4])(-[C;v4])(-[C;v4])-[C;v3]-[C;X4v4](-[C;v4])(-[C;v4])(-[C;v4])",
                              "[C;X3v3](-C)-[C;X4v4](-[C;v4])(-[C;v4])",
                              "[C;X4v4](-[C;v3])-[C;X4v4](-[C;v4])(-[C;v4])"]

    def construct_vector(self):
        self.vector += [get_num_aliphatic_rings(self.mol)]
        self.vector += [get_num_aromatic_rings(self.mol)]
        self.vector += [get_num_rotable_bonds(self.mol)]
        self.vector += [get_molar_mass(self.mol)]
        self.vector += [get_sssr(self.mol)]
        for substructure in self.substructures:
            self.vector += [len(self.mol.GetSubstructMatches(Chem.MolFromSmarts(substructure), uniquify=True))]

        # number of tertiary carbons
        # number of quat carbons
        # molar mass
        # PIONA class

        return self.vector

    def get_vector(self):
        return self.vector

    def get_mol_feature_dim(self):
        return len(self.vector)


def oneHotVector(self, value, choices):
    encoding = [0] * (len(choices) + 1) #if value is not there, last index will be 1
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding
