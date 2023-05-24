from rdkit import Chem
import pandas as pd


def sort_molecule_pool(csv_file):
    # csv_file is a csv_file with a header 'SMILES'
    # [aro_cl, nap_cl, ole_cl, par_cl, iso_cl] list of lists of the sorted SMILES strings
    # function sorts the SMILES molecules in a detailed PIONA composition (depending on PIONA type and chain length)

    pure_hc_database = pd.read_csv(csv_file)
    pure_hc = pure_hc_database['SMILES']
    aromatics, naphthenes, olefins, isoparaffins, paraffins = sort_piona(pure_hc)

    par_cl = sort_chainlength(paraffins)
    iso_cl = sort_chainlength(isoparaffins)
    ole_cl = sort_chainlength(olefins)
    nap_cl = sort_chainlength(naphthenes)
    aro_cl = sort_chainlength(aromatics)

    return [aro_cl, nap_cl, ole_cl, par_cl, iso_cl]


def sort_piona(hc_list):
    # hc list: list of SMILES strings
    # aromatics, naphthenes, olefins, isoparaffins, paraffins: lists of SMILES strings
    # sort a list of hydrocarbons in aromatics, naphthenes, olefins, isoparaffins and paraffins

    aromatics = []
    naphthenes = []
    olefins = []
    isoparaffins = []
    paraffins = []

    for hc in hc_list:
        mol = Chem.MolFromSmiles(hc)
        hc_cor = Chem.MolToSmiles(mol)
        if hc_cor.count('c') > 0:
            aromatics.append(hc)
        elif hc_cor.count('1') > 0:
            naphthenes.append(hc)
        elif hc_cor.count('=') > 0:
            olefins.append(hc)
        elif hc_cor.count('(') > 0:
            isoparaffins.append(hc)
        elif hc_cor.count('C') == len(hc_cor):
            paraffins.append(hc)

    return aromatics, naphthenes, olefins, isoparaffins, paraffins


def sort_chainlength(hc_list):
    # hc list: list of SMILES strings
    # sorted_hc: returns a list of the sorted SMILES strings of hc_list, sorted on chain_length
    # sorted_hc[i] are a list of species with chain length i

    sorted_hc = [[] for j in range(40)]
    for hc in hc_list:
        i = hc.count('C') + hc.count('c')
        sorted_hc[i].append(hc)

    return sorted_hc
