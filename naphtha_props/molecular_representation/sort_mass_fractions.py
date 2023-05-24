import pandas as pd
import numpy as np


def determine_gibbs_energy(hc_list, csv_file):
    # hc list: list of SMILES strings
    # csv_file: csv_file with a header "SMILES" and "deltaGf [kJ/mol]" which contains the Gibbs free energy of
    # molecules provided in hc_list
    # gibbs_energy: list with the same dimensions as hc_list which contains the Gibbs free energy of every molecule

    hc_energy_database = pd.read_csv(csv_file)
    hc_energy = hc_energy_database[['SMILES', 'deltaGf [kJ/mol]']]
    gibbs_energy = [0]*len(hc_list)
    for i in range(len(hc_list)):
        gibbs_energy[i] = hc_energy[hc_energy['SMILES'] == hc_list[i]]['deltaGf [kJ/mol]'].values[0]

    return gibbs_energy


def mole_fraction_equilibrium(free_energy_list):
    # free_energy_list: list of gibbs free energies corresponding to every hydrocarbon (kJ/mol)
    # fractions: list with same dimensionality as free_energy_list, containing the mass fraction of every compound
    # returns a list of the mole fractions of every compound assuming they are in equilibrium
    # as the molar mass in each lump is more or less the same, the mole fraction is assumed equal to the mass fraction

    fractions = [0]*len(free_energy_list)
    boltzmann_factor = []
    for g in free_energy_list:
        boltzmann_factor.append(np.exp(-(g*1000)/(8.314*298)))

    for i in range(len(boltzmann_factor)):
        fractions[i] = boltzmann_factor[i]/sum(boltzmann_factor)

    return fractions


def fraction_per_lump(sorted_hc):
    # sorted_hc: list of SMILES strings sorted per chain length and PIONA composition
    # set_of_compositions: mass fractions per lump formatted similar to sorted_hc per chain length and PIONA

    set_of_compositions = []
    for i in range(5):
        L = []
        for lump in sorted_hc[i]:
            composition = mole_fraction_equilibrium(determine_gibbs_energy(lump, 'hydrocarbon_database.csv'))
            L.append(composition)
        set_of_compositions.append(L)

    return set_of_compositions

