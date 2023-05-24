import pandas as pd
import numpy as np
from sort_SMILES import sort_piona, sort_molecule_pool, sort_chainlength
from sort_mass_fractions import fraction_per_lump, determine_gibbs_energy, mole_fraction_equilibrium


def piona_to_smiles(piona_matrix, molecule_pool, mass_fraction):
    # piona_matrix: (5, 40) np.matrix with mass fraction per PIONA (row 1: A, row 2: N, row 3: O, row 4: P, row 5: I)
    # chain length is the column index
    # molecule_pool: sorted list of SMILES which will represent the feed (same dimensionality as piona_matrix)
    # mass_fraction: sorted list of mass fractions per lumped component (same dimensionality as molecule pool)

    molecules = np.empty(1)
    molecule_mass_fractions = np.empty(1)
    for i in range(5):
        for j in range(40):
            if mass_fraction[i][j] != []:
                mixture_mass_fraction = piona_matrix[i, j]*np.array(mass_fraction[i][j])/100
                if np.sum(mixture_mass_fraction) > 0.0:
                    molecules = np.hstack((molecules, np.ravel(molecule_pool[i][j])))
                    molecule_mass_fractions = np.hstack((molecule_mass_fractions, np.ravel(mixture_mass_fraction)))
    molecules = molecules[1:].reshape(-1, 1)
    molecule_mass_fractions = molecule_mass_fractions[1:].reshape(-1, 1)

    return molecules, molecule_mass_fractions


def piona_vector_to_matrix(piona_vector):
    # [A6, A7, A8, A9, N5, N6, N7, N8, N9, O5, O6,
    #  P4, P5, P6, P7, P8, P9, P10, P11, I5, I6, I7, I8, I9, I10, I11]
    # piona_vector: 1D np.array which expresses piona mass fractions, as formatted above
    # piona_matrix: (5, 40) np.matrix with mass fraction per PIONA (row 1: A, row 2: N, row 3: O, row 4: P, row 5: I)
    # chain length is the column index

    piona_vector_copy = piona_vector
    piona_vector = np.hstack((piona_vector[:3], piona_vector[4:7], piona_vector[8:]))
    piona_vector[2] += piona_vector_copy[3]
    piona_vector[5] += piona_vector_copy[7]
    piona_matrix = np.zeros((5, 40))
    piona_matrix[0, 6:10] = piona_vector[:4].reshape(1, -1)
    piona_matrix[1, 5:10] = piona_vector[4:9].reshape(1, -1)
    piona_matrix[2, 5:7] = piona_vector[9:11].reshape(1, -1)
    piona_matrix[3, 4:12] = piona_vector[11:19].reshape(1, -1)
    piona_matrix[4, 5:12] = piona_vector[19:].reshape(1, -1)

    return piona_matrix


def correction(SMILES, mass_fraction, piona_vector):
    # this function is dependent on the shape of the piona_vector
    # corrects the calculated SMILES and mass_fraction for the detailed analysis of
    # A_1: xylenes, A_2: ethylbenzene, N6_1: cyclohexane, N6_2: methylcyclopentane

    ethylbenz = np.where(SMILES == 'CCC1=CC=CC=C1')
    m_xyl = np.where(SMILES == 'CC1=CC(=CC=C1)C')
    o_xyl = np.where(SMILES == 'CC1=CC=CC=C1C')
    p_xyl = np.where(SMILES == 'CC1=CC=C(C=C1)C')
    xyl = np.where(SMILES == 'Cc1cccc(C)c1')
    methcyclpent = np.where(SMILES == 'CC1CCCC1')
    cyclhex = np.where(SMILES == 'C1CCCCC1')
    mass_fraction[ethylbenz] = piona_vector[3]/100
    mass_fraction[methcyclpent] = piona_vector[7]/100
    mass_fraction[cyclhex] = piona_vector[6]/100
    mass_fraction[m_xyl] = 1/100*piona_vector[2]*(np.exp(-(118.9*1000)/(8.314*298))/(np.exp(-(118.9*1000)/(8.314*298)) +
                                                                               np.exp(-(122.1*1000)/(8.314*298)) +
                                                                               np.exp(-(121.5*1000)/(8.314*298))))
    mass_fraction[o_xyl] = 1/100*piona_vector[2] * (
                np.exp(-(122.1 * 1000) / (8.314 * 298)) / (np.exp(-(118.9 * 1000) / (8.314 * 298)) +
                                                           np.exp(-(122.1 * 1000) / (8.314 * 298)) +
                                                           np.exp(-(121.5 * 1000) / (8.314 * 298))))
    mass_fraction[p_xyl] = 1/100*piona_vector[2] * (
                np.exp(-(121.5 * 1000) / (8.314 * 298)) / (np.exp(-(118.9 * 1000) / (8.314 * 298)) +
                                                           np.exp(-(122.1 * 1000) / (8.314 * 298)) +
                                                           np.exp(-(121.5 * 1000) / (8.314 * 298))))
    mass_fraction[xyl] = 0

    return mass_fraction


def main(piona_vector):
    # piona_vector: 1D np.array that gives the mass fractions of the detailed PIONA following the fixed structure
    # [A6, A7, A8_1, A8_2, A9, N5, N6_1, N6_2, N7, N8, N9, O5, O6, P4, P5, P6, P7, P8, P9, P10, P11,
    # I5, I6, I7, I8, I9, I10, I11]
    # SMILES: 2D column np.array with the SMILES string of the representative compounds
    # mass_fraction: 2D column np.array with the mass fraction of the compounds at the index corresponding
    # to the SMILES array

    piona_matrix = piona_vector_to_matrix(piona_vector)   # convert the piona_vector to a better structured piona_matrix
    molecule_pool = sort_molecule_pool('hydrocarbon_database.csv')  # sort the considered pool of hydrocarbons
    mole_fraction_pool = fraction_per_lump(molecule_pool)  # calculate the mass fraction of each compound per lump
    SMILES, mass_fraction = piona_to_smiles(piona_matrix, molecule_pool, mole_fraction_pool)
    mass_fraction = correction(SMILES, mass_fraction, piona_vector)

    print('The feed is represented by '+str(SMILES.shape[0])+' compounds')
    print('The sum of mass fractions is ' + str(sum(mass_fraction)[0]))

    return SMILES, mass_fraction


piona_vector = np.array([2.29, 1.97, 1.68, 0.37, 1.14, 1.23, 3.17, 3.9, 7.04, 4.42, 2.31, 0.05, 0.03, 2.68,
                                                11.54, 9.7, 4.73, 3.24, 1.77, 0.44, 0.06, 7.65, 12.53, 6.22, 3.77, 3.05,
                                                1.28, 0.12])

print(main(piona_vector))
