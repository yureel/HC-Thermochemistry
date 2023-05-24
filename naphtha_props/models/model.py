import torch
import torch.nn as nn

import naphtha_props.data.data
from naphtha_props.data.data import DataTensor
from naphtha_props.inp import InputArguments
from naphtha_props.data.data import DatapointList

from naphtha_props.models.ffn import FFN
from naphtha_props.models.mpn import MPN
from logging import Logger
import numpy as np
import copy

class Model(nn.Module):
    def __init__(self, inp: InputArguments, logger: Logger = None, data=None):
        super(Model, self).__init__()
        logger = logger.debug if logger is not None else print
        self.postprocess = inp.postprocess
        self.morgan = inp.morgan_fingerprint
        self.morgan_size = inp.morgan_bits
        self.morgan_radius = inp.morgan_radius
        self.feature_size = inp.num_features
        self.cudap = inp.cuda
        self.property = inp.property
        self.mix = inp.mix
        self.data = data
        self.atom_features = None
        self.bond_features = None

        if inp.morgan_fingerprint:
            logger(f"Make FFN model for solvent and solute to encode the morgan fingerprint"
                   f" with number of layers {inp.depth}, hidden size {inp.mpn_hidden}, "
                   f"dropout {inp.mpn_dropout}, "
                   f"activation function {inp.mpn_activation} and bias {inp.mpn_bias}")
            self.morgan_ffn = FFN(self.morgan_size, (inp.mpn_hidden + inp.f_mol_size),
                                  ffn_hidden_size=inp.mpn_hidden, num_layers=inp.depth, dropout=inp.mpn_dropout,
                                  activation=inp.mpn_activation, bias=inp.mpn_bias)

        else:
            logger(f"Make MPN model with depth {inp.depth}, hidden size {inp.mpn_hidden}, dropout {inp.mpn_dropout}, "
                   f"activation function {inp.mpn_activation} and bias {inp.mpn_bias}")
            self.mpn = MPN(depth=inp.depth, hidden_size=inp.mpn_hidden, dropout=inp.mpn_dropout,
                           activation=inp.mpn_activation, bias=inp.mpn_bias, cuda=inp.cuda, atomMessage=False,
                           property=self.property, aggregation=inp.aggregation, aggregation_norm=inp.aggregation_norm)

            logger(f"Make FFN model with number of layers {inp.ffn_num_layers}, hidden size {inp.ffn_hidden}, "
                   f"dropout {inp.ffn_dropout}, activation function {inp.ffn_activation} and bias {inp.ffn_bias}")

            self.ffn = FFN(inp.mpn_hidden + inp.f_mol_size + inp.num_features, inp.num_targets,
                           ffn_hidden_size=inp.ffn_hidden, num_layers=inp.ffn_num_layers, dropout=inp.ffn_dropout,
                           activation=inp.ffn_activation, bias=inp.ffn_bias)

    def convert_features_to_mols(self, tensor, datapointlist):
        i = 0
        for d in datapointlist.get_data():
            for enc in d.get_mol_encoder():
                tensor_atoms = []
                tensor_bonds = []
                tensor_mol = []
                for subtensor in tensor[i]:
                    subtensor = list(subtensor)
                    print(self.model.data.get_data().get_mol_encoder().fa_size)
                    if len(subtensor) == self.model.data.get_data().get_mol_encoder().fa_size:
                        tensor_atoms.append(subtensor)
                    elif len(subtensor) == 39:
                        tensor_bonds.append(subtensor)
                    elif len(subtensor) == 12:
                        tensor_mol = subtensor
                enc.f_atoms = tensor_atoms
                enc.f_bonds = tensor_bonds
                enc.f_mol = tensor_mol
                i += 1
        return datapointlist

    def convert_types_to_molecules(self, vector, Datapoint):
        Datapoint_2 = copy.deepcopy(Datapoint)
        for enc in Datapoint_2.get_mol_encoder():
            if enc:
                atom_list = []
                bond_list = []
                for i in range(len(enc.f_atoms)):
                    atom_list.append(self.atom_features[int(vector[i])])
                for j in range(len(enc.f_bonds)):
                    bond_list.append(self.bond_features[int(vector[j + len(enc.f_atoms)])])
                enc.f_atoms = atom_list
                for k in range(len(enc.f_bonds)):
                    enc.f_bonds[k][-1*enc.fb_size:] = bond_list[k][-1*enc.fb_size:]
                    if k % 2 == 0:
                        enc.f_bonds[k][:-1 * enc.fb_size] = atom_list[
                            min(Datapoint.mol[0].GetBondWithIdx(int(np.floor(k/2))).GetBeginAtom().GetIdx(),
                                Datapoint.mol[0].GetBondWithIdx(int(np.floor(k/2))).GetEndAtom().GetIdx())]
                    else:
                        enc.f_bonds[k][:-1 * enc.fb_size] = atom_list[
                            max(Datapoint.mol[0].GetBondWithIdx(int(np.floor(k/2))).GetBeginAtom().GetIdx(),
                                Datapoint.mol[0].GetBondWithIdx(int(np.floor(k/2))).GetEndAtom().GetIdx())]
                enc.f_mol = vector[
                            len(enc.f_atoms) + len(enc.f_bonds):len(enc.f_atoms) + len(enc.f_bonds) + len(enc.f_mol)]
            Datapoint_2.scaled_features = vector[-3:]
        return Datapoint_2

    def forward(self, data):
        if self.data is not None:
            tensor_og = []
            if self.data.shape == 1:
                d = self.data.get_data()
                for enc in d.get_mol_encoder():
                    if enc:
                        tensor_og.append(enc)
            else:
                for d in self.data.get_data():
                    for enc in d.get_mol_encoder():
                        if enc:
                            tensor_og.append(enc)
            tensor_og = DataTensor(tensor_og, property=self.property)
        if type(data) == naphtha_props.data.data.DatapointList:
            datapoints = data.get_data()
            tensor = []
            for d in data.get_data():
                for enc in d.get_mol_encoder():
                    if enc:
                        tensor.append(enc)
            tensor = DataTensor(tensor, property=self.property)

        # elif data.shape[0] != tensor_og.f_mols.shape[0]:
        #     if data.shape[0] % tensor_og.f_atoms.shape[0] != 0:
        #         print("help", data.shape[0], tensor_og.f_atoms.shape[0])
        #     L = []
        #     output = []
        #     for i in range(data.shape[0] // tensor_og.f_atoms.shape[0]):
        #         L.append(data[tensor_og.f_atoms.shape[0]*i:tensor_og.f_atoms.shape[0]*(i+1), :])
        #         output.append(self.forward(L[i]))
        #     output_tensor = output[0]
        #     if isinstance(data, np.ndarray):
        #         for i in range(1, data.shape[0] // tensor_og.f_atoms.shape[0]):
        #             output_tensor = np.vstack((output_tensor, output[i]))
        #         return output_tensor
        #     for i in range(1, data.shape[0] // tensor_og.f_atoms.shape[0]):
        #         output_tensor = torch.cat((output_tensor, output[i]), 0)
        #     return output_tensor
        input_type = None
        if isinstance(data, np.ndarray):
            input_type = "array"
            L = []
            for i in range(data.shape[0]):
                pseudo_data = self.convert_types_to_molecules(data[i], self.data.get_data())
                L.append(pseudo_data)
            data = DatapointList(L)
            datapoints = data.get_data()
            tensor = []
            for d in data.get_data():
                for enc in d.get_mol_encoder():
                    if enc:
                        tensor.append(enc)
            tensor = DataTensor(tensor, property=self.property)
            # data_tensor = torch.from_numpy(data)
            # tensor = []
            # if self.data.shape == 1:
            #     d = self.data.get_data()
            #     for enc in d.get_mol_encoder():
            #         if enc:
            #             tensor.append(enc)
            # else:
            #     for d in self.data.get_data():
            #         for enc in d.get_mol_encoder():
            #             if enc:
            #                 tensor.append(enc)
            # tensor = DataTensor(tensor, property=self.property)
            # tensor.f_atoms = data_tensor
        elif isinstance(data, torch.Tensor):
            data = data.detach().numpy()
            L = []
            for i in range(data.shape[0]):
                pseudo_data = self.convert_types_to_molecules(data[i], self.data.get_data())
                L.append(pseudo_data)
            data = DatapointList(L)
            datapoints = data.get_data()
            tensor = []
            for d in data.get_data():
                for enc in d.get_mol_encoder():
                    if enc:
                        tensor.append(enc)
            tensor = DataTensor(tensor, property=self.property)
            # tensor = []
            # if self.data.shape == 1:
            #     d = self.data.get_data()
            #     for enc in d.get_mol_encoder():
            #         if enc:
            #             tensor.append(enc)
            # else:
            #     for d in self.data.get_data():
            #         for enc in d.get_mol_encoder():
            #             if enc:
            #                 tensor.append(enc)
            # tensor = DataTensor(tensor, property=self.property)
            # tensor.f_atoms = data
        elif type(data) == naphtha_props.data.data.DataTensor:
            tensor = data
        else:
            datapoints = data.get_data()
            tensor = []
            for d in data.get_data():
                for enc in d.get_mol_encoder():
                    if enc:
                        tensor.append(enc)
            tensor = DataTensor(tensor, property=self.property)

        # datapoints = data.get_data()

        if self.morgan:
            datapoints = data.get_data()
            tensor = []
            for d in data.get_data():
                for enc in d.get_morgan_fingerprint(radius=self.morgan_radius, nBits=self.morgan_size):
                    tensor.append(enc)
            tensor = torch.FloatTensor(tensor)
            mol_encoding = self.morgan_ffn(tensor)
            num_mols = len(datapoints[0].get_mol())
            sizes = list(mol_encoding.size())
            new = sizes[0] / num_mols
            sizes[1] = int(sizes[0] * sizes[1] / new)
            sizes[0] = int(new)
            input = mol_encoding.view(sizes)

        else:
            mol_encoding, atoms_vecs = self.mpn(tensor)
            if self.mix:
                features = data.get_fractions()
                features_2 = []
                scope = []
                count_total = 0
                for f in features:
                    count = 0
                    for k in f:
                        if k is not None:
                            features_2.append([k])
                            count += 1
                    scope.append((count_total, count))
                    count_total += count
                features_2 = torch.FloatTensor(features_2)
                if self.cudap or next(self.parameters()).is_cuda:
                    features_2 = features_2.cuda()
                mol_encoding = torch.mul(mol_encoding, features_2)
                mix_vecs = []
                for i_start, i_size in scope:
                    mix_vec = mol_encoding.narrow(0, i_start, i_size)
                    mix_vec = mix_vec.sum(dim=0)
                    mix_vecs.append(mix_vec)
                mix_vecs = torch.stack(mix_vecs, dim=0)
                input = mix_vecs
            else:
                num_mols = 1
                sizes = list(mol_encoding.size())
                new = sizes[0] / num_mols
                sizes[1] = int(sizes[0] * sizes[1] / new)
                sizes[0] = int(new)
                input = mol_encoding.view(sizes)

        if self.feature_size > 0:
            features = data.get_scaled_features()
            features = torch.FloatTensor(features)
            if self.cudap or next(self.parameters()).is_cuda:
                features = features.cuda()
            input = torch.cat([input, features], dim=1)

        input = input.type(torch.FloatTensor)
        output, logvars = self.ffn(input) #change for att
        del input

        for i in range(0, len(datapoints)):
            datapoints[i].scaled_predictions = output[i]

        return output, logvars
        # if input_type == "array":
        #     return output.detach().numpy()
        # return output
