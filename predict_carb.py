import pandas as pd
from naphtha_props.models.model import Model
import logging
import naphtha_props.inp as inp
import torch
import torch.nn as nn
from naphtha_props.inp import InputArguments
from naphtha_props.data.scaler import Scaler
from naphtha_props.data.data import DatapointList
from naphtha_props.data.data import read_data_from_df
import os
import io, pkgutil
import numpy as np


def predict(model: nn.Module,
            data: DatapointList,
            scaler: Scaler = None):
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()
    preds = []

    batch_size = 1
    num_iters = len(data.get_data()) // batch_size * batch_size
    iter_size = batch_size

    for i in range(0, num_iters, iter_size):
        if (i + iter_size) > len(data.get_data()):
            break
        batch = DatapointList(data.get_data()[i:i+batch_size])
        with torch.no_grad():
            pred, logvars = model(batch)
        pred = pred.data.cpu().numpy().tolist()
        preds.extend(pred)

    if num_iters == 0:
        pred, logvars = [], None

    for i in range(0, len(preds)):
        data.get_data()[i].scaled_predictions = preds[i]

    if scaler is not None:
        preds = scaler.inverse_transform(preds)

    for i in range(0, len(preds)):
        data.get_data()[i].predictions = preds[i]

    # preds = preds.tolist()

    return preds, None


def load_checkpoint(path: str,
                    current_inp: InputArguments,
                    logger: logging.Logger = None,
                    from_package=False) -> Model:
    """
    Loads a model checkpoint.
    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded MoleculeModel.
    """

    debug = logger.debug if logger is not None else print

    # Load model and args
    if from_package:
        state = torch.load(io.BytesIO(pkgutil.get_data('naphtha_props', path)),
                           map_location=lambda storage, loc: storage)
    else:
        if ".pt" not in path:
            path = os.path.join(path, "model.pt")

        state = torch.load(path, map_location=lambda storage, loc: storage)
    inp, loaded_state_dict = state['input'], state['state_dict']

    if current_inp is not None:
        args = current_inp
    # args.cuda = cuda if cuda is not None else args.cuda

    # Build model
    model = Model(args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():

        if param_name not in model_state_dict:
            debug(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            debug(f'Pretrained parameter "{param_name}" '
                  f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                  f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if args.cuda:
        debug('Moving model to cuda')
        model = model.cuda()

    return model


def load_scaler(path: str, from_package=False) -> Scaler:
    if from_package:
        state = torch.load(io.BytesIO(pkgutil.get_data('naphtha_props', path)),
                           map_location=lambda storage, loc: storage)
    else:
        if ".pt" not in path:
            path = os.path.join(path, "model.pt")
        state = torch.load(path, map_location=lambda storage, loc: storage)
    sc_fe = state['scale_features'] if 'scale_features' in state.keys() else True
    same_sc_fe = state['use_same_scaler_for_features'] if 'use_same_scaler_for_features' in state.keys() else True
    scaler = Scaler(mean=state['data_scaler']['means'], std=state['data_scaler']['stds'],
                    mean_f=state['features_scaler']['means'], std_f=state['features_scaler']['stds'],
                    scale_features=sc_fe,
                    use_same_scaler_for_features=same_sc_fe)
    return scaler


type = "carb"  # either "hc", "rad", "carb", for regular hydrocarbons, radicals or carbenium ions
inp = inp.InputArguments()
inp.input_file = "ab_initio_database_"+type+".csv"
inp.num_targets = 9
inp.num_features = 3
df, all_data = read_data_from_df(inp)
inp.f_mol_size = all_data[0].get_mol_encoder()[0].get_sizes()[2]
all_data = DatapointList(all_data)
pred_avg = np.zeros((all_data.shape, 9))
for i in range(5):
    for j in range(10):
        path = os.path.join(os.curdir, type+"_models/run_"+type+"_"+str(i)+"/results/fold_"+str(j)+"/model0/model.pt")
        print(path)
        scaler = load_scaler(path)
        scaler.transform_standard(all_data)
        model = load_checkpoint(path, current_inp=inp)
        print(predict(model, all_data, scaler=scaler)[0])
        pred_avg += predict(model, all_data, scaler=scaler)[0]
pred_avg = pred_avg/50
df_pred = pd.DataFrame(pred_avg, index=df.index, columns=["H_298", "S_298", "Cp_300", "Cp_400", "Cp_500", "Cp_600",
                                                          "Cp_800", "Cp_1000", "Cp_1500"])
df_pred.to_csv("prediction.csv")
