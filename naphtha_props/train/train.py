import copy
import csv
import os
import math

from naphtha_props.inp import InputArguments
from naphtha_props.data.scaler import Scaler
from naphtha_props.data.data import DatapointList, DataTensor
from naphtha_props.data.splitter import Splitter
from naphtha_props.models.model import Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from typing import Callable, List, Union

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, StepLR
from tqdm import trange
import numpy as np
from naphtha_props.train.evaluate import evaluate, predict
import matplotlib.pyplot as plt
from logging import Logger
import pandas as pd
import os
import io, pkgutil


def train(model: nn.Module,
          data: DatapointList,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          n_iter: int = 0,
          logger: logging.Logger = None,
          inp: InputArguments = None) -> (int, float):
    debug = logger.debug if logger is not None else print

    model.train()
    # if inp.seed is not None:
    #    random.seed(inp.seed)
    # random.shuffle(data.get_data())
    data.shuffle()
    loss_sum, iter_count = 0, 0
    batch_size = inp.batch_size
    cuda = inp.cuda

    num_iters = len(data.get_data()) // batch_size * batch_size
    iter_size = batch_size

    for i in trange(0, num_iters, iter_size):
        if (i + iter_size) > len(data.get_data()):
            break
        batch = DatapointList(data.get_data()[i:i + batch_size])

        # make mol graphs and mol vectors now if saving memory
        # training will take longer because encoders have to be constructed again every time

        mask = torch.Tensor([[not np.isnan(x) for x in tb] for tb in batch.get_scaled_targets()])
        targets = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in batch.get_scaled_targets()])

        if next(model.parameters()).is_cuda or inp.cuda:
            mask = mask.cuda()
            targets = targets.cuda()

        # Run model
        model.zero_grad()
        preds, logvars = model(batch)
        loss = loss_func(preds, targets) * mask  # tensor with loss of all datapoints

        loss = loss.sum() / mask.sum()  # sum of all loss in batch
        loss_sum += loss.item()  # sum over all batches in one epoch, so we have loss per epoch
        iter_count += batch_size
        loss.backward()
        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()  # adjust learning rate every batch
        # debug(f"Update of gradients and learning rate, new LR: {scheduler.get_lr()[0]:10.2e}")
        # if isinstance(scheduler, NoamLR):
        #    scheduler.step()

        n_iter += batch_size
    return n_iter, loss_sum


def run_training(inp: InputArguments, all_data: DatapointList, logger: Logger):
    global train_data_list, val_data_list, test_data_list
    logger = logger.debug if logger is not None else print
    inp.num_mols = len(all_data[0].smiles)
    inp.f_mol_size = all_data[0].get_mol_encoder()[0].get_sizes()[2]
    inp.num_targets = len(all_data[0].targets)
    inp.num_features = len(all_data[0].features)
    logger(f"Total number of datapoints read is {len(all_data)}")
    logger(f"Number of molecules: {len(all_data[0].mol)}, features: {len(all_data[0].features)}, "
           f"targets: {len(all_data[0].targets)}")
    initial_seed = inp.seed
    seed = inp.seed
    # if inp.gpu is not None and inp.cuda:
    #     torch.cuda.set_device(inp.gpu)
    test_scores = dict()
    splitter = Splitter(seed=seed, split_ratio=inp.split_ratio)
    if inp.split == "cross-validation":
        train_data_list, val_data_list, test_data_list = splitter.split_crossvalidation(all_data, inp.num_folds, inp.num_models)

    for fold in trange(inp.num_folds):
        seed = initial_seed + fold
        inp.seed = seed
        logger(f"Starting with fold number {fold} having seed {seed}")

        if inp.split == "random":
            train_data, val_data, test_data = splitter.split_random(all_data)
        elif inp.split == "scaffold":
            train_data, val_data, test_data = splitter.split_scaffold(all_data)
        elif inp.split == "cross-validation":
            train_data, val_data, test_data = train_data_list[0][0], val_data_list[0][0], test_data_list[0][0]
        elif inp.split == "train_only":
            train_data, val_data, test_data = all_data, all_data, all_data
        else:
            raise ValueError("splitter not supported")
        logger(f"Splitting data with {len(train_data)} training data, {len(val_data)} validation data and "
               f"{len(test_data)} test data points.")
        train_data_size = len(train_data)
        train_data, val_data, test_data = (DatapointList(d) for d in (train_data, val_data, test_data))

        scaler = Scaler(data=train_data, scale_features=inp.scale_features,
                        use_same_scaler_for_features=inp.use_same_scaler_for_features)
        if inp.scale == "standard":
            [scaler.transform_standard(d) for d in (train_data, val_data, test_data)]
            logger(f"Scaled data with {inp.scale} method")
            logger(f"mean: {scaler.mean[0]:5.2f}, std: {scaler.std[0]:5.2f}")
            df_train_target = pd.DataFrame(train_data.get_scaled_targets())
            df_train_target.to_csv("train_target.csv")
            df_test_target = pd.DataFrame(test_data.get_scaled_targets())
            df_test_target.to_csv("test_target.csv")
            df_val_target = pd.DataFrame(val_data.get_scaled_targets())
            df_val_target.to_csv("val_target.csv")
        elif inp.scale == "minmax":
            [scaler.transform_minmax(d) for d in (train_data, val_data, test_data)]
            logger(f"Scaled data with {inp.scale} method")
            logger(f"min: {scaler.min[0]:5.2f}, max: {scaler.max[0]:5.2f}")
        else:
            raise ValueError("scaler not supported")

        multiple_pretraining_paths = False
        if type(inp.pretraining_path) is not str and inp.pretraining:
            multiple_pretraining_paths = True
            all_pretraining_paths = inp.pretraining_path
            inp.num_models = len(all_pretraining_paths)

        for model_i in trange(inp.num_models):
            if inp.split == "cross-validation":
                train_data, val_data, test_data = train_data_list[fold][model_i], val_data_list[fold][model_i], \
                                                  test_data_list[fold][model_i]

                train_data, val_data, test_data = (DatapointList(d) for d in (train_data, val_data, test_data))
            path = os.path.join(inp.output_dir, f"fold_{fold}", f"model{model_i}")
            if path != '':
                os.makedirs(path, exist_ok=True)
            loss_func = get_loss_func(inp.loss_metric)
            logger(f"Initiate model {model_i}")
            model = Model(inp, logging)
            initialize_weights(model, seed=initial_seed + model_i)

            if inp.cuda:
                logger("Moving model to cuda")
                model = model.cuda()

            logger(f"Save checkpoint file to {path}/model.pt")
            save_checkpoint(os.path.join(path, 'model.pt'), model, inp, scaler)
            if inp.pretraining and inp.pretraining_path is not None:
                if multiple_pretraining_paths:
                    inp.pretraining_path = all_pretraining_paths[model_i]
                logger(f"Load pretraining parameters from {inp.pretraining_path}")

                if inp.only_load_mpn:
                    model_pretrained = load_checkpoint(inp.pretraining_path, current_inp=inp)
                    pretrained_state_dict = model_pretrained.state_dict()
                    new_state_dict = model.state_dict()
                    for key in pretrained_state_dict.keys():
                        if 'mpn' in key:
                            new_state_dict[key] = pretrained_state_dict[key]
                    # Load pretrained weights
                    model.load_state_dict(new_state_dict)
                else:
                    model = load_checkpoint(inp.pretraining_path, current_inp=inp)
                logger(f"Load scaler from pretrained dataset")
                if not inp.only_load_mpn:
                    scaler = load_scaler(inp.pretraining_path)
                    if inp.scale == "standard":
                        [scaler.transform_standard(d) for d in (train_data, val_data, test_data)]
                        logger(f"Scaled data with {inp.scale} method")
                        logger(f"mean: {scaler.mean[0]:5.2f}, std: {scaler.std[0]:5.2f}")
                    else:
                        raise ValueError("minmax scaling not supported for pretraining and model loading")
                if inp.pretraining_fix == "mpn":
                    for param in model.mpn.parameters():
                        param.requires_grad = False
                elif inp.pretraining_fix == "ffn":
                    for param in model.ffn.parameters():
                        param.requires_grad = False
                elif inp.pretraining_fix == "onlylast":
                    for param in model.mpn.parameters():
                        param.requires_grad = False
                    list_modules = [module for module in model.ffn.modules() if type(module) == nn.Linear]
                    for idx, module in enumerate(list_modules):
                        if idx < len(list_modules) - 1:
                            for param in module.parameters():
                                param.requires_grad = False

            if inp.pretraining:
                test_rmse, test_mae = process_results(path="pretraining_run", input=inp, model=model,
                                                      data=[train_data, val_data, test_data], scaler=scaler,
                                                      loss=[])

            # write representation of every molecule to a csv file
            write_learned_representation(model, [train_data, val_data, test_data], "mol_encoding_pretraining")

            optimizer = build_optimizer(model, inp.learning_rates[0])
            scheduler = build_lr_scheduler(optimizer, inp, train_data_size)

            best_score = float('inf') if inp.minimize_score else -float('inf')
            best_epoch = 0
            list_train_loss = []
            list_val_loss = []
            list_lr = []
            for epoch in trange(inp.epochs):
                logger(f"Starting epoch {epoch}/{inp.epochs}")
                (n_iter, loss_sum) = train(model=model, data=train_data, loss_func=loss_func,
                                           optimizer=optimizer, scheduler=scheduler, inp=inp)
                list_train_loss.append(loss_sum)
                if isinstance(scheduler, ExponentialLR) or isinstance(scheduler, StepLR):
                    scheduler.step(epoch=epoch)
                    scheduler.step(epoch=epoch)
                list_lr.append(scheduler.get_lr()[0])
                logger(f"Evaluating validation set of epoch {epoch}/{inp.epochs}")
                val_scores = evaluate(model=model, data=val_data, metric_func=inp.loss_metric, scaler=scaler)
                avg_val_score = np.nanmean(val_scores)
                list_val_loss.append(avg_val_score)
                # logger(f"Scores = {val_scores}")
                logger(f"Average validation score = {avg_val_score:5.2f}")
                # save model with lowest validation score for test
                if (inp.minimize_score and avg_val_score < best_score) or \
                        (not inp.minimize_score and avg_val_score > best_score):
                    logger(f"New best score")
                    best_score, best_epoch = avg_val_score, epoch
                    logger(f"Saving best model to {path}/model.pt")
                    try:
                        save_checkpoint(os.path.join(path, 'model.pt'), model, inp, scaler)
                    except PermissionError:
                        print("skipped writing")
                        pass

            logger(f"Loading best model, from epoch {best_epoch} with validation score {best_score:5.2f}")
            model = load_checkpoint(path, inp)

            # write representation of every molecule to a csv file
            write_learned_representation(model, [train_data, val_data, test_data], "mol_encoding_transfer")

            if inp.print_weigths:
                for param in model.mpn.W_i.parameters():
                    print(param)
            if inp.write_results:
                test_rmse, test_mae = process_results(path=path, input=inp, model=model,
                                                      data=[train_data, val_data, test_data], scaler=scaler,
                                                      loss=[list_train_loss, list_val_loss])

        inp.seed = initial_seed

    if inp.write_results:
        write_ensemble_summary(inp)


def write_learned_representation(model: nn.Module, data: DatapointList, filename):
    tensor = []
    smiles = []
    target = []
    for i in range(3):
        for d in data[i].get_data():
            smiles.append(d.smiles)
            target.append(d.targets)
            for enc in d.get_mol_encoder():
                if enc:
                    tensor.append(enc)
    tensor = DataTensor(tensor, property="solvation")
    mol_encoding, atoms_vecs = model.mpn(tensor)
    a = np.hstack((np.array(smiles).reshape(-1, 1), mol_encoding.detach().numpy()))
    b = np.hstack((np.array(target).reshape(-1, 9), a))
    df = pd.DataFrame(b)
    df.to_csv(filename+".csv")



def write_ensemble_summary(inp: InputArguments):
    names = ["val", "test"]
    for i in names:
        df2 = pd.DataFrame()
        for fold in range(inp.num_folds):
            df = pd.DataFrame()
            prediction_columns = []
            for model in range(inp.num_models):
                path = os.path.join(inp.output_dir, f"fold_{str(fold)}", f"model{str(model)}")
                with open(os.path.join(path, f"{i}_summary.csv"), 'r') as f:
                    df_temp = pd.read_csv(f)
                    new_prediction_columns = [f'{c}_m{model}' for c in df_temp.columns if 'pred' in c]
                    if not 'smiles' in df.columns:
                        df['smiles'] = df_temp.smiles
                        df['inchi'] = df_temp.inchi
                        for t in range(inp.num_targets):
                            df[f'target_{t}'] = df_temp[f'target_{t}']
                    for t, n in zip(range(inp.num_targets), new_prediction_columns):
                        df[n] = df_temp[f'prediction_{t}']
                    prediction_columns.extend(new_prediction_columns)
            df2 = pd.concat([df2, df])
        fig, ax = plt.subplots(nrows=1, ncols=inp.num_targets, figsize=(5 * inp.num_targets, 5), squeeze=False)
        for t in range(inp.num_targets):
            columns = [c for c in df2.columns if f'prediction_{t}' in c]
            df2[f"average_prediction_{t}"] = df2[columns].mean(axis=1)
            df2[f"ensemble_variance_{t}"] = df2[columns].var(axis=1)
            ax[0, t].plot(df2[f"target_{t}"], df2[f"average_prediction_{t}"], '.')
            ax[0, t].set(xlabel=f'target_{t}', ylabel=f'prediction_{t}')
        fig.savefig(os.path.join(inp.output_dir, f'{i}_parity.png'))
        plt.tight_layout()
        plt.close()
        df2.to_csv(os.path.join(inp.output_dir, f"{i}_ensemble_summary.csv"))


def initialize_weights(model: nn.Module, seed=0):
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            torch.manual_seed(seed)
            nn.init.xavier_normal_(param)


def build_optimizer(model: nn.Module, init_lr):
    params = [{'params': model.parameters(), 'lr': init_lr, 'weight_decay': 0}]
    return Adam(params)


def build_lr_scheduler(optimizer: Optimizer, input, train_data_size) -> _LRScheduler:
    """
    Builds a learning rate scheduler.
    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: Arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    if input.lr_scheduler == "Noam":
        return NoamLR(
            optimizer=optimizer,
            warmup_epochs=[input.warm_up_epochs],
            total_epochs=[input.epochs],
            steps_per_epoch=train_data_size // input.batch_size,
            init_lr=[input.learning_rates[0]],
            max_lr=[input.learning_rates[2]],
            final_lr=[input.learning_rates[1]]
        )
    elif input.lr_scheduler == "Step":
        return StepLR(optimizer, step_size=input.step_size, gamma=input.step_decay)
    elif input.lr_scheduler == "Exponential":
        return ExponentialLR(optimizer, gamma=input.exponential_decay)
    else:
        raise ValueError(f'Learning rate scheduler "{input.lr_scheduler}" not supported.')


def save_checkpoint(path: str,
                    model: Model,
                    inp: InputArguments,
                    scaler: Scaler = None,
                    ):
    """
    Saves a model checkpoint.
    :param model: A MoleculeModel.
    :param scaler: A StandardScaler fitted on the data.
    :param features_scaler: A StandardScaler fitted on the features.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        'input': inp,
        'state_dict': model.state_dict(),
        'data_scaler': {
            'means': scaler.mean,
            'stds': scaler.std
        } if scaler is not None else None,
        'features_scaler': {
            'means': scaler.mean_features,
            'stds': scaler.std_features
        } if scaler is not None else None,
        'scale_features': inp.scale_features,
        'use_same_scaler_for_features': inp.use_same_scaler_for_features
    }
    torch.save(state, path)


def get_loss_func(metric):
    # todo check loss function that accounts as much for low values
    if metric == "rmse":
        return nn.MSELoss(reduction='none')
    if metric == "mae":
        return nn.L1Loss(reduction='none')
    if metric == "smooth":
        return nn.SmoothL1Loss()
    raise ValueError(f'Metric for loss function "{metric}" not supported.')


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


def load_checkpoint(path: str,
                    current_inp: InputArguments,
                    logger: logging.Logger = None,
                    from_package=False,
                    only_mpnn=False) -> Model:
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


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.
    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """

    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):
        """
        Initializes the learning rate scheduler.
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        """
        # assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
        #       len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None, epoch: int = None):
        """
        Updates the learning rate by taking a step.
        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


def create_logger(name: str, save_dir: str = None) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.
    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.
    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fh_v = logging.FileHandler(os.path.join(save_dir, 'logger.log'))
        fh_v.setLevel(logging.DEBUG)
        logger.addHandler(fh_v)

    return logger


def process_results(path: str, input: InputArguments, model: nn.Module, data: [], scaler: Scaler, loss: []):
    train_preds, train_ale_unc = predict(model=model, data=data[0], scaler=scaler)
    val_preds, val_ale_unc = predict(model=model, data=data[1], scaler=scaler)
    test_preds, test_ale_unc = predict(model=model, data=data[2], scaler=scaler)
    train_preds = np.array([np.array(i) for i in train_preds])
    val_preds = np.array([np.array(i) for i in val_preds])
    test_preds = np.array([np.array(i) for i in test_preds])
    num_targets = input.num_targets
    # fig, ax = plt.subplots()
    # train_data = DatapointList(data[0].get_data()[0:len(train_preds)])
    # for i in range(0,num_targets):
    #     ax.plot([d.targets[i] for d in train_data.get_data()], train_preds[:,i], 'b.')
    #     ax.set(xlabel='targets', ylabel='predictions')
    #     fig.savefig(os.path.join(path, f'train_parity_target{i}.png'))
    #     plt.close()
    #     fig, ax = plt.subplots()
    #     val_data = DatapointList(data[1].get_data()[0:len(val_preds)])
    #     ax.plot([d.targets[i] for d in val_data.get_data()], val_preds[:,i], 'b.')
    #     ax.set(xlabel='targets', ylabel='predictions')
    #     fig.savefig(os.path.join(path, f'val_parity_target{i}.png'))
    #     plt.close()
    #     fig, ax = plt.subplots()
    #     test_data = DatapointList(data[2].get_data()[0:len(test_preds)])
    #     ax.plot([d.targets[i] for d in test_data.get_data()], test_preds[:,i], 'b.')
    #     ax.set(xlabel='targets', ylabel='predictions')
    #     fig.savefig(os.path.join(path, f'test_parity_target{i}.png'))
    #     plt.close()

    train_data = DatapointList(data[0].get_data()[0:len(train_preds)])
    val_data = DatapointList(data[1].get_data()[0:len(val_preds)])
    test_data = DatapointList(data[2].get_data()[0:len(test_preds)])

    data2 = [train_data, val_data, test_data]
    write_summary(path=path, input=input, data=data2[0], name="train")
    write_summary(path=path, input=input, data=data2[1], name="val")
    write_summary(path=path, input=input, data=data2[2], name="test")

    with open(os.path.join(path, "summary.csv"), 'w+') as f:
        writer = csv.writer(f)
        row = ["name"]
        for i in range(num_targets):
            row.extend([f"rmse_{i}", f"mse_{i}", f"mae_{i}", f"max_{i}"])
        writer.writerow(row)
        names = ["train", "val", "test"]
        for i in range(0, len(names)):
            if len(data2[i].get_data()) > 0:
                row = [names[i]]
                _rmse = rmse(data=data2[i])
                _mse = mse(data=data2[i])
                _mae = mae(data=data2[i])
                _max = max(data=data2[i])
                for j in range(num_targets):
                    row.extend([_rmse[j], _mse[j], _mae[j], _max[j]])
                writer.writerow(row)
    if len(data2[2].get_data()) > 0:
        return rmse(data=data2[2]), mae(data=data2[2])
    else:
        return None, None


def write_summary(path: str, input: InputArguments, data: DatapointList, name: str):
    with open(os.path.join(path, name + "_summary.csv"), 'w+', encoding='utf-8') as f:
        writer = csv.writer(f)
        row = ["inchi"] * input.num_mols + ["smiles"] * input.num_mols
        row.extend([f'target_{i}' for i in range(input.num_targets)])
        row.extend([f'prediction_{i}' for i in range(input.num_targets)])
        if input.split == "scaffold":
            row += ["scaffold"]
        writer.writerow(row)
        for d in data.get_data():
            row = [m for m in d.inchi] + [m for m in d.smiles] + [t for t in d.targets] + [p for p in d.predictions]
            if input.split == "scaffold":
                row += [d.get_scaffold()[len(d.get_scaffold()) - 1]]
            writer.writerow(row)


def rmse(data: DatapointList):
    """
    Computes the root mean squared error.
    :param data:
    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    targets = np.array([np.array(d.targets) for d in data.get_data()])
    preds = np.array([np.array(d.predictions) for d in data.get_data()])
    num_targets = len(targets[0])
    return [math.sqrt(mean_squared_error(targets[:, i], preds[:, i])) for i in range(num_targets)]


def mse(data: DatapointList) -> float:
    """
    Computes the mean squared error.
    :param data:
    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed mse.
    """
    targets = np.array([np.array(d.targets) for d in data.get_data()])
    preds = np.array([np.array(d.predictions) for d in data.get_data()])
    num_targets = len(targets[0])
    return [mean_squared_error(targets[:, i], preds[:, i]) for i in range(num_targets)]


def mae(data: DatapointList) -> float:
    """
    Computes the mean squared error.
    :param data:
    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed mse.
    """
    targets = np.array([np.array(d.targets) for d in data.get_data()])
    preds = np.array([np.array(d.predictions) for d in data.get_data()])
    num_targets = len(targets[0])
    return [mean_absolute_error(targets[:, i], preds[:, i]) for i in range(num_targets)]


def r2(data: DatapointList) -> float:
    """
    Computes the mean squared error.
    :param data:
    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed mse.
    """
    targets = np.array([np.array(d.targets) for d in data.get_data()])
    preds = np.array([np.array(d.predictions) for d in data.get_data()])
    num_targets = len(targets[0])
    return [r2_score(targets[:, i], preds[:, i]) for i in range(num_targets)]


def max(data: DatapointList) -> float:
    """
    Computes the mean squared error.
    :param data:
    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed mse.
    """
    targets = np.array([np.array(d.targets) for d in data.get_data()])
    preds = np.array([np.array(d.predictions) for d in data.get_data()])
    num_targets = len(targets[0])
    return [np.max(np.abs(np.subtract(targets[:, i], preds[:, i]))) for i in range(num_targets)]


def avg_ale_unc(data: DatapointList):
    """
    Computes the root mean squared error.
    :param data:
    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    ale_uncs = [d.aleatoric_uncertainty for d in data.get_data()]
    avg_ale_unc = np.mean(ale_uncs)
    return avg_ale_unc


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


def load_input(path: str) -> InputArguments:
    return torch.load(path, map_location=lambda storage, loc: storage)['input']

