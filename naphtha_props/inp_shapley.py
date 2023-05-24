import torch


class InputArguments:
    def __init__(self):
        self.optimization = True
        self.dir = "C:/Users/yureel/OneDrive - UGent/PhD/Ab Initio/MPNN/"

        # TRANSFER LEARNING
        # self.input_file = self.dir + "ab_initio_database_carbenium_entropy_nofeat.csv"
        # self.model_path = self.dir + "run_1/none"
        # self.output_dir = self.dir + "run_1/results"
        # self.pretraining = True  # False retrain the pretraining model, True use the pretraining model
        # self.split = "cross-validation"  # random or scaffold or cross-validation

        # PRE-TRAINING
        self.pretraining = False  # False retrain the pretraining model, True use the pretraining model
        self.input_file = self.dir + "ab_initio_database_radical_BAC.csv"
        self.model_path = self.dir + "run_0/none"
        self.output_dir = self.dir + "run_0/results"
        self.split = "random"  # random or scaffold or cross-validation or train_only

        # reading and processing data
        # self.input_file = self.dir + "ab_initio_database.csv"
        self.split_ratio = (0.8, 0.1, 0.1)
        self.seed = 0
        self.write_results = True
        self.scale = "standard"  # standard or minmax
        self.scale_features = True
        self.use_same_scaler_for_features = False

        # for featurization
        self.property = 'solvation'  # not used now
        self.add_hydrogens = False  # adds hydrogens to first column smiles
        self.mix = False  # features are fractions of the different molecules in the same order

        # for training
        self.num_folds = 10
        self.num_models = 9
        self.epochs = 40
        self.batch_size = 10
        self.loss_metric = "rmse"
        self.pretraining_path = [self.dir + f'run_0/results/fold_0/model0/model.pt'
                                 for j in range(self.num_models)]
        
        self.pretraining_fix = "onlylast"  # mpn or ffn or none or onlylast
        self.only_load_mpn = False  # only load mpnn parameters and don't scale targets with previous scalar
        self.cuda = True and torch.cuda.is_available()
        self.gpu = 4
        self.learning_rates = (0.0001, 0.00001, 0.0002)  # initial, final, max
        self.lr_scheduler = "Noam"  # Noam or Step or Exponential
        self.warm_up_epochs = 2.0  # you need min 1 with adam optimizer and Noam learning rate scheduler

        # in case of step
        self.step_size = 10 if self.lr_scheduler is 'Step' else None
        self.step_decay = 0.2 if self.lr_scheduler is 'Step' else None
        # in case of exponential
        self.exponential_decay = 0.1 if self.lr_scheduler is 'Exponential' else None
        self.minimize_score = True


        # for mpn
        self.depth = 3
        self.mpn_hidden = 600
        self.mpn_dropout = 0.00
        self.mpn_activation = "LeakyReLU"
        self.mpn_bias = False
        self.aggregation = 'norm'  # way to aggregate atom embeddings, using 'mean', 'sum' or 'norm'alized values
        self.aggregation_norm = 50 if self.aggregation == 'norm' else None

        self.morgan_fingerprint = False
        self.morgan_bits = 16 if self.morgan_fingerprint else None
        self.morgan_radius = 2 if self.morgan_fingerprint else None

        # for ffn
        self.ffn_hidden = 100
        self.ffn_num_layers = 3
        self.ffn_dropout = 0.00
        self.ffn_activation = "LeakyReLU"
        self.ffn_bias = True

        # results
        self.print_weigths = False

        # self.model_path = self.dir + "examples/train_wo_ions/scan_ffn/20_1000_wo_water/fold_1/model0/model.pt"
        self.postprocess = False

        # DONT CHANGE!
        self.num_mols = 2
        self.f_mol_size = 2
        self.num_targets = 1
        self.num_features = 0
