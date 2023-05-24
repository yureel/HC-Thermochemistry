import numpy as np
import random
import rdkit.Chem as Chem

class Splitter:
    """Allows to split the data, either keeping the order in the input or with a random split"""
    def __init__(self, seed=None, split_ratio=(0.8,0.1,0.1)):
        self.seed = seed
        self.x_train = split_ratio[0]
        self.x_test = split_ratio[2]
        self.x_val = split_ratio[1]
        self.solvent_inchis = []

    def split_random(self, data):

        data_shuffled = [i for i in data]
        n = len(data)
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(data_shuffled)
        #data_shuffled= data.shuffle(seed=self.seed)
        train_data, val_data, test_data = self.split(data_shuffled)
        return train_data, val_data, test_data

    def split(self, data):
        n = len(data)
        train_data = data[:int(n*self.x_train)]
        val_data = data[int(n*self.x_train):(int(n*self.x_train)+int(n*self.x_val))]
        test_data = data[(int(n*self.x_train)+int(n*self.x_val)):]
        return train_data, val_data, test_data

    def split_scaffold(self, data):
        # assume scaffold for solute and it is always the last column in the input

        scaffold_dict = dict()
        data_shuffled = [i for i in data]
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(data_shuffled)
        for d in data_shuffled:
            solute_scaffold = d.get_scaffold()[len(d.get_scaffold())-1]
            if solute_scaffold in scaffold_dict.keys():
                scaffold_dict[solute_scaffold].append(d)
            else:
                scaffold_dict[solute_scaffold] = [d]
        #sorted_dict = {k: v for k, v in sorted(scaffold_dict.items(), key=lambda item: item[1])}
        len_test = len(data) * self.x_test
        len_val = len(data) * self.x_val
        len_train = len(data) * self.x_train
        train_data = []
        val_data = []
        test_data = []
        for k in sorted(scaffold_dict, key=lambda k: len(scaffold_dict[k])):
            if len(test_data) < len_test:
                [test_data.append(i) for i in scaffold_dict.get(k)]
            elif len(val_data) < len_val:
                [val_data.append(i) for i in scaffold_dict.get(k)]
            else:
                [train_data.append(i) for i in scaffold_dict.get(k)]
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        return train_data, val_data, test_data

    def split_crossvalidation(self, data, folds, models):
        train_data_list = []
        test_data_list = []
        val_data_list = []

        data_shuffled = [i for i in data]
        n = len(data)
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(data_shuffled)
        for i in range(folds):
            test_data = data_shuffled[int(i*n * self.x_test):int((i+1)*n * self.x_test)]
            data_shuffled_reduced = data_shuffled[:int(i*n * self.x_test)]+data_shuffled[int((i+1)*n * self.x_test):]
            train_data_sublist = []
            test_data_sublist = []
            val_data_sublist = []
            for j in range(models):
                m = np.floor(int(n * self.x_val))
                if int((j+1)*m) % len(data_shuffled_reduced) == 0:
                    val_data = data_shuffled_reduced[int(j * m) % len(data_shuffled_reduced):]
                    train_data = data_shuffled_reduced[:int(j * m) % len(data_shuffled_reduced)]
                else:
                    val_data = data_shuffled_reduced[int(j*m) % len(data_shuffled_reduced):
                                                     int((j+1)*m) % len(data_shuffled_reduced)]
                    train_data = data_shuffled_reduced[:int(j*m) % len(data_shuffled_reduced)] +\
                                 data_shuffled_reduced[int((j+1)*m) % len(data_shuffled_reduced):]
                train_data_sublist.append(train_data)
                test_data_sublist.append(test_data)
                val_data_sublist.append(val_data)


            train_data_list.append(train_data_sublist)
            test_data_list.append(test_data_sublist)
            val_data_list.append(val_data_sublist)
        return train_data_list, val_data_list, test_data_list









