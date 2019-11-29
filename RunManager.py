import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#from tensorboardcolab import *
from IPython.display import display, clear_output
import pandas as pd
import time
import json
import matplotlib.pyplot as plt

from itertools import product
from collections import namedtuple
from collections import OrderedDict

from sklearn.metrics import confusion_matrix
import plotcm
import PIL.Image
from torchvision.transforms import ToTensor

import tensorflow as tf

import os


class ModelSummary():
    def __init__(self, valid_accuracy=0.0, test_accuracy=0.0, run_params=None, network_rpr=None):
        self.valid_accuracy = valid_accuracy
        self.test_accuracy = test_accuracy
        self.run_params = run_params
        self.network_rpr = network_rpr

    def __lt__(self, other):
        return (self.valid_accuracy, self.test_accuracy) < (other.valid_accuracy, other.test_accuracy)

    def __str__(self):
        return 'Valid_accuracy:%f Test_accuracy:%f \nRun_params:\n %s \nNetwork_rpr:\n%s' % ( self.valid_accuracy, self.test_accuracy, self.run_params, self.network_rpr)

    def __repr__(self):
        return 'Valid_accuracy:%f Test_accuracy:%f \nRun_params:\n %s \nNetwork_rpr:\n%s' % ( self.valid_accuracy, self.test_accuracy, self.run_params, self.network_rpr)

class Epoch():
    def __init__(self):
        self.count = 0
        self.loss = 0
        self.num_correct = 0
        self.start_time = None

class RunManager():
    def __init__(self):

        self.epoch = Epoch()

        self.run_params = None
        self.run_count = 0
        self.run_duration = 0;

        self.testLoss = 0
        self.testCorrect = 0

        self.validLoss = 0
        self.validCorrect = 0
        self.valid_split = None

        self.run_data = []
        self.run_start_time = None

        self.names = None
        self.network = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.device = 'cpu'
        self.tb = None

        self.best_test_accuracy = 0
        self.best_models = []

    def begin_run(self, run, network, device, train_loader, valid_loader=None, test_loader=None, valid_split = 0.1, names=None, final_train=False ):
        self.run_start_time = time.time()

        self.final_train=final_train
        self.run_params = run
        self.run_count += 1
        self.testLoss = 0
        self.validLoss = 0

        self.valid_split = valid_split

        self.names = names
        self.network = network
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        #self.tb = SummaryWriter(log_dir=os.path.join(self.base_folder, 'runs'), comment=f'-{run}', filename_suffix=f'-{run}')
        final_train_prefix = '' if self.final_train else '[FT]'
        self.tb = SummaryWriter(comment=f'-{final_train_prefix}-{run}')

        images, labels = next(iter(self.train_loader))
        images, labels = images[:100], labels[:100]
        images, labels = images.to(self.device), labels.to(self.device)
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)

    def end_run(self):
        plot_buf, figure = plotcm.plot_confusion_matrix(self.cm, self.names)
        plt.close(figure)

        image = PIL.Image.open(plot_buf)
        image = ToTensor()(image).unsqueeze(0)

        grid = torchvision.utils.make_grid(image, normalize=True, scale_each=True)
        self.tb.add_image('confusion_matrix', grid)
        valid_accuracy = 0.0
        if self.valid_loader is not None:
            valid_accuracy=self.validCorrect / len(self.valid_loader.sampler)
        test_accuracy = 0.0
        if self.test_loader is not None:
            test_accuracy=self.testCorrect / len(self.test_loader.sampler)
        self.best_models.append(ModelSummary(valid_accuracy=valid_accuracy, test_accuracy=test_accuracy,
                                             run_params=self.run_params, network_rpr=str(self.network)))

        self.tb.close()
        self.epoch.count = 0

    def begin_epoch(self):
        self.epoch.start_time = time.time()

        self.epoch.count += 1
        self.epoch.loss = 0
        self.epoch.num_correct = 0
        self.testLoss = 0
        self.testCorrect = 0
        self.validLoss = 0
        self.validCorrect = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch.start_time
        run_duration = time.time() - self.run_start_time
        self.run_duration = run_duration

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch.count
        results["network"] = self.network.name

        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration

        if self.train_loader is not None:
            loss = self.epoch.loss / len(self.train_loader.sampler)
            accuracy = self.epoch.num_correct / len(self.train_loader.sampler)
            self.tb.add_scalar('Loss/Train', loss, self.epoch.count)
            self.tb.add_scalar('Accuracy/Train', accuracy, self.epoch.count)
            results["loss"] = loss
            results["accuracy"] = accuracy

        if self.valid_loader is not None:
            valid_loss = self.validLoss / len(self.valid_loader.sampler)
            valid_accuracy = self.validCorrect / len(self.valid_loader.sampler)
            self.tb.add_scalar('Loss/Valid ', valid_loss, self.epoch.count)
            self.tb.add_scalar('Accuracy/Valid', valid_accuracy, self.epoch.count)
            results["valid loss"] = valid_loss
            results["valid accuracy"] = valid_accuracy

        if self.test_loader is not None:
            test_loss = self.testLoss / len(self.test_loader.dataset)
            test_accuracy = self.testCorrect / len(self.test_loader.dataset)
            self.tb.add_scalar('Loss/Test', test_loss, self.epoch.count)
            self.tb.add_scalar('Accuracy/Test', test_accuracy, self.epoch.count)
            results["test loss"] = test_loss
            results["test accuracy"] = test_accuracy


        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch.count)
        self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch.count)

        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        clear_output(wait=True)
        display(df.tail(5))

    def _calculate_loss_per_loader(self, loss, data_loader):
        return loss.item() * data_loader.batch_size

    def track_loss(self, loss):
        self.epoch.loss += self._calculate_loss_per_loader(loss, self.train_loader)

    def track_num_correct(self, preds, labels):
        self.epoch.num_correct += self._get_num_correct(preds, labels)

    def track_test_loss_old(self, loss):
        self.testLoss += loss.item() * self.test_loader.batch_size

    def track_test_num_correct_old(self, preds, labels):
        self.testCorrect += self._get_num_correct(preds, labels)

    def track_test_loss(self, loss):
        self.testLoss += self._calculate_loss_per_loader(loss, self.test_loader)

    def track_test_num_correct(self, preds, labels):
        self.testCorrect += self._get_num_correct(preds, labels)

    def track_valid_loss(self, loss):
        self.validLoss += self._calculate_loss_per_loader(loss, self.valid_loader)

    def track_valid_num_correct(self, preds, labels):
        self.validCorrect += self._get_num_correct(preds, labels)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    @torch.no_grad()
    def get_all_preds_labels(self, model, loader):
        all_preds = torch.tensor([], dtype=torch.int64, device=self.device)
        all_labels = torch.tensor([], dtype=torch.int64, device=self.device)
        for batch in loader:
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)

            preds = model(images)

            preds = torch.argmax(preds, dim=1)

            all_preds = torch.cat(
                (all_preds, preds)
                , dim=0
            )
            all_labels = torch.cat(
                (all_labels, labels)
                , dim=0
            )
        return all_preds, all_labels

    def calculate_confusion_matrix(self, loader=None):
        loader = self.valid_loader if loader is None else loader

        all_preds, all_labels = self.get_all_preds_labels(self.network, loader)
        #print(all_preds.shape, all_preds.shape)
        cm = confusion_matrix(all_labels.cpu().numpy(), all_preds.cpu().numpy())
        print(cm)
        self.cm = cm

    def calculate_test_loss(self):
        self.calculate_loss(self.test_loader, self.track_test_loss, self.track_test_num_correct)

    def calculate_valid_loss(self):
        self.calculate_loss(self.valid_loader, self.track_valid_loss, self.track_valid_num_correct)

    def calculate_loss(self, data_loader, track_loss_function, track_num_correct_function):
        with torch.no_grad():
            self.network.eval()
            for batch in data_loader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)

                preds = self.network(images)  # Pass batch
                loss = F.cross_entropy(preds, labels)  # Calculate Loss
                track_loss_function(loss)
                track_num_correct_function(preds, labels)
        self.network.train()

    def save(self, filename):
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
            ).to_csv(f'{filename}.csv')

        with open(f'{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)