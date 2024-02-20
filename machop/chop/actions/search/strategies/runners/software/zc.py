import math
import torch
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.text import Perplexity
from torchmetrics import MeanMetric
from transformers import get_scheduler

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from .base import SWRunnerBase

import torch.nn.functional as F

from ....search_space.zero_cost_nas.foresight.pruners.predictive import find_measures


def get_optimizer(model, optimizer: str, learning_rate, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    match optimizer:
        case "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=learning_rate
            )
        case "adam":
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=learning_rate)
        case "sgd":
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=learning_rate)
        case _:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

    return optimizer


class RunnerZeroCost(SWRunnerBase):
    available_metrics = ("loss", "accuracy", "perplexity")

    def _post_init_setup(self) -> None:
        self.loss = MeanMetric().to(self.accelerator)
        self._setup_metric()

    def _setup_metric(self):
        if self.model_info.is_vision_model:
            match self.task:
                case "classification" | "cls":
                    self.metric = MulticlassAccuracy(
                        num_classes=self.dataset_info.num_classes
                    ).to(self.accelerator)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        elif self.model_info.is_nlp_model:
            match self.task:
                case "classification" | "cls":
                    self.metric = MulticlassAccuracy(
                        num_classes=self.dataset_info.num_classes
                    ).to(self.accelerator)
                case "language_modeling" | "lm":
                    self.metric = Perplexity().to(self.accelerator)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        else:
            raise ValueError(f"model type {self.model_info} is not supported.")

    def nlp_cls_forward(self, batch, model):
        batch.pop("sentence")
        batch = {
            k: v.to(self.accelerator) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        outputs = model(**batch)
        loss = outputs["loss"]
        logits = outputs["logits"]
        labels = batch["labels"]
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        self.metric(logits, labels)
        self.loss(loss)
        return loss

    def nlp_lm_forward(self, batch, model):
        raise NotImplementedError()

    def vision_cls_forward(self, batch, model):
        raise NotImplementedError()

    def forward(self, task: str, batch: dict, model):
        if self.model_info.is_vision_model:
            match self.task:
                case "classification" | "cls":
                    loss = self.vision_cls_forward(batch, model)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        elif self.model_info.is_nlp_model:
            match self.task:
                case "classification" | "cls":
                    loss = self.nlp_cls_forward(batch, model)
                case "language_modeling" | "lm":
                    loss = self.nlp_lm_forward(batch, model)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        else:
            raise ValueError(f"model type {self.model_info} is not supported.")

        return loss

    def compute(self) -> dict[str, float]:
        reduced = {"loss": self.loss.compute().item()}
        if isinstance(self.metric, Perplexity):
            reduced["perplexity"] = self.metric.compute().item()
        elif isinstance(self.metric, MulticlassAccuracy):
            reduced["accuracy"] = self.metric.compute().item()
        else:
            raise ValueError(f"metric {self.metric} is not supported.")
        return reduced

    def __call__(self, data_module, model, sampled_config) -> dict[str, float]:
        
        model = model.model()
        train_dataloader = data_module.train_dataloader()
        dataload_info = ('random', 1, 10)
        device = self.accelerator
        value = find_measures(model, 
                              train_dataloader,
                              dataload_info, # a tuple with (dataload_type = {random, grasp}, number_of_batches_for_random_or_images_per_class_for_grasp, number of classes)
                              device, 
                              loss_fn=F.cross_entropy, 
                              measure_names='grad_norm',
                              measures_arr=None)

        return value
