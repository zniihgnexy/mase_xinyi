# Configuration for the `search_zero_cost` Experiment

## Basics
- `model`: The model used for the experiment, in this case, "vgg7".
- `dataset`: The dataset used for the experiment, in this case, "cifar10".
- `task`: The task for the model, in this case, "cls" (classification).
- `max_epochs`: The maximum number of epochs for training, set to 5.
- `batch_size`: The size of the batches for training, set to 512.
- `learning_rate`: The learning rate for the model, set to 1e-2.
- `accelerator`: The hardware accelerator used, in this case, "gpu".
- `project`: The project name, in this case, "team_project".
- `seed`: The seed for random number generation, set to 42.
- `log_every_n_steps`: The frequency of logging, set to log every 5 steps.

## Search Space
- `name`: The name of the search space, in this case, "graph/software/zero_cost".

### Search Space Setup
- `by`: The method to set up the search space, in this case, "name".

### Search Space Seed Default Config
- `name`: Indicates that layers are not quantized by default, set to "NA".

## Search Config for Vision
- `dataset`: The dataset used, default is "cifar10", but can be selected from [cifar10, cifar10-valid, cifar100, ImageNet16-120].
- `name`: The name of the model, in this case, "infer.tiny".
- `C`: The number of channels, set to 16.
- `N`: The number of cells, set to 5.
- `op_0_0` to `op_3_2`: The operations for each cell.
- `number_classes`: The number of classes, set to 10.

## Search Strategy
- The configuration for the search strategy is defined in this section.