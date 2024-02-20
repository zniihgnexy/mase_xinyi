## Lab 3 for Advanced Deep Learning Systems (ADLS)

#### 1. Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.

###### for counting model size: Using model size as a quality metric, i defined a function as follows:
```python
def get_model_size(mg):
    model_size = 0
    for param in mg.model.parameters():
        print('parameters of element size', param.nelement(), param.element_size())
        model_size += param.nelement() * param.element_size()
    return model_size

```
in the calculation here, i used the chop library to calculate the FLOPs, which is the calculate_modules function. the calculation of different layers are as follows:

(1) for the linear layer:
```python
batch = in_data[0].numel() / in_data[0].shape[-1]
computations = module.weight.numel() * batch
backward_computations = module.weight.numel() * batch
```
(2) for the relu layer:
```python
"computations": in_data[0].numel(),
"backward_computations": in_data[0].numel(),
```
(3) for the batch_norm1d layer:
```python
total_parameters = 2 * module.num_features
computations = 4 * in_data[0].numel()
backward_computations = 4 * in_data[0].numel()
```

these are the three layers i used here.

in my functions, i used my self-defined functions to calculate the model size and FLOPs, and the results are as follows:

(1) model size and FLOPs:
![](lab3pic\modelsize.png)

![](lab3pic\flopdata.png)

from the results above, the "computations" is the FLOPs, and the "total_parameters" is the model size. in order to calculate the full size of the model ,we need to add these numbers together, and the results are as follows:

model size: 468
FLOPs: 1280.0

#### 2. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric. It’s important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).

###### for calculating latency: i defined the latency by using time module in python, and the latency is calculated as follows:

```python
for inputs in data_module.train_dataloader():
    xs, ys = inputs
    start = time.time()
    preds = new_mg.model(xs)
    end = time.time()
    loss = nn.functional.cross_entropy(preds, ys)
    acc = metric(preds, ys)
    accs.append(acc)
    losses.append(loss)
    if j > num_batchs:
        break
    j += 1
```

I simply used the time.time() function to calculate the time difference between the start and end of the model running, and then added them together to get the total latency. Considering the runtime also affected by calculation of accuracy and loss, i only use time module around prediction function, which i only estimate the value of data prediction.

And the results are as follows:

![](lab3pic\accandlatency.png)

Accuracy and loss actually serve as the same quality metric, this is because: 

In this case, we use cross entropy function to calculate the loss, which function is:

![](lab3pic\crossentropy.png)

And the accuracy calculation function is from MulticlassAccuracy Class, because jsc-tiny network is a 5-class classification network. And the calculation of multi-class accuracy is as follows:

![](lab3pic\accuracy.png)

In this particular question, the classification question's accuracy and loss are both using one-hot number for the calculation. In this case, 



#### 3. Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.

using the brute-force search, i defined the method using the optuna.py file already existed in the MASE repo, and the code is as follows:

```python
def sampler_map(self, name):
    match name.lower():
        case "random":
            sampler = optuna.samplers.RandomSampler()
        case "tpe":
            sampler = optuna.samplers.TPESampler()
        case "nsgaii":
            sampler = optuna.samplers.NSGAIISampler()
        case "nsgaiii":
            sampler = optuna.samplers.NSGAIIISampler()
        case "qmc":
            sampler = optuna.samplers.QMCSampler()
        case "brute_force":
            sampler = optuna.samplers.BruteForceSampler()
        case _:
            raise ValueError(f"Unknown sampler name: {name}")
    return sampler
```

in this function, i use the BruteforceSampler() function to define the sampler, referencing the optuna open-source library.

using this specific method by using the command line, i also modified the .toml file as follows:

```python
[search.strategy.setup]
n_jobs = 1
n_trials = 20
timeout = 20000
sampler = "brute-force"
sum_scaled_metrics = false # multi objective
```

Here i used "brute-force" sampler for strategy, so this file can directly use the sampler written in optuna.py file, which is BruteforceSampler() function.

#### 4. Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods.

compare these two methods, i only need to change the sampler in the .toml file:

```python
# when using brute force
sampler = "brute-force"
# when using tpe
sampler = "tpe"
```

i found that the brute-force search is much slower than the TPE based search, and the results are as follows:

This is the TPE strategy output:
![](lab3pic\searchtpe13.png)

in this part, i change the search strategies (sampler) to brute-force, and the results are as follows:

This is the brute-force output:

![](lab3pic\searchbrute13.png)

From the two pictures above, we can see the differences between two different methods. 

When using TPE method, we can see that the running time doesn't have that much difference, but brute-force method runs one second longer. Also, the searchoing results are different. In software metrics calculation, we can see that software metrics calculation using TPE can reach a higher accuracy and lower loss.

this means, when using different search methods, the sample effeciency is different.
When using the brute-force search, the sample effeciency is much lower than the TPE based search, and the reason is that the brute-force search is a method that enumerates all possible combinations of the search space, and then evaluates each combination, and then selects the best combination. This method is very time-consuming, and the sample effeciency is very low. 

However, the TPE based search is a method that uses the Bayesian optimization algorithm to find the best combination, which is much faster than the brute-force search, and the sample effeciency is much higher than the brute-force search.