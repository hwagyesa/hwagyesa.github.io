---
title: "Pytorch Speedrunning"
description: "a weak coder focusing on the wrong aspects :')"
date: 2024-10-09T16:48:20-07:00
draft: false
tags: ["research"]
showToc: true
TocOpen: false
hidemeta: false
comments: false
disableShare: true
hideSummary: true
searchHidden: false
ShowReadingTime: false
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: false
ShowRssButtonInSectionTermList: true
UseHugoToc: false
math: true
---

# Rationale

I don't think I'm _that_ bad at coding, but I'm certainly much slower than I'd
like to be. I did not really learn pytorch until 2022, using a mixture of adhoc
numpy and fast (but who cares) Matlab before that. For preparing for random ML
coding questions in interviews (and, in some useful sense, I think, improving my
general ability!), I'm trying to be a bit strategic and deliberate in my preparation.
In this post I'll record some notes for my own benefit on how a bog-standard
data-train-eval loop in pytorch can be blocked up into recognizable segments and
rolled out quickly from scratch. I am sure this is uninteresting to every good
ML experimenter out there; in writing it out this way for myself, I'm hoping it
will be a useful preparation reference, as well as something that reduces my
barrier to quick experimentation in new problem areas 😄

# The exercise: linear regression

<!-- prettier-ignore -->
We'll spin this out implementing an extremely simple linear regression problem
in pytorch. Something like the following data model:
$$ \boldsymbol y = \boldsymbol X \boldsymbol \beta_o + \sigma \boldsymbol g,$$
with $n$ observations, $d$ dimensions, noise standard deviation $\sigma > 0$,
and everything i.i.d. $\mathcal N(0, 1)$.
In the 'classical' regime where $n \geq d$, we have as usual that the problem
$$ \min_{\boldsymbol \beta} \frac{1}{2n} \|\boldsymbol y - \boldsymbol X \boldsymbol \beta\|_2^2$$
is solved in our random model (almost surely) by
$$ \boldsymbol \beta_{\star} = (\boldsymbol X^\top \boldsymbol X)^{-1} \boldsymbol X^\top \boldsymbol y.$$
It is not too hard to prove further that $\| \boldsymbol \beta_{\star} - \boldsymbol \beta_o \|_2 \lesssim \sqrt{\sigma^2 d / n}$ with overwhelming probability.
So everything should be well-posed when we implement the model and test with
a sample complexity of about $\sigma^2 d$.

For this data model, we will:

1. Code up the training loop.
2. Code up the dataset.
3. Code up the model.
4. Add in eval (with wandb; maybe Hydra).

I'll put notes on these sub-components in the sections below.

## Training loop

For interview coding, it seems to make sense to start by writing out the
'skeleton' of the training loop, then filling in the implementation-specific
details: the model, the data, and different eval components necessary.
This seems like a good structure to use as long as one has enough familiarity
with pytorch to know how to start abstractly.

The following basic structure makes sense to me as a starting point:

```python
def train(
    learning_rate = 1e-3,
    weight_decay = 0.,
    batch_size = 32,
    num_samples_val = ...,
    **kwargs
):
    # TODO:
    # 1. logging (wandb)
    # ...
    # 2. get data
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=False, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, drop_last=False, shuffle=True, batch_size=num_samples_val
    )

    # 3. get model
    model = ...

    # 4. training loop: loss function, optimizer, epoch and batch loop, batch processing.
    loss_fn = MSELoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    for epoch in range(num_epochs):
        for batch, (X, Y) in enumerate(
            train_dataloader
        ):
            loss = process_batch(model, X, Y, loss_fn, optimizer)
            # log stuff to wandb.
    # 5. Tests and logging.
    model.eval()
    # ...
    with torch.no_grad():
        for X, Y in val_dataloader:
            loss = process_batch(model, X, Y, loss_fn, optimizer, training=False)
            print(f"validation loss: {loss.detach()}")
    sleep(1)
```

For writing this out from nothing, one just needs to have an abstract sense of
what one is aiming to do. We'll be training some model with some kind of
gradient descent, and we might as well just use Adam; the objective is going to
be some loss function which we should be able to identify once we have
understood the problem we're trying to solve at a modeling level.
The "thinking" tasks at this level thus seem to be:

1. What's the loss function?
2. What are the parameters I need to pass in? Here I've already written out some
   of the parameters we'll need: optimizer parameters, and some parameters
   associated to the dataset and dataloader. We'll eventually need more dataset
   parameters, possibly, as well as model parameters. In general, we might also
   have parameters associated to logging or the loss function. These can be
   added in as the other sections of the loop skeleton are filled out.
3. What "work" do I need to do in the main loop, encapsulated in
   `process_batch` here? This function can take a pretty simple skeleton form:

```python
def process_batch(model, X, Y, loss_fn, optimizer, training=True):
    preds = model(X)
    loss = loss_fn(preds, Y)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss.detach()
```

We use the model to make predictions, then evaluate the loss. We accumulate
gradients with `backward`, then step the optimizer and zero gradients.

**_NOTE_**: it's helpful here to add some asserts to debug your tensor shapes! Even on
this simple linear regression problem, I spent some amount of time debugging
a case where my predictions variable was `(B, 1)`-shaped but my targets variable
was `(B,)`-shaped and broadcasting messed up the loss.

## Dataset component

The thing to remember here is just that a `Dataloader` (from `torch.utils.data`)
wraps a `Dataset` object (from the same module). For this simple problem, we can
create our own `Dataset` subclass (we'll do this momentarily). Remember the
following for the dataloader:

- Set the batch size here.
- If we're using CUDA, we can add the argument `pin_memory=True` and also use
  the `non_blocking=True` option in our `dataset.to(device)` calls that move
  data to the GPU. This can help performance.
- The other arguments to the dataloader are probably not essential.

### Implementing a dataset for our random linear regression model

Here's code for implementing the dataset, with "thought organization" to follow.

```python
class RandomDataset(Dataset):
    def __init__(
        self,
        data_dim: int,
        num_samples: int,
        ground_truth: torch.nn.Linear | None = None,
        device: torch.device = torch.device("cpu"),
        noise_std: float = 0.1,
        transform: Callable | None = None,
        # target_transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.data = torch.randn(
            (num_samples, data_dim), device=device, dtype=torch.float32
        )
        self.noise_std = noise_std
        if ground_truth is not None:
            self.ground_truth = ground_truth  # can't clone a pytorch module; need to deepcopy or manually
        else:
            self.ground_truth = torch.nn.Linear(
                data_dim, 1, device=device, dtype=torch.float32
            )
            torch.nn.init.normal_(self.ground_truth.weight)
            torch.nn.init.zeros_(self.ground_truth.bias)
            for parameter in self.ground_truth.parameters():
                parameter.requires_grad = False
        with torch.no_grad():
            self.targets = self.ground_truth(self.data) + self.noise_std * torch.randn(
                (num_samples, 1), device=device, dtype=torch.float32
            )
        self.transform = transform

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.data[idx, :]
        if self.transform is not None:
            data = self.transform(data)
        return (data, self.targets[idx])
```

High-level points:

- We subclass `torch.utils.data.Dataset` when we make our own dataset.
- A `torch.utils.data.Dataset` needs to implement `__init__`, `__len__`, and
  `__getitem__`. These are all instance methods; `__getitem__` also takes an
  index (`0` to `__len__() - 1`) and returns a tuple containing the indexed
  sample and its target/label. _These should each be `1D` tensors_.
- For writing the code, we can sort of just write the `__init__` function and
  think about what we need to set up for the data, then pretty easily do the
  other two functions.
- Set device and dtype for things.
- To make things comparable, even though it's most natural to implement the
  model's ground truth parameters with inline matrix operations, I used
  a `torch.nn.Linear` layer here (which will parallel what we do with the model
  below). For this, the arguments are `(in_features, out_features, ...)`. For
  initialization, we use `torch.nn.init`; functions typically have a signature
  like `..._`, and we just apply them to the module's parameters. The linear
  layer's parameters are called `weight` and `bias`. We also set the parameters'
  `requires_grad` to false, and include `torch.no_grad()` context when we
  generate the targets (ChatGPT says this tells pytorch not to track derived
  operations in the computational graph and saves memory).
- I'm including transform (for normalization and data augmentation) here just to
  be consistent with future stuff; will need this for more complicated
  implementations (e.g., CIFAR10).

## The model

Here's the model code, which is already familiar because of how we generated our
dataset above.

```python

class LinearModel(torch.nn.Module):
    def __init__(
        self, data_dim: int, device: torch.device = torch.device("cpu")
    ) -> None:
        super().__init__()
        self.linear = Linear(data_dim, 1, device=device, dtype=torch.float32)
        torch.nn.init.normal_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x) -> torch.Tensor:
        return self.linear(x)
```

Some short notes (reference the above):

- A pytorch model subclasses `torch.nn.Module`. It needs to implement
  `__init__`, where we set up its parameters and structure, and `forward`, an
  instance method that takes an input `x` and generates the layer output from
  it.

## Putting it together

Here's our overall train loop, callable from the command line.

```python

def train(
    data_dim,
    num_samples_train,
    num_samples_val,
    device,
    noise_std=0.1,
    batch_size=32,
    num_epochs=128,
    **kwargs,
):
    # 1. logging (wandb)
    # We could do this with hydra. Chatgpt recommended a kwargs dict structure but hydra is better. Too much rewriting config table.
    # 2. get data
    train_dataset = RandomDataset(
        data_dim=data_dim,
        num_samples=num_samples_train,
        noise_std=noise_std,
        device=device,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=False, shuffle=True
    )
    val_dataset = RandomDataset(
        data_dim=data_dim,
        num_samples=num_samples_val,
        noise_std=noise_std,
        ground_truth=train_dataset.ground_truth,
    )
    val_dataloader = DataLoader(
        val_dataset, drop_last=False, shuffle=True, batch_size=num_samples_val
    )
    # 3. get model
    model = LinearModel(data_dim, device)
    # 4. train
    loss_fn = MSELoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1, weight_decay=0.0)
    model.train()
    for epoch in range(num_epochs):
        for batch, (X, Y) in enumerate(
            train_dataloader
        ):  # enumerate(iterable) gives us the batch idx.
            loss = process_batch(model, X, Y, loss_fn, optimizer)
            print(f"batch {batch}, epoch {epoch}, loss: {loss.item()}")
    # 5. Tests and logging.
    model.eval()
    print(
        f"parameter error: {torch.sum((model.linear.weight - train_dataset.ground_truth.weight)**2)}"
    )
    with torch.no_grad():
        for X, Y in val_dataloader:
            loss = process_batch(model, X, Y, loss_fn, optimizer, training=False)
            print(f"validation loss: {loss.detach()}")
    sleep(1)


def process_batch(model, X, Y, loss_fn, optimizer, training=True):
    preds = model(X)
    loss = loss_fn(preds, Y)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss.detach()


if __name__ == "__main__":
    num_samples_train = 128
    config = {
        "data_dim": 16,
        "num_samples_train": num_samples_train,
        "num_samples_val": 128,
        "noise_std": 0.0,
        "batch_size": num_samples_train,
        "num_epochs": 2**10,
        "device": torch.device("cpu"),
    }
    train(**config)
```

We can write all this code in a bottom-up format without losing too much
momentum!

# Some concluding thoughts

- I didn't include much logging code above. It would be best to do this with
  wandb and hydra, but I guess there are not parameters to do this in an
  interview.
- The code above is not particularly optimized. ChatGPT gives some
  recommendations about enabling mixed-precision training (seems to be mild
  overhead) and implementing incremental logging. It would be fun to implement
  this exercise in a distributed training setting, as well.
- For doing more complex stuff, it probably makes sense to think about a way to
  write incremental tests (e.g., test the dataset is doing the right thing, test
  the model is doing the right thing) as one goes. Otherwise when something
  breaks it is a little bit annoying to drill down on what it is in this kind of
  "bottom up" implementation scheme.
