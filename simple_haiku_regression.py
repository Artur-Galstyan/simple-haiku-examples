import torch
import haiku as hk
from torch.utils import data
from torch.utils.data.dataset import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from datasets import load_dataset


class HousesData(Dataset):
    def __init__(self, file_path) -> None:
        raw_data = pd.read_csv(file_path)
        x = raw_data.values[:, :-1]
        y = raw_data.values[:, -1]
        min_max_scaler = MinMaxScaler()
        self.x = min_max_scaler.fit_transform(x)
        self.y = min_max_scaler.fit_transform(y.reshape(-1, 1))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_data():
    dataset = load_dataset(
        "inria-soda/tabular-benchmark", data_files="reg_num/houses.csv", split="train"
    )
    dataset.set_format("pandas")
    df = dataset[:]

    df.to_csv("houses_csv/houses.csv")
    houses_data = HousesData("houses_csv/houses.csv")
    training_data, test_data = torch.utils.data.random_split(
        houses_data,
        [int(len(houses_data) * 0.7), len(houses_data) - int(len(houses_data) * 0.7)],
    )

    train_dataloader = data.DataLoader(training_data, batch_size=32, shuffle=True)
    test_dataloader = data.DataLoader(test_data, batch_size=32, shuffle=True)
    return train_dataloader, test_dataloader


train_dataloader, test_dataloader = get_data()

sample = next(iter(train_dataloader))[0].numpy()
label = next(iter(train_dataloader))[1].numpy()
model = hk.without_apply_rng(hk.transform(lambda x: hk.Linear(output_size=1)(x)))


rng_key = jax.random.PRNGKey(42)
params = model.init(rng=rng_key, x=sample)
output = model.apply(x=sample, params=params)

learning_rate = 0.1
optimizer = optax.adam(learning_rate=learning_rate)
opt_state = optimizer.init(params)

compute_loss = lambda params, x, y: jnp.mean(
    optax.l2_loss(model.apply(x=x, params=params), y)
)

losses = []

for x, y in train_dataloader:
    x = x.numpy()
    y = y.numpy()
    losses.append(compute_loss(params, x, y))
    grads = jax.grad(compute_loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)


# plt.plot(losses)
# plt.show()

losses_test = []
for x, y in test_dataloader:
    x = x.numpy()
    y = y.numpy()
    losses_test.append(compute_loss(params, x, y))

plt.plot(losses_test)
plt.show()
