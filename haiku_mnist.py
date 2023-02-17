import torchvision
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import wandb

# wandb.init(project="FashionMNIST-Haiku", entity="littlehelpers")


# transforms
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


def one_hot_encode(label):
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoder.fit(np.arange(10).reshape(-1, 1))
    return one_hot_encoder.transform(np.array(label).reshape(1, 1)).reshape(-1)


BATCH_SIZE = 64


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# constant for classes
classes = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
)

# Load the FashionMNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(
    "fashion",
    train=True,
    download=True,
    transform=transform,
    target_transform=one_hot_encode,
)
test_dataset = torchvision.datasets.FashionMNIST(
    "fashion",
    train=False,
    download=True,
    transform=transform,
    target_transform=one_hot_encode,
)

trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)

dataiter = iter(trainloader)
images, labels = next(dataiter)

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
# matplotlib_imshow(img_grid, one_channel=True)


def get_metrics(logits: np.ndarray, labels: np.ndarray):
    return {
        "accuracy": get_batch_accuracy(logits, labels),
        "loss": softmax_cross_entropy_loss(logits, labels),
    }


@jax.jit
def softmax_cross_entropy_loss(logits: np.ndarray, labels: np.ndarray):
    return -jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))


def get_batch_accuracy(logits: np.ndarray, labels: np.ndarray):
    return jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1))


class CNN(hk.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def __call__(self, x):
        x = hk.Conv2D(16, kernel_shape=3, stride=1, padding="SAME")(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(8, kernel_shape=3, stride=1, padding="SAME")(x)
        x = jax.nn.relu(x)
        x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)

        x = hk.Flatten()(x)

        x = hk.Linear(64)(x)
        x = jax.nn.relu(x)

        x = hk.Linear(self.num_classes)(x)
        x = jax.nn.softmax(x)
        return x


def forward_fn(x):
    model = CNN()
    return model(x)


model = hk.transform(forward_fn)

# Initialize the model
rng_seq = hk.PRNGSequence(42)

images, labels = next(iter(trainloader))

params = model.init(next(rng_seq), jnp.ones(images.shape))
learning_rate = 0.001

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

N_EPOCHS = 10


for epoch in range(N_EPOCHS):
    for images, labels in trainloader:
        images = images.numpy()
        labels = labels.numpy()

        def loss_fn(params):
            logits = model.apply(params, None, images)
            return softmax_cross_entropy_loss(logits, labels)

        grads = jax.grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    logits = model.apply(params, None, images)
    metrics = get_metrics(logits, labels)
    # wandb.log(metrics)
    # print(metrics)

    print(
        f"Epoch {epoch} - loss: {metrics['loss']:.4f}, accuracy: {metrics['accuracy']:.4f}"
    )

# Test the model

for images, labels in testloader:
    images = images.numpy()
    labels = labels.numpy()

    logits = model.apply(params, None, images)
    metrics = get_metrics(logits, labels)
    print(metrics)

    print(f"Test - loss: {metrics['loss']:.4f}, accuracy: {metrics['accuracy']:.4f}")
