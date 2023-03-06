from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby

from simulai.models import AutoencoderVariational
from simulai.optimization import Optimizer
from simulai.metrics import L2Norm
from simulai.file import SPFile

dataset = load_dataset("mnist")

dataset.set_format("numpy")

train_data = dataset["train"]["image"]
test_data = dataset["test"]["image"]

train_label = dataset['train']['label']
test_label = dataset['test']['label']
labels = np.hstack([train_label, test_label])

max_value = train_data.max()
min_value = train_data.min()

train_data = (train_data - min_value)/(max_value - min_value)
test_data = (test_data - min_value)/(max_value - min_value)

indices = np.arange(train_data.shape[0] + test_data.shape[0])
labels_indices = np.stack([labels, indices], axis=1)
labels_indices_sorted = labels_indices[labels_indices[:,0].argsort()]

keys_groups = dict()
for k, group in groupby(labels_indices_sorted, key=lambda x: x[0]):
    keys_groups[k] = np.array(list(group))[:, 1].astype(int)


def model():

    from simulai.models import AutoencoderVariational

    autoencoder = AutoencoderVariational(
        input_dim=(None, 1, 28, 28),
        latent_dim=2,
        activation="relu",
        architecture="cnn",
        case="2d",
        shallow=True,
    )

    return autoencoder

mnist_ae = model()

mnist_ae.summary()

optimizer_config = {"lr": 1e-4}

params = {"lambda_1": 0.0, "lambda_2": 0.0, "use_mean": True, "relative": False, "beta": 250}

optimizer = Optimizer("adam", params=optimizer_config,
            lr_decay_scheduler_params={
                "name": "ExponentialLR",
                "gamma": 0.9,
                "decay_frequency": 2_00,
            },
            )

n_epochs = 10_000


optimizer.fit(
    op=mnist_ae,
    input_data=train_data,
    target_data=train_data,
    n_epochs=n_epochs,
    loss="vaermse",
    params=params,
    batch_size=1_00,
)

saver = SPFile(compact=False)
saver.write(
    save_dir=".",
    name="mnist_ae",
    model=mnist_ae,
    template=model,
)

test_data_eval = mnist_ae.eval(input_data=test_data[::10])

error = L2Norm()(data=test_data_eval, reference_data=test_data[::10],
                 relative_norm=True)
print(f"Error :{100*error} %")

n_samples = train_data.shape[0] + test_data.shape[0]

batches = np.array_split(np.arange(n_samples), int(n_samples / 1_00))
data = np.vstack([train_data, test_data])

projection_list = list()

for i, batch in enumerate(batches):
    print(f"Projecting batch {i}/{len(batches)}")
    projected = mnist_ae.project(input_data=data[batch])
    projection_list.append(projected)

projected = np.vstack(projection_list)

np.savez("projected.npz", projected=projected, keys_groups=keys_groups)


for k, batch in keys_groups.items():

    batch_data = projected[batch]

    plt.scatter(batch_data[:,0], batch_data[:,1], label=f"{k}")

plt.legend()

plt.savefig("mnist_embedding.png")

plt.show()
