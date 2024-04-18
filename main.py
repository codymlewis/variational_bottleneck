from typing import Optional
import math
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import flax.linen as nn
import optax
import einops
import datasets
import sklearn.metrics as skm
from tqdm import trange


class VBTrainState(train_state.TrainState):
    "Extended train state to keep track of rng used in bottleneck"
    key: jax.Array


class VariationalBottleNeck(nn.Module):
    "The variational bottleneck layer"
    K: int = 256
    rng_collection: str = "bottleneck"

    @nn.compact
    def __call__(self, x, rng: Optional[jax.random.PRNGKey] = None):
        batch_size = x.shape[0]
        in_shape = x.shape[1:]

        if len(in_shape) > 1 and x.shape[1] > 1 and len(in_shape) != 3:
            x = nn.Conv(1, (1, 1), padding="SAME")(x)
        if len(x.shape) > 2:
            x = x.reshape(batch_size, -1)

        statistics = nn.Dense(2 * self.K)(x)
        mu = statistics[:, :self.K]
        std = nn.softplus(statistics[:, self.K:])
        if rng is None:
            rng = self.make_rng(self.rng_collection)
        eps = jax.random.normal(rng, std.shape, dtype=std.dtype)
        encoding = mu + eps * std
        x = nn.Dense(x.shape[1])(encoding)
        x = x.reshape((batch_size, *in_shape))
        return x, mu, std


class VBMLP(nn.Module):
    "A simple LeNet-300-100 with a variational bottleneck"
    classes: int = 10

    @nn.compact
    def __call__(self, x, training=False):
        x = einops.rearrange(x, 'b h w c -> b (h w c)')
        x = nn.Dense(300)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x, mu, std = VariationalBottleNeck()(x)
        x = nn.Dense(self.classes)(x)
        x = nn.softmax(x)
        if training:
            return x, mu, std
        return x


@jax.jit
def update_step(state, X, Y, beta: float = 1e-3):
    "Function for a step of training the model"
    vb_train_key = jax.random.fold_in(key=state.key, data=state.step)

    def loss_fn(params):
        logits, mu, std = state.apply_fn(params, X, training=True, rngs={'bottleneck': vb_train_key})
        logits = jnp.clip(logits, 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        ce_loss = -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
        vb_loss = -0.5 * einops.reduce(1 + 2 * jnp.log(std) - mu**2 - std**2, 'b x -> b', 'sum').mean() / jnp.log(2)
        return ce_loss + beta * vb_loss

    state = state.replace(key=vb_train_key)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state


def accuracy(state, X, Y, batch_size=1000):
    """
    Calculate the accuracy of the model across the given dataset

    Arguments:
    - model: Model function that performs predictions given parameters and samples
    - variables: Parameters and other learned values used by the model
    - X: The samples
    - Y: The corresponding labels for the samples
    - batch_size: Amount of samples to compute the accuracy on at a time
    """
    @jax.jit
    def _apply(batch_X, rng_key):
        return jnp.argmax(state.apply_fn(state.params, batch_X, rngs={'bottleneck': rng_key}), axis=-1)

    preds, Ys = [], []
    rng_key = state.key
    for i in range(0, len(Y), batch_size):
        rng_key = jax.random.fold_in(rng_key, data=i)
        i_end = min(i + batch_size, len(Y))
        preds.append(_apply(X[i:i_end], rng_key))
        Ys.append(Y[i:i_end])
    return skm.accuracy_score(jnp.concatenate(Ys), jnp.concatenate(preds))


def load_dataset():
    "Load the fmnist dataset https://arxiv.org/abs/1708.07747"
    ds = datasets.load_dataset("fashion_mnist")
    ds = ds.map(
        lambda e: {
            'X': einops.rearrange(np.array(e['image'], dtype=np.float32) / 255, "h (w c) -> h w c", c=1),
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    input_shape = (28, 28, 1)
    features['X'] = datasets.Array3D(shape=input_shape, dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    return {t: {k: ds[t][k] for k in ds[t].column_names} for t in ds.keys()}


if __name__ == "__main__":
    seed = 42
    batch_size = 128

    rng = np.random.default_rng(seed)
    dataset = load_dataset()
    model = VBMLP()
    params_key, vb_key = jax.random.split(jax.random.PRNGKey(seed))
    state = VBTrainState.create(
        apply_fn=model.apply,
        params=model.init(params_key, dataset['train']['X'][:1]),
        tx=optax.sgd(0.01),
        key=vb_key,
    )

    print("Training the model...")
    for _ in (pbar := trange(10)):
        idxs = np.array_split(
            rng.permutation(len(dataset['train']['Y'])), math.ceil(len(dataset['train']['Y']) / batch_size)
        )
        loss_sum = 0.0
        for idx in idxs:
            loss, state = update_step(state, dataset['train']['X'][idx], dataset['train']['Y'][idx])
            loss_sum += loss
        pbar.set_postfix_str(f"LOSS: {loss_sum / len(idxs):.5f}")

    print(f"Trained model accuracy: {accuracy(state, dataset['test']['X'], dataset['test']['Y']):%}")
