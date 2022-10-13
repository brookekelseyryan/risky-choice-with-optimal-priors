import jax
import jax.numpy as jnp
import optax
import flax
from flax import linen as nn
from flax.training import train_state
from flax.serialization import (
    to_state_dict, msgpack_serialize, from_bytes
)
import os
import wandb
import numpy as np
from typing import Callable
import tqdm
from tqdm.notebook import tqdm

from hurd.jax_utils import select_array_inputs
import math
from hurd.internal_datasets import load_c13k_data


class FlatModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=36, name="input")(x)
        x = nn.softmax(x)
        x = nn.Dense(features=36, name="dense1")(x)
        x = nn.softmax(x)
        x = nn.Dense(features=36, name="dense2")(x)
        x = nn.softmax(x)
        x = nn.Dense(features=2, name="output")(x)
        x = nn.softmax(x)
        return x

    def predict(self,
                state: train_state.TrainState,
                test_dataset,
                config):

        # wandb logging
        restored_state = load_checkpoint("checkpoint.msgpack", state)

        num_test_batches = math.ceil(len(test_dataset) / config.batch_size)
        test_batch_metrics = []
        test_datagen = test_dataset.iter_batch(batch_size=config.batch_size, random_state=config.seed)

        for (batch_idx, batch) in enumerate(tqdm(test_datagen, total=num_test_batches)):
            outcomes, probabilities, labels = batch.as_array(return_targets=True)
            problems = select_array_inputs(outcomes, probabilities)
            metrics = val_step(state, problems, labels)
            test_batch_metrics.append(metrics)

        test_batch_metrics = accumulate_metrics(test_batch_metrics)
        print(
            'Test: Loss: %.4f, accuracy: %.2f' % (
                test_batch_metrics['loss'],
                test_batch_metrics['accuracy'] * 100
            )
        )

        wandb.log({
            "Test Loss": test_batch_metrics['loss'],
            "Test Accuracy": test_batch_metrics['accuracy']
        })

        return state, restored_state

    def fit(self,
            train_dataset,
            val_dataset,
            state: train_state.TrainState,
            config):

        num_train_batches = math.ceil(len(train_dataset) / config.batch_size)
        num_val_batches = math.ceil(len(val_dataset) / config.batch_size)

        # check that wandb parameters have been set before proceeding
        if config is None or config.batch_size is None or config.epochs is None:
            raise ValueError("config.batch_size and config.epochs must be defined")

        "-----------------TRAINING LOOP--------------------"
        for epoch in tqdm(range(1, config.epochs + 1)):
            # initialize the best validation loss
            best_val_loss = 1e6

            "------------------TRAINING BATCH------------------"
            train_batch_metrics = []
            train_datagen = train_dataset.iter_batch(batch_size=config.batch_size, random_state=config.seed)

            for (batch_idx, batch) in enumerate(tqdm(train_datagen, total=num_train_batches)):
                outcomes, probabilities, labels = batch.as_array(return_targets=True)
                problems = select_array_inputs(outcomes, probabilities)
                state, metrics = train_step(state, problems, labels)
                train_batch_metrics.append(metrics)

            train_batch_metrics = accumulate_metrics(train_batch_metrics)
            print(
                'TRAIN (%d/%d): Loss: %.4f, accuracy: %.2f' % (
                    epoch, config.epochs, train_batch_metrics['loss'],
                    train_batch_metrics['accuracy'] * 100
                )
            )

            "------------------VALIDATION BATCH------------------"
            val_batch_metrics = []
            val_datagen = val_dataset.iter_batch(batch_size=config.batch_size, random_state=config.seed)

            for (batch_idx, batch) in enumerate(tqdm(val_datagen, total=num_val_batches)):
                outcomes, probabilities, labels = batch.as_array(return_targets=True)
                problems = select_array_inputs(outcomes, probabilities)
                metrics = val_step(state, problems, labels)
                val_batch_metrics.append(metrics)

            val_batch_metrics = accumulate_metrics(val_batch_metrics)
            print(
                'VAL (%d/%d):  Loss: %.4f, accuracy: %.2f\n' % (
                    epoch, config.epochs, val_batch_metrics['loss'],
                    val_batch_metrics['accuracy'] * 100
                )
            )

            "------------------wandb LOGGING------------------"
            wandb.log({
                "Train Loss": train_batch_metrics['loss'],
                "Train Accuracy": train_batch_metrics['accuracy'],
                "Validation Loss": val_batch_metrics['loss'],
                "Validation Accuracy": val_batch_metrics['accuracy']
            }, step=epoch)

            if val_batch_metrics['loss'] < best_val_loss:
                save_checkpoint("checkpoint.msgpack", state, epoch)

        return state


def init_train_state(model, random_key, shape, learning_rate) -> train_state.TrainState:
    # Initialize the Model
    variables = model.init(random_key, jnp.ones(shape))
    # Create the optimizer
    optimizer = optax.adam(learning_rate)
    # Create a State
    return train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=variables['params']
    )


# TODO: eventually move these functions somewhere else.
def cross_entropy_loss(*, logits, labels):
    one_hot_encoded_labels = jax.nn.one_hot(labels, num_classes=2)
    return optax.softmax_cross_entropy(
        logits=logits, labels=one_hot_encoded_labels
    ).mean()


def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


@jax.jit
def train_step(state: train_state.TrainState, problem, label):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, problem)
        loss = cross_entropy_loss(logits=logits, labels=label)
        return loss, logits

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=label)
    return state, metrics


@jax.jit
def val_step(state: train_state.TrainState, problem, label):
    logits = state.apply_fn({'params': state.params}, problem)
    return compute_metrics(logits=logits, labels=label)


def save_checkpoint(ckpt_path, state, epoch):
    with open(ckpt_path, "wb") as outfile:
        outfile.write(msgpack_serialize(to_state_dict(state)))
    artifact = wandb.Artifact(
        f'{wandb.run.name}-checkpoint', type='dataset'
    )
    artifact.add_file(ckpt_path)
    wandb.log_artifact(artifact, aliases=["latest", f"epoch_{epoch}"])


def load_checkpoint(ckpt_file, state):
    artifact = wandb.use_artifact(
        f'{wandb.run.name}-checkpoint:latest'
    )
    artifact_dir = artifact.download()
    ckpt_path = os.path.join(artifact_dir, ckpt_file)
    with open(ckpt_path, "rb") as data_file:
        byte_data = data_file.read()
    return from_bytes(state, byte_data)


def accumulate_metrics(metrics):
    metrics = jax.device_get(metrics)
    return {
        k: np.mean([metric[k] for metric in metrics])
        for k in metrics[0]
    }

