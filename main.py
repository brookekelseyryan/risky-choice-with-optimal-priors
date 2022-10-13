import jax
import jax.numpy as jnp

import wandb
from hurd.internal_datasets import load_c13k_data
from models.flat_model import FlatModel, init_train_state


def human_proportions():
    """
    Can this model learn the human proportions better?
    First let's just try to have it predict the human proportions
    """

    wandb.init(project="choices13k-optimal",
               entity="brookeryan",
               name="flat-model-frozen-predict-humans")
    # Setting up configs to be synced by the Weights & Biases run
    config = wandb.config
    config.seed = 42
    config.batch_size = 36
    config.validation_split = 0.1
    config.learning_rate = 0.01
    config.epochs = 200

    # import some data
    data = load_c13k_data(fb_filter="EV_B_highest_target")


if __name__ == '__main__':
    wandb.init(project="choices13k-optimal",
               entity="brookeryan",
               name="flat-model")
    # Setting up configs to be synced by the Weights & Biases run
    config = wandb.config
    config.seed = 42
    config.batch_size = 108
    config.validation_split = 0.1
    config.learning_rate = 0.01
    config.epochs = 200

    # import some data
    data = load_c13k_data(fb_filter="EV_B_highest_target")

    # data can be split with a method yielding an iterator
    splitter = data.split(p=(1 - config.validation_split), n_splits=1, shuffle=True, random_state=1)
    (train_data, val_data) = list(splitter)[0]

    # set up the params
    rng = jax.random.PRNGKey(config.seed)
    x = jnp.ones(shape=(config.batch_size, 36))
    model = FlatModel()
    params = model.init(rng, x)
    tree = jax.tree_map(lambda x: x.shape, params)

    print(tree)

    state = init_train_state(
        model, rng, (config.batch_size, 36), config.learning_rate
    )

    # not sure if we need to save the state here or not ? or if it should be attached to the model?
    state = model.fit(train_dataset=train_data, val_dataset=val_data, config=config, state=state)

    model.predict(test_dataset=val_data, config=config, state=state)

