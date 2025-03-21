import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from jax import random
from jax import lax
from linreg_utils.model import linear_model


# Define the updated neural network model with dropout and L2 regularization
class TransformationNN(nn.Module):
    output_dim: int
    weight_decay: float = 1e-3  # L2 regularization strength

    @nn.compact  # Use @compact to wrap the method defining the layers
    def __call__(self, x):
        # Neural network transformation T(X)
        x = nn.Dense(features=128)(x)
        x = nn.sigmoid(x)

        x = nn.Dense(features=64)(x)
        x = nn.sigmoid(x)

        x = nn.Dense(features=32)(x)
        x = nn.sigmoid(x)

        x = nn.Dense(features=32)(x)
        x = nn.tanh(x)

        x = nn.Dense(features=32)(x)
        x = nn.leaky_relu(x)

        z_pred = nn.Dense(features=self.output_dim)(x)  # Output Z
        return z_pred


# Define T2 (a simple neural network for the parameter beta)
class T2(nn.Module):
    output_dim: int
    weight_decay: float = 1e-3

    @nn.compact
    def __call__(self, theta_complement):
        beta_out = nn.Dense(self.output_dim)(theta_complement)
        return beta_out


# Shuffle and split the dataset
def split_data(x, y, key, train_ratio=0.8):

    num_train = int(train_ratio * x.shape[0])
    indices = random.permutation(key, x.shape[0])

    train_idx = indices[:num_train]
    test_idx = indices[num_train:]

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return x_train, x_test, y_train, y_test


def l2_regularization(params1, weight_decay):
    # Compute L2 regularization for params1
    reg_loss_1 = sum(jnp.sum(param**2) for param in jax.tree.leaves(params1))

    # Combine the regularization terms for both models
    return weight_decay * reg_loss_1


def inference(model, model_params, phi_model, phi_params, x, theta, theta_complement):
    phi = phi_model.apply({"params": phi_params}, theta_complement)
    Z = model.apply({"params": model_params}, x)
    param_space = jnp.concatenate([phi, theta])
    y_pred = linear_model(params=param_space, X=Z)
    return y_pred


# Mean Squared Error loss function with L2 regularization
def mse_loss(
    model,
    model_params,
    phi_model,
    phi_params,
    x,
    y_true,
    theta,
    theta_complement,
    weight_decay=1e-4,
):
    y_pred = inference(
        model, model_params, phi_model, phi_params, x, theta, theta_complement
    )
    mse = jnp.mean((y_pred - y_true) ** 2)
    reg_loss = l2_regularization(model_params, weight_decay)
    return mse + reg_loss


# Evaluation function without dropout
def evaluate_model(
    model,
    model_params,
    phi_model,
    phi_params,
    x,
    y_true,
    theta,
    theta_complement,
    test=True,
):
    y_pred = inference(
        model, model_params, phi_model, phi_params, x, theta, theta_complement
    )
    mse = jnp.mean((y_pred - y_true) ** 2)
    print(f"{'Test' if test else 'Train'} MSE: {mse}")
    return mse


def mse_loss_fn(
    combo_params,
    model,
    phi_model,
    x,
    y_true,
    theta,
    theta_complement,
    weight_decay=1e-4,
):
    return mse_loss(
        model,
        combo_params["x2z"],
        phi_model,
        combo_params["thetaComp2phi"],
        x,
        y_true,
        theta,
        theta_complement,
        weight_decay,
    )


def training_loss_fn(
    combo_params, model, phi_model, x, y, theta, theta_complement, test=True
):
    return evaluate_model(
        model,
        combo_params["x2z"],
        phi_model,
        combo_params["thetaComp2phi"],
        x,
        y,
        theta,
        theta_complement,
        test,
    )


# Training function with regularization and batch processing
def train_model(
    model,
    phi_model,
    x_train,
    y_train,
    theta,
    theta_complement,
    num_epochs=1000,
    batch_size=32,
    learning_rate=0.001,
    seed=0,
):
    # Initialize the model parameters
    init_rng_model, init_rng_phi = random.split(random.PRNGKey(seed), 2)
    model_params = model.init(init_rng_model, x_train)["params"]
    phi_params = phi_model.init(init_rng_phi, theta_complement)["params"]

    # Define the optimizer with weight decay (L2 regularization)
    combo_params = {"x2z": model_params, "thetaComp2phi": phi_params}
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(combo_params)

    # Training step with L2 regularization and dropout
    @jax.jit
    def step(combo_params, opt_state, x_batch, y_batch, theta, theta_complement):
        loss, grads = jax.value_and_grad(mse_loss_fn)(
            combo_params, model, phi_model, x_batch, y_batch, theta, theta_complement
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        combo_params = optax.apply_updates(combo_params, updates)
        return combo_params, opt_state, loss

    # Training loop with batch processing
    num_samples = x_train.shape[0]
    num_batches = num_samples // batch_size
    print(f"Sample size: {num_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {num_batches}")
    print(f"================================")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {model.weight_decay}")
    print(f"================================")

    for epoch in range(num_epochs):
        # Process mini-batches
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            x_batch = x_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]

            combo_params, opt_state, loss = step(
                combo_params, opt_state, x_batch, y_batch, theta, theta_complement
            )

            # Optionally print loss at regular intervals
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Batch Loss: {loss}")
                loss = mse_loss_fn(
                    combo_params,
                    model,
                    phi_model,
                    x_batch,
                    y_batch,
                    theta,
                    theta_complement,
                )
                print(f"Epoch {epoch}, Training Loss: {loss}")
                training_loss_fn(
                    combo_params,
                    model,
                    phi_model,
                    x_batch,
                    y_batch,
                    theta,
                    theta_complement,
                )

    return combo_params
