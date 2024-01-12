import jax.numpy as jnp
from jax import jit


# functional form
@jit
def linear_model(params, X):
    return jnp.matmul(X, params)


# Perform linear regression using linear algebra
@jit
def linear_regression(X, y):
    X_transpose = jnp.transpose(X)
    X_transpose_X_inv = jnp.linalg.inv(jnp.matmul(X_transpose, X))
    coeff = jnp.matmul(jnp.matmul(X_transpose_X_inv, X_transpose), y)
    return coeff


def loss_fn_mse(y_trues, y_preds):
    assert y_trues.shape == y_preds.shape
    return jnp.mean(
        jnp.array(
            [
                jnp.power(y_true - y_pred, 2)
                for (y_true, y_pred) in zip(y_trues, y_preds)
            ]
        )
    )
