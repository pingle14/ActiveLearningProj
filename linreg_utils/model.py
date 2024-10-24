import jax.numpy as jnp
from jax import jit


# functional form
@jit
def linear_model(params, X):
    return jnp.matmul(X, params)


# Perform linear regression using linear algebra
# @jit
# def linear_regression(X, y):
#     known_intercept = 0
#     X_slope = X[:, 1:]  # Assuming the first column is for the intercept
#     X_transpose = jnp.transpose(X_slope)

#     # Calculate the slope coefficients
#     X_transpose_X_inv = jnp.linalg.inv(jnp.matmul(X_transpose, X_slope))
#     slope_coeff = jnp.matmul(
#         jnp.matmul(X_transpose_X_inv, X_transpose), y - known_intercept
#     )
#     # Re-add a column of zeros
#     coeff = jnp.concatenate((jnp.zeros((1,)), slope_coeff))

#     return coeff


# @jit
# def linear_regression(X, y):
#     X_transpose = jnp.transpose(X)
#     X_transpose_X_inv = jnp.linalg.inv(jnp.matmul(X_transpose, X))
#     coeff = jnp.matmul(jnp.matmul(X_transpose_X_inv, X_transpose), y)
#     return coeff


# @jit
# def FAILED_linear_regression(X, y):
#     # Center the independent variable X
#     X_mean = jnp.mean(X, axis=0)
#     X_centered = X - X_mean

#     # Estimate the coefficients using centered X
#     X_transpose = jnp.transpose(X_centered)
#     X_transpose_X_inv = jnp.linalg.inv(jnp.matmul(X_transpose, X_centered))
#     coeff = jnp.matmul(jnp.matmul(X_transpose_X_inv, X_transpose), y)

#     return coeff


@jit
def linear_regression(X, y):
    # Add a column of ones to X for the intercept
    X_with_intercept = jnp.hstack((jnp.ones((X.shape[0], 1)), X))

    # Estimate the coefficients
    X_transpose = jnp.transpose(X_with_intercept)
    X_transpose_X_inv = jnp.linalg.inv(jnp.matmul(X_transpose, X_with_intercept))
    coeff_with_intercept = jnp.matmul(X_transpose_X_inv, X_transpose)

    # Calculate the coefficients including intercept
    final_coeff = jnp.matmul(coeff_with_intercept, y)

    return final_coeff[1:]


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
