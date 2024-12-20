import jax.numpy as jnp
from jax import random


def generate_data(
    linearity_percentage,
    sample_size,
    key=random.PRNGKey(9355442),
    covariate_size=5,
    coeff=None,
    measurement_error=False,
):
    key_X, key_coeff, key_e, key_err = random.split(key, 4)

    # generate random coefficents
    if coeff is None:
        coeff = random.normal(key_coeff, shape=(covariate_size,))
    else:
        covariate_size = coeff.shape[0]

    # generate random covariates
    X = jnp.reshape(
        random.normal(key_X, shape=(sample_size * covariate_size,)),
        (covariate_size, -1),
    ).T
    X = X.at[:, 0].set(1)

    # generate random noise
    epsilon = random.normal(key_e, shape=(sample_size,))

    # generate random noise
    error = (
        0.0 + 5.0 * (X[:, 1] - 5.0) ** 2 / 25.0 if measurement_error else X[:, 1] * 0.0
    )
    epsilon_error = (error) * random.normal(key_err, shape=(sample_size,))

    # compute outcome
    linear_component = jnp.matmul(X, coeff)
    nonlinear_component = jnp.exp(jnp.matmul(X, coeff))
    y = (
        (linearity_percentage * linear_component)
        + ((1 - linearity_percentage) * nonlinear_component)
        + epsilon
        + epsilon_error
    )

    return X, y, error, coeff


def generate_non_linear_data(
    sample_size,
    key=random.PRNGKey(9355442),
    covariate_size=5,
    coeff=None,
    measurement_error=False,
):
    key_X, key_coeff, key_e, key_err = random.split(key, 4)

    # generate random coefficents
    if coeff is None:
        coeff = random.normal(key_coeff, shape=(covariate_size,))
    else:
        covariate_size = coeff.shape[0]

    # generate random covariates
    X = jnp.reshape(
        random.normal(key_X, shape=(sample_size * covariate_size,)),
        (covariate_size, -1),
    ).T
    X = X.at[:, 0].set(1)

    # generate random noise
    epsilon = random.normal(key_e, shape=(sample_size,))
    # generate random noise
    error = (
        0.0 + 5.0 * (X[:, 1] - 5.0) ** 2 / 25.0 if measurement_error else X[:, 1] * 0.0
    )
    epsilon_error = (error) * random.normal(key_err, shape=(sample_size,))

    # compute outcome
    y = jnp.exp(jnp.matmul(X, coeff)) + epsilon + epsilon_error

    return X, y, error, coeff


def generate_linear_data(
    sample_size,
    key=random.PRNGKey(9355442),
    covariate_size=5,
    coeff=None,
    measurement_error=False,
):
    key_X, key_coeff, key_e, key_err = random.split(key, 4)

    # generate random coefficents
    if coeff is None:
        coeff = random.normal(key_coeff, shape=(covariate_size,))
    else:
        covariate_size = coeff.shape[0]

    # generate random covariates
    X = jnp.reshape(
        random.normal(key_X, shape=(sample_size * covariate_size,)),
        (covariate_size, -1),
    ).T
    X = X.at[:, 0].set(1)

    # generate random noise
    epsilon = random.normal(key_e, shape=(sample_size,))

    # generate random noise
    error = (
        0.0 + 5.0 * (X[:, 1] - 5.0) ** 2 / 25.0 if measurement_error else X[:, 1] * 0.0
    )
    epsilon_error = (error) * random.normal(key_err, shape=(sample_size,))

    # compute outcome
    y = jnp.matmul(X, coeff) + epsilon + epsilon_error

    return X, y, error, coeff
