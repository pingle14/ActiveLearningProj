import jax.numpy as jnp
from jax import vmap, lax
from jax.scipy.linalg import det
from query_strategies.data_gen import generate_data
from query_strategies.strategy import (
    estimate_variance,
    Strategy,
)


class AdjustedFisher(Strategy):
    def __init__(
        self,
        initial_sample_sz=20,
        pool_sz=100,
        budget=10,
        iter=10,
        true_coeff=None,
        given_key=None,
    ):
        super(AdjustedFisher, self).__init__(
            name="AdjustedFisher",
            initial_sample_sz=initial_sample_sz,
            pool_sz=pool_sz,
            budget=budget,
            iter=iter,
            true_coeff=true_coeff,
            given_key=given_key,
        )
        self.fisher_information_vmaped = vmap(
            self.fisher_information,
            in_axes=(None, None, 0, 0, None, None),
            out_axes=0,
        )

    def top_k_indices(self, array, k):
        if k == 1:
            indices = jnp.argmax(array)
        elif k > 1:
            indices = jnp.argsort(array)[::-1][:k]
        else:
            raise ValueError("k should be an integer above 0.")
        return indices

    # Estimate Fisher information
    def fisher_information(
        self, params, variance, X, err, n_params, start_index
    ):
        if n_params is None:
            df = self.grad_f(params, X)
        else:
            df = lax.dynamic_slice(
                self.grad_f(params, X), (start_index,), (n_params,)
            )
        fi_adjusted = jnp.outer(df, df) / (variance + err**2)
        return fi_adjusted

    def update_sample_gradient_strategy(
        self, X_new, y_new, err_new, params, n_params=None, trace=True
    ):
        Xres = X_new - jnp.mean(self.labeled_X, axis=0)
        variance = estimate_variance(
            params, self.labeled_y, self.labeled_X, self.error
        )
        fi = self.fisher_information_vmaped(
            params, variance, Xres, err_new, n_params, 1
        )

        objective_func = jnp.trace(fi, axis1=1, axis2=2) if trace else det(fi)
        indices = self.top_k_indices(objective_func, self.budget)

        sampled_feature_vectors = X_new[indices, :]

        self.labeled_X = (
            jnp.append(self.labeled_X, sampled_feature_vectors, axis=0)
            if self.budget > 1
            else jnp.append(
                self.labeled_X, sampled_feature_vectors.reshape((1, -1)), axis=0
            )
        )
        self.labeled_y = jnp.append(self.labeled_y, y_new[indices])
        self.error = jnp.append(self.error, err_new[indices])

    def choose_sample(self, key):
        # Get new data
        X, y, error, _ = generate_data(
            self.initial_sample_sz if self.labeled_X is None else self.pool_sz,
            coeff=self.true_coeff,
            key=key,
        )

        if self.labeled_X is None:
            # Init self.labeled: (initial sample)
            self.labeled_X = X
            self.labeled_y = y
            self.error = error
        else:
            # Subsequent samples
            self.update_sample_gradient_strategy(
                X, y, error, self.current_params, trace=True
            )
