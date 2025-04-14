import jax.numpy as jnp
from jax import vmap, lax
from jax.scipy.linalg import det
from query_strategies.strategy import Strategy


class AdjustedFisher(Strategy):
    def __init__(
        self,
        model_inference_fn,
        model_training_fn,
        generate_data,
        initial_sample_sz=20,
        pool_sz=100,
        budget=10,
        iter=10,
        true_coeff=None,
        given_key=None,
        measurement_error=False,
    ):
        super(AdjustedFisher, self).__init__(
            name="AdjustedFisher",
            model_inference_fn=model_inference_fn,
            model_training_fn=model_training_fn,
            generate_data=generate_data,
            initial_sample_sz=initial_sample_sz,
            pool_sz=pool_sz,
            budget=budget,
            iter=iter,
            true_coeff=true_coeff,
            given_key=given_key,
            measurement_error=measurement_error,
        )
        self.fisher_information_vmaped = vmap(
            self.fisher_information,
            in_axes=(None, None, 0, 0, None, None),
            out_axes=0,
        )
        self.param_start = 1
        self.num_params = 1

    def top_k_indices(self, array, k):
        if k == 1:
            indices = jnp.argmax(array)
        elif k > 1:
            indices = jnp.argsort(array)[::-1][:k]
        else:
            raise ValueError("k should be an integer above 0.")
        return indices

    # Estimate Fisher information
    def fisher_information(self, params, variance, X, err, n_params, start_index):
        if n_params is None:
            df = self.grad_f(params, X)
        else:
            df = lax.dynamic_slice(self.grad_f(params, X), (start_index,), (n_params,))
        fi_adjusted = jnp.outer(df, df) / (variance + err**2)
        return fi_adjusted

    def update_sample(self, key, X, y, error):
        trace = True
        Xres = X - jnp.mean(self.labeled_X, axis=0)
        variance = self.estimate_variance(
            # NOTE: HYP 1: jnp.asarray([0 if i == 0 else 1 for i in range(len(self.current_params))]),
            self.current_params,
            self.labeled_y,
            self.labeled_X,
            self.error,
        )
        fi = self.fisher_information_vmaped(
            self.current_params,
            variance,
            Xres,
            error,
            self.num_params,
            self.param_start,
        )
        "For each row (dim 0), calculates sum of diagonal (trace) of Jacobian num_feat x num_feat mtrx"
        objective_func = jnp.trace(fi, axis1=1, axis2=2) if trace else det(fi)
        indices = self.top_k_indices(objective_func, self.budget)
        sampled_feature_vectors = X[indices, :]
        self.labeled_X = (
            jnp.append(self.labeled_X, sampled_feature_vectors, axis=0)
            if self.budget > 1
            else jnp.append(
                self.labeled_X, sampled_feature_vectors.reshape((1, -1)), axis=0
            )
        )
        self.labeled_y = jnp.append(self.labeled_y, y[indices])

        if self.measurement_error:
            self.error = jnp.append(self.error, error[indices])
