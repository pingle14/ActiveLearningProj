import jax.numpy as jnp
from jax import vmap, lax
from query_strategies.strategy import Strategy


class BAIT(Strategy):
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
    ):
        super(BAIT, self).__init__(
            name="BAIT",
            model_inference_fn=model_inference_fn,
            model_training_fn=model_training_fn,
            generate_data=generate_data,
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
        self.include_crit_vmapped = vmap(
            self.include_criteria,
            in_axes=(0, None, None),
            out_axes=0,
        )
        self.prune_crit_vmapped = vmap(
            self.prune_criteria,
            in_axes=(0, None, None),
            out_axes=0,
        )

    # # Choose 2B points: oversample
    # """
    # U = unlabeled generated points
    # S = labeled points
    # all_fishy = mean fisher for {U \cup S}
    # labeled_fishy = mean fisher for {S}
    # """
    def fisher_information(
        self, params, variance, X, err, n_params, start_index
    ):
        if n_params is None:
            df = self.grad_f(params, X)
        else:
            df = lax.dynamic_slice(
                self.grad_f(params, X), (start_index,), (n_params,)
            )
        fi = jnp.outer(df, df) / (variance + err**2)
        return fi

    def include_criteria(self, info_i, M, avg_fiU):
        sum_M_info = M + info_i
        intermed = jnp.linalg.inv(sum_M_info)
        return jnp.trace(intermed * avg_fiU)

    def prune_criteria(self, info_i, M, avg_fiU):
        intermed = jnp.linalg.inv(M - info_i)
        return jnp.trace(intermed * avg_fiU)

    def update_sample(self, key, X, y, error):
        n_params = None
        lam = 1
        # I_{U or S} = fisher info for {U or S}
        variance = self.estimate_variance(
            self.current_params, self.labeled_y, self.labeled_X, self.error
        )
        fi_U = self.fisher_information_vmaped(
            self.current_params, variance, X, error, n_params, 1
        )
        avg_fiU = jnp.mean(fi_U)

        fi_S = self.fisher_information_vmaped(
            self.current_params,
            variance,
            self.labeled_X,
            self.error,
            n_params,
            1,
        )
        avg_fiS = jnp.mean(fi_S)
        M = lam + avg_fiS

        # OVERSAMPLE 2b points
        chosen_sampleX = []
        chosen_sampleY = []
        chosen_sample_err = []
        for b in range(2 * self.budget):
            # chosen_sample.
            includes = self.include_crit_vmapped(fi_U, M, avg_fiU)

            new_pt_indx = jnp.argmin(includes)
            M += fi_U[new_pt_indx]
            chosen_sampleX.append(X[new_pt_indx])
            chosen_sampleY.append(y[new_pt_indx])
            chosen_sample_err.append(error[new_pt_indx])

            # Remove new_pt from unlabeled sample so dont re-sample:
            fi_U = jnp.delete(fi_U, new_pt_indx, 0)
            X = jnp.delete(X, new_pt_indx, 0)

        # PRUNE b points .. from running sample or overall sample?
        for b in range(self.budget):
            prunes = self.prune_crit_vmapped(
                jnp.array(chosen_sampleX), M, avg_fiU
            )
            bad_pt_indx = jnp.argmin(prunes)
            M -= prunes[bad_pt_indx]
            chosen_sampleX = (
                chosen_sampleX[:bad_pt_indx] + chosen_sampleX[bad_pt_indx + 1 :]
            )
            chosen_sampleY = (
                chosen_sampleY[:bad_pt_indx] + chosen_sampleY[bad_pt_indx + 1 :]
            )
            chosen_sample_err = (
                chosen_sample_err[:bad_pt_indx]
                + chosen_sample_err[bad_pt_indx + 1 :]
            )

        # Concat Chosen_sample to labeledX
        chosen_sampleX = jnp.array(chosen_sampleX)
        chosen_sampleY = jnp.array(chosen_sampleY)
        chosen_sample_err = jnp.array(chosen_sample_err)
        self.labeled_X = (
            jnp.append(self.labeled_X, jnp.array(chosen_sampleX), axis=0)
            if self.budget > 1
            else jnp.append(
                self.labeled_X,
                jnp.array(chosen_sampleX).reshape((1, -1)),
                axis=0,
            )
        )
        self.labeled_y = jnp.append(self.labeled_y, jnp.array(chosen_sampleY))
        self.error = jnp.append(self.error, jnp.array(chosen_sample_err))
