import jax.numpy as jnp
from jax import vmap, lax
from jax.scipy.linalg import det
from query_strategies.data_gen import generate_data
from query_strategies.strategy import (
    estimate_variance,
    Strategy,
)


class BAIT(Strategy):
    def __init__(
        self,
        initial_sample_sz=20,
        pool_sz=100,
        budget=10,
        iter=10,
        true_coeff=None,
        given_key=None,
    ):
        super(BAIT, self).__init__(
            name="BAIT",
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
        # print(f"M_type = {type(M)} ... info_i tpye = {type(info_i)}")
        sum_M_info = M + info_i
        intermed = jnp.linalg.inv(sum_M_info)
        return jnp.trace(intermed * avg_fiU)

    def prune_criteria(self, info_i, M, avg_fiU):
        intermed = jnp.linalg.inv(M - info_i)
        return jnp.trace(intermed * avg_fiU)

    def bait_algo(self, params, X_new, Y_new, err_new, n_params=None, lam=1):
        # I_{U or S} = fisher info for {U or S}
        variance = estimate_variance(
            params, self.labeled_y, self.labeled_X, self.error
        )
        fi_U = self.fisher_information_vmaped(
            params, variance, X_new, err_new, n_params, 1
        )
        avg_fiU = jnp.mean(fi_U)

        fi_S = self.fisher_information_vmaped(
            params, variance, self.labeled_X, self.error, n_params, 1
        )
        avg_fiS = jnp.mean(fi_S)
        M = lam + avg_fiS

        # OVERSAMPLE 2b points
        chosen_sampleX = []
        chosen_sampleY = []
        chosen_sample_err = []
        # print("OVERSAMPLE 2b pts")
        for b in range(2 * self.budget):
            # chosen_sample.
            includes = self.include_crit_vmapped(fi_U, M, avg_fiU)

            new_pt_indx = jnp.argmin(includes)
            M += fi_U[new_pt_indx]
            chosen_sampleX.append(X_new[new_pt_indx])
            chosen_sampleY.append(Y_new[new_pt_indx])
            chosen_sample_err.append(err_new[new_pt_indx])

            # Remove new_pt from unlabeled sample so dont re-sample:
            fi_U = jnp.delete(fi_U, new_pt_indx, 0)
            X_new = jnp.delete(X_new, new_pt_indx, 0)

        # PRUNE b points .. from running sample or overall sample?
        # print("Pruning b pts")
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
        # print(
        #     f"CHOSEN SAMPLE DIMS: X: {chosen_sampleX.shape}, Y: {chosen_sampleY.shape}"
        # )
        self.labeled_X = (
            jnp.append(self.labeled_X, jnp.array(chosen_sampleX), axis=0)
            if self.budget > 1
            else jnp.append(
                self.labeled_X,
                jnp.array(chosen_sampleX).reshape((1, -1)),
                axis=0,
            )
        )
        # print(f"DONE APPENDING X")
        self.labeled_y = jnp.append(self.labeled_y, jnp.array(chosen_sampleY))
        # print(f"DONE APPENDING Y")
        self.error = jnp.append(self.error, jnp.array(chosen_sample_err))
        # print(f"DONE APPENDING ERR")

    def choose_sample(self, key):
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
            self.bait_algo(self.current_params, X, y, error)
