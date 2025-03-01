import numpy as np

class TjelmelandTransitions:
    """
    Builds Tjelmeland's multi-proposal transition matrices (Peskun/Barker),
    optionally forcing or not forcing a jump away from each state.
    """

    def __init__(self, log_pi, log_q, force_jump=False):
        """
        Parameters
        ----------
        log_pi : callable
            log_pi(x) -> float, log of target density at x (up to additive const).
        log_q : callable
            log_q(x_from, x_to) -> float, log of proposal density q(x_from->x_to).
        force_jump : bool
            If True, sets T[i,i] = 0.0 so we always move away from x_i.
            If False, we allow leftover probability on the diagonal.
        """
        self.log_pi = log_pi
        self.log_q = log_q
        self.force_jump = force_jump

    def _build_log_w_matrix(self, x_set):
        """
        Builds log_w[i,j] = log( pi(x_j)* q(x_j->x_i ) ), for i!=j, else -inf.
        Returns
        -------
        log_w : np.ndarray, shape (n,n)
        """
        n = len(x_set)
        log_w = np.full((n, n), -np.inf)
        for i in range(n):
            for j in range(n):
                if i != j:
                    log_w[i, j] = (self.log_pi(x_set[j]) +
                                   self.log_q(x_set[j], x_set[i]))
        return log_w

    def peskun_transition_matrix(self, x_set):
        """
        Tjelmeland's Peskun kernel:
          T[i,j] ~ w_{i->j} = pi(x_j)* q(x_j-> x_i)
        """
        n = len(x_set)
        log_w = self._build_log_w_matrix(x_set)

        T = np.zeros((n, n), dtype=float)
        for i in range(n):
            # Exponentiate row i stably
            row_i = log_w[i,:]  # shape (n,)
            # We only want the off-diagonal terms (i-> j for j != i).
            finite_mask = np.isfinite(row_i)
            if not np.any(finite_mask):
                # row is all -inf => no moves => stay put
                T[i,i] = 1.0
                continue

            # Find max to shift
            rmax = np.max(row_i[finite_mask])
            exps = np.zeros(n, dtype=float)
            for j in range(n):
                if j != i and np.isfinite(row_i[j]):
                    exps[j] = np.exp(row_i[j] - rmax)

            denom = exps.sum()
            if denom > 0:
                # Fill off-diagonal
                T[i,:] = exps / denom
                if self.force_jump:
                    # Force T[i,i] = 0, then re-normalize row
                    T[i,i] = 0.0
                    s2 = T[i,:].sum()
                    if s2 > 0:
                        T[i,:] /= s2
                else:
                    # Let leftover = 1 - sum_{j != i}
                    leftover = 1.0 - T[i,:].sum()
                    if leftover < 0.0:
                        leftover = 0.0  # clamp small negative due to float
                    T[i,i] = leftover
                    # Row might exceed 1 due to rounding, re-normalize safely
                    row_sum = T[i,:].sum()
                    if row_sum > 0:
                        T[i,:] /= row_sum
            else:
                T[i,i] = 1.0

        return T

    def barker_transition_matrix(self, x_set):
        """
        Tjelmeland's Barker kernel:
          phi[i,j] = w[i,j]/(w[i,j] + w[j,i]), w[i,j]>=0
          T[i,j] ~ phi[i,j] for j!=i, then row-normalize.
        """
        n = len(x_set)
        log_w = self._build_log_w_matrix(x_set)

        # Exponentiate in a stable manner
        w = np.zeros((n, n), dtype=float)
        for i in range(n):
            row_i = log_w[i,:]
            finite_mask = np.isfinite(row_i)
            if not np.any(finite_mask):
                # entire row i is -inf => no moves => w[i,:]=0
                continue
            # shift by max
            rmax = np.max(row_i[finite_mask])
            for j in range(n):
                if i != j and np.isfinite(row_i[j]):
                    w[i,j] = np.exp(row_i[j] - rmax)

        # Build phi[i,j] = w[i,j]/(w[i,j] + w[j,i]), i!=j
        # Vectorized approach: denom = w + w.T
        denom = w + w.T
        phi = np.zeros((n, n), dtype=float)
        mask = (denom > 0)
        phi[mask] = w[mask] / denom[mask]
        # phi diagonal = 0
        np.fill_diagonal(phi, 0.0)

        # Now row-normalize phi to get T
        T = np.zeros((n, n), dtype=float)
        for i in range(n):
            row_vals = phi[i, :]
            s = row_vals.sum()
            if s > 0:
                T[i,:] = row_vals / s
                if self.force_jump:
                    # Force T[i,i]=0
                    T[i,i] = 0.0
                    row_sum = T[i,:].sum()
                    if row_sum > 0:
                        T[i,:] /= row_sum
                else:
                    leftover = 1.0 - T[i,:].sum()
                    if leftover < 0.0:
                        leftover = 0.0
                    T[i,i] = leftover
                    row_sum = T[i,:].sum()
                    if row_sum > 0:
                        T[i,:] /= row_sum
            else:
                # no moves => stay
                T[i,i] = 1.0

        return T

    def sample_next_state(self, i_current, T):
        """
        Given a transition matrix T and the current index i_current,
        sample the next index from T[i_current,:].
        """
        row_probs = T[i_current, :]
        return np.random.choice(len(row_probs), p=row_probs)
