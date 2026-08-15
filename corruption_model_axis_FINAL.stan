
/*
  Structured Label Corruption Model v6 — Soft Boundary, One-Sided (1 -> 0)
  ========================================================================
  Corruption mechanism (underdiagnosis): a true positive inside the
  region is recorded as negative with probability q. Negatives are
  never flipped. Changes from v5:

  1. Direction: likelihood matches the 1->0 operator used in
     Chapters 1 and 2:
        P(y = 1 | x) = (1 - q r) p
        P(y = 0 | x) = (1 - p) + q r p
     with p = inv_logit(beta0 + x.beta), r = inv_logit(kappa (x_b - c)).
     Under this mechanism z_i = 1 is only possible when y_i = 0.
  2. Intercept beta0 added (the DGP has one; omitting it biases
     beta, c and q).
  3. Corruption terms computed on the log scale via log_inv_logit
     for numerical stability far from the boundary.
  4. c prior centre AND sd passed as data — the centre must not be
     set to the simulation truth (that would be circular).
  5. log_lik in generated quantities for LOO/WAIC.
*/

data {
  int<lower=0> n;
  int<lower=0> p;
  matrix[n, p] X;
  array[n] int<lower=0, upper=1> y;
  int<lower=1, upper=p> boundary_dim;
  real kappa;              // boundary sharpness (fixed; sensitivity in Sec. 8)
  real c_prior_mean;
  real<lower=0> c_prior_sd;
}

parameters {
  real beta0;
  vector[p] beta;
  real c;
  real<lower=0, upper=1> q;
}

model {
  beta0 ~ normal(0, 2);
  beta  ~ normal(0, 1);
  c     ~ normal(c_prior_mean, c_prior_sd);
  q     ~ beta(2, 5);

  {
    vector[n] eta   = beta0 + X * beta;
    vector[n] log_r = log_inv_logit(kappa * (X[, boundary_dim] - c));
    for (i in 1:n) {
      real log_p  = log_inv_logit(eta[i]);      // log P(y' = 1 | x)
      real log_1p = log1m_inv_logit(eta[i]);    // log P(y' = 0 | x)
      real log_qr = log(q) + log_r[i];          // log P(z = 1 | x)
      if (y[i] == 1) {
        // observed positive: must be a clean true positive
        target += log1m_exp(log_qr) + log_p;
      } else {
        // observed negative: true negative OR hidden (corrupted) positive
        target += log_sum_exp(log_1p, log_qr + log_p);
      }
    }
  }
}

generated quantities {
  vector[n] z_prob;   // P(z_i = 1 | y_i, theta); zero by construction when y_i = 1
  vector[n] r;
  vector[n] log_lik;
  {
    vector[n] eta   = beta0 + X * beta;
    vector[n] log_r = log_inv_logit(kappa * (X[, boundary_dim] - c));
    for (i in 1:n) {
      real log_p  = log_inv_logit(eta[i]);
      real log_1p = log1m_inv_logit(eta[i]);
      real log_qr = log(q) + log_r[i];
      r[i] = exp(log_r[i]);
      if (y[i] == 1) {
        z_prob[i]  = 0;
        log_lik[i] = log1m_exp(log_qr) + log_p;
      } else {
        real log_corr = log_qr + log_p;
        real log_tot  = log_sum_exp(log_1p, log_corr);
        z_prob[i]  = exp(log_corr - log_tot);
        log_lik[i] = log_tot;
      }
    }
  }
}
