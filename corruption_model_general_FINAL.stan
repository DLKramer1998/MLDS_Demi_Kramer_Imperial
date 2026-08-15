
/*
  Structured Label Corruption Model v7 — General Hyperplane Boundary
  ===================================================================
  Extension of v6: the corruption boundary is a general hyperplane
      R = { x : alpha . x > c },   ||alpha|| = 1,
  with the full normal vector alpha estimated jointly with (beta0, beta, c, q).
  This implements the supervisor's specification with alpha in R^p, rather
  than fixing the boundary axis as domain knowledge.

  Identifiability notes:
  - ||alpha|| = 1 fixes the scale non-identifiability of (alpha, c).
    Implemented via the normalized-vector trick: alpha_raw ~ normal(0, 1),
    alpha = alpha_raw / ||alpha_raw||. With an isotropic Gaussian prior
    on alpha_raw this induces no preferred orientation.
  - There is no sign/reflection ambiguity for the region itself:
    {x : alpha.x > c} and {x : -alpha.x > -c} are complementary halfspaces,
    not the same set. These constraints remove parameterisation ambiguity;
    practical recovery from finite data is assessed empirically below.
  - Everything else (one-sided 1->0 likelihood, soft boundary, marginalised
    latents, closed-form z_prob, log_lik) matches v6.
*/

data {
  int<lower=0> n;
  int<lower=0> p;
  matrix[n, p] X;
  array[n] int<lower=0, upper=1> y;
  real kappa;              // soft-boundary steepness (fixed)
  real c_prior_mean;
  real<lower=0> c_prior_sd;
}

parameters {
  real beta0;
  vector[p] beta;
  vector[p] alpha_raw;     // unnormalised boundary normal
  real c;
  real<lower=0, upper=1> q;
}

transformed parameters {
  vector[p] alpha = alpha_raw / norm2(alpha_raw);
}

model {
  beta0     ~ normal(0, 2);
  beta      ~ normal(0, 1);
  alpha_raw ~ normal(0, 1);      // induces uniform prior on the unit sphere
  c         ~ normal(c_prior_mean, c_prior_sd);
  q         ~ beta(2, 5);

  {
    vector[n] eta   = beta0 + X * beta;
    vector[n] log_r = log_inv_logit(kappa * (X * alpha - c));
    for (i in 1:n) {
      real log_p  = log_inv_logit(eta[i]);      // log P(y' = 1 | x)
      real log_1p = log1m_inv_logit(eta[i]);    // log P(y' = 0 | x)
      real log_qr = log(q) + log_r[i];          // log P(z = 1 | x)
      if (y[i] == 1) {
        target += log1m_exp(log_qr) + log_p;
      } else {
        target += log_sum_exp(log_1p, log_qr + log_p);
      }
    }
  }
}

generated quantities {
  vector[n] z_prob;
  vector[n] r;
  vector[n] log_lik;
  {
    vector[n] eta   = beta0 + X * beta;
    vector[n] log_r = log_inv_logit(kappa * (X * alpha - c));
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
