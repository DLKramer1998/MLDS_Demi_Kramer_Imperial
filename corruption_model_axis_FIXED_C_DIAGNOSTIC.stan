
data {
  int<lower=1> n;
  int<lower=1> p;
  matrix[n, p] X;
  array[n] int<lower=0, upper=1> y;

  int<lower=1, upper=p> boundary_dim;
  real<lower=0> kappa;
  real c_fixed;
}

parameters {
  real beta0;
  vector[p] beta;
  real<lower=0, upper=1> q;
}

model {
  beta0 ~ normal(0, 2);
  beta ~ normal(0, 1);
  q ~ beta(2, 5);

  for (i in 1:n) {
    real p_true = inv_logit(
      beta0 + dot_product(beta, X[i])
    );
    real r = inv_logit(
      kappa * (
        X[i, boundary_dim] - c_fixed
      )
    );
    real w = q * r;

    if (y[i] == 1) {
      target += log(p_true) + log1m(w);
    } else {
      target += log_sum_exp(
        log1m(p_true),
        log(p_true) + log(w)
      );
    }
  }
}
