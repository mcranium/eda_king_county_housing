data {
    int<lower=0> N; // nummber of datapoints
    vector[N] x; // covariates
    vector[N] y; // variates
}

parameters {
    real alpha;
    real beta;
    real <lower=0> sigma;
}

model {
    // priors
    alpha ~ normal(0,10);
    beta ~ normal(0,10);
    sigma ~ normal(0,10);
    
    y ~ normal(alpha + beta * x, sigma); // likelihood
}

// somehow the section below breaks the code
// generated quantities {
//     vector[N] y_sim; //simulated data based on the posterior
//     for(i in 1:N)
//         y_sim[i] = normal_rng(alpha + beta * x[i], sigma);
// }