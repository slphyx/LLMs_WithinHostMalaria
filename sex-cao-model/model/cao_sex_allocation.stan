// Matched Cao assay-control (B0) / persistent sex-allocation (B2) refit.
//
// Both arms use the assay-calibrated split-sex observation model and exactly
// equal female/male maturation. B0 (allocation_active=0) retains one global
// male allocation fraction. B2 (allocation_active=1) adds persistent
// participant-level allocation:
//   rho_i = inv_logit(logit(rho_global) + tau_rho * z_i).
// The global rho keeps Cao-Sex's beta(2, 8) prior. No maturation contrast is
// present, so this executable cannot combine allocation and timing mechanisms.

functions {
  real integrated_flow(real rate) {
    if (rate < 1e-8) {
      return 1 - 0.5 * rate + rate * rate / 6;
    }
    return -expm1(-rate) / rate;
  }

  vector cao_initialisation(
    int nStages,
    real ipl,
    real iplMu,
    real iplSigma,
    real pmf
  ) {
    vector[nStages] pAge;
    for (i in 1:nStages) {
      pAge[i] = log_sum_exp(
        log(pmf) + normal_lpdf(i - 0.5 | iplMu, iplSigma),
        normal_lpdf(i - nStages - 0.5 | iplMu, iplSigma)
      );
    }
    return ipl * softmax(pAge);
  }

  // Columns are total circulating 18S target, female mature stock, and male
  // mature stock. Both matched arms evaluate the same split-sex recurrence.
  matrix cao_sex_trajectory(
    int Niter,
    int last_iObs,
    int nStages,
    int a_s,
    vector conc,
    array[] int output_needed,
    vector theta,
    real rho_male
  ) {
    matrix[Niter, 3] out = rep_matrix(0, Niter, 3);
    vector[nStages] P;
    vector[nStages] P_G;
    vector[nStages] P_new;
    vector[nStages] P_G_new;
    vector[4] immature_female;
    vector[4] immature_female_new;
    vector[4] immature_male;
    vector[4] immature_male_new;
    real mature_female;
    real mature_male;
    real ec50_gamma = pow(theta[6], theta[7]);
    vector[Niter] drug_killing = rep_vector(0, Niter);
    real parasite_survival = exp(-theta[9]);
    real mature_survival = exp(-theta[12]);
    real mature_flow = integrated_flow(theta[12]);
    // Maturation is fixed equal by construction in both B0 and B2.
    real m_female = theta[10];
    real m_male = theta[10];
    real rate_female = m_female + theta[11];
    real rate_male = m_male + theta[11];
    real survival_female = exp(-rate_female);
    real survival_male = exp(-rate_male);
    real flow_female = integrated_flow(rate_female);
    real flow_male = integrated_flow(rate_male);

    for (i in 1:last_iObs) {
      real conc_gamma = pow(conc[i], theta[7]);
      drug_killing[i] = theta[5] * conc_gamma / (ec50_gamma + conc_gamma);
    }
    P = cao_initialisation(
      nStages, theta[1], theta[2], theta[3], theta[4]
    );
    P_G = rep_vector(0, nStages);
    immature_female = rep_vector(0, 4);
    immature_male = rep_vector(0, 4);
    mature_female = 0;
    mature_male = 0;

    if (output_needed[1] == 1) {
      out[1, 1] = sum(P[1:(a_s - 1)]);
    }

    for (i in 2:last_iObs) {
      real kd = 0.5 * (drug_killing[i - 1] + drug_killing[i]);
      real source;
      real treated_survival = exp(-kd) * parasite_survival;

      P_new = rep_vector(0, nStages);
      P_G_new = rep_vector(0, nStages);
      P_new[2:nStages] = P[1:(nStages - 1)] * treated_survival;
      P_new[1] = theta[4] * P[nStages] * treated_survival;
      P_new[a_s + 1] = (1 - theta[8]) * P[a_s] * treated_survival;

      P_G_new[(a_s + 2):nStages] =
        P_G[(a_s + 1):(nStages - 1)] * parasite_survival;
      P_G_new[a_s + 1] = theta[8] * P[a_s] * treated_survival;
      P_G_new[1] = theta[4] * P_G[nStages] * parasite_survival;
      P_G_new[2:(a_s - 1)] = P_G[1:(a_s - 2)] * parasite_survival;

      // This is the unchanged Cao source into immature stage I.
      source = P_G[a_s - 1] * parasite_survival;

      immature_female_new[1] = immature_female[1] * survival_female
        + (1 - rho_male) * source * flow_female;
      immature_male_new[1] = immature_male[1] * survival_male
        + rho_male * source * flow_male;
      for (stage in 2:4) {
        immature_female_new[stage] =
          immature_female[stage] * survival_female
          + m_female * immature_female[stage - 1] * flow_female;
        immature_male_new[stage] = immature_male[stage] * survival_male
          + m_male * immature_male[stage - 1] * flow_male;
      }
      mature_female = mature_female * mature_survival
        + m_female * immature_female[4] * mature_flow;
      mature_male = mature_male * mature_survival
        + m_male * immature_male[4] * mature_flow;
      immature_female = immature_female_new;
      immature_male = immature_male_new;

      P = P_new;
      P_G = P_G_new;
      if (output_needed[i] == 1) {
        out[i, 1] = sum(P[1:(a_s - 1)]) + sum(P_G[1:(a_s - 1)])
          + mature_female + mature_male;
        out[i, 2] = mature_female;
        out[i, 3] = mature_male;
      }
    }
    return out;
  }

  real total_log_lik_one(
    int is_censored,
    real ln_y,
    real prediction,
    real sigma_total,
    real total_lod
  ) {
    real log_mu = log(prediction);
    if (is_censored == 1) {
      return normal_lcdf(log(total_lod) | log_mu, sigma_total);
    }
    return normal_lpdf(ln_y | log_mu, sigma_total);
  }

  real sex_log_lik_one(
    int is_nondetection,
    int is_male,
    real ln_y,
    real mature_stock,
    real log_scale,
    vector log_sigma_sex,
    vector alpha_detect,
    vector detection_slope,
    real sex_lod,
    real latent_floor
  ) {
    int sex_index = is_male + 1;
    real latent = latent_floor + exp(log_scale) * mature_stock;
    real eta = alpha_detect[sex_index]
      + detection_slope[sex_index] * log1p(latent / sex_lod);
    if (is_nondetection == 1) {
      return bernoulli_logit_lpmf(0 | eta);
    }
    return bernoulli_logit_lpmf(1 | eta)
      + normal_lpdf(ln_y | log(latent), exp(log_sigma_sex[sex_index]))
      - ln_y;
  }

  real partial_subject_sum(
    array[] int subject_slice,
    int slice_start,
    int slice_end,
    int Niter,
    int nStages,
    int a_s,
    array[] int subject_last_iObs,
    array[] int output_needed,
    matrix PIPconc,
    matrix thetaInd,
    int nTotalObs,
    array[] int total_subject,
    array[] int total_iObs,
    vector lnTotalCount,
    array[] int total_is_censored,
    int nSexObs,
    array[] int sex_subject,
    array[] int sex_iObs,
    array[] int sex_is_male,
    array[] int sex_is_nondetection,
    array[] int sex_include,
    vector lnSexCount,
    real sigma_total,
    real total_lod,
    real log_scale,
    vector rho_male_subject,
    vector log_sigma_sex,
    vector alpha_detect,
    vector detection_slope,
    real sex_lod,
    real latent_floor
  ) {
    real lp = 0;
    for (slice_index in 1:size(subject_slice)) {
      int subject = subject_slice[slice_index];
      int last_iObs = subject_last_iObs[subject];
      matrix[Niter, 3] trajectory = cao_sex_trajectory(
        Niter,
        last_iObs,
        nStages,
        a_s,
        PIPconc[1:Niter, subject],
        output_needed,
        thetaInd[1:12, subject],
        rho_male_subject[subject]
      );
      for (obs in 1:nTotalObs) {
        if (total_subject[obs] == subject) {
          lp += total_log_lik_one(
            total_is_censored[obs],
            lnTotalCount[obs],
            trajectory[total_iObs[obs], 1],
            sigma_total,
            total_lod
          );
        }
      }
      for (obs in 1:nSexObs) {
        if (sex_subject[obs] == subject && sex_include[obs] == 1) {
          int sex_column = sex_is_male[obs] + 2;
          lp += sex_log_lik_one(
            sex_is_nondetection[obs],
            sex_is_male[obs],
            lnSexCount[obs],
            trajectory[sex_iObs[obs], sex_column],
            log_scale,
            log_sigma_sex,
            alpha_detect,
            detection_slope,
            sex_lod,
            latent_floor
          );
        }
      }
    }
    return lp;
  }
}

data {
  int<lower=1> Niter;
  int<lower=1> nPar;
  int<lower=1> nSubjects;
  int<lower=3> nStages;
  int<lower=2, upper=nStages - 1> a_s;

  int<lower=1> nTotalObs;
  array[nTotalObs] int<lower=1, upper=nSubjects> total_subject;
  array[nTotalObs] int<lower=1, upper=Niter> total_iObs;
  vector[nTotalObs] lnTotalCount;
  array[nTotalObs] int<lower=0, upper=1> total_is_censored;

  int<lower=1> nSexObs;
  array[nSexObs] int<lower=1, upper=nSubjects> sex_subject;
  array[nSexObs] int<lower=1, upper=Niter> sex_iObs;
  array[nSexObs] int<lower=0, upper=1> sex_is_male;
  array[nSexObs] int<lower=0, upper=1> sex_is_nondetection;
  array[nSexObs] int<lower=0, upper=1> sex_include;
  vector<lower=0>[nSexObs] sex_count;
  vector[nSexObs] lnSexCount;

  matrix<lower=0>[Niter, nSubjects] PIPconc;
  vector[nPar] a;
  vector[nPar] b;
  real<lower=0> total_lod;
  real<lower=0> sex_lod;
  real<lower=0> latent_floor;
  int<lower=0, upper=1> allocation_active;
  int<lower=1> grainsize;
}

transformed data {
  array[nSubjects] int subject_ids;
  array[nSubjects] int subject_last_iObs = rep_array(1, nSubjects);
  array[Niter] int output_needed = rep_array(0, Niter);
  if (nPar != 12) {
    reject("the published Cao hierarchy requires nPar = 12");
  }
  for (parameter in 1:nPar) {
    if (a[parameter] >= b[parameter]) {
      reject("every Cao lower bound must be below its upper bound");
    }
  }
  for (obs in 1:nSexObs) {
    if (sex_count[obs] <= 0 || abs(log(sex_count[obs]) - lnSexCount[obs]) > 1e-8) {
      reject("sex_count and lnSexCount must be positive and consistent");
    }
  }
  for (obs in 1:nTotalObs) {
    int subject = total_subject[obs];
    output_needed[total_iObs[obs]] = 1;
    subject_last_iObs[subject] = max(subject_last_iObs[subject], total_iObs[obs]);
  }
  for (obs in 1:nSexObs) {
    int subject = sex_subject[obs];
    output_needed[sex_iObs[obs]] = 1;
    subject_last_iObs[subject] = max(subject_last_iObs[subject], sex_iObs[obs]);
  }
  for (subject in 1:nSubjects) {
    subject_ids[subject] = subject;
  }
}

parameters {
  real<lower=a[1], upper=b[1]> popmean_ipl;
  real<lower=a[2], upper=b[2]> popmean_iplMu;
  real<lower=a[3], upper=b[3]> popmean_iplSigma;
  real<lower=a[4], upper=b[4]> popmean_pmf;
  real<lower=a[5], upper=b[5]> popmean_kmax;
  real<lower=a[6], upper=b[6]> popmean_EC50;
  real<lower=a[7], upper=b[7]> popmean_gamma;
  real<lower=a[8], upper=b[8]> popmean_f;
  real<lower=a[9], upper=b[9]> popmean_deltaP;
  real<lower=a[10], upper=b[10]> popmean_m;
  real<lower=a[11], upper=b[11]> popmean_deltaG;
  real<lower=a[12], upper=b[12]> popmean_deltaG5;

  real<lower=0> sigma_total;
  cholesky_factor_corr[nPar] L;
  vector<lower=0>[nPar] omega;
  matrix[nPar, nSubjects] etaStd;

  real log_amplitude_female;
  real log_amplitude_male;
  vector[2] log_sigma_sex;
  vector[2] alpha_detect;
  vector<lower=0>[2] detection_slope;

  // These coordinates are present in both arms. In B0 they are prior-only,
  // making the likelihood nesting exact within the shared executable.
  vector[nSubjects] z_rho;
  real<lower=0> tau_rho;
}

transformed parameters {
  // This rotation is exactly equivalent to sampling log_scale and logit(rho)
  // but aligns the coordinates with the female and male assay amplitudes.
  real log_scale = log_sum_exp(log_amplitude_female, log_amplitude_male);
  real logit_rho_male = log_amplitude_male - log_amplitude_female;
  real<lower=0, upper=1> rho_male = inv_logit(logit_rho_male);
  vector<lower=0, upper=1>[nSubjects] rho_male_subject =
    inv_logit(logit_rho_male + allocation_active * tau_rho * z_rho);
  matrix[nPar, nSubjects] thetaInd;
  {
    vector[nPar] thetaPop;
    vector[nPar] phiPop;
    matrix[nPar, nSubjects] phiInd;

    thetaPop[1] = popmean_ipl;
    thetaPop[2] = popmean_iplMu;
    thetaPop[3] = popmean_iplSigma;
    thetaPop[4] = popmean_pmf;
    thetaPop[5] = popmean_kmax;
    thetaPop[6] = popmean_EC50;
    thetaPop[7] = popmean_gamma;
    thetaPop[8] = popmean_f;
    thetaPop[9] = popmean_deltaP;
    thetaPop[10] = popmean_m;
    thetaPop[11] = popmean_deltaG;
    thetaPop[12] = popmean_deltaG5;

    phiPop = log((thetaPop - a) ./ (b - thetaPop));
    phiInd = rep_matrix(phiPop, nSubjects)
      + diag_pre_multiply(omega, L * etaStd);
    thetaInd = rep_matrix(a, nSubjects) .* (1 - inv_logit(phiInd))
      + rep_matrix(b, nSubjects) .* inv_logit(phiInd);
  }
}

model {
  // Cao's published priors.  The bounded population means have the original
  // implicit uniform priors on their natural scales.
  L ~ lkj_corr_cholesky(2);
  omega ~ normal(0, 1);
  to_vector(etaStd) ~ normal(0, 1);
  sigma_total ~ cauchy(0, 5);

  // Frozen observation-head priors, identical in B0 and B2.
  log_scale ~ normal(0, 2.5);
  rho_male ~ beta(2, 8);
  // Jacobian for the global logit(rho) coordinate. The transform from
  // (female amplitude, male amplitude) to (log_scale, logit(rho_global)) has
  // unit determinant. Participant rho_i enters once in the biological source
  // split; because the sex stocks are linear under common maturation, the
  // observation amplitudes are correspondingly scale*(1-rho_i) and scale*rho_i.
  target += log_inv_logit(logit_rho_male) + log_inv_logit(-logit_rho_male);
  log_sigma_sex ~ normal(0, 1);
  alpha_detect ~ normal(0, 2.5);
  detection_slope ~ normal(0, 1.5);
  z_rho ~ std_normal();
  // The lower bound makes this a half-normal(0, 0.5) prior.
  tau_rho ~ normal(0, 0.5);

  target += reduce_sum(
    partial_subject_sum,
    subject_ids,
    grainsize,
    Niter,
    nStages,
    a_s,
    subject_last_iObs,
    output_needed,
    PIPconc,
    thetaInd,
    nTotalObs,
    total_subject,
    total_iObs,
    lnTotalCount,
    total_is_censored,
    nSexObs,
    sex_subject,
    sex_iObs,
    sex_is_male,
    sex_is_nondetection,
    sex_include,
    lnSexCount,
    sigma_total,
    total_lod,
    log_scale,
    rho_male_subject,
    log_sigma_sex,
    alpha_detect,
    detection_slope,
    sex_lod,
    latent_floor
  );
}

generated quantities {
  vector[nTotalObs] log_lik_total = rep_vector(0, nTotalObs);
  vector[nSexObs] log_lik_sex = rep_vector(0, nSexObs);
  vector[nSubjects] log_lik_total_subject = rep_vector(0, nSubjects);
  vector[nSubjects] log_lik_sex_subject = rep_vector(0, nSubjects);

  for (subject in 1:nSubjects) {
    int last_iObs = subject_last_iObs[subject];
    matrix[Niter, 3] trajectory = cao_sex_trajectory(
      Niter,
      last_iObs,
      nStages,
      a_s,
      PIPconc[1:Niter, subject],
      output_needed,
      thetaInd[1:12, subject],
      rho_male_subject[subject]
    );
    for (obs in 1:nTotalObs) {
      if (total_subject[obs] == subject) {
        log_lik_total[obs] = total_log_lik_one(
          total_is_censored[obs],
          lnTotalCount[obs],
          trajectory[total_iObs[obs], 1],
          sigma_total,
          total_lod
        );
        log_lik_total_subject[subject] += log_lik_total[obs];
      }
    }
    for (obs in 1:nSexObs) {
      if (sex_subject[obs] == subject) {
        int sex_column = sex_is_male[obs] + 2;
        log_lik_sex[obs] = sex_log_lik_one(
          sex_is_nondetection[obs],
          sex_is_male[obs],
          lnSexCount[obs],
          trajectory[sex_iObs[obs], sex_column],
          log_scale,
          log_sigma_sex,
          alpha_detect,
          detection_slope,
          sex_lod,
          latent_floor
        );
        log_lik_sex_subject[subject] += log_lik_sex[obs];
      }
    }
  }
}
