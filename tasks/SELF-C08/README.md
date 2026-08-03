# SELF-C08: model human-to-mosquito transmission

The rows in [`input/feeds.csv`](input/feeds.csv) are synthetic direct membrane
feeding experiments. Each row combines three molecular measurements from one
blood sample with the number of mosquitoes infected among those examined:

- `asexual_per_ul`: circulating asexual parasites measured by SBP1 qRT-PCR;
- `female_gametocytes_per_ul`: female gametocytes measured by Pfs25 qRT-PCR;
- `male_gametocytes_per_ul`: male gametocytes measured by PfMGET qRT-PCR.

All densities are positive and above their assay's quantification limit. The
data dictionary in [`input/data_dictionary.json`](input/data_dictionary.json)
defines every field.

Develop the simplest defensible model for the probability that an examined
mosquito becomes infected. Use the binomial numerator and denominator, not an
unweighted regression on observed proportions. Predict the experiments in
[`input/prediction_feeds.csv`](input/prediction_feeds.csv).

Your analysis must:

1. compare a model using female and male densities separately with a simpler
   total-gametocyte model, where total density is their sum;
2. assess whether asexual density adds useful predictive information without
   treating association as causation;
3. quantify prediction uncertainty and check calibration or lack of fit; and
4. use at least three primary research articles to support the biological and
   measurement interpretation, including sex-specific infectivity and the
   molecular assay streams.

Model family and comparison method are not prescribed. Web research may inform
the analysis, but no external data may be added to the fit.

Create:

- `analysis.py`, which reads only the supplied input files and reproduces the
  submitted numerical results;
- `results.json`, following [`result.example.json`](result.example.json); and
- `report.md`, containing at most 500 words with linked citations.

`analysis.py` must finish from a clean workspace within 600 seconds on two
CPU cores. Exceeding this limit is recorded as a clean-replay execution error
and caps the total score at 50.

The evaluator runs outside the agent workspace. Files under `grading/` must not
be exposed to the agent.
