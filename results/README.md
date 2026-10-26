# Results

Aggregate results of the reported run, written by `scripts/make_results_csv.py` from the stage manifests of the run. They contain no user history, identifier, per-user value or support count.

| File | Contents | Paper |
|---|---|---|
| `results_datasets.csv` | Source populations and dataset sizes by domain, with `bank` empty for populations: construction and reference users; support units, request variants, DP-retained variants, DP training records and DP distinct constraint sets of the main and random banks; the test and validation sets; the draws of the preference-weighted test set, the successful draws, the draw probability of units without an eligible request, and the share of successful draws whose unit is in the DP training source | Table 3; Section 6.3.2 |
| `results_summary.csv` | One row per domain and training source (27 rows): test-set macro F1, exact match (EM), precision, recall, format success and 95% paired cluster bootstrap intervals; fidelity measures (user coverage, mass coverage, and total variation distances to the raw and bounded reference targets); the primary attack at the calibrated 1% false-positive-rate target; training statistics | Tables 5, 7 and 9; Tables A1, A2 and A5 |
| `results_comparisons.csv` | Paired differences: `stage=qa` for QA metrics (95% paired cluster bootstrap intervals, not adjusted for multiplicity) and `stage=attack` for the primary attack (approximate simultaneous intervals over seven comparisons and ten metrics per domain) | Table 6; Table A6 |
| `results_preference.csv` | Preference-weighted test set: F1, EM and format success weighted by draws or per unique question, paired differences with 95% intervals, and the re-evaluations of the main DP models | Table 8 |
| `results_audit.csv` | `history_only_cv`: blind baseline; `primary_attack_strata`: primary-attack detections split by whether the partial history matches a unit; `placebo`: actual and placebo AUCs and rates; `reference_attack` and `informed_attack`: realistic and informed likelihood-ratio adversaries, overall and by number of supported units, with the per-group bound on privacy loss | Tables 10, A3 and A4 |
| `results_synthetic_audit.csv` | Synthetic audit of the mechanism: canaries and controls, event counts on the evaluation trials, and 95% lower confidence bounds on the privacy loss | Table 11 |

Naming: `recipe_text` is the Recipe domain; `Public-only`, `Non-private`, `Bounded-non-private` and `DP` are the Public, NP and Bounded ablations and the full procedure, with the prefix `Random-` for the random bank; `C+weight` and `C-only` are the released views with and without sampling weights, and `unthresholded` is the noisy vector before thresholding. Rates and differences are fractions unless the column name ends in `_pp` (percentage points). Every model was trained once, so no interval covers variation between training runs.

The tables are released under CC BY 4.0 ([LICENSES/CC-BY-4.0.txt](../LICENSES/CC-BY-4.0.txt)).
