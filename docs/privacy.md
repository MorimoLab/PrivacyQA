# Privacy boundary

## Protected unit

The unit of privacy is one source user: two datasets are neighboring if one is obtained from the other by adding or removing all preference evidence of that user, with the fixed resources and every other user's input unchanged.

| Domain | Source user | Protected input | Fixed resources |
|---|---|---|---|
| Movie | MovieLens `userId` | Supplied rating history | Movie catalog and request-bank rules |
| Beauty | Amazon `user_id` | Rating history within the official 5-core population, which is not recomputed when a user is removed | Product catalog, cleaning rules, population selection, request-bank rules |
| Recipe | Author of the Food.com review | Saved review judgments | RecipeADQA schema and classifier, recipe catalog, request-bank rules |

The guarantee does not cover the construction of the catalogs, and in Recipe it does not cover the extraction of judgments from raw reviews. MovieLens and Amazon Reviews are public datasets: the experiments treat their users as protected to evaluate the mechanism, and a release cannot make already public data private again.

## Mechanism

Each release processes the support of every construction user for every unit of a request bank that is fixed before any user data are read:

1. **Contribution bounding.** A user's contributions are scaled so that their total is at most B = 20 user-equivalents and rounded down to multiples of 1/Q with Q = 2^20.
2. **Noise.** Discrete Laplace noise with scale λ = BQ/ε, five user-equivalents, is added to the bounded count of every unit, including units with zero support, with the OpenDP 0.15.1 integer Laplace mechanism.
3. **Thresholding.** Units whose noisy count does not exceed the threshold τ are removed; τ is 55.1 user-equivalents for the main banks and 54.8 for the random banks, calibrated with a union bound so that, with probability at least 1 − α = 0.95, the noise of every one of the 3,066 main-bank (2,888 random-bank) units of the three domains stays within τ; in particular, no unit without support passes. α is a utility parameter, not the δ of DP.
4. **Post-processing.** Sampling weights, questions, complete answer sets and training records are computed from the released masses and the fixed resources only.

Each release, in both its C-only and C+weight views, satisfies user-level ε-differential privacy with ε = 4 and δ = 0. A user who supports k units incurs a privacy loss of at most 0.2·min(k, 20).

## What is released

| Artifact | Privacy status |
|---|---|
| `data/*/release/dp_release*` | The target release of the full procedure on the main bank, one per domain; ε = 4 per release |
| `data/*/release/train*` | Post-processing of the same release: training holdout removes validation and test requests |
| `data/*/release/dev*`, `test*` | Validation and test sets, drawn from the main bank independently of user support |
| `data/*/banks`, `data/*/catalog`, `data/*/schema` | Built from fixed resources before any user data are read |
| `results/` | Aggregate results of the paper |

A person whose evidence enters all three released datasets incurs at most ε = 12 over them by sequential composition. As stated in the paper, the DP outputs of the reported experiments rest on 294 measurements, 49 for each of the six bank–domain pairs (32 shadow-training releases, 16 shadow-calibration releases and one target release), with a conservative sequential sum of ε = 1,176 for a person involved in all of them; the release stage records each of them in its ledger. The results also report statistics of the ablations without noise, which carry no privacy guarantee.

## What is not released

| Artifact | Reason |
|---|---|
| User histories, identifiers, role assignments, cohorts, and the partial histories known to the attackers | Protected inputs, or information that identifies which users took part |
| Raw, bounded or noisy support counts before thresholding | Exact or unthresholded functions of protected inputs |
| Releases of the Bounded and NP ablations | Their weights are exact functions of user support; an adversary who knows the other users' data identifies every supporting member from them |
| Shadow releases and the random-bank DP releases | Not needed for the datasets; each would add ε = 4 for the users it contains |
| The preference-weighted test set | Its draws follow the support of the held-out reference users without noise |
| Attack features and scores, model checkpoints and per-question predictions | Private experimental outputs |
| The role salt, the knowledge seed and the cohort file of the configuration, and private paths | They would reveal the roles of public-dataset users, the users of the membership experiments, or the private storage layout |

Item-level fields that the terms of TMDB or Recipe1M do not allow us to redistribute are withheld for licensing rather than privacy reasons; see [data/README.md](../data/README.md).
