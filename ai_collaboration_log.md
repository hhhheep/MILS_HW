# AI Collaboration Log

## 1. How AI Was Used

AI was used as a learning and planning assistant, not as the final answer generator.

The main uses were:

1. Understanding the assignment requirements.
2. Choosing CIFAR-10-C as a suitable robustness benchmark.
3. Turning vague robustness intuitions into testable claims.
4. Designing robustness measures such as relative accuracy drop, wrong confidence, and failure overlap.
5. Helping organize the experiment into results.csv, claim_audit.csv, figures, and failure cases.
6. Helping draft the report structure.

## 2. AI-Generated Claims

The AI suggested several natural-language robustness claims, including:

| Claim ID | AI-style claim | How we made it testable |
|---|---|---|
| C1 | Gaussian noise is more damaging than brightness. | Compare relative drop between gaussian_noise and brightness under the same model and severity. |
| C2 | Higher corruption severity lowers accuracy. | Check whether severity 1 -> 3 -> 5 accuracy is monotonically non-increasing. |
| C3 | JPEG compression is easier than Gaussian noise. | Compare average relative drop of jpeg_compression and gaussian_noise. |
| C4 | The stronger model should be more robust. | Compare average relative drop of resnet56 and resnet20. |
| C5 | At high severity, models fail on the same hard samples more often. | Compare average failure overlap at severity 1 and severity 5. |

## 3. Where AI Helped

AI was helpful because it converted broad ideas into measurable hypotheses.

For example, instead of simply saying:

> Gaussian noise is hard.

The experiment rewrote it as:

> For the same model and severity, gaussian_noise should have larger relative accuracy drop than brightness.

This made the claim testable with results.csv.

AI also helped identify that robustness should not only be measured by accuracy. Therefore, the experiment also reports wrong confidence and failure overlap.

## 4. Where AI Could Have Misled Us

AI could have misled us in several ways:

1. It may state broad claims too confidently.
2. It may imply that a deeper or stronger model is always more robust.
3. It may imply that Gaussian noise is universally harder than JPEG compression.
4. It may treat softmax confidence as reliable confidence, even though softmax can be overconfident.
5. It may make explanations sound generally true without specifying dataset, model, corruption type, severity, and metric.

For example, the claim:

> The stronger model is more robust.

is too broad. In this report, resnet56 is only treated as the stronger/deeper setting because it has higher clean CIFAR-10 accuracy. It is not a specially trained robust model.

## 5. How We Verified AI Claims

We verified AI claims by:

1. Using official CIFAR-10 labels instead of AI-generated labels.
2. Evaluating both models on exactly the same corruption conditions.
3. Using all 10,000 images for each selected corruption/severity condition.
4. Saving numerical results into results.csv.
5. Auditing each claim with explicit evidence from tables, figures, and failure cases.
6. Reporting limitations when a claim is supported only within this controlled setting.

The final audit decisions were based on experimental evidence, not on AI's explanation.

## 6. Final Reflection

AI was useful for generating hypotheses and organizing the analysis, but it was not reliable enough to be accepted directly. The most important step was converting AI statements into testable hypotheses and checking them against results.csv.

The main lesson is:

> AI can help propose robustness claims, but the claims only become meaningful after they are tied to measurable evidence.
