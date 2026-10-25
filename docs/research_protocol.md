# Research protocol — version 0.1

## Research question
Can numerical and neural HJM pricing methods retain accurate *hedge sensitivities* and no-arbitrage consistency under changes in yield-curve and volatility regimes?

This is a proposed investigation, **not an established literature gap or novel result**.

## Stage gates
1. Establish exact HJM drift and bond-price identities.
2. Validate Gaussian HJM against tractable bond-option and caplet references.
3. Measure weak/strong time-step convergence and maturity-grid error.
4. Reproduce selected published neural HJM experiments under disclosed assumptions.
5. Only then test neural surrogates, DV01/vega accuracy, stress robustness and cost.

## Experiment controls
- Fixed training/validation/test split, including withheld curve regimes.
- Explicit random seeds and hardware/runtime recording.
- Independent numerical reference, not the same model under test.
- Compare errors in price and sensitivity, including tails, not just mean error.
- Report calibration/training cost separately from inference cost.
- Log failures, nonconvergence and numerical instabilities.
- Never label a result 'arbitrage-free' merely because training loss is small.

## Immediate limitations
Current simulator uses Euler steps on **absolute-maturity nodes** with deterministic factor volatility. It is not a Musiela SPDE discretisation, does not compute a numeraire, and cannot yet test discounted-bond martingales or hedge P&L. These are upcoming tasks, not implemented features.

## Literature workflow
Maintain a verified bibliography with DOI/arXiv identifiers, baseline implementation details, dates and licence checks. No novelty claim until a structured prior-art review is completed.
