# Test Report

Date: 2026-09-01

## Result

All automated tests passed:

```text
6 passed, 1 warning
```

The warning is an expected scikit-learn convergence warning because the end-to-end test intentionally uses only 25 training iterations to keep tests fast. The production command uses more iterations.

## Tests

1. CSV loading, duplicate timestamp aggregation and missing-hour interpolation — PASS
2. Contextual feature creation and sliding-window generation — PASS
3. Invalid window input validation — PASS
4. Robust normal screening and score normalization — PASS
5. Local spike, local drop and sustained/global event classification — PASS
6. End-to-end pipeline with three injected anomaly events — PASS

The end-to-end test verifies that all three known synthetic anomaly events overlap at least one detected anomalous window.
