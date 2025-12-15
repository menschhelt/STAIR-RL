# Benchmark Comparison Report

Generated: 2025-12-16 00:31:36

## Summary

| strategy     | total_return   | annual_return   |   sharpe_ratio |   sortino_ratio | max_drawdown   |   calmar_ratio | cvar_95   | volatility   |   total_turnover |   avg_turnover |   total_cost |
|:-------------|:---------------|:----------------|---------------:|----------------:|:---------------|---------------:|:----------|:-------------|-----------------:|---------------:|-------------:|
| min_variance | -89.95%        | -92.00%         |         -0.212 |          -0.217 | -95.16%        |         -0.967 | -1.45%    | 204.88%      |          735.981 |     0.00769718 |     0.662383 |
| equal_risk   | -93.41%        | -94.97%         |         -0.61  |          -0.598 | -95.60%        |         -0.993 | -1.10%    | 182.92%      |          266.881 |     0.00279114 |     0.240193 |
| markowitz    | -99.77%        | -99.88%         |         -0.653 |          -0.657 | -99.82%        |         -1.001 | -2.10%    | 304.25%      |          950.889 |     0.00994477 |     0.8558   |
| equal_weight | -98.77%        | -99.21%         |         -1.123 |          -1.103 | -99.16%        |         -1     | -1.32%    | 211.59%      |          247.4   |     0.00258741 |     0.22266  |
| cap_weight   | -99.91%        | -99.95%         |         -1.506 |          -1.507 | -99.95%        |         -1     | -1.66%    | 260.75%      |          398.619 |     0.00416891 |     0.358757 |


## Best Performers

- **Best Sharpe Ratio**: min_variance

- **Best Annual Return**: min_variance


## Relative Performance (vs Equal-Weight)

| strategy     |   excess_return |   sharpe_improvement |   dd_improvement |   turnover_diff |
|:-------------|----------------:|---------------------:|-----------------:|----------------:|
| min_variance |      0.0720281  |             0.910699 |      -0.039959   |     0.00510977  |
| equal_risk   |      0.0423498  |             0.513311 |      -0.0355725  |     0.000203738 |
| markowitz    |     -0.00669801 |             0.469781 |       0.00660673 |     0.00735736  |
| cap_weight   |     -0.00746242 |            -0.382372 |       0.00786934 |     0.0015815   |



## Statistical Significance

- **cap_weight**: p=0.2583, significant at 95%: No

- **markowitz**: p=0.8662, significant at 95%: No

- **min_variance**: p=0.3505, significant at 95%: No

- **equal_risk**: p=0.0524, significant at 95%: No
