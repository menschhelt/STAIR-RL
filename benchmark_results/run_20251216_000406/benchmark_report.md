# Benchmark Comparison Report

Generated: 2025-12-16 00:08:45

## Summary

| strategy     | total_return   | annual_return   |   sharpe_ratio |   sortino_ratio | max_drawdown   |   calmar_ratio | cvar_95   | volatility   |   total_turnover |   avg_turnover |   total_cost |
|:-------------|:---------------|:----------------|---------------:|----------------:|:---------------|---------------:|:----------|:-------------|-----------------:|---------------:|-------------:|
| min_variance | -89.63%        | -91.34%         |         -0.185 |          -0.189 | -95.16%        |          -0.96 | -1.44%    | 203.81%      |          743.874 |     0.00763801 |     0.669486 |
| markowitz    | -99.76%        | -99.85%         |         -0.615 |          -0.619 | -99.82%        |          -1    | -2.09%    | 302.14%      |          958.325 |     0.00983997 |     0.862492 |
| equal_weight | -98.78%        | -99.14%         |         -1.115 |          -1.095 | -99.16%        |          -1    | -1.32%    | 209.81%      |          255.514 |     0.00262359 |     0.229963 |
| cap_weight   | -99.91%        | -99.95%         |         -1.519 |          -1.521 | -99.95%        |          -1    | -1.66%    | 258.56%      |          407.688 |     0.0041861  |     0.36692  |


## Best Performers

- **Best Sharpe Ratio**: min_variance

- **Best Annual Return**: min_variance


## Relative Performance (vs Equal-Weight)

| strategy     |   excess_return |   sharpe_improvement |   dd_improvement |   turnover_diff |
|:-------------|----------------:|---------------------:|-----------------:|----------------:|
| min_variance |      0.0780601  |             0.930113 |      -0.039959   |      0.00501442 |
| markowitz    |     -0.00708165 |             0.499925 |       0.00660673 |      0.00721638 |
| cap_weight   |     -0.00808314 |            -0.404242 |       0.00786934 |      0.00156251 |



## Statistical Significance

- **cap_weight**: p=0.2387, significant at 95%: No

- **markowitz**: p=0.8322, significant at 95%: No

- **min_variance**: p=0.3378, significant at 95%: No
