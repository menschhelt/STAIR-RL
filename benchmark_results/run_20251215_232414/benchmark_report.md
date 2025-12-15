# Benchmark Comparison Report

Generated: 2025-12-15 23:27:51

## Summary

| strategy     | total_return   | annual_return   |   sharpe_ratio |   sortino_ratio | max_drawdown   |   calmar_ratio | cvar_95   | volatility   |   total_turnover |   avg_turnover |   total_cost |
|:-------------|:---------------|:----------------|---------------:|----------------:|:---------------|---------------:|:----------|:-------------|-----------------:|---------------:|-------------:|
| cap_weight   | 235.14%        | 254.24%         |          1.752 |           1.711 | -85.31%        |          2.981 | -1.72%    | 245.04%      |          302.425 |     0.00300882 |     0.272183 |
| equal_weight | -44.09%        | -45.56%         |          0.659 |           0.624 | -90.32%        |         -0.504 | -1.37%    | 192.80%      |          256     |     0.00254693 |     0.2304   |
| min_variance | -42.55%        | -43.99%         |          0.471 |           0.47  | -89.75%        |         -0.49  | -1.18%    | 164.54%      |          750.898 |     0.00747065 |     0.675808 |
| markowitz    | -94.59%        | -95.27%         |         -0.101 |          -0.099 | -98.52%        |         -0.967 | -1.69%    | 236.48%      |         1013.86  |     0.0100868  |     0.912472 |


## Best Performers

- **Best Sharpe Ratio**: cap_weight

- **Best Annual Return**: cap_weight


## Relative Performance (vs Equal-Weight)

| strategy     |   excess_return |   sharpe_improvement |   dd_improvement |   turnover_diff |
|:-------------|----------------:|---------------------:|-----------------:|----------------:|
| cap_weight   |       2.998     |             1.09319  |      -0.0501439  |     0.000461884 |
| min_variance |       0.0156992 |            -0.187752 |      -0.00570377 |     0.00492372  |
| markowitz    |      -0.497027  |            -0.759684 |       0.0820445  |     0.0075399   |



## Statistical Significance

- **cap_weight**: p=0.0087, significant at 95%: Yes

- **markowitz**: p=0.3422, significant at 95%: No

- **min_variance**: p=0.7748, significant at 95%: No
