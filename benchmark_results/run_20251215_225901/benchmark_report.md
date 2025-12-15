# Benchmark Comparison Report

Generated: 2025-12-15 23:00:24

## Summary

| strategy     | total_return   | annual_return   |   sharpe_ratio |   sortino_ratio | max_drawdown   |   calmar_ratio | cvar_95   | volatility   |   total_turnover |   avg_turnover |   total_cost |
|:-------------|:---------------|:----------------|---------------:|----------------:|:---------------|---------------:|:----------|:-------------|-----------------:|---------------:|-------------:|
| min_variance | 310.16%        | 0.35%           |          0.085 |           0.084 | -66.54%        |          0.005 | -1.07%    | 7.62%        |          734.975 |    0.00731223  |     0.661477 |
| equal_weight | 48.03%         | 0.10%           |          0.058 |           0.054 | -92.33%        |          0.001 | -1.32%    | 9.31%        |            2     |    1.98979e-05 |     0.0018   |
| cap_weight   | -42.87%        | -0.14%          |          0.036 |           0.035 | -94.70%        |         -0.001 | -1.39%    | 9.93%        |          207.487 |    0.00206428  |     0.186738 |
| markowitz    | -54.46%        | -0.20%          |          0.029 |           0.028 | -93.65%        |         -0.002 | -1.38%    | 9.69%        |          989.931 |    0.00984878  |     0.890938 |


## Best Performers

- **Best Sharpe Ratio**: min_variance

- **Best Annual Return**: min_variance


## Relative Performance (vs Equal-Weight)

| strategy     |   excess_return |   sharpe_improvement |   dd_improvement |   turnover_diff |
|:-------------|----------------:|---------------------:|-----------------:|----------------:|
| min_variance |      0.00256088 |            0.0266162 |       -0.257884  |      0.00729234 |
| cap_weight   |     -0.00238662 |           -0.0216655 |        0.0236252 |      0.00204438 |
| markowitz    |     -0.00295393 |           -0.029063  |        0.0131797 |      0.00982889 |



## Statistical Significance

- **cap_weight**: p=0.3537, significant at 95%: No

- **markowitz**: p=0.4426, significant at 95%: No

- **min_variance**: p=0.8184, significant at 95%: No
