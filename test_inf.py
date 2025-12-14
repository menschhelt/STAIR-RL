import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import importlib
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
n_timestamps = 200
n_symbols = 50
symbols = [f'SYM{i:03d}' for i in range(n_symbols)]
timestamps = pd.date_range('2024-01-01', periods=n_timestamps, freq='h')

base_price = np.random.randn(n_timestamps, n_symbols).cumsum(axis=0) * 0.5 + 100
data = {
    'open': pd.DataFrame(base_price + np.random.randn(n_timestamps, n_symbols) * 0.5, index=timestamps, columns=symbols),
    'high': pd.DataFrame(base_price + np.abs(np.random.randn(n_timestamps, n_symbols)) * 1.0, index=timestamps, columns=symbols),
    'low': pd.DataFrame(base_price - np.abs(np.random.randn(n_timestamps, n_symbols)) * 1.0, index=timestamps, columns=symbols),
    'close': pd.DataFrame(base_price + np.random.randn(n_timestamps, n_symbols) * 0.3, index=timestamps, columns=symbols),
    'volume': pd.DataFrame(np.abs(np.random.randn(n_timestamps, n_symbols)) * 1000000 + 100000, index=timestamps, columns=symbols),
}
typical_price = (data['high'] + data['low'] + data['close']) / 3
data['vwap'] = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
data['returns'] = data['close'].pct_change()
data['amount'] = data['volume'] * data['close']

alpha_dirs = [('alphas/alpha_101', 'alpha_101'), ('alphas/alpha_191', 'alpha_191')]
success = 0
warn_list = []
failed = []

for alpha_dir, prefix in alpha_dirs:
    files = sorted([f for f in os.listdir(alpha_dir) if f.startswith(prefix) and f.endswith('.py')])
    for f in files:
        name = f[:-3]
        try:
            if prefix == 'alpha_101':
                module = importlib.import_module(f'alphas.alpha_101.{name}')
            else:
                module = importlib.import_module(f'alphas.alpha_191.{name}')
            importlib.reload(module)
            alpha_class = getattr(module, name)
            alpha = alpha_class()
            result = alpha.calculate(data, None)

            if not isinstance(result, pd.DataFrame):
                failed.append(f'{name}: not DataFrame')
                continue
            if result.shape != (n_timestamps, n_symbols):
                failed.append(f'{name}: shape {result.shape}')
                continue

            warmup = 50
            result_after = result.iloc[warmup:]
            inf_count = np.isinf(result_after.values).sum()
            nan_count = result_after.isna().sum().sum()

            if inf_count > 0:
                warn_list.append(f'{name}: {inf_count} inf')
            elif nan_count / result_after.size > 0.05:
                warn_list.append(f'{name}: {nan_count/result_after.size*100:.1f}% NaN')
            success += 1
        except Exception as e:
            failed.append(f'{name}: {str(e)[:50]}')

print(f'Success: {success}/292')
if warn_list:
    print(f'Warnings ({len(warn_list)}):')
    for w in warn_list:
        print(f'  {w}')
else:
    print('No NaN/Inf issues!')
if failed:
    print(f'Failed ({len(failed)}):')
    for f in failed[:10]:
        print(f'  {f}')
