"""dual-ema-vt
Vibe-Trading signal_engine translated to QuantDinger Strategy API V2.
Universe: USStock:SPY, USStock:QQQ
Frequency: 1d bars; long-only target weights from VT generate().
Risk: weights come from the embedded VT engine; QD protections (stop_loss)
are declared by the engine's order calls if any.
"""

import pandas as pd
def _ma(series, n):
    return series.rolling(n).mean()

FAST, SLOW = 20, 50

def _vt_generate(data_map):
    weights = {}
    for sym, df in data_map.items():
        close = df['close']
        fast, slow = (_ma(close, FAST), _ma(close, SLOW))
        weights[sym] = (fast > slow).astype(float) * 0.5
    return weights

def initialize(context):
    g.symbols = ['USStock:SPY', 'USStock:QQQ']
    context.set_universe(list(g.symbols))
    context.subscribe(
        frequency='1d',
        fields=["open", "high", "low", "close", "volume"],
    )
    context.set_warmup(120)
    context.set_benchmark('USStock:SPY')

def handle_data(context, data):
    import pandas as pd
    warmup = int(context.params.get('vt_warmup', 120))
    data_map = {}
    for symbol in g.symbols:
        frame = get_history(warmup, '1d', ['open', 'high', 'low', 'close', 'volume'], symbol)
        if frame is None or len(frame) == 0:
            continue
        data_map[symbol] = frame
    if not data_map:
        return
    weights = _vt_generate(data_map)
    for symbol, target in weights.items():
        qd_sym = g.symbols.get(symbol, symbol) if hasattr(g.symbols, 'get') else symbol
        if hasattr(target, 'iloc'):
            target = float(target.iloc[-1]) if len(target) else 0.0
        elif hasattr(target, 'item'):
            target = float(target)
        else:
            target = float(target)
        order_target_percent(
            qd_sym,
            target,
            reason='vt_signal_engine',
        )
