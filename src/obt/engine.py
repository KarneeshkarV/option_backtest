"""Wire strategy signals, synthetic premiums and costs into a vectorbt run.

The engine owns three guarantees that individual strategies are not trusted to
implement, because a strategy that forgets one produces results that look fine:

1. **Signals are shifted one bar.** A rule computed from bar ``t``'s close
   fills at ``t+1``. Without this every backtest trades on information it did
   not have.
2. **Every position closes at its session's last bar.** This is what makes it
   structurally impossible for a trade to span the multi-month holes in the
   data, and what makes the strike-pinning in :mod:`obt.chain` safe.
3. **Strikes are pinned to resolved position opens**, never to raw entry
   signals.

Short legs are supported mechanically but **margin is not modelled** --
vectorbt has no notion of SPAN/ELM. A short-premium strategy's return on
capital will therefore be optimistic, and :class:`BacktestResult` carries a
warning saying so rather than leaving it to be rediscovered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from obt.calendar import ExpiryCalendar, block_edge_sessions
from obt.chain import LegSpec, lot_size_for, pinned_leg
from obt.costs import IndiaOptionsCosts, SlippageParams, vbt_fees
from obt.session import TRADING_DAYS_PER_YEAR
from obt.signals import last_bar_of_day, resolve_trades, shift_signals
from obt.strategies.spec import get_strategy
from obt.vol.base import VolModel
from obt.vol.spec import get_vol_model

#: Rupees of starting capital. Sized so a one-lot strategy is never truncated
#: by running out of cash -- at ₹2 lakh a losing run exhausts the account
#: partway through the sample and the remaining trades silently never happen,
#: which looks like a finished backtest but is a censored one. The engine
#: checks for that explicitly rather than relying on this default being large
#: enough.
DEFAULT_INIT_CASH = 2_000_000.0


@dataclass(frozen=True)
class BacktestResult:
    """A completed run plus everything needed to judge whether to believe it."""

    portfolio: vbt.Portfolio
    leg_frame: pd.DataFrame
    trades: pd.DataFrame
    assumptions: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def stats(self) -> dict[str, float]:
        pf = self.portfolio
        return {
            "total_return_pct": float(pf.total_return()) * 100.0,
            # Sharpe from *daily* returns, not vectorbt's minute-frequency
            # annualization, so it is directly comparable to the spot baseline.
            "sharpe": self.daily_sharpe,
            "max_drawdown_pct": float(pf.max_drawdown()) * 100.0,
            "trades": int(len(self.trades)),
            "win_rate_pct": (
                float((self.trades["pnl"] > 0).mean()) * 100.0
                if len(self.trades)
                else float("nan")
            ),
            "avg_pnl": (
                float(self.trades["pnl"].mean()) if len(self.trades) else float("nan")
            ),
        }

    @property
    def daily_sharpe(self) -> float:
        """Annualized Sharpe of the daily equity curve."""
        value = self.portfolio.value()
        if isinstance(value, pd.DataFrame):
            value = value.iloc[:, 0]
        daily = value.groupby(value.index.date).last().pct_change().dropna()
        if daily.empty or daily.std() == 0:
            return float("nan")
        return float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def _resolve_legs(
    signals_leg: LegSpec | None,
    signals_legs: pd.Series | None,
    open_mask: np.ndarray,
    bars: pd.DataFrame,
) -> list[tuple[LegSpec, np.ndarray]]:
    """Group resolved opens by the leg they use.

    A strategy that trades calls on some days and puts on others produces two
    price series. Each is priced independently and run as its own vectorbt
    column, then combined -- which is correct because the two never hold a
    position simultaneously.
    """
    if signals_legs is None:
        if signals_leg is None:
            raise ValueError("strategy returned neither `leg` nor `legs`")
        return [(signals_leg, open_mask)]

    # `signals_legs` has already been shifted in lockstep with `entries`, so a
    # resolved open always lands on the bar carrying its leg.
    values = signals_legs.to_numpy()
    positions_by_leg: dict[LegSpec, list[int]] = {}
    for position in np.flatnonzero(open_mask):
        leg = values[position]
        if leg is None:
            raise ValueError(
                f"position opens at bar {position} but the strategy supplied "
                "no leg for it"
            )
        positions_by_leg.setdefault(leg, []).append(int(position))

    groups: list[tuple[LegSpec, np.ndarray]] = []
    for leg, positions in positions_by_leg.items():
        mask = np.zeros(len(open_mask), dtype=bool)
        mask[positions] = True
        groups.append((leg, mask))
    return groups


def _extract_trades(portfolio: vbt.Portfolio, bars: pd.DataFrame) -> pd.DataFrame:
    """Per-trade records with entry/exit dates attached."""
    records = portfolio.trades.records_readable
    if records.empty:
        return pd.DataFrame(
            columns=["entry_time", "exit_time", "entry_date", "exit_date", "pnl"]
        )
    rename = {
        "Entry Timestamp": "entry_time",
        "Exit Timestamp": "exit_time",
        "PnL": "pnl",
        "Return": "return",
        "Size": "size",
        "Avg Entry Price": "entry_price",
        "Avg Exit Price": "exit_price",
    }
    out = records.rename(columns=rename)
    keep = [c for c in rename.values() if c in out.columns]
    out = out[keep].copy()
    out["entry_date"] = pd.to_datetime(out["entry_time"]).dt.date
    out["exit_date"] = pd.to_datetime(out["exit_time"]).dt.date
    return out


def run(
    bars: pd.DataFrame,
    strategy_name: str,
    *,
    vol: VolModel | str = "gk_vrp",
    vol_kwargs: dict[str, Any] | None = None,
    costs: IndiaOptionsCosts | None = None,
    slippage: SlippageParams | None = None,
    init_cash: float = DEFAULT_INIT_CASH,
    params: dict[str, Any] | None = None,
    calendar_time: bool = False,
) -> BacktestResult:
    """Run ``strategy_name`` over ``bars`` and return a judged result."""
    spec = get_strategy(strategy_name)
    vol_model = (
        get_vol_model(vol, **(vol_kwargs or {})) if isinstance(vol, str) else vol
    )
    costs = costs or IndiaOptionsCosts()
    slippage = slippage or SlippageParams()
    calendar = ExpiryCalendar.from_bars(bars)

    merged = {**spec.defaults, **(params or {})}
    signals = spec.signal_fn(bars, **merged)

    # Guarantee 1: act on the next bar, not the signalling one. The per-bar leg
    # choice must shift with the entries it belongs to, or the leg lookup below
    # misses every open.
    entries = shift_signals(signals.entries)
    exits = shift_signals(signals.exits)
    shifted_legs = signals.legs.shift(1) if signals.legs is not None else None

    # Guarantee 2 + 3: resolve to real positions with a forced end-of-day exit.
    is_last = last_bar_of_day(bars)
    resolved = resolve_trades(entries, exits, is_last)
    open_mask = resolved["open_mask"]
    close_mask = resolved["close_mask"]

    lot_size, lot_from_history = lot_size_for(bars["date"].iloc[0])

    groups = _resolve_legs(signals.leg, shifted_legs, open_mask, bars)
    if not groups:
        raise ValueError(f"strategy {strategy_name!r} produced no trades")

    premium_columns: dict[str, pd.Series] = {}
    entry_columns: dict[str, np.ndarray] = {}
    exit_columns: dict[str, np.ndarray] = {}
    leg_frames: dict[str, pd.DataFrame] = {}
    is_short = False

    for leg, leg_open in groups:
        priced = pinned_leg(
            bars, leg_open, leg, vol_model, calendar, calendar_time=calendar_time
        )
        # Close only the trades this leg actually opened.
        leg_close = close_mask & _closes_for_opens(leg_open, open_mask, close_mask)
        name = leg.label
        premium_columns[name] = priced["premium"]
        entry_columns[name] = leg_open
        exit_columns[name] = leg_close
        leg_frames[name] = priced
        is_short = is_short or leg.direction == "short"

    close = pd.DataFrame(premium_columns, index=bars.index)
    entry_df = pd.DataFrame(entry_columns, index=bars.index)
    exit_df = pd.DataFrame(exit_columns, index=bars.index)

    reference_leg = groups[0][0]
    effective_lot = lot_size if lot_from_history else reference_leg.lot_size
    quantity = reference_leg.lots * effective_lot

    portfolio = vbt.Portfolio.from_signals(
        close,
        entry_df,
        exit_df,
        size=float(quantity),
        size_type="amount",
        direction="shortonly" if is_short else "longonly",
        init_cash=float(init_cash),
        fees=vbt_fees(costs),
        slippage=float(slippage.premium_pct),
        freq=_bar_freq(),
        group_by=True,
        cash_sharing=True,
    )

    trades = _extract_trades(portfolio, bars)

    warnings: list[str] = [
        "Premiums are Black-76 model output, not observed quotes. "
        "Every P&L number inherits the vol model's assumptions."
    ]

    # A losing run can spend the account partway through the sample, after
    # which vectorbt rejects orders for want of cash. The backtest still
    # "completes", but the tail of the sample is censored and every aggregate
    # is computed over a truncated period. Compare intended vs executed.
    intended = int(open_mask.sum())
    executed = int(len(trades))
    if executed < intended:
        warnings.append(
            f"CENSORED SAMPLE: {intended - executed} of {intended} intended "
            f"trades were never executed, almost certainly because cash ran "
            f"out (last trade {trades['entry_date'].max() if executed else 'n/a'}, "
            f"data ends {bars['date'].iloc[-1]}). Raise init_cash and re-run "
            "before reading any of these statistics."
        )
    # The calendar reads holidays off data presence, which misfires at the last
    # session of each covered block: no bars follow, so it looks like a holiday
    # run and the expiry rolls back onto the block edge. Trades there are priced
    # against a contract that expires sooner than the real one did.
    edge_sessions = set(block_edge_sessions(bars, calendar))
    if edge_sessions and executed:
        affected = int(trades["entry_date"].isin(edge_sessions).sum())
        if affected:
            warnings.append(
                f"{affected} of {executed} trades sit on the trailing edge of a "
                f"data block ({len(edge_sessions)} such sessions), where the "
                "expiry rolls back only because the data stops rather than "
                "because the exchange was shut. Their options are priced too "
                "cheap by a shortened time to expiry."
            )

    if not lot_from_history:
        warnings.append(
            f"No point-in-time lot history found; assumed a flat lot size of "
            f"{quantity // max(groups[0][0].lots, 1)}. NIFTY's lot size changed "
            "during this sample, so early-period notionals are approximate."
        )
    if is_short:
        warnings.append(
            "Short legs present: vectorbt models no SPAN/ELM margin, so "
            "return-on-capital is optimistic."
        )

    assumptions = {
        "strategy": strategy_name,
        "params": merged,
        "vol_model": vol_model.label,
        "slippage_pct": slippage.premium_pct * 100.0,
        "fee_fraction": vbt_fees(costs),
        "stt_rate": costs.stt_rate,
        "init_cash": init_cash,
        "lot_from_history": lot_from_history,
        "quantity_per_trade": quantity,
        "time_basis": "calendar" if calendar_time else "trading",
    }

    return BacktestResult(
        portfolio=portfolio,
        leg_frame=pd.concat(leg_frames, axis=1),
        trades=trades,
        assumptions=assumptions,
        warnings=warnings,
    )


def _closes_for_opens(
    leg_open: np.ndarray, open_mask: np.ndarray, close_mask: np.ndarray
) -> np.ndarray:
    """Map each close back to whether its opening bar belonged to this leg."""
    owner = np.zeros(len(open_mask), dtype=bool)
    active = False
    for i in range(len(open_mask)):
        if open_mask[i]:
            active = bool(leg_open[i])
        owner[i] = active
    return owner


def _bar_freq() -> str:
    """Frequency string for vectorbt's annualization."""
    return "1min"


def buy_and_hold_spot(bars: pd.DataFrame) -> dict[str, float]:
    """Baseline: hold the index over the same covered bars.

    Not directly comparable to a leveraged option strategy, but it answers the
    question every backtest should have to answer -- did this beat doing
    nothing?
    """
    close = bars["close"]
    total_return = float(close.iloc[-1] / close.iloc[0] - 1.0)
    daily = close.groupby(bars["date"]).last().pct_change().dropna()
    sharpe = (
        float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        if daily.std() > 0
        else float("nan")
    )
    running_max = close.cummax()
    max_dd = float((close / running_max - 1.0).min())
    return {
        "total_return_pct": total_return * 100.0,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100.0,
    }
