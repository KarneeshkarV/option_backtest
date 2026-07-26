"""NSE F&O options cost stack.

``screener.backtester.costs.IndiaDeliveryCosts`` is explicitly the **cash
market** stack and does not apply here -- F&O has a different STT basis (levied
on premium, sell side only), a much higher exchange transaction charge, and
flat per-order brokerage instead of a percentage. This model implements the
same ``CostModel`` Protocol so screener's ``vbt_fee_fraction`` helper works on
it unchanged.

The number that actually decides whether an intraday option strategy is
profitable is **not** in here -- it is :class:`SlippageParams`. Statutory fees
run ~0.1% of premium; the bid-ask spread on a NIFTY weekly option routinely
costs 0.5-2% per side. Fees are modelled precisely because they are knowable;
slippage is swept because it is not.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field
from screener.backtester.costs import CostModel, Side

#: STT on options premium rose from 0.0625% to 0.1% (sell side) on 2024-10-01.
STT_RATE_CHANGE_DATE = date(2024, 10, 1)
STT_RATE_BEFORE = 0.000625
STT_RATE_AFTER = 0.001


class IndiaOptionsCosts(BaseModel):
    """NSE index-options statutory fee stack, as fractions of **premium**.

    Defaults are the post-2024-10-01 schedule. Applying them to the whole
    sample makes 2021-2024 trades slightly *more* expensive than they really
    were -- a conservative bias, which is the safe direction to be wrong in.
    Use :meth:`for_date` when that matters.

    Rates are constructor fields so regulatory changes need no code edits.
    """

    model_config = ConfigDict(frozen=True)

    brokerage_per_order: float = Field(default=20.0, ge=0)
    """Flat rupees per order (discount-broker standard). Not a fraction."""

    stt_rate: float = Field(default=STT_RATE_AFTER, ge=0)
    """0.1% of premium, **sell side only**."""

    exchange_txn_rate: float = Field(default=0.0003503, ge=0)
    """NSE options transaction charge, ~0.03503% of premium, both sides."""

    sebi_turnover_rate: float = Field(default=0.000001, ge=0)
    stamp_duty_rate: float = Field(default=0.00003, ge=0)
    """0.003% of premium, buy side only."""

    ipft_rate: float = Field(default=0.000005, ge=0)
    gst_rate: float = Field(default=0.18, ge=0)
    """18% on (brokerage + exchange txn + SEBI)."""

    @classmethod
    def for_date(cls, as_of: date, **overrides: float) -> IndiaOptionsCosts:
        """Instance with the STT rate in force on ``as_of``."""
        rate = STT_RATE_AFTER if as_of >= STT_RATE_CHANGE_DATE else STT_RATE_BEFORE
        return cls(stt_rate=rate, **overrides)

    def side_cost_fraction(self, side: Side, notional: float) -> float:
        breakdown = self.side_cost_breakdown(side, notional)
        gross = abs(float(notional))
        if gross <= 0.0:
            return 0.0
        return sum(breakdown.values()) / gross

    def side_cost_breakdown(
        self, side: Side, notional: float, shares: float | None = None
    ) -> dict[str, float]:
        gross = abs(float(notional))
        brokerage_frac = self.brokerage_per_order / gross if gross > 0 else 0.0
        gst_frac = self.gst_rate * (
            brokerage_frac + self.exchange_txn_rate + self.sebi_turnover_rate
        )
        return {
            "brokerage": self.brokerage_per_order,
            "stt": gross * self.stt_rate if side == "sell" else 0.0,
            "stamp_duty": gross * self.stamp_duty_rate if side == "buy" else 0.0,
            "exchange_txn": gross * self.exchange_txn_rate,
            "sebi": gross * self.sebi_turnover_rate,
            "gst": gross * gst_frac,
            "ipft": gross * self.ipft_rate,
        }


class SlippageParams(BaseModel):
    """Execution slippage as a fraction of premium, applied per fill.

    This dominates the statutory stack by roughly an order of magnitude for
    intraday options, and unlike fees it cannot be looked up -- it depends on
    strike liquidity, time of day and order size, none of which spot data can
    tell us. Treat the default as a placeholder and read the sensitivity table,
    not a single run.
    """

    model_config = ConfigDict(frozen=True)

    premium_pct: float = Field(default=0.0075, ge=0)
    """Half-spread plus impact, as a fraction of premium. Default 0.75%."""


#: Representative option premium (rupees) used to amortize the flat brokerage
#: into a fraction. A NIFTY weekly ATM option on one lot of 75 at ~1% of a
#: 23,000 spot is roughly 230 * 75.
FEE_NOTIONAL_DEFAULT = 230.0 * 75.0


def vbt_fees(
    costs: CostModel | None = None, *, notional: float = FEE_NOTIONAL_DEFAULT
) -> float:
    """Per-side fee fraction for vectorbt's single scalar ``fees`` parameter.

    Delegates to screener's ``vbt_fee_fraction``, which averages the buy and
    sell sides so a round trip totals the correct amount (STT is sell-only, and
    stamp duty buy-only, so neither side alone is right).
    """
    from screener.backtester.vbt.sweep import vbt_fee_fraction

    model = costs if costs is not None else IndiaOptionsCosts()
    return float(vbt_fee_fraction(model, notional=notional))
