"""
Quarterly Correlation Break Strategy - Python Version
======================================================

This strategy:
1. Tracks daily quarters (Q1-Q4) and 90-minute sub-quarters (Q11-Q14, etc.)
2. Detects when NQ sweeps previous quarter's low/high wick while ES/YM don't
3. Waits for first candle to close above/below the sweep level = entry signal
4. Position sizing based on $1000 risk per trade
5. 70% TP at opposite quarter level, 30% to break-even

Quarters (NY Time):
- Q1: 18:00-24:00
- Q2: 00:00-06:00
- Q3: 06:00-12:00
- Q4: 12:00-18:00

Each daily quarter has 4 x 90-min sub-quarters.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum
import pytz


class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class QuarterData:
    """Stores data for tracking min/max body candles in a quarter"""
    low_body: float = np.nan
    low_wick: float = np.nan
    high_body: float = np.nan
    high_wick: float = np.nan
    low_90min_q: int = 0
    high_90min_q: int = 0


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: int
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    tp1_hit: bool = False


@dataclass
class SweepState:
    """Tracks sweep detection state"""
    long_sweep_detected: bool = False
    short_sweep_detected: bool = False
    sweep_wick_level: float = np.nan
    sweep_bar_idx: int = 0


class QuarterlyCorrelationStrategy:
    def __init__(
        self,
        risk_amount: float = 1000.0,
        tp1_percent: float = 70.0,
        point_value: float = 20.0,  # NQ E-mini = $20 per point
        initial_capital: float = 100000.0,
        timezone: str = "America/New_York"
    ):
        self.risk_amount = risk_amount
        self.tp1_percent = tp1_percent / 100.0
        self.point_value = point_value
        self.initial_capital = initial_capital
        self.timezone = pytz.timezone(timezone)

        # State
        self.prev_quarter: QuarterData = QuarterData()
        self.curr_quarter: QuarterData = QuarterData()
        self.prev_quarter_es: QuarterData = QuarterData()
        self.curr_quarter_es: QuarterData = QuarterData()
        self.prev_quarter_ym: QuarterData = QuarterData()
        self.curr_quarter_ym: QuarterData = QuarterData()

        self.sweep_state: SweepState = SweepState()
        self.current_trade: Optional[Trade] = None
        self.trades: List[Trade] = []

        self.prev_daily_q: int = 0
        self.capital: float = initial_capital

    def get_daily_quarter(self, hour: int) -> int:
        """
        Get daily quarter based on hour (NY time)
        Q1: 18:00-24:00 (18-23)
        Q2: 00:00-06:00 (0-5)
        Q3: 06:00-12:00 (6-11)
        Q4: 12:00-18:00 (12-17)
        """
        if hour >= 18:
            return 1
        elif hour >= 12:
            return 4
        elif hour >= 6:
            return 3
        else:
            return 2

    def get_90min_quarter(self, hour: int, minute: int) -> int:
        """
        Get 90-minute sub-quarter (1-4) within each daily quarter
        Each daily quarter has 4 x 90-minute periods
        """
        mins = hour * 60 + minute

        if 1080 <= mins < 1440:  # Q1: 18:00-24:00 (1080-1440 mins)
            q = int((mins - 1080) / 90) + 1
        elif mins < 360:  # Q2: 00:00-06:00 (0-360 mins)
            q = int(mins / 90) + 1
        elif mins < 720:  # Q3: 06:00-12:00 (360-720 mins)
            q = int((mins - 360) / 90) + 1
        elif mins < 1080:  # Q4: 12:00-18:00 (720-1080 mins)
            q = int((mins - 720) / 90) + 1
        else:
            q = 4

        return min(q, 4)

    def is_excluded_transition(self, prev_daily_q: int, curr_daily_q: int,
                                prev_90_q: int, curr_90_q: int) -> bool:
        """
        Check if we're at an excluded transition (between daily quarters)
        Excluded: Q14->Q21, Q24->Q31, Q34->Q41, Q44->Q11
        """
        if prev_daily_q == 1 and curr_daily_q == 2 and prev_90_q == 4 and curr_90_q == 1:
            return True
        if prev_daily_q == 2 and curr_daily_q == 3 and prev_90_q == 4 and curr_90_q == 1:
            return True
        if prev_daily_q == 3 and curr_daily_q == 4 and prev_90_q == 4 and curr_90_q == 1:
            return True
        if prev_daily_q == 4 and curr_daily_q == 1 and prev_90_q == 4 and curr_90_q == 1:
            return True
        return False

    def calculate_position_size(self, entry: float, stop_loss: float) -> int:
        """Calculate position size based on risk amount"""
        points = abs(entry - stop_loss)
        risk_per_contract = points * self.point_value
        if risk_per_contract > 0:
            return max(1, round(self.risk_amount / risk_per_contract))
        return 1

    def reset_quarter_data(self, quarter_data: QuarterData):
        """Reset quarter tracking data"""
        quarter_data.low_body = np.nan
        quarter_data.low_wick = np.nan
        quarter_data.high_body = np.nan
        quarter_data.high_wick = np.nan
        quarter_data.low_90min_q = 0
        quarter_data.high_90min_q = 0

    def copy_quarter_data(self, source: QuarterData, dest: QuarterData):
        """Copy quarter data from source to destination"""
        dest.low_body = source.low_body
        dest.low_wick = source.low_wick
        dest.high_body = source.high_body
        dest.high_wick = source.high_wick
        dest.low_90min_q = source.low_90min_q
        dest.high_90min_q = source.high_90min_q

    def update_quarter_data(self, quarter_data: QuarterData,
                            o: float, h: float, l: float, c: float,
                            min_q: int):
        """Update quarter tracking with new candle data"""
        low_body = min(o, c)
        high_body = max(o, c)

        # Track candle with lowest body
        if np.isnan(quarter_data.low_body) or low_body < quarter_data.low_body:
            quarter_data.low_body = low_body
            quarter_data.low_wick = l  # Wick of candle with lowest body
            quarter_data.low_90min_q = min_q

        # Track candle with highest body
        if np.isnan(quarter_data.high_body) or high_body > quarter_data.high_body:
            quarter_data.high_body = high_body
            quarter_data.high_wick = h  # Wick of candle with highest body
            quarter_data.high_90min_q = min_q

    def process_bar(self, idx: int,
                    nq_row: pd.Series, es_row: pd.Series, ym_row: pd.Series,
                    prev_nq: pd.Series, prev_es: pd.Series, prev_ym: pd.Series) -> dict:
        """
        Process a single bar and return signals/state
        """
        result = {
            'long_sweep_now': False,
            'short_sweep_now': False,
            'long_signal': False,
            'short_signal': False,
            'trade_entry': None,
            'trade_exit': None,
            'daily_q': 0,
            'min_q': 0
        }

        # Get time info
        dt = nq_row.name
        if isinstance(dt, pd.Timestamp):
            dt_ny = dt.tz_convert(self.timezone) if dt.tz else dt.tz_localize('UTC').tz_convert(self.timezone)
        else:
            dt_ny = dt

        hour = dt_ny.hour
        minute = dt_ny.minute

        daily_q = self.get_daily_quarter(hour)
        min_q = self.get_90min_quarter(hour, minute)

        result['daily_q'] = daily_q
        result['min_q'] = min_q

        # Check for quarter change
        q_change = daily_q != self.prev_daily_q and self.prev_daily_q != 0

        if q_change:
            # Copy current to previous
            self.copy_quarter_data(self.curr_quarter, self.prev_quarter)
            self.copy_quarter_data(self.curr_quarter_es, self.prev_quarter_es)
            self.copy_quarter_data(self.curr_quarter_ym, self.prev_quarter_ym)

            # Reset current
            self.reset_quarter_data(self.curr_quarter)
            self.reset_quarter_data(self.curr_quarter_es)
            self.reset_quarter_data(self.curr_quarter_ym)

            # Reset sweep flags on quarter change
            self.sweep_state.long_sweep_detected = False
            self.sweep_state.short_sweep_detected = False

        # Update current quarter tracking
        self.update_quarter_data(
            self.curr_quarter,
            nq_row['open'], nq_row['high'], nq_row['low'], nq_row['close'],
            min_q
        )
        self.update_quarter_data(
            self.curr_quarter_es,
            es_row['open'], es_row['high'], es_row['low'], es_row['close'],
            min_q
        )
        self.update_quarter_data(
            self.curr_quarter_ym,
            ym_row['open'], ym_row['high'], ym_row['low'], ym_row['close'],
            min_q
        )

        # Check for excluded transitions
        prev_min_q = self.get_90min_quarter(
            (dt_ny - timedelta(minutes=15)).hour,  # approximate previous bar
            (dt_ny - timedelta(minutes=15)).minute
        )
        is_excluded = self.is_excluded_transition(self.prev_daily_q, daily_q, prev_min_q, min_q)

        # Candle alignment check (previous candle must be same direction on all 3)
        nq_prev_bull = prev_nq['close'] > prev_nq['open']
        es_prev_bull = prev_es['close'] > prev_es['open']
        ym_prev_bull = prev_ym['close'] > prev_ym['open']
        candles_aligned = (nq_prev_bull and es_prev_bull and ym_prev_bull) or \
                          (not nq_prev_bull and not es_prev_bull and not ym_prev_bull)

        # Get body values
        nq_low_body = min(nq_row['open'], nq_row['close'])
        nq_high_body = max(nq_row['open'], nq_row['close'])
        es_low_body = min(es_row['open'], es_row['close'])
        es_high_body = max(es_row['open'], es_row['close'])
        ym_low_body = min(ym_row['open'], ym_row['close'])
        ym_high_body = max(ym_row['open'], ym_row['close'])

        # ================================================================
        # LONG SWEEP DETECTION
        # NQ sweeps previous quarter's low wick, ES/YM don't
        # ================================================================
        if not np.isnan(self.prev_quarter.low_wick):
            nq_swept_low = nq_row['low'] < self.prev_quarter.low_wick
            es_not_swept_low = np.isnan(self.prev_quarter_es.low_wick) or \
                               es_row['low'] > self.prev_quarter_es.low_wick
            ym_not_swept_low = np.isnan(self.prev_quarter_ym.low_wick) or \
                               ym_row['low'] > self.prev_quarter_ym.low_wick
            es_body_ok = np.isnan(self.prev_quarter_es.low_body) or \
                         es_low_body > self.prev_quarter_es.low_body
            ym_body_ok = np.isnan(self.prev_quarter_ym.low_body) or \
                         ym_low_body > self.prev_quarter_ym.low_body
            same_90_low = min_q == self.prev_quarter.low_90min_q

            long_sweep_now = (nq_swept_low and es_not_swept_low and ym_not_swept_low and
                             es_body_ok and ym_body_ok and same_90_low and
                             not is_excluded and candles_aligned)

            if long_sweep_now and not self.sweep_state.long_sweep_detected and \
               not self.sweep_state.short_sweep_detected:
                self.sweep_state.long_sweep_detected = True
                self.sweep_state.sweep_wick_level = self.prev_quarter.low_wick
                self.sweep_state.sweep_bar_idx = idx
                result['long_sweep_now'] = True

        # ================================================================
        # SHORT SWEEP DETECTION
        # NQ sweeps previous quarter's high wick, ES/YM don't
        # ================================================================
        if not np.isnan(self.prev_quarter.high_wick):
            nq_swept_high = nq_row['high'] > self.prev_quarter.high_wick
            es_not_swept_high = np.isnan(self.prev_quarter_es.high_wick) or \
                                es_row['high'] < self.prev_quarter_es.high_wick
            ym_not_swept_high = np.isnan(self.prev_quarter_ym.high_wick) or \
                                ym_row['high'] < self.prev_quarter_ym.high_wick
            es_body_ok_s = np.isnan(self.prev_quarter_es.high_body) or \
                           es_high_body < self.prev_quarter_es.high_body
            ym_body_ok_s = np.isnan(self.prev_quarter_ym.high_body) or \
                           ym_high_body < self.prev_quarter_ym.high_body
            same_90_high = min_q == self.prev_quarter.high_90min_q

            short_sweep_now = (nq_swept_high and es_not_swept_high and ym_not_swept_high and
                              es_body_ok_s and ym_body_ok_s and same_90_high and
                              not is_excluded and candles_aligned)

            if short_sweep_now and not self.sweep_state.short_sweep_detected and \
               not self.sweep_state.long_sweep_detected:
                self.sweep_state.short_sweep_detected = True
                self.sweep_state.sweep_wick_level = self.prev_quarter.high_wick
                self.sweep_state.sweep_bar_idx = idx
                result['short_sweep_now'] = True

        # ================================================================
        # ENTRY SIGNALS - first candle that closes above/below sweep level
        # ================================================================
        if self.sweep_state.long_sweep_detected:
            if nq_row['close'] > self.sweep_state.sweep_wick_level:
                result['long_signal'] = True
                self.sweep_state.long_sweep_detected = False

        if self.sweep_state.short_sweep_detected:
            if nq_row['close'] < self.sweep_state.sweep_wick_level:
                result['short_signal'] = True
                self.sweep_state.short_sweep_detected = False

        # ================================================================
        # POSITION MANAGEMENT
        # ================================================================

        # Check for exit on existing position
        if self.current_trade is not None:
            trade = self.current_trade

            if trade.direction == TradeDirection.LONG:
                # Check SL hit
                if nq_row['low'] <= trade.stop_loss:
                    trade.exit_time = dt
                    trade.exit_price = trade.stop_loss
                    trade.exit_reason = "SL" if not trade.tp1_hit else "BE"
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity * self.point_value
                    result['trade_exit'] = trade
                    self.trades.append(trade)
                    self.capital += trade.pnl
                    self.current_trade = None
                # Check TP hit
                elif nq_row['high'] >= trade.take_profit:
                    if not trade.tp1_hit:
                        # Partial exit at TP1 (70%)
                        tp1_qty = int(trade.quantity * self.tp1_percent)
                        remaining_qty = trade.quantity - tp1_qty
                        tp1_pnl = (trade.take_profit - trade.entry_price) * tp1_qty * self.point_value
                        self.capital += tp1_pnl
                        trade.tp1_hit = True
                        trade.quantity = remaining_qty
                        trade.stop_loss = trade.entry_price  # Move to break-even

            elif trade.direction == TradeDirection.SHORT:
                # Check SL hit
                if nq_row['high'] >= trade.stop_loss:
                    trade.exit_time = dt
                    trade.exit_price = trade.stop_loss
                    trade.exit_reason = "SL" if not trade.tp1_hit else "BE"
                    trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity * self.point_value
                    result['trade_exit'] = trade
                    self.trades.append(trade)
                    self.capital += trade.pnl
                    self.current_trade = None
                # Check TP hit
                elif nq_row['low'] <= trade.take_profit:
                    if not trade.tp1_hit:
                        # Partial exit at TP1 (70%)
                        tp1_qty = int(trade.quantity * self.tp1_percent)
                        remaining_qty = trade.quantity - tp1_qty
                        tp1_pnl = (trade.entry_price - trade.take_profit) * tp1_qty * self.point_value
                        self.capital += tp1_pnl
                        trade.tp1_hit = True
                        trade.quantity = remaining_qty
                        trade.stop_loss = trade.entry_price  # Move to break-even

        # New entries
        if self.current_trade is None:
            if result['long_signal']:
                entry_price = nq_row['close']
                stop_loss = nq_row['low']  # Entry candle's low
                take_profit = self.prev_quarter.high_wick  # Opposite quarter level
                quantity = self.calculate_position_size(entry_price, stop_loss)

                trade = Trade(
                    entry_time=dt,
                    direction=TradeDirection.LONG,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    quantity=quantity
                )
                self.current_trade = trade
                result['trade_entry'] = trade

            elif result['short_signal']:
                entry_price = nq_row['close']
                stop_loss = nq_row['high']  # Entry candle's high
                take_profit = self.prev_quarter.low_wick  # Opposite quarter level
                quantity = self.calculate_position_size(entry_price, stop_loss)

                trade = Trade(
                    entry_time=dt,
                    direction=TradeDirection.SHORT,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    quantity=quantity
                )
                self.current_trade = trade
                result['trade_entry'] = trade

        self.prev_daily_q = daily_q

        return result

    def run_backtest(self, nq_data: pd.DataFrame, es_data: pd.DataFrame, ym_data: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest on provided data

        Data format expected:
        - Index: DatetimeIndex (UTC or with timezone)
        - Columns: open, high, low, close (lowercase)
        """
        results = []

        # Align data by index
        common_idx = nq_data.index.intersection(es_data.index).intersection(ym_data.index)
        nq_data = nq_data.loc[common_idx]
        es_data = es_data.loc[common_idx]
        ym_data = ym_data.loc[common_idx]

        for i in range(1, len(nq_data)):
            idx = i
            nq_row = nq_data.iloc[i]
            es_row = es_data.iloc[i]
            ym_row = ym_data.iloc[i]

            prev_nq = nq_data.iloc[i-1]
            prev_es = es_data.iloc[i-1]
            prev_ym = ym_data.iloc[i-1]

            result = self.process_bar(idx, nq_row, es_row, ym_row, prev_nq, prev_es, prev_ym)
            result['datetime'] = nq_row.name
            result['close'] = nq_row['close']
            result['capital'] = self.capital
            results.append(result)

        return pd.DataFrame(results)

    def get_trade_summary(self) -> dict:
        """Get summary statistics of all trades"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'max_win': 0,
                'max_loss': 0,
                'profit_factor': 0
            }

        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(self.trades) * 100 if self.trades else 0,
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'max_win': max(wins) if wins else 0,
            'max_loss': min(losses) if losses else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        }

    def print_trades(self):
        """Print all trades"""
        print("\n" + "="*80)
        print("TRADE LOG")
        print("="*80)

        for i, trade in enumerate(self.trades, 1):
            print(f"\nTrade #{i}: {trade.direction.value}")
            print(f"  Entry: {trade.entry_time} @ {trade.entry_price:.2f}")
            print(f"  Exit:  {trade.exit_time} @ {trade.exit_price:.2f} ({trade.exit_reason})")
            print(f"  Qty:   {trade.quantity}")
            print(f"  P&L:   ${trade.pnl:,.2f}")

        print("\n" + "="*80)
        summary = self.get_trade_summary()
        print("SUMMARY")
        print("="*80)
        print(f"Total Trades:    {summary['total_trades']}")
        print(f"Win Rate:        {summary['win_rate']:.1f}%")
        print(f"Total P&L:       ${summary['total_pnl']:,.2f}")
        print(f"Avg P&L:         ${summary['avg_pnl']:,.2f}")
        print(f"Max Win:         ${summary['max_win']:,.2f}")
        print(f"Max Loss:        ${summary['max_loss']:,.2f}")
        print(f"Profit Factor:   {summary['profit_factor']:.2f}")
        print(f"Final Capital:   ${self.capital:,.2f}")


def load_sample_data():
    """
    Example function to load data.
    Replace this with your actual data loading logic.

    You can use various sources:
    - CSV files
    - APIs (TradeStation, Interactive Brokers, etc.)
    - Database
    """
    # Example: Load from CSV
    # nq_data = pd.read_csv('NQ_15min.csv', index_col='datetime', parse_dates=True)
    # es_data = pd.read_csv('ES_15min.csv', index_col='datetime', parse_dates=True)
    # ym_data = pd.read_csv('YM_15min.csv', index_col='datetime', parse_dates=True)

    print("="*80)
    print("QUARTERLY CORRELATION BREAK STRATEGY - PYTHON VERSION")
    print("="*80)
    print("\nTo run this strategy, you need to provide 15-minute OHLC data for:")
    print("  - NQ (Nasdaq E-mini futures)")
    print("  - ES (S&P 500 E-mini futures)")
    print("  - YM (Dow E-mini futures)")
    print("\nData format (CSV example):")
    print("  datetime,open,high,low,close")
    print("  2024-01-02 18:00:00,16500.25,16510.50,16495.00,16505.75")
    print("  ...")
    print("\nExample usage:")
    print("  strategy = QuarterlyCorrelationStrategy()")
    print("  nq = pd.read_csv('NQ_15min.csv', index_col='datetime', parse_dates=True)")
    print("  es = pd.read_csv('ES_15min.csv', index_col='datetime', parse_dates=True)")
    print("  ym = pd.read_csv('YM_15min.csv', index_col='datetime', parse_dates=True)")
    print("  results = strategy.run_backtest(nq, es, ym)")
    print("  strategy.print_trades()")
    print("="*80)

    return None, None, None


def generate_mock_data(days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate mock data for testing purposes.
    This is just for demonstration - use real data for actual backtesting!
    """
    np.random.seed(42)

    start_date = datetime(2024, 1, 1, 18, 0)  # Start at Q1
    periods = days * 24 * 4  # 15-minute bars

    dates = pd.date_range(start=start_date, periods=periods, freq='15min', tz='America/New_York')

    def generate_ohlc(base_price, volatility, n):
        prices = [base_price]
        for _ in range(n - 1):
            change = np.random.randn() * volatility
            prices.append(prices[-1] + change)

        data = []
        for i, p in enumerate(prices):
            high = p + abs(np.random.randn() * volatility * 0.5)
            low = p - abs(np.random.randn() * volatility * 0.5)
            open_p = np.random.uniform(low, high)
            close_p = np.random.uniform(low, high)
            data.append({
                'open': open_p,
                'high': high,
                'low': low,
                'close': close_p
            })
        return data

    nq_data = pd.DataFrame(generate_ohlc(16500, 20, periods), index=dates)
    es_data = pd.DataFrame(generate_ohlc(4800, 5, periods), index=dates)
    ym_data = pd.DataFrame(generate_ohlc(37500, 50, periods), index=dates)

    return nq_data, es_data, ym_data


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Run with mock data for demonstration
        print("\n*** RUNNING WITH MOCK DATA FOR DEMONSTRATION ***\n")

        nq_data, es_data, ym_data = generate_mock_data(days=60)

        strategy = QuarterlyCorrelationStrategy(
            risk_amount=1000.0,
            tp1_percent=70.0,
            point_value=20.0,
            initial_capital=100000.0
        )

        results = strategy.run_backtest(nq_data, es_data, ym_data)
        strategy.print_trades()

        # Show some signals
        signals = results[(results['long_signal']) | (results['short_signal'])]
        if not signals.empty:
            print("\n" + "="*80)
            print("SIGNAL LOG")
            print("="*80)
            for _, row in signals.head(20).iterrows():
                signal_type = "LONG" if row['long_signal'] else "SHORT"
                print(f"{row['datetime']} - {signal_type} @ {row['close']:.2f} (Q{row['daily_q']}{row['min_q']})")
    else:
        load_sample_data()
