"""
Quarterly Correlation Break Strategy - Python Backtesting Version
==================================================================

Strategia oparta na złamaniu korelacji między NQ, ES i YM.

Kwartały dzienne (NY Time):
- Q1: 18:00-24:00
- Q2: 00:00-06:00
- Q3: 06:00-12:00
- Q4: 12:00-18:00

Każdy kwartał dzienny ma 4 x 90-minutowe sub-kwartały (np. Q11-Q14 dla Q1).

Koncept:
1. Wyznacz min/max poprzedniego kwartału dziennego (patrząc na body, nie wick)
2. Zapamiętaj wick świecy z najniższym/najwyższym body
3. Czekaj na złamanie korelacji: NQ zbiera low/high, ES/YM nie
4. Złamanie musi być w tym samym kwartale 90-min co low/high poprzedniego kwartału
5. Świeca przed złamaniem musi być taka sama na wszystkich 3 aktywach
6. Czekaj na zamknięcie świecy ponad/poniżej wicka
7. Wejście na otwarciu następnej świecy
8. SL na wicku świecy która złamała korelację
9. TP1 (70%) na opozycyjnej stronie poprzedniego kwartału
10. Pozostałe 30% - SL na break even
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
import pytz


class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class QuarterData:
    """Przechowuje dane o min/max w kwartale"""
    # Świeca z najniższym body (close)
    lowest_body_close: float = np.nan
    lowest_body_open: float = np.nan
    lowest_body_wick: float = np.nan  # Low tej świecy
    lowest_body_high: float = np.nan  # High tej świecy
    lowest_body_90min_q: int = 0

    # Świeca z najwyższym body (close)
    highest_body_close: float = np.nan
    highest_body_open: float = np.nan
    highest_body_wick: float = np.nan  # High tej świecy
    highest_body_low: float = np.nan  # Low tej świecy
    highest_body_90min_q: int = 0


@dataclass
class Trade:
    """Reprezentuje pojedynczy trade"""
    entry_time: datetime
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    initial_quantity: int
    current_quantity: int
    sweep_wick: float  # Wick świecy która złamała korelację
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    realized_pnl: float = 0.0
    tp1_hit: bool = False


@dataclass
class SweepState:
    """Stan detekcji sweepa"""
    # Long sweep (zbieranie low)
    long_sweep_detected: bool = False
    long_sweep_candle_low: float = np.nan  # Wick (low) świecy która złamała
    long_sweep_candle_high: float = np.nan
    long_sweep_bar_idx: int = 0
    long_sweep_90min_q: int = 0
    long_waiting_for_close: bool = False  # Czekamy na zamknięcie ponad poziomem

    # Short sweep (zbieranie high)
    short_sweep_detected: bool = False
    short_sweep_candle_high: float = np.nan  # Wick (high) świecy która złamała
    short_sweep_candle_low: float = np.nan
    short_sweep_bar_idx: int = 0
    short_sweep_90min_q: int = 0
    short_waiting_for_close: bool = False

    # Poziom do przebicia (wick poprzedniego kwartału)
    target_level: float = np.nan


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

        # Stan kwartałów dla każdego instrumentu
        self.prev_quarter_nq: QuarterData = QuarterData()
        self.curr_quarter_nq: QuarterData = QuarterData()
        self.prev_quarter_es: QuarterData = QuarterData()
        self.curr_quarter_es: QuarterData = QuarterData()
        self.prev_quarter_ym: QuarterData = QuarterData()
        self.curr_quarter_ym: QuarterData = QuarterData()

        self.sweep_state: SweepState = SweepState()
        self.current_trade: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.closed_trades: List[Trade] = []

        self.prev_daily_q: int = 0
        self.capital: float = initial_capital

        # Flaga sygnału wejścia (używana do wejścia na następnej świecy)
        self.pending_long_entry: bool = False
        self.pending_short_entry: bool = False
        self.pending_entry_sl: float = np.nan
        self.pending_entry_tp: float = np.nan

        # Debug/logging
        self.debug_log: List[Dict] = []

    def get_daily_quarter(self, hour: int) -> int:
        """
        Pobierz kwartał dzienny na podstawie godziny (NY time)
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
        Pobierz 90-minutowy sub-kwartał (1-4) w ramach kwartału dziennego
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

    def is_excluded_transition(self, prev_daily_q: int, curr_daily_q: int) -> bool:
        """
        Sprawdź czy jesteśmy na wykluczonym przejściu między kwartałami dziennymi
        Wykluczenia: między Q1-Q2, Q2-Q3, Q3-Q4, Q4-Q1
        """
        if prev_daily_q != curr_daily_q and prev_daily_q != 0:
            return True
        return False

    def is_same_90min_subquarter(self, prev_q_90min: int, curr_90min: int,
                                   prev_daily_q: int, curr_daily_q: int) -> bool:
        """
        Sprawdź czy aktualny 90-min kwartał odpowiada 90-min kwartałowi z poprzedniego dnia
        Przykład: Jeśli low Q2 było w Q21, to szukamy złamania w Q31
        """
        # Złamanie musi być w pierwszym 90-min kwartale następnego dziennego kwartału
        # który odpowiada 90-min kwartałowi gdzie było low/high

        # Mapowanie: Q21 -> Q31 (ten sam numer sub-kwartału)
        return prev_q_90min == curr_90min

    def calculate_position_size(self, entry: float, stop_loss: float) -> int:
        """Oblicz wielkość pozycji na podstawie ryzyka"""
        points = abs(entry - stop_loss)
        risk_per_contract = points * self.point_value
        if risk_per_contract > 0:
            return max(1, int(self.risk_amount / risk_per_contract))
        return 1

    def reset_quarter_data(self, quarter_data: QuarterData):
        """Reset danych kwartału"""
        quarter_data.lowest_body_close = np.nan
        quarter_data.lowest_body_open = np.nan
        quarter_data.lowest_body_wick = np.nan
        quarter_data.lowest_body_high = np.nan
        quarter_data.lowest_body_90min_q = 0
        quarter_data.highest_body_close = np.nan
        quarter_data.highest_body_open = np.nan
        quarter_data.highest_body_wick = np.nan
        quarter_data.highest_body_low = np.nan
        quarter_data.highest_body_90min_q = 0

    def copy_quarter_data(self, source: QuarterData, dest: QuarterData):
        """Kopiuj dane kwartału"""
        dest.lowest_body_close = source.lowest_body_close
        dest.lowest_body_open = source.lowest_body_open
        dest.lowest_body_wick = source.lowest_body_wick
        dest.lowest_body_high = source.lowest_body_high
        dest.lowest_body_90min_q = source.lowest_body_90min_q
        dest.highest_body_close = source.highest_body_close
        dest.highest_body_open = source.highest_body_open
        dest.highest_body_wick = source.highest_body_wick
        dest.highest_body_low = source.highest_body_low
        dest.highest_body_90min_q = source.highest_body_90min_q

    def update_quarter_data(self, quarter_data: QuarterData,
                            o: float, h: float, l: float, c: float,
                            min_q: int):
        """Aktualizuj dane kwartału nową świecą"""
        # Szukamy świecy z najniższym ZAMKNIĘCIEM (body)
        body_low = min(o, c)
        body_high = max(o, c)

        # Świeca z najniższym body (close)
        if np.isnan(quarter_data.lowest_body_close) or c < quarter_data.lowest_body_close:
            quarter_data.lowest_body_close = c
            quarter_data.lowest_body_open = o
            quarter_data.lowest_body_wick = l  # Dolny wick tej świecy
            quarter_data.lowest_body_high = h
            quarter_data.lowest_body_90min_q = min_q

        # Świeca z najwyższym body (close)
        if np.isnan(quarter_data.highest_body_close) or c > quarter_data.highest_body_close:
            quarter_data.highest_body_close = c
            quarter_data.highest_body_open = o
            quarter_data.highest_body_wick = h  # Górny wick tej świecy
            quarter_data.highest_body_low = l
            quarter_data.highest_body_90min_q = min_q

    def check_candles_aligned(self, prev_nq: pd.Series, prev_es: pd.Series,
                               prev_ym: pd.Series) -> bool:
        """
        Sprawdź czy poprzednie świece są wyrównane (wszystkie bullish lub wszystkie bearish)
        """
        nq_bull = prev_nq['close'] > prev_nq['open']
        es_bull = prev_es['close'] > prev_es['open']
        ym_bull = prev_ym['close'] > prev_ym['open']

        all_bull = nq_bull and es_bull and ym_bull
        all_bear = (not nq_bull) and (not es_bull) and (not ym_bull)

        return all_bull or all_bear

    def process_bar(self, idx: int,
                    nq_row: pd.Series, es_row: pd.Series, ym_row: pd.Series,
                    prev_nq: pd.Series, prev_es: pd.Series, prev_ym: pd.Series) -> dict:
        """
        Przetwarza pojedynczy bar i zwraca sygnały/stan
        """
        result = {
            'long_sweep_detected': False,
            'short_sweep_detected': False,
            'long_signal': False,
            'short_signal': False,
            'trade_entry': None,
            'trade_exit': None,
            'daily_q': 0,
            'min_q': 0,
            'excluded': False
        }

        # Pobierz informacje o czasie
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

        # ================================================================
        # PENDING ENTRY - wejście na otwarciu tej świecy
        # ================================================================
        if self.pending_long_entry and self.current_trade is None:
            entry_price = nq_row['open']
            quantity = self.calculate_position_size(entry_price, self.pending_entry_sl)

            trade = Trade(
                entry_time=dt,
                direction=TradeDirection.LONG,
                entry_price=entry_price,
                stop_loss=self.pending_entry_sl,
                take_profit=self.pending_entry_tp,
                initial_quantity=quantity,
                current_quantity=quantity,
                sweep_wick=self.pending_entry_sl
            )
            self.current_trade = trade
            result['trade_entry'] = trade

            self.pending_long_entry = False
            self.pending_entry_sl = np.nan
            self.pending_entry_tp = np.nan

        if self.pending_short_entry and self.current_trade is None:
            entry_price = nq_row['open']
            quantity = self.calculate_position_size(entry_price, self.pending_entry_sl)

            trade = Trade(
                entry_time=dt,
                direction=TradeDirection.SHORT,
                entry_price=entry_price,
                stop_loss=self.pending_entry_sl,
                take_profit=self.pending_entry_tp,
                initial_quantity=quantity,
                current_quantity=quantity,
                sweep_wick=self.pending_entry_sl
            )
            self.current_trade = trade
            result['trade_entry'] = trade

            self.pending_short_entry = False
            self.pending_entry_sl = np.nan
            self.pending_entry_tp = np.nan

        # ================================================================
        # SPRAWDŹ ZMIANĘ KWARTAŁU
        # ================================================================
        q_change = daily_q != self.prev_daily_q and self.prev_daily_q != 0

        if q_change:
            # Kopiuj current -> previous
            self.copy_quarter_data(self.curr_quarter_nq, self.prev_quarter_nq)
            self.copy_quarter_data(self.curr_quarter_es, self.prev_quarter_es)
            self.copy_quarter_data(self.curr_quarter_ym, self.prev_quarter_ym)

            # Reset current
            self.reset_quarter_data(self.curr_quarter_nq)
            self.reset_quarter_data(self.curr_quarter_es)
            self.reset_quarter_data(self.curr_quarter_ym)

            # Reset sweep state
            self.sweep_state = SweepState()
            self.pending_long_entry = False
            self.pending_short_entry = False

        # ================================================================
        # AKTUALIZUJ DANE KWARTAŁU
        # ================================================================
        self.update_quarter_data(
            self.curr_quarter_nq,
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

        # ================================================================
        # SPRAWDŹ WYKLUCZENIA
        # ================================================================
        is_excluded = self.is_excluded_transition(self.prev_daily_q, daily_q)
        result['excluded'] = is_excluded

        # Sprawdź wyrównanie świec
        candles_aligned = self.check_candles_aligned(prev_nq, prev_es, prev_ym)

        # ================================================================
        # ZARZĄDZANIE ISTNIEJĄCĄ POZYCJĄ
        # ================================================================
        if self.current_trade is not None:
            trade = self.current_trade

            if trade.direction == TradeDirection.LONG:
                # Sprawdź SL
                if nq_row['low'] <= trade.stop_loss:
                    trade.exit_time = dt
                    trade.exit_price = trade.stop_loss
                    trade.exit_reason = "SL" if not trade.tp1_hit else "BE"
                    pnl = (trade.exit_price - trade.entry_price) * trade.current_quantity * self.point_value
                    trade.realized_pnl += pnl
                    result['trade_exit'] = trade
                    self.closed_trades.append(trade)
                    self.capital += pnl
                    self.current_trade = None
                # Sprawdź TP
                elif nq_row['high'] >= trade.take_profit:
                    if not trade.tp1_hit:
                        # Częściowe wyjście na TP1 (70%)
                        tp1_qty = int(trade.initial_quantity * self.tp1_percent)
                        remaining_qty = trade.initial_quantity - tp1_qty
                        tp1_pnl = (trade.take_profit - trade.entry_price) * tp1_qty * self.point_value
                        self.capital += tp1_pnl
                        trade.realized_pnl += tp1_pnl
                        trade.tp1_hit = True
                        trade.current_quantity = remaining_qty
                        trade.stop_loss = trade.entry_price  # Break-even

            elif trade.direction == TradeDirection.SHORT:
                # Sprawdź SL
                if nq_row['high'] >= trade.stop_loss:
                    trade.exit_time = dt
                    trade.exit_price = trade.stop_loss
                    trade.exit_reason = "SL" if not trade.tp1_hit else "BE"
                    pnl = (trade.entry_price - trade.exit_price) * trade.current_quantity * self.point_value
                    trade.realized_pnl += pnl
                    result['trade_exit'] = trade
                    self.closed_trades.append(trade)
                    self.capital += pnl
                    self.current_trade = None
                # Sprawdź TP
                elif nq_row['low'] <= trade.take_profit:
                    if not trade.tp1_hit:
                        # Częściowe wyjście na TP1 (70%)
                        tp1_qty = int(trade.initial_quantity * self.tp1_percent)
                        remaining_qty = trade.initial_quantity - tp1_qty
                        tp1_pnl = (trade.entry_price - trade.take_profit) * tp1_qty * self.point_value
                        self.capital += tp1_pnl
                        trade.realized_pnl += tp1_pnl
                        trade.tp1_hit = True
                        trade.current_quantity = remaining_qty
                        trade.stop_loss = trade.entry_price  # Break-even

        # ================================================================
        # DETEKCJA SWEEP I SYGNAŁÓW
        # Tylko jeśli nie mamy otwartej pozycji
        # ================================================================
        if self.current_trade is None and not is_excluded:
            # ============================================================
            # LONG SWEEP - NQ zbiera low poprzedniego kwartału
            # ============================================================
            if not np.isnan(self.prev_quarter_nq.lowest_body_wick):
                prev_low_wick_nq = self.prev_quarter_nq.lowest_body_wick
                prev_low_body_nq = min(self.prev_quarter_nq.lowest_body_close,
                                        self.prev_quarter_nq.lowest_body_open)
                prev_low_90min_nq = self.prev_quarter_nq.lowest_body_90min_q

                prev_low_wick_es = self.prev_quarter_es.lowest_body_wick
                prev_low_body_es = min(self.prev_quarter_es.lowest_body_close,
                                        self.prev_quarter_es.lowest_body_open) if not np.isnan(self.prev_quarter_es.lowest_body_close) else np.nan

                prev_low_wick_ym = self.prev_quarter_ym.lowest_body_wick
                prev_low_body_ym = min(self.prev_quarter_ym.lowest_body_close,
                                        self.prev_quarter_ym.lowest_body_open) if not np.isnan(self.prev_quarter_ym.lowest_body_close) else np.nan

                # Warunki dla long sweep:
                # 1. NQ zbiera low wick poprzedniego kwartału
                nq_swept_low = nq_row['low'] < prev_low_wick_nq

                # 2. ES nie zbiera ani wicka ani body poprzedniego kwartału
                es_no_sweep = True
                if not np.isnan(prev_low_wick_es):
                    es_no_sweep = (es_row['low'] > prev_low_wick_es and
                                   min(es_row['open'], es_row['close']) > prev_low_body_es)

                # 3. YM nie zbiera ani wicka ani body poprzedniego kwartału
                ym_no_sweep = True
                if not np.isnan(prev_low_wick_ym):
                    ym_no_sweep = (ym_row['low'] > prev_low_wick_ym and
                                   min(ym_row['open'], ym_row['close']) > prev_low_body_ym)

                # 4. Musi być w tym samym 90-min kwartale
                same_90min = self.is_same_90min_subquarter(prev_low_90min_nq, min_q,
                                                           self.prev_daily_q, daily_q)

                # 5. Świece muszą być wyrównane
                # 6. Nie mamy jeszcze sweepа
                if (nq_swept_low and es_no_sweep and ym_no_sweep and same_90min and
                    candles_aligned and not self.sweep_state.long_sweep_detected and
                    not self.sweep_state.short_sweep_detected):

                    self.sweep_state.long_sweep_detected = True
                    self.sweep_state.long_sweep_candle_low = nq_row['low']
                    self.sweep_state.long_sweep_candle_high = nq_row['high']
                    self.sweep_state.long_sweep_bar_idx = idx
                    self.sweep_state.long_sweep_90min_q = min_q
                    self.sweep_state.target_level = prev_low_wick_nq
                    self.sweep_state.long_waiting_for_close = True
                    result['long_sweep_detected'] = True

            # ============================================================
            # SHORT SWEEP - NQ zbiera high poprzedniego kwartału
            # ============================================================
            if not np.isnan(self.prev_quarter_nq.highest_body_wick):
                prev_high_wick_nq = self.prev_quarter_nq.highest_body_wick
                prev_high_body_nq = max(self.prev_quarter_nq.highest_body_close,
                                         self.prev_quarter_nq.highest_body_open)
                prev_high_90min_nq = self.prev_quarter_nq.highest_body_90min_q

                prev_high_wick_es = self.prev_quarter_es.highest_body_wick
                prev_high_body_es = max(self.prev_quarter_es.highest_body_close,
                                         self.prev_quarter_es.highest_body_open) if not np.isnan(self.prev_quarter_es.highest_body_close) else np.nan

                prev_high_wick_ym = self.prev_quarter_ym.highest_body_wick
                prev_high_body_ym = max(self.prev_quarter_ym.highest_body_close,
                                         self.prev_quarter_ym.highest_body_open) if not np.isnan(self.prev_quarter_ym.highest_body_close) else np.nan

                # Warunki dla short sweep:
                # 1. NQ zbiera high wick poprzedniego kwartału
                nq_swept_high = nq_row['high'] > prev_high_wick_nq

                # 2. ES nie zbiera ani wicka ani body poprzedniego kwartału
                es_no_sweep = True
                if not np.isnan(prev_high_wick_es):
                    es_no_sweep = (es_row['high'] < prev_high_wick_es and
                                   max(es_row['open'], es_row['close']) < prev_high_body_es)

                # 3. YM nie zbiera ani wicka ani body poprzedniego kwartału
                ym_no_sweep = True
                if not np.isnan(prev_high_wick_ym):
                    ym_no_sweep = (ym_row['high'] < prev_high_wick_ym and
                                   max(ym_row['open'], ym_row['close']) < prev_high_body_ym)

                # 4. Musi być w tym samym 90-min kwartale
                same_90min = self.is_same_90min_subquarter(prev_high_90min_nq, min_q,
                                                           self.prev_daily_q, daily_q)

                # 5. Świece muszą być wyrównane
                # 6. Nie mamy jeszcze sweepa
                if (nq_swept_high and es_no_sweep and ym_no_sweep and same_90min and
                    candles_aligned and not self.sweep_state.short_sweep_detected and
                    not self.sweep_state.long_sweep_detected):

                    self.sweep_state.short_sweep_detected = True
                    self.sweep_state.short_sweep_candle_high = nq_row['high']
                    self.sweep_state.short_sweep_candle_low = nq_row['low']
                    self.sweep_state.short_sweep_bar_idx = idx
                    self.sweep_state.short_sweep_90min_q = min_q
                    self.sweep_state.target_level = prev_high_wick_nq
                    self.sweep_state.short_waiting_for_close = True
                    result['short_sweep_detected'] = True

            # ============================================================
            # CZEKANIE NA ZAMKNIĘCIE PONAD/PONIŻEJ POZIOMU
            # ============================================================
            if self.sweep_state.long_waiting_for_close:
                # Czekamy na zamknięcie świecy POWYŻEJ wicka poprzedniego kwartału
                if nq_row['close'] > self.sweep_state.target_level:
                    # Sygnał LONG - wejście na otwarciu NASTĘPNEJ świecy
                    result['long_signal'] = True
                    self.pending_long_entry = True
                    self.pending_entry_sl = self.sweep_state.long_sweep_candle_low  # SL na wicku świecy sweep
                    self.pending_entry_tp = self.prev_quarter_nq.highest_body_wick  # TP na przeciwnej stronie
                    self.sweep_state.long_waiting_for_close = False
                    self.sweep_state.long_sweep_detected = False

            if self.sweep_state.short_waiting_for_close:
                # Czekamy na zamknięcie świecy PONIŻEJ wicka poprzedniego kwartału
                if nq_row['close'] < self.sweep_state.target_level:
                    # Sygnał SHORT - wejście na otwarciu NASTĘPNEJ świecy
                    result['short_signal'] = True
                    self.pending_short_entry = True
                    self.pending_entry_sl = self.sweep_state.short_sweep_candle_high  # SL na wicku świecy sweep
                    self.pending_entry_tp = self.prev_quarter_nq.lowest_body_wick  # TP na przeciwnej stronie
                    self.sweep_state.short_waiting_for_close = False
                    self.sweep_state.short_sweep_detected = False

        self.prev_daily_q = daily_q

        return result

    def run_backtest(self, nq_data: pd.DataFrame, es_data: pd.DataFrame,
                     ym_data: pd.DataFrame) -> pd.DataFrame:
        """
        Uruchom backtest na dostarczonych danych

        Format danych:
        - Index: DatetimeIndex (UTC lub z timezone)
        - Kolumny: open, high, low, close (lowercase)
        """
        results = []

        # Wyrównaj dane po indeksie
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
        """Pobierz statystyki wszystkich transakcji"""
        trades = self.closed_trades

        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'max_win': 0,
                'max_loss': 0,
                'profit_factor': 0,
                'final_capital': self.capital
            }

        pnls = [t.realized_pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        return {
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(trades) * 100 if trades else 0,
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'max_win': max(wins) if wins else 0,
            'max_loss': min(losses) if losses else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            'final_capital': self.capital
        }

    def print_trades(self):
        """Wydrukuj wszystkie transakcje"""
        print("\n" + "="*80)
        print("LOG TRANSAKCJI")
        print("="*80)

        for i, trade in enumerate(self.closed_trades, 1):
            print(f"\nTransakcja #{i}: {trade.direction.value}")
            print(f"  Wejście: {trade.entry_time} @ {trade.entry_price:.2f}")
            print(f"  Wyjście: {trade.exit_time} @ {trade.exit_price:.2f} ({trade.exit_reason})")
            print(f"  SL:      {trade.sweep_wick:.2f}")
            print(f"  TP:      {trade.take_profit:.2f}")
            print(f"  Ilość:   {trade.initial_quantity}")
            print(f"  TP1 Hit: {trade.tp1_hit}")
            print(f"  P&L:     ${trade.realized_pnl:,.2f}")

        print("\n" + "="*80)
        summary = self.get_trade_summary()
        print("PODSUMOWANIE")
        print("="*80)
        print(f"Łączna liczba transakcji: {summary['total_trades']}")
        print(f"Wygrane:                  {summary['winning_trades']}")
        print(f"Przegrane:                {summary['losing_trades']}")
        print(f"Win Rate:                 {summary['win_rate']:.1f}%")
        print(f"Łączny P&L:               ${summary['total_pnl']:,.2f}")
        print(f"Średni P&L:               ${summary['avg_pnl']:,.2f}")
        print(f"Max zysk:                 ${summary['max_win']:,.2f}")
        print(f"Max strata:               ${summary['max_loss']:,.2f}")
        print(f"Profit Factor:            {summary['profit_factor']:.2f}")
        print(f"Kapitał końcowy:          ${summary['final_capital']:,.2f}")


def generate_mock_data(days: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generuj przykładowe dane do testów.
    Użyj prawdziwych danych do rzeczywistego backtestingu!
    """
    np.random.seed(42)

    start_date = datetime(2024, 1, 1, 18, 0)  # Start od Q1
    periods = days * 24 * 4  # 15-minutowe bary

    dates = pd.date_range(start=start_date, periods=periods, freq='15min', tz='America/New_York')

    def generate_ohlc(base_price, volatility, n, correlation_factor=0.8):
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

    # Generuj skorelowane dane z drobnymi różnicami
    nq_base = generate_ohlc(16500, 25, periods)

    # ES i YM będą miały podobny kierunek, ale różne poziomy
    nq_data = pd.DataFrame(nq_base, index=dates)

    # ES - skorelowany z NQ ale z mniejszą zmiennością
    es_data = []
    es_base = 4800
    for i, nq in enumerate(nq_base):
        nq_change = (nq['close'] - 16500) / 16500
        es_change = nq_change * 0.9 + np.random.randn() * 0.001
        es_close = es_base * (1 + es_change)
        es_range = abs(np.random.randn() * 3)
        es_data.append({
            'open': es_close + np.random.randn() * 2,
            'high': es_close + es_range,
            'low': es_close - es_range,
            'close': es_close
        })
    es_data = pd.DataFrame(es_data, index=dates)

    # YM - skorelowany z NQ ale z większą zmiennością
    ym_data = []
    ym_base = 37500
    for i, nq in enumerate(nq_base):
        nq_change = (nq['close'] - 16500) / 16500
        ym_change = nq_change * 1.1 + np.random.randn() * 0.002
        ym_close = ym_base * (1 + ym_change)
        ym_range = abs(np.random.randn() * 40)
        ym_data.append({
            'open': ym_close + np.random.randn() * 20,
            'high': ym_close + ym_range,
            'low': ym_close - ym_range,
            'close': ym_close
        })
    ym_data = pd.DataFrame(ym_data, index=dates)

    return nq_data, es_data, ym_data


def load_data_from_csv(nq_path: str, es_path: str, ym_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Załaduj dane z plików CSV

    Format CSV:
    datetime,open,high,low,close
    2024-01-02 18:00:00,16500.25,16510.50,16495.00,16505.75
    """
    nq_data = pd.read_csv(nq_path, index_col='datetime', parse_dates=True)
    es_data = pd.read_csv(es_path, index_col='datetime', parse_dates=True)
    ym_data = pd.read_csv(ym_path, index_col='datetime', parse_dates=True)

    return nq_data, es_data, ym_data


if __name__ == "__main__":
    import sys

    print("\n" + "="*80)
    print("QUARTERLY CORRELATION BREAK STRATEGY - PYTHON BACKTEST")
    print("="*80)

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Uruchom z przykładowymi danymi
        print("\n*** URUCHAMIANIE Z PRZYKŁADOWYMI DANYMI ***\n")

        nq_data, es_data, ym_data = generate_mock_data(days=90)

        strategy = QuarterlyCorrelationStrategy(
            risk_amount=1000.0,
            tp1_percent=70.0,
            point_value=20.0,
            initial_capital=100000.0
        )

        results = strategy.run_backtest(nq_data, es_data, ym_data)
        strategy.print_trades()

        # Pokaż sygnały
        signals = results[(results['long_signal']) | (results['short_signal'])]
        if not signals.empty:
            print("\n" + "="*80)
            print("LOG SYGNAŁÓW")
            print("="*80)
            for _, row in signals.head(20).iterrows():
                signal_type = "LONG" if row['long_signal'] else "SHORT"
                print(f"{row['datetime']} - {signal_type} @ {row['close']:.2f} (Q{row['daily_q']}.{row['min_q']})")

        # Pokaż sweep detections
        sweeps = results[(results['long_sweep_detected']) | (results['short_sweep_detected'])]
        if not sweeps.empty:
            print("\n" + "="*80)
            print("LOG SWEEP DETECTION")
            print("="*80)
            for _, row in sweeps.head(20).iterrows():
                sweep_type = "LONG SWEEP" if row['long_sweep_detected'] else "SHORT SWEEP"
                print(f"{row['datetime']} - {sweep_type} @ {row['close']:.2f} (Q{row['daily_q']}.{row['min_q']})")

    else:
        print("\nUżycie:")
        print("  python quarterly_correlation_strategy.py --demo    # Uruchom z przykładowymi danymi")
        print("\nAby użyć z prawdziwymi danymi:")
        print("  from quarterly_correlation_strategy import QuarterlyCorrelationStrategy, load_data_from_csv")
        print("  nq, es, ym = load_data_from_csv('NQ_15min.csv', 'ES_15min.csv', 'YM_15min.csv')")
        print("  strategy = QuarterlyCorrelationStrategy()")
        print("  results = strategy.run_backtest(nq, es, ym)")
        print("  strategy.print_trades()")
        print("\nFormat danych CSV:")
        print("  datetime,open,high,low,close")
        print("  2024-01-02 18:00:00,16500.25,16510.50,16495.00,16505.75")
