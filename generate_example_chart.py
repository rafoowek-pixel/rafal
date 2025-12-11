"""
Generuje przykładowy wykres pozycji LONG według strategii Quarterly Correlation Break
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Ustawienia
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(18, 10))
fig.suptitle('QUARTERLY CORRELATION BREAK - Przykład pozycji LONG', fontsize=16, fontweight='bold', color='white')

# Kolory
GREEN = '#26a69a'
RED = '#ef5350'
BLUE = '#42a5f5'
YELLOW = '#ffeb3b'
ORANGE = '#ff9800'
GRAY = '#666666'

def draw_candle(ax, x, open_p, high, low, close, width=0.6):
    """Rysuje świecę"""
    color = GREEN if close >= open_p else RED
    body_bottom = min(open_p, close)
    body_height = abs(close - open_p)

    # Wick (knot)
    ax.plot([x, x], [low, high], color=color, linewidth=1.5)

    # Body
    rect = patches.Rectangle((x - width/2, body_bottom), width, body_height,
                              linewidth=1, edgecolor=color, facecolor=color)
    ax.add_patch(rect)

def setup_ax(ax, title, ylabel='Cena'):
    ax.set_facecolor('#1a1a2e')
    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
    ax.set_ylabel(ylabel, color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, linestyle='--')
    for spine in ax.spines.values():
        spine.set_color(GRAY)

# ============================================
# NQ - główny instrument (z wejściem)
# ============================================
ax1 = axes[0]
setup_ax(ax1, 'NQ (Nasdaq) - WEJŚCIE')

# Poprzedni kwartał Q2 (świece 1-4)
# Świeca 1 - najniższe body (Q21)
draw_candle(ax1, 1, 100, 102, 94, 96)  # Bearish z długim dolnym knotem
# Świeca 2
draw_candle(ax1, 2, 96, 105, 95, 104)  # Bullish - najwyższe body
# Świeca 3
draw_candle(ax1, 3, 104, 106, 100, 101)  # Bearish
# Świeca 4
draw_candle(ax1, 4, 101, 103, 98, 99)  # Bearish

# Następny kwartał Q3 (świece 5-8)
# Świeca 5 - przed sweepem (bearish jak ES i YM)
draw_candle(ax1, 5, 99, 100, 96, 97)  # Bearish
# Świeca 6 - SWEEP (zbiera low, zamyka powyżej)
draw_candle(ax1, 6, 97, 99, 92, 98)  # Bullish - przebija 94, zamyka na 98
# Świeca 7 - ENTRY
draw_candle(ax1, 7, 98, 103, 97, 102)  # Bullish - wejście na open (98)
# Świeca 8
draw_candle(ax1, 8, 102, 106, 101, 105)  # Bullish - idzie do TP

# Linie poziome
ax1.axhline(y=94, color=YELLOW, linestyle='--', linewidth=2, alpha=0.8, label='LOW WICK Q2 (poziom sweep)')
ax1.axhline(y=96, color=ORANGE, linestyle=':', linewidth=1.5, alpha=0.6, label='LOW BODY Q2')
ax1.axhline(y=105, color=BLUE, linestyle='--', linewidth=2, alpha=0.8, label='HIGH Q2 (TP)')
ax1.axhline(y=92, color=RED, linestyle='-', linewidth=2, alpha=0.8, label='STOP LOSS (wick sweep)')

# Strzałki i etykiety
ax1.annotate('SWEEP\n(przebija low)', xy=(6, 92), xytext=(6, 88),
            fontsize=9, color=YELLOW, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=YELLOW, lw=2))

ax1.annotate('ENTRY\n(open)', xy=(7, 98), xytext=(8.5, 95),
            fontsize=10, color=GREEN, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))

ax1.annotate('TP 70%', xy=(7.5, 105), xytext=(9, 107),
            fontsize=10, color=BLUE, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.5))

ax1.annotate('SL', xy=(6, 92), xytext=(4.5, 90),
            fontsize=10, color=RED, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))

# Oznaczenia kwartałów
ax1.axvline(x=4.5, color=GRAY, linestyle='-', linewidth=2, alpha=0.5)
ax1.text(2.5, 108, 'Q2 (poprzedni)', ha='center', fontsize=11, color='white', fontweight='bold')
ax1.text(6.5, 108, 'Q3 (aktualny)', ha='center', fontsize=11, color='white', fontweight='bold')

ax1.text(1, 86, 'Q21', ha='center', fontsize=9, color=GRAY)
ax1.text(6, 86, 'Q31', ha='center', fontsize=9, color=GRAY)
ax1.text(1, 84, '(low tu)', ha='center', fontsize=8, color=YELLOW)
ax1.text(6, 84, '(sweep tu)', ha='center', fontsize=8, color=YELLOW)

ax1.set_xlim(0, 10)
ax1.set_ylim(82, 112)
ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
ax1.legend(loc='upper left', fontsize=8)

# ============================================
# ES - brak sweepa
# ============================================
ax2 = axes[1]
setup_ax(ax2, 'ES (S&P 500) - BRAK SWEEP')

# Poprzedni kwartał Q2
draw_candle(ax2, 1, 100, 102, 95, 97)  # Najniższe body
draw_candle(ax2, 2, 97, 104, 96, 103)
draw_candle(ax2, 3, 103, 105, 100, 101)
draw_candle(ax2, 4, 101, 103, 99, 100)

# Następny kwartał Q3
draw_candle(ax2, 5, 100, 101, 97, 98)  # Bearish (wyrównane z NQ)
draw_candle(ax2, 6, 98, 100, 96, 99)   # NIE przebija 95!
draw_candle(ax2, 7, 99, 102, 98, 101)
draw_candle(ax2, 8, 101, 104, 100, 103)

# Linie
ax2.axhline(y=95, color=YELLOW, linestyle='--', linewidth=2, alpha=0.8, label='LOW WICK Q2')
ax2.axhline(y=97, color=ORANGE, linestyle=':', linewidth=1.5, alpha=0.6, label='LOW BODY Q2')

ax2.axvline(x=4.5, color=GRAY, linestyle='-', linewidth=2, alpha=0.5)
ax2.text(2.5, 108, 'Q2', ha='center', fontsize=11, color='white', fontweight='bold')
ax2.text(6.5, 108, 'Q3', ha='center', fontsize=11, color='white', fontweight='bold')

# Zaznacz że NIE przebija
ax2.annotate('NIE PRZEBIJA\nlow Q2!', xy=(6, 96), xytext=(6, 90),
            fontsize=10, color=GREEN, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))

ax2.text(5, 85, 'ES nie zbiera low = KORELACJA ZŁAMANA',
         ha='center', fontsize=10, color=GREEN, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#1a3a1a', edgecolor=GREEN))

ax2.set_xlim(0, 10)
ax2.set_ylim(82, 112)
ax2.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
ax2.legend(loc='upper left', fontsize=8)

# ============================================
# YM - brak sweepa
# ============================================
ax3 = axes[2]
setup_ax(ax3, 'YM (Dow) - BRAK SWEEP')

# Poprzedni kwartał Q2
draw_candle(ax3, 1, 100, 102, 94, 96)
draw_candle(ax3, 2, 96, 103, 95, 102)
draw_candle(ax3, 3, 102, 104, 99, 100)
draw_candle(ax3, 4, 100, 102, 98, 99)

# Następny kwartał Q3
draw_candle(ax3, 5, 99, 100, 96, 97)  # Bearish (wyrównane)
draw_candle(ax3, 6, 97, 99, 95, 98)   # NIE przebija 94!
draw_candle(ax3, 7, 98, 101, 97, 100)
draw_candle(ax3, 8, 100, 103, 99, 102)

# Linie
ax3.axhline(y=94, color=YELLOW, linestyle='--', linewidth=2, alpha=0.8, label='LOW WICK Q2')
ax3.axhline(y=96, color=ORANGE, linestyle=':', linewidth=1.5, alpha=0.6, label='LOW BODY Q2')

ax3.axvline(x=4.5, color=GRAY, linestyle='-', linewidth=2, alpha=0.5)
ax3.text(2.5, 108, 'Q2', ha='center', fontsize=11, color='white', fontweight='bold')
ax3.text(6.5, 108, 'Q3', ha='center', fontsize=11, color='white', fontweight='bold')

ax3.annotate('NIE PRZEBIJA\nlow Q2!', xy=(6, 95), xytext=(6, 89),
            fontsize=10, color=GREEN, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))

ax3.text(5, 85, 'YM nie zbiera low = KORELACJA ZŁAMANA',
         ha='center', fontsize=10, color=GREEN, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#1a3a1a', edgecolor=GREEN))

ax3.set_xlim(0, 10)
ax3.set_ylim(82, 112)
ax3.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
ax3.legend(loc='upper left', fontsize=8)

# Podsumowanie na dole
fig.text(0.5, 0.02,
         'WARUNKI: 1) NQ zbiera low Q2  2) ES/YM NIE zbierają  3) Świece przed sweepem wyrównane (wszystkie bearish)  '
         '4) Sweep zamyka powyżej wicka  5) Entry na open następnej świecy  6) SL na wick sweepa  7) TP na high Q2',
         ha='center', fontsize=10, color='white',
         bbox=dict(boxstyle='round', facecolor='#2a2a4a', edgecolor=BLUE, alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('example_trade_long.png', dpi=150, facecolor='#0d0d1a', edgecolor='none', bbox_inches='tight')
print("Zapisano: example_trade_long.png")
plt.close()
