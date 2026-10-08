# -*- coding: utf-8 -*-
"""
Fig. 2 (new): APCNet architecture — publication-style redraw (v6, wider gaps & arrows-to-edge).
v6 fixes:
- All inter-box gaps widened to >= 0.35 (previously 0.15-0.30).
- FancyBboxPatch pad reduced to 0.02 so the declared box edge IS the visible edge;
  every arrow tip now stops exactly at the neighbour box edge (no arrow enters a box).
- Head boxes (Res Head / Rain Prob / Storm) spaced >= 0.35.
Output: fig_p3_pub/fig3_architecture.png (300 dpi) + .pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# Times New Roman 统一字体
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
})

fig, ax = plt.subplots(1, 1, figsize=(15, 9))
ax.set_xlim(0, 16.1)
ax.set_ylim(0, 9)
ax.axis('off')

# Colour-blind-safe soft palette (Okabe-Ito based tints)
C_INPUT = '#D6E4FF'      # blue tint (input)
C_KIN   = '#CDE8D6'      # green tint (kinematic)
C_THERM = '#FDE9C8'      # orange tint (thermodynamic)
C_ASPP  = '#F6CCCD'      # red tint (ASPP)
C_HEAD  = '#E3E3E3'      # grey (heads)
C_GATE  = '#BDE3EF'      # cyan tint (gating)
C_OUT   = '#B8E3C0'      # deep green (output)
C_SKIP  = '#0072B2'      # blue (skip)
C_MOD   = '#D55E00'      # vermilion (modulation)

def box(x, y, w, h, text, color, fontsize=15, bold=False, text_color='#1A1B1C', lw=1.6, pad=0.02):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=%.2f" % pad,
                       facecolor=color, edgecolor='#3A3F44', linewidth=lw)
    ax.add_patch(p)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, weight=weight, color=text_color)
    return (x, y, w, h)

def o_arrow(x1, y1, x2, y2, color='#3A3F44', lw=1.8):
    """L-shaped orthogonal arrow; tip ends exactly at (x2, y2)."""
    if abs(x2 - x1) >= abs(y2 - y1):
        mid = (x2, y1)
    else:
        mid = (x1, y2)
    ax.plot([x1, mid[0]], [y1, mid[1]], color=color, lw=lw, solid_capstyle='round')
    ax.plot([mid[0], x2], [mid[1], y2], color=color, lw=lw, solid_capstyle='round')
    ax.add_patch(FancyArrowPatch(mid, (x2, y2), arrowstyle='-|>', color=color, lw=lw, mutation_scale=18))

def arrow(x1, y1, x2, y2, color='#3A3F44', style='-|>', lw=1.8):
    """Straight arrow; tip ends exactly at (x2, y2)."""
    if style == 'ortho':
        o_arrow(x1, y1, x2, y2, color=color, lw=lw)
        return
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        color=color, linewidth=lw, mutation_scale=20)
    ax.add_patch(a)

# ============ Title ============
ax.text(7.5, 8.55, 'APCNet: Kinematic\u2013Thermodynamic Decoupled Architecture',
        ha='center', fontsize=26, weight='bold')
ax.text(7.5, 8.12, 'Input: 8 channels \u00d7 6 timesteps (CAPE, PWAT, U850, V850, U500, V500, V-VEL, GFS-Precip)   |   Grid 25\u00d737   |   ~1.38 M params',
        ha='center', fontsize=14, color='#333')

# ============ Input ============
box(0.3, 4.1, 1.7, 1.1, 'Input\n8ch \u00d7 6 steps\n25\u00d737', C_INPUT, 14, bold=True)

# ============ Kinematic branch (upper) — gaps >= 0.35 ============
box(2.35, 6.4, 1.55, 0.85, 'Temporal\nAttention\n(3ch, 6 steps)', C_KIN, 13)          # right 3.90
box(4.25, 6.3, 1.70, 1.05, 'Enc-1\nConv 3\u219224\nConv 24\u219248\nBN+SiLU', C_KIN, 13)  # right 5.95
box(6.30, 6.55, 0.95, 0.5, 'MaxPool\n2\u00d72', C_KIN, 13)                             # right 7.25
box(7.65, 6.3, 1.80, 1.05, 'Enc-2\nASPP 48\u219296\nConv 96\u219296', C_ASPP, 13)       # right 9.45
box(9.90, 6.3, 1.65, 1.05, 'Bottleneck\nASPP 96\u2192192', C_ASPP, 14, bold=True)      # right 11.55
box(11.90, 6.55, 0.95, 0.5, 'Upsample\n2\u00d72', C_KIN, 13)                           # right 12.85
box(13.25, 6.1, 1.95, 1.15, 'Dec-1\nConcat(up+e1)\n=240\nConv 240\u219296\nConv 96\u219296', C_KIN, 12)  # right 15.20
box(13.25, 4.75, 1.95, 0.95, 'Spatial\nAttention\nConv 96\u21921\n(7\u00d77)+Sigmoid', C_KIN, 12)       # right 15.20

arrow(1.15, 5.2, 2.35, 6.825, style='ortho')     # Input top-mid -> TA left-mid
arrow(3.90, 6.825, 4.25, 6.825)                  # TA -> Enc-1 (both box midpoints y=6.825)
arrow(5.95, 6.8125, 6.30, 6.8125)                # Enc-1 -> MaxPool (HORIZONTAL, mid of 6.825/6.8)
arrow(7.25, 6.8125, 7.65, 6.8125)                # MaxPool -> Enc-2 (HORIZONTAL, mid of 6.8/6.825)
arrow(9.45, 6.825, 9.90, 6.825)                  # Enc-2 -> Bottleneck (both midpoints y=6.825)
arrow(11.55, 6.8125, 11.90, 6.8125)              # Bottleneck -> Upsample (HORIZONTAL, mid of 6.825/6.8)
arrow(12.85, 6.7375, 13.25, 6.7375)              # Upsample -> Dec-1 (HORIZONTAL, mid of 6.8/6.675)
arrow(14.22, 6.1, 14.22, 5.70)                   # Dec-1 -> SA (tip at SA top-edge midpoint)

# skip connection (e1 -> Dec-1 concat): 正交折线（垂直-水平-垂直），
# 从 Enc-1 上边框中点 (5.10, 7.35) 出发，向上至 y=7.75（高于 Enc-2/Bottleneck 顶 7.35）、
# 水平右行至 Dec-1 上边框中点 (14.225, 7.25) 正上方、垂直向下进入 Dec-1 上边框
ax.plot([5.10, 5.10], [7.35, 7.75], color=C_SKIP, lw=1.5, solid_capstyle='round')
ax.plot([5.10, 14.225], [7.75, 7.75], color=C_SKIP, lw=1.5, solid_capstyle='round')
ax.plot([14.225, 14.225], [7.75, 7.25], color=C_SKIP, lw=1.5, solid_capstyle='round')
ax.add_patch(FancyArrowPatch((14.225, 7.75), (14.225, 7.25), arrowstyle='-|>', color=C_SKIP, lw=1.5, mutation_scale=18))
ax.text(9.68, 7.45, 'e1 skip (48ch) \u2192 Dec-1 concat', fontsize=13, color=C_SKIP, style='italic')

# ============ Thermodynamic branch (lower) ============
box(2.35, 2.1, 1.55, 1.0, 'Last step\n7ch thermo\n(CAPE, PWAT,\nU, V, VVEL\u2026)', C_THERM, 12.5)  # right 3.90
box(4.25, 2.15, 1.95, 1.0, 'FiLM-1\nSE-Attn\nConv 7\u219296\u2192192\n\u03b3, \u03b2 = Conv 7\u2192192', C_THERM, 12)  # right 6.20
box(6.55, 2.15, 1.95, 1.0, 'FiLM-2\nx\u00b7(1+tanh(\u03b3))\n+\u03b2', C_THERM, 13)  # right 8.50

arrow(1.15, 4.1, 2.35, 2.6, style='ortho')       # Input bottom-mid -> Last step left-mid
arrow(3.90, 2.625, 4.25, 2.625)                  # Last step -> FiLM-1 (HORIZONTAL, mid of 2.6/2.65)
arrow(6.20, 2.65, 6.55, 2.65)                    # FiLM-1 -> FiLM-2 (both midpoints y=2.65)

# FiLM-2 -> Spatial Attention modulation: 从 FiLM-2 上边框中点 (7.525, 3.15) 垂直向上至 y=5.225、
# 水平右行、箭头终点 SA 左边框中点 (13.25, 5.225)
ax.plot([7.525, 7.525], [3.15, 5.225], color=C_MOD, lw=1.6, solid_capstyle='round')
ax.plot([7.525, 13.25], [5.225, 5.225], color=C_MOD, lw=1.6, solid_capstyle='round')
ax.add_patch(FancyArrowPatch((12.95, 5.225), (13.25, 5.225), arrowstyle='-|>', color=C_MOD, lw=1.6, mutation_scale=16))
ax.text(10.4, 4.92, 'thermodynamic modulation (\u03b3, \u03b2)', fontsize=13, color=C_MOD, style='italic')

# ============ Heads (unified boxes, well separated, gap 0.35) ============
box(11.40, 3.0, 1.0, 0.7, 'Res Head\nConv 96\u21921\n(1\u00d71)\u00d710', C_HEAD, 10.5)   # right 12.40
box(12.75, 3.0, 1.0, 0.7, 'Rain Prob\nConv 96\u21921\n+Sigmoid', C_HEAD, 10.5)          # right 13.75
box(14.10, 3.0, 1.0, 0.7, 'Storm\nConv 96\u21925', C_HEAD, 10.5)                       # right 15.10

# SA bottom (14.22, 4.75) -> trunk down to y=4.3 -> three orthogonal branches into heads
ax.plot([14.22, 14.22], [4.75, 4.3], color='#3A3F44', lw=1.8, solid_capstyle='round')
for hx in (11.90, 13.25, 14.60):
    ax.plot([14.22, hx], [4.3, 4.3], color='#3A3F44', lw=1.8, solid_capstyle='round')
    ax.plot([hx, hx], [4.3, 3.7], color='#3A3F44', lw=1.8, solid_capstyle='round')
    ax.add_patch(FancyArrowPatch((hx, 4.3), (hx, 3.7), arrowstyle='-|>', color='#3A3F44', lw=1.8, mutation_scale=18))

# ============ Gated fusion ============
box(11.55, 1.15, 2.5, 1.15,
    'Gated Fusion\npred = GFS + gate\u00b7res\ngate = rain_prob^power\nstorm\u2192gate = 1',
    C_GATE, 11.5)                                                                       # right 14.05

arrow(11.90, 3.0, 11.90, 2.30)                  # Res Head -> Gated Fusion (tip at top edge 2.30)
arrow(13.25, 3.0, 13.00, 2.30, style='ortho')   # Rain Prob -> Gated Fusion
arrow(14.60, 3.0, 14.05, 1.725, style='ortho')  # Storm bottom-mid -> Gated Fusion right-mid

# GFS base
box(9.55, 1.2, 1.55, 0.8, 'GFS base\n(Precip ch)', C_INPUT, 12.5)                       # right 11.10
arrow(11.10, 1.6625, 11.55, 1.6625)             # GFS base -> Gated Fusion (HORIZONTAL, mid of 1.6/1.725)

# ============ Output ============
box(11.90, 0.05, 1.85, 0.8, 'Corrected Precip\n(mm/3h, 25\u00d737)', C_OUT, 13, bold=True)  # top 0.85
arrow(12.82, 1.15, 12.82, 0.85)                 # Gated Fusion -> Output (tip at Output top edge)

# ============ ASPP detail ============
ax.text(0.3, 0.15, 'ASPP:  1\u00d71 Conv | 3\u00d73 dil=6 | 3\u00d73 dil=12 | GlobalAvgPool+1\u00d71 \u2192 concat \u2192 fuse',
        fontsize=12.5, color='#722F37',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF3F3', edgecolor='#C53030', alpha=0.95))

# ============ Legend ============
legend_elements = [
    mpatches.Patch(facecolor=C_KIN, edgecolor='#3A3F44', label='Kinematic backbone'),
    mpatches.Patch(facecolor=C_THERM, edgecolor='#3A3F44', label='Thermodynamic modulation'),
    mpatches.Patch(facecolor=C_ASPP, edgecolor='#3A3F44', label='ASPP (atrous conv)'),
    mpatches.Patch(facecolor=C_GATE, edgecolor='#3A3F44', label='Adaptive gating'),
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
          fontsize=13.5, framealpha=0.95, ncol=4, borderaxespad=0.5)

import os
out_dir = r'D:\liaohe\论文三\04_定稿投稿代_2026_R3投稿包与归档\投稿系统上传\figures_300dpi'
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, 'Fig02.png')
out_pdf = os.path.join(out_dir, 'Fig02.pdf')
plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print('saved', out_png)
print('saved', out_pdf)
