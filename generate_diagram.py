"""
dlshogi WideResNet10 Architecture Diagram (PNG output)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig = plt.figure(figsize=(20, 28))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 20)
ax.set_ylim(0, 28)
ax.axis('off')
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

COL_INPUT   = '#4a90d9'
COL_CONV    = '#7b68ee'
COL_BN      = '#20b2aa'
COL_RESBLOCK= '#9370db'
COL_POLICY  = '#ff8c00'
COL_VALUE   = '#32cd32'
COL_OUTPUT  = '#dc143c'
COL_TEXT    = 'white'
COL_ARROW   = '#aaaaaa'
COL_DETAIL  = '#cccccc'


def box(ax, x, y, w, h, color, label, sublabel=None, fontsize=9, radius=0.15):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          facecolor=color, edgecolor='white', linewidth=0.8, alpha=0.92)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + 0.13, label, ha='center', va='center',
                color=COL_TEXT, fontsize=fontsize, fontweight='bold')
        ax.text(x, y - 0.17, sublabel, ha='center', va='center',
                color=COL_DETAIL, fontsize=7)
    else:
        ax.text(x, y, label, ha='center', va='center',
                color=COL_TEXT, fontsize=fontsize, fontweight='bold')


def arrow(ax, x1, y1, x2, y2, color=COL_ARROW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))


def vline(ax, x, y1, y2, color=COL_ARROW, lw=1.0):
    ax.plot([x, x], [y1, y2], color=color, lw=lw, zorder=2)


# Title
ax.text(10, 27.4, 'dlshogi WideResNet10 Architecture',
        ha='center', va='center', color='white', fontsize=16, fontweight='bold')
ax.text(10, 27.0, 'Shogi Policy-Value Network  (WCSC29 model, 2019)',
        ha='center', va='center', color='#aaaaaa', fontsize=10)

# ── INPUT ────────────────────────────────────────────────────────────────────
y_in = 25.8
box(ax, 5.5, y_in, 4.4, 0.7, COL_INPUT,
    'FEATURES1  (1, 62, 9, 9)',
    'Piece pos (0-13,31-44) + Attacks (14-27,45-58) + Attack cnt (28-30,59-61)',
    fontsize=8)
box(ax, 14.5, y_in, 4.0, 0.7, COL_INPUT,
    'FEATURES2  (1, 57, 9, 9)',
    'Hand pieces (0-55) + Check flag (56)',
    fontsize=8)

box(ax, 10, 24.7, 5.0, 0.6, COL_INPUT,
    'Concat  ->  (1, 119, 9, 9)',
    'FEATURES1 + FEATURES2  (62+57=119 ch)',
    fontsize=8)

arrow(ax, 5.5, y_in - 0.35, 8.5, 24.7 + 0.3)
arrow(ax, 14.5, y_in - 0.35, 11.5, 24.7 + 0.3)

# ── Entry Block ──────────────────────────────────────────────────────────────
y_entry = 23.5
box(ax, 7.0, y_entry, 3.6, 0.6, COL_CONV,
    'l1_1_1  Conv 3x3  119->192ch', 'kernel=3, pad=1', fontsize=7.5)
box(ax, 10.0, y_entry, 3.2, 0.6, COL_CONV,
    'l1_1_2  Conv 1x1  119->192ch', 'kernel=1', fontsize=7.5)
box(ax, 13.2, y_entry, 3.2, 0.6, COL_CONV,
    'l1_2  Conv 1x1  119->192ch', 'skip connection', fontsize=7.5)

arrow(ax, 10, 24.7 - 0.3, 7.0, y_entry + 0.3)
arrow(ax, 10, 24.7 - 0.3, 10.0, y_entry + 0.3)
arrow(ax, 10, 24.7 - 0.3, 13.2, y_entry + 0.3)

y_bn1 = 22.4
box(ax, 8.5, y_bn1, 3.8, 0.6, COL_BN,
    'Add + BN (norm1_1) + ReLU', 'l1_1_1 + l1_1_2  (192ch)', fontsize=7.5)
box(ax, 13.2, y_bn1, 3.2, 0.6, COL_BN,
    'BN (norm1_2)', 'skip branch  (192ch)', fontsize=7.5)

arrow(ax, 7.0, y_entry - 0.3, 8.0, y_bn1 + 0.3)
arrow(ax, 10.0, y_entry - 0.3, 9.0, y_bn1 + 0.3)
arrow(ax, 13.2, y_entry - 0.3, 13.2, y_bn1 + 0.3)

y_entry_out = 21.3
box(ax, 10.0, y_entry_out, 4.8, 0.6, COL_CONV,
    'Add Entry Output  ->  ReLU',
    '(1, 192, 9, 9)', fontsize=8.5)

arrow(ax, 8.5, y_bn1 - 0.3, 9.0, y_entry_out + 0.3)
arrow(ax, 13.2, y_bn1 - 0.3, 11.5, y_entry_out + 0.3)

# ── Residual Blocks x10 ───────────────────────────────────────────────────────
y_res_top = 20.2
box(ax, 10.0, y_res_top, 10.0, 0.7, COL_RESBLOCK,
    'x10  Residual Blocks  ->  (1, 192, 9, 9)',
    'Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN  +  skip -> ReLU',
    fontsize=9)
arrow(ax, 10, y_entry_out - 0.3, 10, y_res_top + 0.35)

# Residual block detail
rx, ry_top = 16.8, 19.5
ax.text(rx, ry_top, '[ Single Residual Block ]', ha='center', va='center',
        color='#cccccc', fontsize=7, style='italic')
ry = ry_top - 0.5
for lbl, sub in [
    ('Conv 3x3', '192->192ch, pad=1'),
    ('BN + ReLU', 'Batch Normalization'),
    ('Conv 3x3', '192->192ch, pad=1'),
    ('BN', 'Batch Normalization'),
    ('Add skip', '+ input'),
    ('ReLU', 'activation'),
]:
    box(ax, rx, ry, 3.2, 0.42, COL_RESBLOCK, lbl, sub, fontsize=6.5, radius=0.1)
    if ry > ry_top - 3.3:
        vline(ax, rx, ry - 0.21, ry - 0.52, color='#888888')
    ry -= 0.55

# skip arrow on detail
ax.annotate('', xy=(rx - 1.6, ry_top - 3.0), xytext=(rx - 1.6, ry_top - 0.45),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8))

# ── Split ─────────────────────────────────────────────────────────────────────
y_split = 19.1
arrow(ax, 10, y_res_top - 0.35, 10, y_split + 0.35)

ax.plot([5.5, 14.5], [y_split, y_split], '--', color='#888888', lw=0.8)
ax.plot([5.5, 5.5], [y_split, y_split - 0.2], color='#888888', lw=0.8)
ax.plot([14.5, 14.5], [y_split, y_split - 0.2], color='#888888', lw=0.8)
ax.text(10, y_split + 0.15, '(1, 192, 9, 9)', ha='center', color='#aaaaaa', fontsize=7)

# ── Policy Head ──────────────────────────────────────────────────────────────
py = 18.5
ax.text(5.5, py + 0.6, 'Policy Head', ha='center', color=COL_POLICY,
        fontsize=10, fontweight='bold')

for lbl, sub, yy in [
    ('l22  Conv 1x1', '192->2187ch (27 dirs x 81 sq)', py),
    ('l22_2  Bias', 'bias add  (2187,)', py - 0.75),
    ('Flatten', '(1, 2187)', py - 1.5),
    ('Softmax', 'probability distribution', py - 2.25),
]:
    box(ax, 5.5, yy, 4.2, 0.55, COL_POLICY, lbl, sub, fontsize=7.5, radius=0.1)
    if yy > py - 2.25:
        arrow(ax, 5.5, yy - 0.275, 5.5, yy - 0.475)

box(ax, 5.5, py - 3.0, 4.2, 0.55, COL_OUTPUT,
    'Policy Output  (2187,)',
    '27 directions x 81 squares -> move probability', fontsize=7.5, radius=0.1)
arrow(ax, 5.5, py - 2.525, 5.5, py - 2.725)

# ── Value Head ───────────────────────────────────────────────────────────────
vy = 18.5
ax.text(14.5, vy + 0.6, 'Value Head', ha='center', color=COL_VALUE,
        fontsize=10, fontweight='bold')

for lbl, sub, yy in [
    ('l22_v  Conv 1x1', '192->1ch', vy),
    ('Flatten', '(1, 81)', vy - 0.75),
    ('l23_v  Linear', '81->256', vy - 1.5),
    ('ReLU', 'activation', vy - 2.25),
    ('l24_v  Linear', '256->1', vy - 3.0),
    ('tanh', 'value score [-1, +1]', vy - 3.75),
]:
    box(ax, 14.5, yy, 4.2, 0.55, COL_VALUE, lbl, sub, fontsize=7.5, radius=0.1)
    if yy > vy - 3.75:
        arrow(ax, 14.5, yy - 0.275, 14.5, yy - 0.475)

box(ax, 14.5, vy - 4.5, 4.2, 0.55, COL_OUTPUT,
    'Value Output  (1,)',
    '+1 = Black winning  -1 = White winning', fontsize=7.5, radius=0.1)
arrow(ax, 14.5, vy - 4.025, 14.5, vy - 4.225)

# Arrows from split
arrow(ax, 5.5, y_split - 0.2, 5.5, py + 0.275)
arrow(ax, 14.5, y_split - 0.2, 14.5, vy + 0.275)

# ── Stats / Legend ────────────────────────────────────────────────────────────
lx, ly = 0.3, 5.5
ax.text(lx, ly, 'Key Parameters', color='white', fontsize=9,
        fontweight='bold', va='top')
stats = [
    ('Input channels',    '62 (FEATURES1) + 57 (FEATURES2) = 119 ch'),
    ('Internal channels', '192 ch  (wide = 2x standard)'),
    ('Residual blocks',   '10 blocks'),
    ('Policy output dim', '2187 = 27 directions x 81 squares'),
    ('Value output dim',  '1  (tanh scalar)'),
    ('Training data',     'WCSC29 self-play games'),
    ('Estimated strength','Amateur 3-5 dan'),
]
for i, (k, v) in enumerate(stats):
    ax.text(lx,       ly - 0.5 - i*0.45, f'  {k}: ', color='#aaaaaa',
            fontsize=7.5, va='top')
    ax.text(lx + 3.2, ly - 0.5 - i*0.45, v, color='white',
            fontsize=7.5, va='top')

# Color legend
legend_items = [
    (COL_INPUT,    'Input features'),
    (COL_CONV,     'Convolution'),
    (COL_BN,       'BN / Activation'),
    (COL_RESBLOCK, 'Residual block'),
    (COL_POLICY,   'Policy head'),
    (COL_VALUE,    'Value head'),
    (COL_OUTPUT,   'Network output'),
]
ax.text(0.3, 2.5, 'Legend:', color='white', fontsize=8, fontweight='bold', va='top')
for i, (c, lbl) in enumerate(legend_items):
    px = 0.3 + (i % 4) * 2.5
    py_leg = 2.0 - (i // 4) * 0.5
    ax.add_patch(FancyBboxPatch((px, py_leg - 0.15), 0.28, 0.28,
                                boxstyle="round,pad=0.02",
                                facecolor=c, edgecolor='white', linewidth=0.5))
    ax.text(px + 0.4, py_leg, lbl, color='white', fontsize=7.5, va='center')

plt.tight_layout(pad=0)
out = '/workspace/dlshogi_wideresnet10_architecture.png'
plt.savefig(out, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor(), edgecolor='none')
print(f'Saved: {out}')
