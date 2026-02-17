"""
make_magfreq_norm.py  –  Spatiotemporal normalization appendix figure.

Three clean sections:
  A. Normalization formula + symbol key
  B. Cliff base estimation (table)
  C. Area + observation period comparison table (both methods)

Output: figures/appendix/magfreq_norm.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

plt.rcParams['font.family']        = 'sans-serif'
plt.rcParams['font.sans-serif']    = ['DejaVu Sans', 'Arial']  # DejaVu has all subscript glyphs
plt.rcParams['mathtext.fontset']   = 'dejavusans'
plt.rcParams['axes.spines.top']    = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.left']  = False
plt.rcParams['axes.spines.bottom']= False

# ── colour palette ────────────────────────────────────────────────────────────
C_HEAD  = '#1a3a5c'
C_ROW1  = '#edf3f9'
C_ROW2  = 'white'
C_ACC   = '#a50f15'
C_HDRFG = 'white'
C_HDRB  = '#4a7db5'
C_NPZB  = '#d4e8d4'   # light green tint for NPZ column
C_SIMB  = '#fef3cd'   # light yellow tint for Simple column
C_KEY   = '#fdecea'   # highlight for key row

# ============================================================================
# 1.  COMPUTE VALUES
# ============================================================================
npz   = np.load('results/data_cubes/DelMar_cube.npz', allow_pickle=True)
elev  = npz['elevation_m']
ero   = npz['erosion']
dep   = npz['deposition']
dates = npz['date_strings']

combined     = np.abs(ero) + np.abs(dep)
has_data_any = (combined > 0).any(axis=2)

base_list, top_list = [], []
for i in range(has_data_any.shape[0]):
    idxs = np.where(has_data_any[i])[0]
    if len(idxs) > 0:
        base_list.append(elev[idxs[0]])
        top_list.append(elev[idxs[-1]])

base_arr    = np.array(base_list)
top_arr     = np.array(top_list)
n_poly      = len(base_arr)
base_z0     = float(np.median(base_arr))   # 2.50 m

heights_npz  = top_arr - base_z0
area_npz_m2  = heights_npz.sum() * 0.25
area_npz_hm2 = area_npz_m2 / 1e4
h_mean_npz   = heights_npz.mean()

along_m       = 2285.0
eff_h_simple  = 18.0 - base_z0
area_sim_m2   = along_m * eff_h_simple
area_sim_hm2  = area_sim_m2 / 1e4

all_starts = [s.split('_to_')[0] for s in dates]
all_ends   = [s.split('_to_')[1] for s in dates]
t_start    = datetime.strptime(min(all_starts), '%Y%m%d')
t_end      = datetime.strptime(max(all_ends),   '%Y%m%d')
t_days     = (t_end - t_start).days
t_years    = t_days / 365.25
n_int      = len(dates)

denom_npz = area_npz_hm2 * t_years
denom_sim = area_sim_hm2 * t_years

# ============================================================================
# 2.  TABLE DATA
# ============================================================================

# ── Table B: cliff base stats ────────────────────────────────────────────────
base_rows = [
    ['Mean',                f'{np.mean(base_arr):.2f} m NAVD88',   ''],
    ['Median  (selected)',  f'{base_z0:.2f} m NAVD88',             'Robust to scan-occlusion outliers'],
    ['Standard deviation',  f'{np.std(base_arr):.2f} m',           ''],
    ['IQR',  f'{np.percentile(base_arr,25):.2f}\u2013{np.percentile(base_arr,75):.2f} m NAVD88', ''],
    ['Range', f'{base_arr.min():.2f}\u2013{base_arr.max():.2f} m NAVD88',
              'Max likely scan-occlusion artifact'],
    ['Polygons with data',  f'{n_poly:,} of 9,140',               ''],
]

# ── Table C: parameters (both methods) ──────────────────────────────────────
param_rows = [
    ['Cliff base  z\u2080',
     f'{base_z0:.2f}', f'{base_z0:.2f}', 'm NAVD88',
     'Same for both; median from NPZ (Section B)'],
    ['Mean cliff height',
     f'{h_mean_npz:.2f}', f'{eff_h_simple:.1f}', 'm',
     'NPZ: data-derived  |  Simple: 18.0 \u2212 z\u2080'],
    ['Along-shore extent',
     f'{n_poly * 0.25:,.0f}', '2,285', 'm',
     'NPZ: 9,124 polygons \u00d7 0.25 m  |  Simple: field survey'],
    ['Polygon / cell width',
     '0.25', '\u2014', 'm',
     'NPZ data cube resolution'],
    ['Cliff face area  A\u209b\u209c',
     f'{area_npz_hm2:.3f}', f'{area_sim_hm2:.3f}', 'hm\u00b2',
     f'Ratio (Simple/NPZ) = {area_sim_hm2/area_npz_hm2:.2f}\u00d7'],
    ['Observation period  T',
     f'{t_years:.2f}', f'{t_years:.2f}', 'yr',
     f'{t_start.strftime("%d %b %Y")} \u2013 {t_end.strftime("%d %b %Y")}  ({t_days:,} days)'],
    ['Survey intervals',
     f'{n_int:,}', f'{n_int:,}', '\u2014',
     'Del Mar, all consecutive pairs'],
    ['Normalization denominator  A\u209b\u209c \u00d7 T',
     f'{denom_npz:.2f}', f'{denom_sim:.2f}', 'hm\u00b2 yr',
     'Divides N(\u2265V) to yield F\u209b\u209c  \u2014  KEY VALUE'],
]

# ============================================================================
# 3.  FIGURE
# ============================================================================
fig = plt.figure(figsize=(14, 17), facecolor='white')
gs  = gridspec.GridSpec(
    4, 1,
    figure=fig,
    top=0.95, bottom=0.03,
    left=0.04, right=0.96,
    hspace=0.55,
    height_ratios=[1.8, 2.2, 3.5, 4.5],
)

# helper: draw a clean table in an axes
def draw_table(ax, rows, col_headers, col_widths,
               accent_rows=(), accent_col=None,
               accent_col_cols=()):
    """
    rows         : list of lists (str)
    col_headers  : list of str (column header labels)
    col_widths   : relative widths (must sum to 1)
    accent_rows  : row indices (0-based in rows list) to highlight
    accent_col   : column index (1-based) to tint two data columns
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    n_rows    = len(rows)
    n_cols    = len(col_headers)
    row_h     = 1.0 / (n_rows + 1)   # +1 for header
    col_xs    = [0.0]
    for w in col_widths[:-1]:
        col_xs.append(col_xs[-1] + w)

    def cell_bg(ax, x, y, w, h, color):
        ax.add_patch(plt.Rectangle(
            (x, y), w, h,
            transform=ax.transAxes, zorder=1,
            facecolor=color, edgecolor='none', clip_on=False))

    # header row
    hy = 1.0 - row_h
    for j, (hdr, w, x) in enumerate(zip(col_headers, col_widths, col_xs)):
        bg = C_NPZB if (accent_col and j in accent_col_cols[0:1]) else \
             C_SIMB if (accent_col and j in accent_col_cols[1:2]) else C_HDRB
        cell_bg(ax, x, hy, w, row_h, bg)
        ax.text(x + w/2, hy + row_h*0.5, hdr,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=9, fontweight='bold',
                color=C_HEAD if bg in (C_NPZB, C_SIMB) else C_HDRFG,
                zorder=2)

    # data rows
    for i, row in enumerate(rows):
        ry  = hy - (i + 1) * row_h
        is_accent = i in accent_rows
        stripe = C_ROW1 if i % 2 == 0 else C_ROW2
        for j, (val, w, x) in enumerate(zip(row, col_widths, col_xs)):
            bg = C_KEY if is_accent else \
                 C_NPZB if (accent_col and j in accent_col_cols[0:1] and not is_accent) else \
                 C_SIMB if (accent_col and j in accent_col_cols[1:2] and not is_accent) else stripe
            cell_bg(ax, x, ry, w, row_h, bg)
            ha = 'left' if j == 0 or j == n_cols-1 else 'center'
            pad = 0.01 if ha == 'left' else 0
            ax.text(x + pad + (w/2 if ha == 'center' else 0),
                    ry + row_h * 0.5, val,
                    transform=ax.transAxes, ha=ha, va='center',
                    fontsize=8.5,
                    color=C_ACC if is_accent else '#111111',
                    fontweight='bold' if is_accent else 'normal',
                    zorder=2)

    # thin grid lines
    for i in range(n_rows + 2):
        y = 1.0 - i * row_h
        ax.axhline(y, color='#cccccc', lw=0.5, zorder=3)

# ── TITLE ─────────────────────────────────────────────────────────────────────
ax_title = fig.add_axes([0.04, 0.955, 0.92, 0.040])
ax_title.axis('off')
ax_title.text(0.5, 0.80,
    'Spatiotemporal Normalization of Rockfall Magnitude\u2013Frequency',
    ha='center', va='top', fontsize=15, fontweight='bold', color=C_HEAD)
ax_title.text(0.5, 0.10,
    'Del Mar, California   \u00b7   Procedure and Derived Parameters',
    ha='center', va='bottom', fontsize=10.5, color='#555555', fontstyle='italic')

# ============================================================================
# PANEL A  –  Normalization Framework
# ============================================================================
ax_a = fig.add_subplot(gs[0])
ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1); ax_a.axis('off')
ax_a.text(0.0, 1.0, 'A.  Normalization Framework',
          transform=ax_a.transAxes, va='top',
          fontsize=11, fontweight='bold', color=C_HEAD)

ax_a.text(0.0, 0.84,
    'Following Janeras et al. (2023), raw cumulative event counts N(\u2265V) are divided by the\n'
    'monitored cliff face area A\u209b\u209c (hm\u00b2) and observation period T (yr) to yield a normalized\n'
    'frequency density F\u209b\u209c that is comparable across sites and studies:',
    transform=ax_a.transAxes, va='top', fontsize=10, color='#222222')

ax_a.text(0.5, 0.38,
    r'$F_{st}(V) = \dfrac{N(\geq V)}{A_{st} \cdot T}$'
    r'$\qquad\longrightarrow\qquad$'
    r'$F_{st} = \alpha \cdot V^{-B}$',
    transform=ax_a.transAxes, ha='center', va='top',
    fontsize=16, color=C_ACC)

# symbol key as a small inline table
sym_data = [
    [r'$N(\geq V)$',        'Cumulative count of events with volume \u2265 V'],
    [r'$A_{st}$',    'Monitored cliff face area  (hm\u00b2)'],
    [r'$T$',                'Observation period  (yr)'],
    [r'$F_{st}$',    'Normalized frequency density  (hm\u207b\u00b2 yr\u207b\u00b9)'],
    [r'$\alpha,\; B$',      'Power law activity rate and scaling exponent'],
]
ax_a.text(0.02, 0.08, 'Symbol key:',
          transform=ax_a.transAxes, va='top',
          fontsize=9, fontweight='bold', color='#444444')
for k, (sym, desc) in enumerate(sym_data):
    ax_a.text(0.11, 0.08 - k * 0.105,
              sym, transform=ax_a.transAxes, va='top', fontsize=10, color=C_HEAD)
    ax_a.text(0.20, 0.08 - k * 0.105,
              desc, transform=ax_a.transAxes, va='top', fontsize=9.5, color='#333333')

ax_a.add_patch(plt.Rectangle((0, -0.08), 1, 1.10,
    transform=ax_a.transAxes, zorder=0,
    facecolor='#f5f9fd', edgecolor=C_HEAD, lw=1.0, clip_on=False))

# ============================================================================
# PANEL B  –  Cliff Base Estimation
# ============================================================================
ax_b = fig.add_subplot(gs[1])
ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1); ax_b.axis('off')
ax_b.text(0.0, 1.02, 'B.  Cliff Base Elevation (z\u2080)',
          transform=ax_b.transAxes, va='bottom',
          fontsize=11, fontweight='bold', color=C_HEAD)
ax_b.text(0.0, 0.97,
    'For each of the 9,140 along-shore polygons in the Del Mar data cube, '
    'the lowest 0.25-m elevation\nbin with any non-zero erosion or deposition signal is identified. '
    'The beach-wide cliff base z\u2080 is the median.',
    transform=ax_b.transAxes, va='top', fontsize=10, color='#222222')

# leave top 25% for text, use bottom 75% for table
ax_b_tbl = ax_b.inset_axes([0.0, 0.0, 1.0, 0.68])
draw_table(
    ax_b_tbl,
    rows=base_rows,
    col_headers=['Statistic', 'Value', 'Notes'],
    col_widths=[0.28, 0.28, 0.44],
    accent_rows=(1,),   # median row
)

ax_b.add_patch(plt.Rectangle((0, -0.05), 1, 1.10,
    transform=ax_b.transAxes, zorder=0,
    facecolor='#f5f9fd', edgecolor=C_HEAD, lw=1.0, clip_on=False))

# ============================================================================
# PANEL C  –  Parameters (both methods)
# ============================================================================
ax_c = fig.add_subplot(gs[2])
ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1); ax_c.axis('off')
ax_c.text(0.0, 1.02, 'C.  Cliff Face Area  \u2014  Two Approaches',
          transform=ax_c.transAxes, va='bottom',
          fontsize=11, fontweight='bold', color=C_HEAD)
ax_c.text(0.0, 0.975,
    'NPZ (data-driven): per-polygon cliff height from the data cube (0.25 m polygon width).\n'
    'Simple (literature): representative length \u00d7 mean cliff height from field surveys.',
    transform=ax_c.transAxes, va='top', fontsize=10, color='#222222')

ax_c_tbl = ax_c.inset_axes([0.0, 0.0, 1.0, 0.80])
draw_table(
    ax_c_tbl,
    rows=param_rows,
    col_headers=['Parameter', 'NPZ method', 'Simple method', 'Units', 'Notes'],
    col_widths=[0.26, 0.10, 0.12, 0.09, 0.43],
    accent_rows=(7,),   # normalization denominator
    accent_col=True,
    accent_col_cols=[1, 2],
)

ax_c.add_patch(plt.Rectangle((0, -0.05), 1, 1.10,
    transform=ax_c.transAxes, zorder=0,
    facecolor='#f5f9fd', edgecolor=C_HEAD, lw=1.0, clip_on=False))

# ============================================================================
# PANEL D  –  Observation Period  (compact)
# ============================================================================
ax_d = fig.add_subplot(gs[3])
ax_d.set_xlim(0, 1); ax_d.set_ylim(0, 1); ax_d.axis('off')
ax_d.text(0.0, 1.02, 'D.  Observation Period',
          transform=ax_d.transAxes, va='bottom',
          fontsize=11, fontweight='bold', color=C_HEAD)

ax_d.text(0.5, 0.96,
    r'$T = \dfrac{' + str(t_days) + r'\;\mathrm{days}}{365.25} = '
    + f'{t_years:.2f}' + r'\;\mathrm{yr}$'
    + f'        ({t_start.strftime("%d %b %Y")} \u2013 {t_end.strftime("%d %b %Y")},'
    + f' {n_int} intervals)',
    transform=ax_d.transAxes, ha='center', va='top',
    fontsize=13, color=C_ACC)

obs_rows = [
    ['First survey start',              t_start.strftime('%d %B %Y'),  '',     ''],
    ['Last survey end',                 t_end.strftime('%d %B %Y'),    '',     ''],
    ['Total calendar days',             f'{t_days:,}',                 '',     ''],
    ['Fractional years  T',             f'{t_years:.2f} yr',           '',     ''],
    ['Number of survey intervals',      f'{n_int:,}',                  '',     ''],
    ['Normalization denom.  (NPZ)',      f'{denom_npz:.2f} hm\u00b2 yr',
                                        f'{area_npz_hm2:.3f} hm\u00b2 \u00d7 {t_years:.2f} yr', ''],
    ['Normalization denom.  (Simple)',  f'{denom_sim:.2f} hm\u00b2 yr',
                                        f'{area_sim_hm2:.3f} hm\u00b2 \u00d7 {t_years:.2f} yr', ''],
]
ax_d_tbl = ax_d.inset_axes([0.0, 0.08, 1.0, 0.65])
draw_table(
    ax_d_tbl,
    rows=obs_rows,
    col_headers=['Item', 'Value', 'Calculation', 'Notes'],
    col_widths=[0.30, 0.22, 0.28, 0.20],
    accent_rows=(5, 6),
)

# caption
ax_d.text(0.0, 0.055,
    'Data source: results/data_cubes/DelMar_cube.npz.  '
    'Cliff base z\u2080 = median lowest-occupied elevation bin (0.25-m resolution) per polygon.  '
    'Cliff top per polygon = highest occupied bin across all survey intervals.  '
    'Reference: Janeras et al. (2023), Earth Surface Dynamics, 11, 1119\u20131138.',
    transform=ax_d.transAxes, va='top', fontsize=8, color='#666666', fontstyle='italic',
    wrap=True)

ax_d.add_patch(plt.Rectangle((0, -0.02), 1, 1.07,
    transform=ax_d.transAxes, zorder=0,
    facecolor='#f5f9fd', edgecolor=C_HEAD, lw=1.0, clip_on=False))

# ============================================================================
# SAVE
# ============================================================================
out = 'figures/appendix/magfreq_norm.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
plt.close()
