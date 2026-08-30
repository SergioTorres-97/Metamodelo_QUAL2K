"""
figura_srcc_explicacion.py
Figura didáctica: cómo se calcula el SRCC espacial por sección del río.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

rng = np.random.default_rng(42)

# ── Datos simulados ───────────────────────────────────────────────────────────
N_RUNS  = 10          # corridas (simplificado para visualización)
N_SECS  = 12          # secciones del río (km)
kms     = np.linspace(0, 55, N_SECS)

# Parámetro: alpha_1 muestreado con LHS uniforme [0.10, 0.80]
alpha1  = np.linspace(0.10, 0.80, N_RUNS)
rng.shuffle(alpha1)

# Perfiles de DBO: aumenta con alpha_1 más hacia aguas abajo
mat = np.zeros((N_RUNS, N_SECS))
for i, a in enumerate(alpha1):
    base  = 5 + 30 * (kms / kms.max())          # tendencia base
    efecto = a * 20 * (kms / kms.max()) ** 1.5   # efecto de alpha_1 crece aguas abajo
    noise  = rng.normal(0, 1.5, N_SECS)
    mat[i] = base + efecto + noise
mat = np.clip(mat, 0, None)

# SRCC por sección
rhos = []
for s in range(N_SECS):
    rho, _ = spearmanr(alpha1, mat[:, s])
    rhos.append(rho)
rhos = np.array(rhos)

# ── Figura ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11), facecolor="#F7F9FC")
gs  = gridspec.GridSpec(
    3, 3,
    figure=fig,
    hspace=0.55, wspace=0.38,
    left=0.07, right=0.97, top=0.91, bottom=0.07,
)

AZUL   = "#3C5488"
ROJO   = "#E64B35"
VERDE  = "#00A087"
GRIS   = "#8491B4"
BG     = "#F7F9FC"

# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 (arriba izq): LHS — distribución del parámetro
# ─────────────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(range(N_RUNS), np.sort(alpha1), color=AZUL, edgecolor="white", width=0.7)
ax1.set_xticks(range(N_RUNS))
ax1.set_xticklabels([f"c{i}" for i in range(N_RUNS)], fontsize=7)
ax1.set_ylabel("Valor de α₁", fontsize=9, fontweight="bold")
ax1.set_title("① Muestreo LHS\n(parámetro α₁ en [0.10, 0.80])",
              fontsize=9, fontweight="bold", color=AZUL)
ax1.set_ylim(0, 1.0)
ax1.set_facecolor(BG)
ax1.grid(axis="y", linestyle="--", color="#CCCCCC", linewidth=0.6, alpha=0.8)
for spine in ax1.spines.values():
    spine.set_edgecolor("#CCCCCC")

# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 (arriba centro): Perfiles de DBO por corrida
# ─────────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
cmap = plt.cm.get_cmap("Blues", N_RUNS + 3)
for i in range(N_RUNS):
    orden = np.argsort(alpha1)[i]
    ax2.plot(kms, mat[i], color=cmap(3 + orden), linewidth=1.2, alpha=0.85)

sm = plt.cm.ScalarMappable(cmap="Blues",
     norm=plt.Normalize(vmin=alpha1.min(), vmax=alpha1.max()))
sm.set_array([])
cb = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
cb.set_label("α₁", fontsize=8, fontweight="bold")
cb.ax.tick_params(labelsize=7)

ax2.set_xlabel("Distancia longitudinal (km)", fontsize=8, fontweight="bold")
ax2.set_ylabel("DBO (mg/L)", fontsize=9, fontweight="bold")
ax2.set_title("② Perfiles de DBO simulados\n(una línea por corrida LHS)",
              fontsize=9, fontweight="bold", color=AZUL)
ax2.set_facecolor(BG)
ax2.grid(linestyle="--", color="#CCCCCC", linewidth=0.6, alpha=0.8)
for spine in ax2.spines.values():
    spine.set_edgecolor("#CCCCCC")

# ─────────────────────────────────────────────────────────────────────────────
# Panel 3 (arriba der): Heatmap de la matriz corridas × secciones
# ─────────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
im  = ax3.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax3.set_xlabel("Sección del río (km)", fontsize=8, fontweight="bold")
ax3.set_ylabel("Corrida LHS", fontsize=8, fontweight="bold")
ax3.set_xticks(range(N_SECS))
ax3.set_xticklabels([f"{k:.0f}" for k in kms], fontsize=6, rotation=45)
ax3.set_yticks(range(N_RUNS))
ax3.set_yticklabels([f"c{i}" for i in range(N_RUNS)], fontsize=7)
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04).set_label("DBO (mg/L)", fontsize=8)
ax3.set_title("③ Matriz de resultados\n(corridas × secciones)",
              fontsize=9, fontweight="bold", color=AZUL)

# Resaltar columna s=8 para explicar el zoom
col_demo = 8
for spine in [ax3.spines['left'], ax3.spines['right'],
              ax3.spines['top'],  ax3.spines['bottom']]:
    spine.set_edgecolor("#CCCCCC")
ax3.axvline(col_demo - 0.5, color=ROJO, lw=2)
ax3.axvline(col_demo + 0.5, color=ROJO, lw=2)
ax3.text(col_demo, -1.5, "↓", color=ROJO, ha="center", fontsize=11, fontweight="bold",
         transform=ax3.transData, clip_on=False)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 4 (medio izq): Zoom — scatter alpha_1 vs DBO en sección demo
# ─────────────────────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
y_demo = mat[:, col_demo]
ax4.scatter(alpha1, y_demo, color=ROJO, s=55, zorder=3, edgecolors="white", linewidths=0.5)

# Línea de tendencia de rangos (visual)
idx_sorted = np.argsort(alpha1)
ax4.plot(alpha1[idx_sorted], np.poly1d(np.polyfit(alpha1, y_demo, 1))(alpha1[idx_sorted]),
         color=GRIS, lw=1.5, linestyle="--", label="Tendencia")

rho_demo, _ = spearmanr(alpha1, y_demo)
ax4.set_xlabel("Valor de α₁ (parámetro)", fontsize=8, fontweight="bold")
ax4.set_ylabel(f"DBO en km {kms[col_demo]:.1f} (mg/L)", fontsize=8, fontweight="bold")
ax4.set_title(f"④ Scatter en km {kms[col_demo]:.1f}\n"
              f"SRCC = {rho_demo:+.3f}",
              fontsize=9, fontweight="bold", color=ROJO)
ax4.set_facecolor(BG)
ax4.grid(linestyle="--", color="#CCCCCC", linewidth=0.6, alpha=0.8)
for spine in ax4.spines.values():
    spine.set_edgecolor("#CCCCCC")

# Anotar rangos
for i, (x, y) in enumerate(zip(alpha1, y_demo)):
    ax4.annotate(f"c{i}", (x, y), textcoords="offset points",
                 xytext=(4, 3), fontsize=6, color="#555555")

# ─────────────────────────────────────────────────────────────────────────────
# Panel 5 (medio centro): Cómo se convierte a rangos
# ─────────────────────────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
rangos_x = np.argsort(np.argsort(alpha1)) + 1
rangos_y = np.argsort(np.argsort(y_demo)) + 1
ax5.scatter(rangos_x, rangos_y, color=VERDE, s=55, zorder=3,
            edgecolors="white", linewidths=0.5)
ax5.plot([1, N_RUNS], [1, N_RUNS], color=GRIS, lw=1.2, linestyle="--", alpha=0.6)
for i, (rx, ry) in enumerate(zip(rangos_x, rangos_y)):
    ax5.annotate(f"c{i}", (rx, ry), textcoords="offset points",
                 xytext=(4, 3), fontsize=6, color="#555555")
ax5.set_xlabel("Rango de α₁", fontsize=8, fontweight="bold")
ax5.set_ylabel(f"Rango DBO km {kms[col_demo]:.1f}", fontsize=8, fontweight="bold")
ax5.set_title(f"⑤ Conversión a rangos\n(Spearman usa rangos, no valores)",
              fontsize=9, fontweight="bold", color=VERDE)
ax5.set_facecolor(BG)
ax5.grid(linestyle="--", color="#CCCCCC", linewidth=0.6, alpha=0.8)
for spine in ax5.spines.values():
    spine.set_edgecolor("#CCCCCC")

# ─────────────────────────────────────────────────────────────────────────────
# Panel 6 (medio der): Repetir para todas las secciones
# ─────────────────────────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
cols_demo = [2, 5, 8, 10]
colors_d  = [AZUL, VERDE, ROJO, "#F39B7F"]
for ci, col in enumerate(cols_demo):
    y_c = mat[:, col]
    ax6.scatter(alpha1, y_c, color=colors_d[ci], s=30, alpha=0.7,
                edgecolors="none", label=f"km {kms[col]:.0f}")
    rho_c, _ = spearmanr(alpha1, y_c)
ax6.set_xlabel("Valor de α₁", fontsize=8, fontweight="bold")
ax6.set_ylabel("DBO (mg/L)", fontsize=8, fontweight="bold")
ax6.set_title("⑥ Mismo cálculo en CADA sección\n(se obtiene un SRCC por km)",
              fontsize=9, fontweight="bold", color=AZUL)
ax6.legend(fontsize=7, framealpha=0.9, title="Sección", title_fontsize=7)
ax6.set_facecolor(BG)
ax6.grid(linestyle="--", color="#CCCCCC", linewidth=0.6, alpha=0.8)
for spine in ax6.spines.values():
    spine.set_edgecolor("#CCCCCC")

# ─────────────────────────────────────────────────────────────────────────────
# Panel 7 (abajo, ancho completo): Perfil espacial de SRCC
# ─────────────────────────────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, :])
colores_barra = [ROJO if r > 0 else AZUL for r in rhos]
bars = ax7.bar(kms, rhos, width=3.8, color=colores_barra,
               edgecolor="white", linewidth=0.5, zorder=3)
ax7.axhline(0,    color="#333333", lw=1.0, zorder=2)
ax7.axhline( 0.6, color=ROJO,  lw=1.0, linestyle=":", alpha=0.7, label="|SRCC| = 0.6 (alta influencia)")
ax7.axhline(-0.6, color=AZUL,  lw=1.0, linestyle=":", alpha=0.7)
ax7.axhline( 0.3, color=GRIS,  lw=0.8, linestyle=":", alpha=0.5, label="|SRCC| = 0.3 (influencia moderada)")
ax7.axhline(-0.3, color=GRIS,  lw=0.8, linestyle=":", alpha=0.5)

# Anotar valor en cada barra
for km, rho in zip(kms, rhos):
    ax7.text(km, rho + (0.03 if rho >= 0 else -0.06),
             f"{rho:+.2f}", ha="center", fontsize=7,
             color=ROJO if rho > 0 else AZUL, fontweight="bold")

ax7.set_xlim(-3, 58)
ax7.set_ylim(-1.1, 1.1)
ax7.set_xlabel("Distancia longitudinal (km)  ←  aguas arriba  |  aguas abajo  →",
               fontsize=9, fontweight="bold")
ax7.set_ylabel("SRCC  (α₁ vs DBO)", fontsize=9, fontweight="bold")
ax7.set_title("⑦ Perfil espacial de SRCC — resultado final\n"
              "Cada barra = correlación entre α₁ y la DBO en esa sección del río",
              fontsize=10, fontweight="bold", color=AZUL)
ax7.set_facecolor(BG)
ax7.grid(axis="y", linestyle="--", color="#CCCCCC", linewidth=0.6, alpha=0.8)
for spine in ax7.spines.values():
    spine.set_edgecolor("#CCCCCC")

patch_pos = mpatches.Patch(color=ROJO, label="SRCC > 0: relación directa (↑α₁ → ↑DBO)")
patch_neg = mpatches.Patch(color=AZUL, label="SRCC < 0: relación inversa (↑α₁ → ↓DBO)")
ax7.legend(handles=[patch_pos, patch_neg,
           mpatches.Patch(color="none", label="|SRCC| > 0.6: alta influencia"),
           mpatches.Patch(color="none", label="|SRCC| < 0.3: baja influencia")],
           fontsize=8, loc="lower right", framealpha=0.9, ncol=2)

# ─────────────────────────────────────────────────────────────────────────────
# Título general
# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Análisis de Sensibilidad Espacial — SRCC (Spearman Rank Correlation Coefficient)\n"
    "¿Cómo influye el parámetro α₁ sobre la DBO en cada sección del río?",
    fontsize=12, fontweight="bold", color="#2C3E50", y=0.98,
)

out = r"D:\Maestria\Proyecto_de_grado\PYQ2K\resultados\figura_srcc_didactica.png"
import os; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Figura guardada: {out}")
