"""Generate CDES Entity-Relationship Diagram as PNG."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "cdes_erd.png"

fig, ax = plt.subplots(figsize=(18, 13))
ax.set_xlim(0, 16)
ax.set_ylim(0, 13)
ax.axis('off')

# Title
ax.text(8, 12.5, 'COMPASS Data Exchange Standard (CDES) v0.1.0',
        ha='center', fontsize=18, fontweight='bold', color='#1B3A6B')
ax.text(8, 12.05, 'Entity-Relationship Diagram — Capstone Scope (Registry + Financial + Compliance domains)',
        ha='center', fontsize=11, style='italic', color='#505050')

# Domain colors
NAVY = '#1B3A6B'
BLUE = '#2E75B6'
GREEN = '#1F6B3A'
ORANGE = '#8B4500'
LIGHT_BLUE = '#D6E4F5'
LIGHT_GREEN = '#D9EFE3'
LIGHT_ORANGE = '#FDE8CC'
GRAY = '#505050'

# Entity box drawer
def entity(x, y, w, h, name, fields, fill, edge):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.1",
                         linewidth=2, edgecolor=edge, facecolor=fill, zorder=2)
    ax.add_patch(box)
    # Header bar
    header = FancyBboxPatch((x, y+h-0.55), w, 0.55,
                            boxstyle="round,pad=0,rounding_size=0.1",
                            linewidth=0, facecolor=edge, zorder=3)
    ax.add_patch(header)
    ax.text(x + w/2, y + h - 0.28, name, ha='center', va='center',
            fontsize=10.5, fontweight='bold', color='white', zorder=4)
    for i, fld in enumerate(fields):
        weight = 'bold' if fld.endswith('*') else 'normal'
        color = edge if fld.endswith('*') else '#1A1A1A'
        ax.text(x + 0.15, y + h - 0.85 - i*0.28, fld.rstrip('*'),
                fontsize=8.5, color=color, fontweight=weight, zorder=4)

# Relationship drawer
def relate(x1, y1, x2, y2, label=''):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            connectionstyle="arc3,rad=0", arrowstyle='-',
                            linewidth=1.4, color='#888888', zorder=1)
    ax.add_patch(arrow)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, label, fontsize=7.5, color=GRAY,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'),
                ha='center', va='center', zorder=5)

# ── REGISTRY DOMAIN (Blue, top-left) ────────────────────────────────────────
entity(0.3, 9.0, 3.0, 2.6, "ASSOCIATIONS",
       ["caid*", "legal_name", "association_type", "state",
        "total_units", "year_built", "coastal_flag"],
       LIGHT_BLUE, NAVY)

entity(4.2, 9.0, 2.5, 2.6, "UNITS",
       ["unit_id*", "caid →", "unit_number",
        "unit_type", "occupancy_status"],
       LIGHT_BLUE, NAVY)

# Domain label
ax.text(2.5, 11.7, 'REGISTRY DOMAIN', fontsize=10, fontweight='bold',
        color=NAVY, alpha=0.7)

# ── FINANCIAL DOMAIN (Green, middle) ────────────────────────────────────────
entity(0.3, 5.0, 2.7, 2.6, "FEE_SCHEDULES",
       ["fee_schedule_id*", "caid →", "effective_date",
        "monthly_assessment", "frequency"],
       LIGHT_GREEN, GREEN)

entity(3.4, 5.0, 2.8, 2.6, "PAYMENT_LEDGER",
       ["ledger_entry_id*", "caid →", "unit_id →",
        "transaction_date", "transaction_type", "amount"],
       LIGHT_GREEN, GREEN)

entity(6.6, 5.0, 2.8, 2.6, "DELINQUENCY",
       ["delinquency_id*", "caid →", "unit_id →",
        "snapshot_date", "days_delinquent", "aging_bucket"],
       LIGHT_GREEN, GREEN)

entity(9.8, 5.0, 2.7, 2.6, "RESERVE_FUNDS",
       ["reserve_id*", "caid →", "snapshot_date",
        "percent_funded", "current_balance"],
       LIGHT_GREEN, GREEN)

entity(12.9, 5.0, 2.7, 2.6, "ANNUAL_BUDGETS",
       ["budget_id*", "caid →", "fiscal_year",
        "total_revenue", "audit_status"],
       LIGHT_GREEN, GREEN)

ax.text(8.0, 7.7, 'FINANCIAL DOMAIN', fontsize=10, fontweight='bold',
        color=GREEN, alpha=0.7)

# ── COMPLIANCE DOMAIN (Orange, bottom) ─────────────────────────────────────
entity(0.3, 1.2, 3.0, 2.6, "VIOLATIONS",
       ["violation_id*", "caid →", "unit_id →",
        "issue_date", "violation_category", "status"],
       LIGHT_ORANGE, ORANGE)

entity(4.5, 1.2, 3.0, 2.6, "WORK_ORDERS",
       ["work_order_id*", "caid →", "category",
        "priority", "status", "deferred_days"],
       LIGHT_ORANGE, ORANGE)

entity(8.7, 1.2, 3.0, 2.6, "SIRS_FILINGS",
       ["sirs_id*", "caid →", "filing_date",
        "structural_findings", "compliance_status"],
       LIGHT_ORANGE, ORANGE)

entity(12.4, 1.2, 3.2, 2.6, "AUDIT_LOG",
       ["audit_id*", "actor_id", "actor_role",
        "action", "entity_type", "success"],
       '#F0F0F0', GRAY)

ax.text(6.0, 3.9, 'COMPLIANCE DOMAIN', fontsize=10, fontweight='bold',
        color=ORANGE, alpha=0.7)
ax.text(14.0, 3.9, 'SECURITY', fontsize=10, fontweight='bold',
        color=GRAY, alpha=0.7)

# ── RELATIONSHIPS ──────────────────────────────────────────────────────────
# Associations → Units
relate(3.3, 10.3, 4.2, 10.3, '1:N')
# Associations → all Financial entities
relate(1.6, 9.0, 1.6, 7.6, '1:N')
relate(2.1, 9.0, 4.5, 7.6)
relate(2.4, 9.0, 7.8, 7.6)
relate(2.7, 9.0, 11.0, 7.6)
relate(3.0, 9.0, 13.8, 7.6)
# Units → Payment Ledger, Delinquency, Violations
relate(5.4, 9.0, 4.5, 7.6)
relate(5.6, 9.0, 7.8, 7.6)
relate(4.8, 9.0, 1.8, 3.8)
# Associations → Compliance
relate(1.5, 9.0, 1.5, 3.8)
relate(2.0, 9.0, 5.5, 3.8)
relate(2.5, 9.0, 10.0, 3.8)

# ── LEGEND ──────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=LIGHT_BLUE, edgecolor=NAVY, linewidth=2, label='Registry Domain'),
    mpatches.Patch(facecolor=LIGHT_GREEN, edgecolor=GREEN, linewidth=2, label='Financial Domain'),
    mpatches.Patch(facecolor=LIGHT_ORANGE, edgecolor=ORANGE, linewidth=2, label='Compliance Domain'),
    mpatches.Patch(facecolor='#F0F0F0', edgecolor=GRAY, linewidth=2, label='Security / Audit'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=8.5,
          bbox_to_anchor=(1.0, -0.02), frameon=True, framealpha=0.95)

# Footer
ax.text(0.3, 0.5, '* = Primary Key   |   → = Foreign Key reference   |   1:N = One-to-Many relationship',
        fontsize=8, style='italic', color=GRAY)
ax.text(15.7, 0.5, 'D. Avella · STU MS Cybersecurity & Analytics · Week 1 Deliverable',
        fontsize=8, style='italic', color=GRAY, ha='right')

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
print(f"ERD saved to: {OUT}")
