"""
COMPASS — Security Architecture Diagram (Phase 4)
Renders the data-flow / trust-boundary diagram to docs/security_architecture.png.

Shows: ingestion -> CDES validation -> encrypted storage -> analytics/models
-> role-based dashboard, with trust boundaries and NIST SP 800-53 control
annotations. Rendered in grayscale to match the document standard while
remaining a visual representation (color is permitted for figures, but a
security architecture reads cleanly in grayscale).

Author: Diego Avella
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "docs" / "security_architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

GREY_DK = "#222222"; GREY_MD = "#666666"; GREY_LT = "#DDDDDD"; WHITE = "#FFFFFF"

ax.text(8, 8.6, "COMPASS Security Architecture — Data Flow & Trust Boundaries",
        ha="center", fontsize=16, fontweight="bold", color=GREY_DK)
ax.text(8, 8.15, "NIST SP 800-53 Rev. 5 control annotations · FedRAMP High target baseline",
        ha="center", fontsize=10, style="italic", color=GREY_MD)


def box(x, y, w, h, title, controls, fill=WHITE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                 linewidth=1.6, edgecolor=GREY_DK, facecolor=fill))
    ax.text(x + w/2, y + h - 0.32, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=GREY_DK)
    ax.text(x + w/2, y + h - 0.72, controls, ha="center", va="top",
            fontsize=8.0, color=GREY_MD)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=16, linewidth=1.6, color=GREY_DK))


# Trust boundary 1: external / ingestion
ax.add_patch(Rectangle((0.3, 5.1), 4.6, 2.7, linewidth=1.2, edgecolor=GREY_MD,
             facecolor="none", linestyle=(0, (6, 4))))
ax.text(0.45, 7.65, "Trust boundary: External ingestion zone", fontsize=8.5,
        color=GREY_MD, style="italic")

box(0.6, 5.4, 4.0, 1.9, "1. Data Ingestion",
    "Association / management-company\nsubmissions over TLS 1.3\n\nSC-8 (transit encryption)\nIA-2 (submitter auth)\nSI-10 (input validation)", fill=GREY_LT)

# Trust boundary 2: government platform
ax.add_patch(Rectangle((5.3, 1.0), 10.4, 6.8, linewidth=1.2, edgecolor=GREY_DK,
             facecolor="none", linestyle=(0, (6, 4))))
ax.text(5.45, 7.65, "Trust boundary: COMPASS government platform (FedRAMP High)",
        fontsize=8.5, color=GREY_DK, style="italic")

box(5.6, 5.4, 4.2, 1.9, "2. CDES Validation",
    "JSON Schema conformance,\nreferential integrity, range checks\n\nSI-7 (integrity)\nSI-3 (malicious-input defense)\nAU-2 (event logging)")

box(10.5, 5.4, 4.7, 1.9, "3. Encrypted Data Store",
    "PostgreSQL, AES-256 at rest;\nPII tokenized / masked\n\nSC-28 (rest encryption)\nAC-3 (access enforcement)\nCP-9 (backup)")

box(5.6, 2.7, 4.2, 1.9, "4. Analytics & Models",
    "Models 1\u20133 (delinquency,\nreserve failure, anomaly)\n\nCM-6 (config baselines)\nSI-2 (flaw remediation)\nSA-11 (dev. testing)")

box(10.5, 2.7, 4.7, 1.9, "5. Role-Based Dashboard",
    "Public tier vs. government tier;\nPII masked in public view\n\nAC-2 (account mgmt)\nAC-6 (least privilege)\nIA-5 (authenticators)")

# Audit log spanning the platform (bottom)
box(5.6, 1.05, 9.6, 1.25, "Immutable Audit Log (cross-cutting)",
    "Every ingestion, prediction, and access event recorded\nAU-3 (record content) · AU-12 (audit generation) · RA-3 (risk assessment)\nIR-4 (incident handling) · PL-4 (rules of behavior)")

# arrows
arrow(4.6, 6.35, 5.6, 6.35)      # ingestion -> validation
arrow(9.8, 6.35, 10.5, 6.35)     # validation -> store
arrow(12.85, 5.4, 12.85, 4.6)    # store -> dashboard (down)
arrow(10.5, 3.65, 9.8, 3.65)     # dashboard <- analytics
arrow(7.7, 5.4, 7.7, 4.6)        # validation -> analytics (down)
arrow(9.8, 3.2, 10.5, 3.2)       # analytics -> dashboard

plt.tight_layout()
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"Security architecture diagram saved: {OUT}")
