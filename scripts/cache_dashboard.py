"""
Cache Generation Dashboard
============================
Reads smart_cache.log and renders a physician-friendly dashboard showing
the progress of AI-generated lung ultrasound frame rendering.

Design: Clean thumbnails with pass/fail badges in the grid.
Detailed metrics in a summary table below.

Run anytime:
    python3 scripts/cache_dashboard.py
"""

import re
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import OrderedDict

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "checkpoints" / "smart_cache.log"
PROGRESS_DIR = ROOT / "checkpoints" / "benchmarks" / "cache_progress"
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

ZONE_ORDER = [
    "UPPER_BLUE_L", "UPPER_BLUE_R",
    "LOWER_BLUE_L", "LOWER_BLUE_R",
    "PLAPS_L", "PLAPS_R",
    "DIAPHRAGM_L", "DIAPHRAGM_R",
]

ZONE_LABELS = [
    "Upper BLUE\nLeft",
    "Upper BLUE\nRight",
    "Lower BLUE\nLeft",
    "Lower BLUE\nRight",
    "PLAPS\nLeft",
    "PLAPS\nRight",
    "Diaphragm\nLeft",
    "Diaphragm\nRight",
]

SCENARIO_LABELS = {
    "normal": "Normal (A-Profile)",
    "left_pneumothorax": "Left Pneumothorax",
    "right_pneumothorax": "Right Pneumothorax",
    "left_pneumothorax_with_lung_point": "Left PTX + Lung Point",
    "pulmonary_edema": "Pulmonary Edema (B-Profile)",
    "pulmonary_edema_with_effusion": "Pulm. Edema + Effusion",
    "ards": "ARDS (White Lung)",
    "left_pneumonia": "Left Pneumonia",
    "right_pneumonia": "Right Pneumonia",
    "bilateral_pneumonia": "Bilateral Pneumonia",
    "right_pleural_effusion": "Right Pleural Effusion",
    "bilateral_effusion": "Bilateral Effusion",
    "left_hemothorax": "Left Hemothorax",
    "pneumonia_with_effusion": "Pneumonia + Effusion",
    "copd_exacerbation": "COPD Exacerbation",
}

PATHOLOGY_LABELS = {
    "normal_a_profile": "Normal (A-lines)",
    "pneumothorax": "Pneumothorax",
    "b_lines_focal": "Focal B-Lines",
    "b_lines_diffuse": "Diffuse B-Lines",
    "consolidation": "Consolidation",
    "pleural_effusion": "Pleural Effusion",
    "ards_white_lung": "ARDS / White Lung",
    "lung_point": "Lung Point",
    "pleural_thickening": "Pleural Thickening",
    "interstitial_syndrome": "Interstitial Syndrome",
}

TOTAL_ZONES = 120


def parse_log(log_path: Path) -> dict:
    if not log_path.exists():
        return {"scenarios": OrderedDict(), "zones": [], "retries": 0}

    text = log_path.read_text()
    result = {"scenarios": OrderedDict(), "zones": [], "retries": 0}
    current_scenario = None

    for line in text.split("\n"):
        m = re.match(r"=== Scenario: (\S+) ===", line)
        if m:
            current_scenario = m.group(1)
            if current_scenario not in result["scenarios"]:
                result["scenarios"][current_scenario] = {}

        if "RETRY" in line:
            result["retries"] += 1

        m = re.match(
            r"\s*\[(\d+)/120\]\s+(\S+):\s+(\S+)\s+\(([\d.]+)s\)\s+\[(PASS|FAIL)\]\s+(.*)",
            line,
        )
        if m and current_scenario:
            zone_data = {
                "idx": int(m.group(1)),
                "zone": m.group(2),
                "pathology": m.group(3),
                "time_s": float(m.group(4)),
                "status": m.group(5),
                "scenario": current_scenario,
                "metrics": {},
            }
            for pair in m.group(6).split(" | "):
                kv = pair.strip().split("=")
                if len(kv) == 2:
                    try:
                        zone_data["metrics"][kv[0]] = float(kv[1])
                    except ValueError:
                        zone_data["metrics"][kv[0]] = kv[1]

            result["zones"].append(zone_data)
            result["scenarios"][current_scenario][zone_data["zone"]] = zone_data

    return result


def load_fonts():
    fonts = {}
    defs = [("sm", "DejaVuSans.ttf", 12), ("md", "DejaVuSans.ttf", 14),
            ("lg", "DejaVuSans.ttf", 17), ("xl", "DejaVuSans-Bold.ttf", 24),
            ("bold", "DejaVuSans-Bold.ttf", 14), ("bold_sm", "DejaVuSans-Bold.ttf", 12),
            ("bold_lg", "DejaVuSans-Bold.ttf", 16),
            ("table", "DejaVuSansMono.ttf", 12), ("table_bold", "DejaVuSansMono-Bold.ttf", 12)]
    for name, filename, size in defs:
        try:
            fonts[name] = ImageFont.truetype(
                f"/usr/share/fonts/truetype/dejavu/{filename}", size)
        except (OSError, IOError):
            fonts[name] = ImageFont.load_default()
    return fonts


def draw_bar(draw, x, y, w, h, progress, fill, bg=(45, 45, 55)):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=3, fill=bg)
    fw = max(0, int(w * min(progress, 1.0)))
    if fw > 0:
        draw.rounded_rectangle([x, y, x + fw, y + h], radius=3, fill=fill)


def render_dashboard(data: dict, output_path: Path):
    scenarios = data["scenarios"]
    zones = data["zones"]
    fonts = load_fonts()

    if not zones:
        print("No zones completed yet.")
        return

    # ── Stats ──
    n_done = len(zones)
    n_pass = sum(1 for z in zones if z["status"] == "PASS")
    n_fail = n_done - n_pass
    pass_pct = 100 * n_pass / n_done if n_done else 0
    total_time = sum(z["time_s"] for z in zones)
    avg_time = total_time / n_done if n_done else 0
    eta_h = (TOTAL_ZONES - n_done) * avg_time / 3600

    # Pathology stats
    path_stats = OrderedDict()
    for z in zones:
        p = z["pathology"]
        if p not in path_stats:
            path_stats[p] = {"pass": 0, "total": 0}
        path_stats[p]["total"] += 1
        if z["status"] == "PASS":
            path_stats[p]["pass"] += 1

    # Collect failed zones for detail table
    failed_list = [z for z in zones if z["status"] == "FAIL"]

    # ── Layout ──
    thumb = 150
    pathology_label_h = 22
    pad = 6
    cell_w = thumb + pad * 2
    cell_h = thumb + pathology_label_h + pad * 2
    scen_w = 280
    header_h = 110
    col_hdr_h = 55
    n_cols = 8
    n_rows = len(scenarios)

    # Section 2: Pathology quality table
    table_header_h = 50
    table_row_h = 28
    table_h = table_header_h + len(path_stats) * table_row_h + 20

    # Section 3: Failed zones detail (if any)
    fail_header_h = 50 if failed_list else 0
    fail_row_h = 22
    max_fail_show = min(len(failed_list), 15)
    fail_h = fail_header_h + max_fail_show * fail_row_h + 20 if failed_list else 0

    # Legend
    legend_h = 60

    W = scen_w + n_cols * cell_w + 30
    H = header_h + col_hdr_h + n_rows * cell_h + 30 + table_h + fail_h + legend_h

    img = Image.new("RGB", (W, H), (16, 18, 26))
    draw = ImageDraw.Draw(img)

    # ════════════════════════════════════════
    # SECTION 1: HEADER + GRID
    # ════════════════════════════════════════

    # Title
    draw.text((20, 14), "MoCoLUS AI Ultrasound Generation", fill=(200, 215, 245), font=fonts["xl"])
    draw.text((20, 46), "Generating realistic POCUS frames for all clinical scenarios",
              fill=(120, 130, 150), font=fonts["md"])

    # Progress bar
    bx, by, bw, bh = 20, 74, W - 40, 20
    bar_col = (55, 185, 100) if pass_pct > 75 else (220, 165, 50) if pass_pct > 50 else (200, 70, 70)
    draw_bar(draw, bx, by, bw, bh, n_done / TOTAL_ZONES, bar_col)
    pct_text = f"{n_done} of {TOTAL_ZONES} zones completed ({100 * n_done / TOTAL_ZONES:.0f}%)"
    tw = draw.textlength(pct_text, font=fonts["bold"])
    draw.text((bx + bw // 2 - tw // 2, by + 2), pct_text,
              fill=(255, 255, 255), font=fonts["bold"])

    # Quick stats
    sy = 100
    draw.text((20, sy - 2),
              f"Passed: {n_pass}    Failed: {n_fail}    "
              f"Quality Rate: {pass_pct:.0f}%    "
              f"Retries: {data['retries']}    "
              f"Est. Remaining: {eta_h:.1f} hours",
              fill=(140, 150, 170), font=fonts["sm"])

    # Column headers
    col_y = header_h
    for c, label in enumerate(ZONE_LABELS):
        x = scen_w + c * cell_w + pad
        lines = label.split("\n")
        for i, line in enumerate(lines):
            tw = draw.textlength(line, font=fonts["bold_sm"])
            draw.text((x + thumb // 2 - tw // 2, col_y + 10 + i * 18),
                      line, fill=(140, 160, 190), font=fonts["bold_sm"])

    draw.line([(15, col_y + col_hdr_h - 3), (W - 15, col_y + col_hdr_h - 3)],
              fill=(45, 50, 60), width=1)

    # Load thumbnails
    prog_path = PROGRESS_DIR / "progress.png"
    prog_img = Image.open(prog_path) if prog_path.exists() else None

    # Grid
    gy0 = header_h + col_hdr_h

    for r, (scen_key, scen_zones) in enumerate(scenarios.items()):
        ry = gy0 + r * cell_h

        if r % 2 == 0:
            draw.rectangle([scen_w - 5, ry, W - 15, ry + cell_h], fill=(20, 22, 32))

        # Scenario label
        label = SCENARIO_LABELS.get(scen_key, scen_key.replace("_", " ").title())
        draw.text((18, ry + cell_h // 2 - 8), label, fill=(190, 200, 220), font=fonts["bold"])

        for c, zone_name in enumerate(ZONE_ORDER):
            cx = scen_w + c * cell_w + pad
            cy = ry + pad

            if zone_name not in scen_zones:
                draw.rounded_rectangle(
                    [cx, cy, cx + thumb, cy + thumb],
                    radius=5, fill=(26, 28, 38), outline=(40, 42, 52), width=1)
                tw = draw.textlength("Pending", font=fonts["sm"])
                draw.text((cx + thumb // 2 - tw // 2, cy + thumb // 2 - 7),
                          "Pending", fill=(55, 60, 72), font=fonts["sm"])
                continue

            zd = scen_zones[zone_name]
            ok = zd["status"] == "PASS"
            border = (50, 195, 90) if ok else (220, 65, 65)

            # Paste actual ControlNet thumbnail
            src_sz = 80
            got_thumb = False
            if prog_img is not None:
                px, py = c * src_sz, r * src_sz
                if px + src_sz <= prog_img.width and py + src_sz <= prog_img.height:
                    t = prog_img.crop((px, py, px + src_sz, py + src_sz))
                    t = t.resize((thumb, thumb), Image.LANCZOS)
                    if t.mode != "RGB":
                        t = t.convert("RGB")
                    img.paste(t, (cx, cy))
                    got_thumb = True

            if not got_thumb:
                draw.rectangle([cx, cy, cx + thumb, cy + thumb], fill=(32, 34, 44))

            # Border
            draw.rounded_rectangle([cx - 2, cy - 2, cx + thumb + 2, cy + thumb + 2],
                                   radius=5, outline=border, width=3)

            # Badge
            badge = "Pass" if ok else "Fail"
            badge_bg = (30, 145, 60, 220) if ok else (180, 45, 45, 220)
            badge_w = draw.textlength(badge, font=fonts["bold_sm"]) + 10
            bx_ = cx + thumb - badge_w - 4
            by_ = cy + 5
            draw.rounded_rectangle([bx_, by_, bx_ + badge_w, by_ + 20],
                                   radius=4, fill=badge_bg)
            draw.text((bx_ + 5, by_ + 2), badge, fill=(255, 255, 255), font=fonts["bold_sm"])

            # Pathology label below thumbnail
            path_label = PATHOLOGY_LABELS.get(zd["pathology"], zd["pathology"])
            tw = draw.textlength(path_label, font=fonts["sm"])
            lx = cx + max(0, (thumb - tw) // 2)
            draw.text((lx, cy + thumb + 4), path_label, fill=(130, 140, 160), font=fonts["sm"])

    # ════════════════════════════════════════
    # SECTION 2: PATHOLOGY QUALITY TABLE
    # ════════════════════════════════════════

    sec2_y = gy0 + n_rows * cell_h + 25
    draw.line([(15, sec2_y), (W - 15, sec2_y)], fill=(45, 50, 60), width=1)
    draw.text((20, sec2_y + 8), "Quality Summary by Pathology",
              fill=(190, 200, 220), font=fonts["bold_lg"])
    draw.text((20, sec2_y + 30),
              "Each pathology type is checked for brightness, contrast, clinical similarity to real images, and pleural line visibility.",
              fill=(110, 120, 140), font=fonts["sm"])

    # Table header
    ty = sec2_y + table_header_h
    cols = [
        ("Pathology", 20, 240),
        ("Zones", 260, 60),
        ("Passed", 330, 70),
        ("Failed", 410, 70),
        ("Pass Rate", 490, 100),
        ("Avg Similarity", 610, 120),
        ("Avg Brightness", 750, 120),
        ("Pleural Visible", 890, 120),
    ]
    for label, x, w in cols:
        draw.text((x, ty), label, fill=(150, 165, 190), font=fonts["bold_sm"])
    ty += 22
    draw.line([(20, ty), (W - 20, ty)], fill=(40, 45, 55), width=1)
    ty += 4

    for p, s in path_stats.items():
        rate = s["pass"] / s["total"] if s["total"] else 0
        label = PATHOLOGY_LABELS.get(p, p)
        row_color = (80, 200, 120) if rate >= 0.8 else (220, 170, 50) if rate >= 0.5 else (200, 80, 80)

        # Compute per-pathology averages
        p_zones = [z for z in zones if z["pathology"] == p]
        avg_ssim = np.mean([z["metrics"].get("ssim_vs_real", 0) for z in p_zones])
        avg_mean = np.mean([z["metrics"].get("mean", 0) for z in p_zones])
        avg_plr = np.mean([z["metrics"].get("pleural_rate", 0) for z in p_zones])

        draw.text((20, ty), label, fill=(170, 180, 195), font=fonts["table"])
        draw.text((270, ty), str(s["total"]), fill=(160, 170, 185), font=fonts["table"])
        draw.text((345, ty), str(s["pass"]), fill=(80, 200, 120), font=fonts["table"])
        draw.text((425, ty), str(s["total"] - s["pass"]),
                  fill=(200, 80, 80) if s["total"] - s["pass"] > 0 else (80, 200, 120),
                  font=fonts["table"])

        # Rate with bar
        draw.text((500, ty), f"{rate:.0%}", fill=row_color, font=fonts["table_bold"])
        draw_bar(draw, 540, ty + 4, 40, 10, rate, row_color)

        # Avg similarity
        sim_color = (140, 190, 140) if avg_ssim > 0.05 else (210, 160, 100)
        draw.text((620, ty), f"{avg_ssim:.3f}", fill=sim_color, font=fonts["table"])

        # Avg brightness
        bright_ok = 0.1 < avg_mean < 0.7
        draw.text((760, ty), f"{avg_mean:.3f}", fill=(140, 190, 140) if bright_ok else (210, 110, 110),
                  font=fonts["table"])

        # Avg pleural
        plr_ok = avg_plr >= 0.5
        plr_label = f"{avg_plr:.0%}"
        if avg_plr < 0.5:
            plr_label += " (obscured)"
        draw.text((900, ty), plr_label,
                  fill=(140, 190, 140) if plr_ok else (210, 130, 100), font=fonts["table"])

        ty += table_row_h

    # ════════════════════════════════════════
    # SECTION 3: FAILED ZONES DETAIL
    # ════════════════════════════════════════

    if failed_list:
        sec3_y = sec2_y + table_h + 5
        draw.line([(15, sec3_y), (W - 15, sec3_y)], fill=(45, 50, 60), width=1)
        draw.text((20, sec3_y + 8), "Failed Zones — Detail",
                  fill=(210, 150, 150), font=fonts["bold_lg"])
        draw.text((20, sec3_y + 30),
                  "Note: Some failures are clinically expected (e.g., diffuse B-lines obscure the pleural line in real imaging too).",
                  fill=(140, 130, 120), font=fonts["sm"])

        fy = sec3_y + fail_header_h
        for i, z in enumerate(failed_list[:max_fail_show]):
            scen_label = SCENARIO_LABELS.get(z["scenario"], z["scenario"])
            path_label = PATHOLOGY_LABELS.get(z["pathology"], z["pathology"])
            zone_short = z["zone"].replace("_", " ")
            m = z["metrics"]
            reasons = []
            if m.get("pleural_rate", 1) < 0.5:
                reasons.append("Pleural line obscured")
            if not (0.1 < m.get("mean", 0.5) < 0.7):
                reasons.append("Brightness out of range")
            if m.get("std", 1) < 0.04:
                reasons.append("Too smooth")
            if m.get("ssim_vs_real", 1) < 0.03:
                reasons.append("Low similarity")
            reason_str = ", ".join(reasons) if reasons else "Below threshold"

            text = f"  {scen_label}  >  {zone_short}  >  {path_label}  —  {reason_str}"
            draw.text((20, fy), text, fill=(190, 140, 140), font=fonts["sm"])
            fy += fail_row_h

    # ════════════════════════════════════════
    # LEGEND
    # ════════════════════════════════════════

    legend_y = H - legend_h + 5
    draw.line([(15, legend_y), (W - 15, legend_y)], fill=(45, 50, 60), width=1)
    draw.text((20, legend_y + 8), "How to read this dashboard:", fill=(150, 160, 180), font=fonts["bold"])
    draw.rounded_rectangle([20, legend_y + 30, 36, legend_y + 44], radius=2, fill=(50, 195, 90))
    draw.text((42, legend_y + 28),
              "Pass = Frame meets all quality checks (brightness, contrast, pleural line visibility, similarity to real clinical images)",
              fill=(130, 140, 160), font=fonts["sm"])
    draw.rounded_rectangle([20, legend_y + 48, 36, legend_y + 62], radius=2, fill=(220, 65, 65))
    draw.text((42, legend_y + 46),
              "Fail = Below one or more thresholds — often clinically expected (e.g., B-lines/ARDS naturally obscure the pleural line)",
              fill=(130, 140, 160), font=fonts["sm"])

    img.save(output_path, quality=95)
    print(f"Dashboard: {output_path}")
    print(f"  {n_done}/{TOTAL_ZONES} | {n_pass} pass | {n_fail} fail | ETA {eta_h:.1f}h")


def main():
    data = parse_log(LOG_PATH)
    render_dashboard(data, PROGRESS_DIR / "dashboard.png")


if __name__ == "__main__":
    main()
