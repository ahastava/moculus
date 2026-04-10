# Diaphragm Data Sourcing — Phase 3 Runbook

How to source, label, and merge diaphragm-visible POCUS frames for the
zone-aware fine-tune (Phase 5). This is the data-layer companion to
[ARCHITECTURE.md](ARCHITECTURE.md) and [RETRAINING.md](RETRAINING.md).

```
data/diaphragm_staging/  ──→  candidates  ──→  labels.csv  ──→  metadata.csv  ──→  diaphragm bank  ──→  LoRA fine-tune
   (you populate)             (256x256)         (you label)       (merged)        (built)              (trained)
```

The `data/` directory is gitignored, so the staging tree only exists
on local disk. Bootstrap it with:

```bash
mkdir -p data/diaphragm_staging/{covid_blues,litfl_ems,manual}
```

## Quick start (the whole pipeline in one block)

```bash
# 1. Drop sourced media files into the per-source subdirs below
#    (see "Sources" section for what to download from where)

# 2. Run the generic ingest pipeline
python3 -m src.acquire_diaphragm_data \
    --staging data/diaphragm_staging \
    --output  data/diaphragm_candidates \
    --verbose

# 3. Open scripts/labeling_review.html in your browser
#    Click "Load candidates" → select all diap_cand_*.png files
#    Use 0-9 for class, Q-Y for zone, Ctrl+S to save labels CSV

# 4. Merge labels into metadata.csv
python3 scripts/merge_diaphragm_labels.py \
    --labels         data/diaphragm_candidates/diaphragm_labels.csv \
    --candidates-dir data/diaphragm_candidates

# 5. Rebuild the zone-aware anatomy bank with the new patches
python3 scripts/build_diaphragm_bank.py

# 6. Phase 5 — kick off the LoRA fine-tune (in tmux so you can disconnect)
tmux new -s moculus_train
python3 -m src.train_realistic \
    --resume checkpoints/realistic_v2_finetune/latest.pt \
    --finetune --lora --lora-rank 16 --lora-alpha 32 \
    --output-dir checkpoints/realistic_v2_diaphragm_lora \
    --epochs 20 --batch-size 16 --lr 3e-4 \
    --cfg-dropout 0.10 --sample-every 3 --save-every 5
# Ctrl+B then D to detach. Reattach: tmux attach -t moculus_train

# 7. Phase 6 — regenerate the lower-zone slice of the cache
python3 scripts/smart_cache_gen.py \
    --only-zones LOWER_BLUE_L,LOWER_BLUE_R,PLAPS_L,PLAPS_R,DIAPHRAGM_L,DIAPHRAGM_R \
    --diaphragm-model checkpoints/realistic_v2_diaphragm_lora/latest_lora.pt \
    --max-retries 3
```

---

## Sources

### Auto-ingestible (no manual labeling needed)

#### `covid_blues/` — COVID-BLUES dataset

The COVID-BLUES dataset already exists at
`data/_sources/covid_blues_meta/lus_videos/` and contains 362 lung
ultrasound videos taken at the **6 BLUE protocol points** for each of
63 patients. The filenames encode the zone directly:

```
patient_<ID>_L1.mp4  →  Upper BLUE Left   (zone_region 0 — UPPER_GENERIC)
patient_<ID>_L2.mp4  →  Lower BLUE Left   (zone_region 1)
patient_<ID>_L3.mp4  →  PLAPS Left        (zone_region 3)
patient_<ID>_R1.mp4  →  Upper BLUE Right  (zone_region 0)
patient_<ID>_R2.mp4  →  Lower BLUE Right  (zone_region 2)
patient_<ID>_R3.mp4  →  PLAPS Right       (zone_region 4)
```

Severity scores (0-3) per video are in
`data/_sources/covid_blues_meta/severity.csv` along with explicit
`A-lines` and `B-lines` annotations from medical experts.

**Source**: [COVID-BLUES on GitHub](https://github.com/NinaWie/COVID-BLUeS-dataset)
**Paper**: [Wiedemann et al. (2025), IEEE J Biomed Health Inform](https://ieeexplore.ieee.org/abstract/document/10903196)
**License**: **CC BY-NC-ND 4.0** — Non-commercial, attribution required.
The "no derivatives" clause is the catch — using these frames for ML
training is technically a derivative work and is in a legal gray area.
The MoCoLUS project already cites COVID-BLUES under research/educational
fair use; you should review with your institution before publishing.

**Auto-ingest**: Run `scripts/ingest_covid_blues.py` (built in Phase 3
for this exact dataset). It reads filenames + severity.csv, extracts
frames at 2 fps, and writes pre-labeled candidates with zone_region
already set — **no manual labeling needed**. See that script's
`--help` for options.

#### `litfl_ems/` — LITFL pathology videos

`data/_sources/covidx_us_ems_videos/` contains 59 LITFL videos labeled
by pathology in the filename:

```
113_litfl_other_Pleural_Effusion.mp4
133_litfl_other_Pneumothorax.mp4
103_litfl_other_Lung_Point.mp4
...
```

**Source**: [LITFL POCUS](https://litfl.com/) (aggregated by COVIDx-US)
**License**: **CC BY-NC-SA 4.0** — Non-commercial, share-alike,
**derivatives ARE permitted** under the same license. This is the
cleanest license for ML training in this directory.

**Caveat**: These videos do NOT have zone labels in the filename. They're
individual case captures, not BLUE-protocol scans. For diaphragm-relevant
classes (effusion, consolidation), the conventional imaging approach is
PLAPS or Diaphragm view, so a default zone_region of `PLAPS_L`/`PLAPS_R`
or `DIAPHRAGM_L`/`DIAPHRAGM_R` is a reasonable best-guess. The labeling
UI lets you confirm or adjust per-frame.

### Manual sourcing required

#### `manual/` — anything else you download by hand

Drop any other media here from sources that need manual download:

- **POCUS Atlas** (`thepocusatlas.com`) — CC BY-NC 4.0. Lung section.
  Save individual case images. Each case page has a "Download" link.
- **GrepMed** (`grepmed.com/?q=lung+ultrasound+diaphragm`) — Educational
  fair use. Tag-search for `#diaphragm`, `#spine-sign`, `#curtain-sign`.
- **NEJM Lung Ultrasound supplements** — Requires NEJM institutional
  access. Save the supplementary videos from
  [Lung Ultrasound, Lichtenstein et al.](https://www.nejm.org/doi/full/10.1056/NEJMvcm2108203).
  Permissions required for derivative use.
- **Butterfly Network teaching cases** — Login-gated educational content.
  Save individually from the iOS/Android app or web library.
- **University of Rochester LUS dataset** — Research access request.
  Email the corresponding author for the data use agreement.

For these, the per-frame labels are unknown until you visually inspect
them, so the manual labeling UI (`scripts/labeling_review.html`) is
required.

---

## Licensing summary

| Source | License | Derivatives OK? | Notes |
|---|---|---|---|
| COVID-BLUES | CC BY-NC-ND 4.0 | ⚠️ Gray area | "no derivatives" — review with your institution |
| LITFL via COVIDx-US | CC BY-NC-SA 4.0 | ✅ Yes | Cleanest license for ML training |
| POCUS Atlas | CC BY-NC 4.0 | ✅ Yes | Standard fair-use citation |
| GrepMed | Fair use | ⚠️ Variable | Per-case basis |
| NEJM | Permissions required | ❌ Unless granted | Institutional access needed |
| Butterfly | Educational fair use | ⚠️ Variable | Login-gated |
| Rochester LUS | DUA required | ✅ With agreement | Email authors |

**Recommendation**: Start with the LITFL videos (`litfl_ems/`) since
the license is the cleanest. COVID-BLUES is a much larger dataset but
the licensing is more restrictive — talk to your institution first if
you plan to publish anything based on the trained model.

---

## Coverage targets

For the LoRA fine-tune to learn meaningful zone-conditional deltas, you
want at least 30-50 frames per `(pathology_class, zone_region)` cell
across the diaphragm-relevant cells:

| Pathology class | Zones to cover | Min frames per zone |
|---|---|---|
| 4 (Consolidation) | PLAPS_L, PLAPS_R, DIAPHRAGM_L, DIAPHRAGM_R | 30-50 each |
| 5 (Pleural Effusion) | All 6 lower zones | 30-50 each |
| 6 (ARDS) | All 6 lower zones | 30-50 each |
| 9 (Interstitial) | All 6 lower zones | 30-50 each |

Total target: ~500-1000 labeled frames after dedup. The auto-ingest
on COVID-BLUES alone can produce 4-8x this with no manual labeling,
which is why it's the recommended starting point if licensing is OK.

After running ingest + merge, `scripts/merge_diaphragm_labels.py`
prints a per-cell coverage report so you can see which cells need
more data before kicking off Phase 5 training.

---

## Files referenced by this runbook

| File | Purpose |
|---|---|
| [src/acquire_diaphragm_data.py](src/acquire_diaphragm_data.py) | Generic ingest (any local media → 256×256 candidates) |
| [scripts/ingest_covid_blues.py](scripts/ingest_covid_blues.py) | COVID-BLUES auto-ingest (zones from filename, no manual labeling) |
| [scripts/labeling_review.html](scripts/labeling_review.html) | Single-page labeling UI (open in any browser) |
| [scripts/merge_diaphragm_labels.py](scripts/merge_diaphragm_labels.py) | Merge labels CSV → metadata.csv with zone_region column |
| [scripts/build_diaphragm_bank.py](scripts/build_diaphragm_bank.py) | Build the zone-aware anatomy bank |
| [src/train_realistic.py](src/train_realistic.py) | LoRA fine-tune entry point (Phase 5) |
| [scripts/smart_cache_gen.py](scripts/smart_cache_gen.py) | Cache regen with --only-zones + --diaphragm-model (Phase 6) |

## Local-disk staging structure (gitignored)

After running `mkdir -p data/diaphragm_staging/{covid_blues,litfl_ems,manual}`:

```
data/diaphragm_staging/
├── covid_blues/   # staging slot for COVID-BLUES manual ingest (mostly
│                  # unused since ingest_covid_blues.py reads from
│                  # data/_sources/covid_blues_meta/ directly)
├── litfl_ems/     # staging slot for LITFL videos (mostly unused for
│                  # the same reason — they're in data/_sources/)
└── manual/        # drop arbitrary downloaded media here for the
                   # generic src.acquire_diaphragm_data pipeline
```
