"""
Extract per-read features from SRR28314028 BAMs.
Label source: UMI dedup ground truth.

- raw.bam:        all mapped reads, no dup flag (label unknown at this stage)
- umi_dedup.bam:  reads kept by umi_tools (label = 0 = unique molecule)
- mark_only.bam:  Picard-marked reads, dup flag = coord-based prediction

Label rule:
  QNAME in umi_dedup.bam  -> 0 (unique, kept by UMI dedup)
  QNAME in raw.bam \ umi_dedup  -> 1 (PCR duplicate, removed by UMI dedup)

Features (NOT using UMI itself):
  position: start, end, strand (encoded)
  alignment: MAPQ, AS (alignment score), nM (mismatches), NM_tag
  cigar: n_M, n_I, n_D, n_S_left, n_S_right, read_length
  sequence: GC_content, mean_base_q, std_base_q, min_base_q
  context: local_density_50bp, local_density_500bp, position_in_chr_pct
  picard_pred: Picard's coord-based dup flag (for baseline, NOT for training)

Output: results_ml/features.parquet
"""

import pysam
import pandas as pd
import numpy as np
from collections import Counter
import sys, os, time

BAM_DIR = "/home/eagle/try3/bench/results_real/bam"
RAW_BAM = f"{BAM_DIR}/SRR28314028.raw.bam"
UMI_BAM = f"{BAM_DIR}/SRR28314028.umi_dedup.bam"
MARK_BAM = f"{BAM_DIR}/SRR28314028.mark_only.bam"
OUT_PATH = "/home/eagle/try3/bench/results_ml/features.parquet"


def gc_content(seq):
    if not seq:
        return 0.0
    seq = seq.upper()
    gc = sum(1 for b in seq if b in "GC")
    return gc / len(seq)


def parse_cigar(cigartuples):
    """Return dict of cigar op counts."""
    ops = {key: 0 for key in ["M", "I", "D", "N", "S", "H", "P", "=", "X"]}
    op_map = {0: "M", 1: "I", 2: "D", 3: "N", 4: "S", 5: "H", 6: "P", 7: "=", 8: "X"}
    for op, length in cigartuples:
        ops[op_map.get(op, "?")] += length
    return ops


def extract_features(read, local_density_cache=None):
    """Extract single-read features."""
    seq = read.query_sequence or ""
    qual = read.query_qualities or np.array([])
    cig = parse_cigar(read.cigartuples or [])

    f = {
        "qname": read.query_name,
        "start": read.reference_start,
        "end": read.reference_end,
        "strand": 1 if read.is_reverse else 0,
        "mapq": read.mapping_quality,
        "read_len": read.query_length,
        "n_M": cig["M"],
        "n_I": cig["I"],
        "n_D": cig["D"],
        "n_S_left": cig["S"] if not read.is_reverse else 0,  # approximation
        "n_S_right": cig["S"],
        "n_softclip": cig["S"],
        "n_match_mismatch": cig["M"] + cig["="] + cig["X"],
        "gc_content": gc_content(seq),
        "mean_q": float(np.mean(qual)) if len(qual) > 0 else 0.0,
        "std_q": float(np.std(qual)) if len(qual) > 0 else 0.0,
        "min_q": float(np.min(qual)) if len(qual) > 0 else 0.0,
        "as_score": read.get_tag("AS") if read.has_tag("AS") else 0,
        "nm_mismatch": read.get_tag("nM") if read.has_tag("nM") else 0,
        "is_secondary": int(read.is_secondary),
        "is_supplementary": int(read.is_supplementary),
    }
    return f


def main():
    t0 = time.time()
    print(f"[1/4] Loading UMI-kept QNAMEs from {UMI_BAM} ...")
    umi_kept = set()
    with pysam.AlignmentFile(UMI_BAM) as bam:
        for r in bam.fetch(until_eof=True):
            if not r.is_unmapped:
                umi_kept.add(r.query_name)
    print(f"      {len(umi_kept):,} unique-molecule reads kept by UMI dedup")

    print(f"[2/4] Loading Picard dup flags from {MARK_BAM} ...")
    picard_dup = {}
    with pysam.AlignmentFile(MARK_BAM) as bam:
        for r in bam.fetch(until_eof=True):
            if not r.is_unmapped:
                picard_dup[r.query_name] = int(r.is_duplicate)
    print(
        f"      {sum(picard_dup.values()):,} marked as dup by Picard (of {len(picard_dup):,})"
    )

    print(f"[3/4] Extracting features from {RAW_BAM} (mapped reads only) ...")
    rows = []
    with pysam.AlignmentFile(RAW_BAM) as bam:
        # collect all mapped reads first for local density
        all_reads = []
        for r in bam.fetch(until_eof=True):
            if not r.is_unmapped:
                all_reads.append(r)
        print(f"      {len(all_reads):,} mapped reads in raw.bam")

        # build position histogram for local density (per-chr bins)
        positions = [r.reference_start for r in all_reads]
        pos_counter = Counter(positions)

        for r in all_reads:
            feat = extract_features(r)
            # local density: reads within +-50bp and +-500bp
            s = r.reference_start
            density_50 = sum(
                pos_counter[p] for p in range(s - 50, s + 51) if p in pos_counter
            )
            density_500 = sum(
                pos_counter[p] for p in range(s - 500, s + 501) if p in pos_counter
            )
            feat["density_50bp"] = density_50
            feat["density_500bp"] = density_500
            # position on chr as percentage
            feat["pos_pct"] = s / 50_000_000.0  # chr22 ~ 50Mb

            # UMI ground-truth label
            feat["umi_label"] = 0 if r.query_name in umi_kept else 1
            # Picard's prediction (for baseline)
            feat["picard_pred"] = picard_dup.get(r.query_name, 0)

            rows.append(feat)

    df = pd.DataFrame(rows)
    print(f"\n[4/4] Feature table shape: {df.shape}")
    print(
        f"      UMI labels: 0 (unique) = {(df.umi_label == 0).sum():,}, "
        f"1 (dup) = {(df.umi_label == 1).sum():,}"
    )
    print(f"      Picard pred: dup = {(df.picard_pred == 1).sum():,}")

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
    print(f"Elapsed: {time.time() - t0:.1f}s")

    # Quick label-vs-picard agreement
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(df.umi_label, df.picard_pred)
    print(f"\nPicard vs UMI confusion (rows=UMI truth, cols=Picard pred):")
    print(f"               Picard=uniq  Picard=dup")
    print(f"  UMI=unique      {cm[0, 0]:>7,}      {cm[0, 1]:>7,}")
    print(f"  UMI=dup         {cm[1, 0]:>7,}      {cm[1, 1]:>7,}")
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
    print(f"  Picard accuracy vs UMI: {acc:.3f}")


if __name__ == "__main__":
    main()
