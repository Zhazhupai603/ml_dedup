"""
Whole-genome feature extraction (efficient, streaming).

Strategy:
  1. Pass 1: scan raw.bam → collect (qname, chr, start, end, strand, key_features)
             + build position-density Counter (chrom, pos) → n_reads_at_pos
             + load Picard dup flag from mark_only.bam
             + load UMI-kept set from umi_dedup.bam
  2. Pass 2: for each read compute all features + label, write to parquet

Memory: O(N_mapped_reads) ~ a few hundred MB for ~3M reads.
"""

import pysam
import pandas as pd
import numpy as np
from collections import Counter
import os, sys, time, gc
import argparse


def gc_content(seq):
    if not seq:
        return 0.0
    seq = seq.upper()
    gc = sum(1 for b in seq if b in "GC")
    return gc / len(seq)


def parse_cigar(cigartuples):
    if not cigartuples:
        return {"M": 0, "I": 0, "D": 0, "N": 0, "S": 0, "H": 0, "P": 0, "=": 0, "X": 0}
    ops = {k: 0 for k in "MIDNSHP=X"}
    op_map = {0: "M", 1: "I", 2: "D", 3: "N", 4: "S", 5: "H", 6: "P", 7: "=", 8: "X"}
    for op, length in cigartuples:
        if op in op_map:
            ops[op_map[op]] += length
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bam-dir", default="/home/eagle/try3/bench/results_wg/bam")
    ap.add_argument("--sample", default="SRR28314028")
    ap.add_argument(
        "--out", default="/home/eagle/try3/bench/results_wg/features.parquet"
    )
    ap.add_argument("--max-reads", type=int, default=0, help="0 = all reads")
    args = ap.parse_args()

    SRR = args.sample
    RAW_BAM = f"{args.bam_dir}/{SRR}.raw.bam"
    UMI_BAM = f"{args.bam_dir}/{SRR}.umi_dedup.bam"
    MARK_BAM = f"{args.bam_dir}/{SRR}.mark_only.bam"
    OUT_PATH = args.out

    t0 = time.time()
    print(f"[1/5] Loading UMI-kept QNAMEs from {UMI_BAM} ...")
    umi_kept = set()
    n = 0
    with pysam.AlignmentFile(UMI_BAM) as bam:
        for r in bam.fetch(until_eof=True):
            if not r.is_unmapped:
                umi_kept.add(r.query_name)
                n += 1
                if n % 1_000_000 == 0:
                    print(
                        f"      ... {n:,} reads loaded ({len(umi_kept):,} unique kept)"
                    )
    print(f"      {len(umi_kept):,} unique-molecule reads kept by UMI dedup")

    print(f"[2/5] Loading Picard dup flags from {MARK_BAM} ...")
    picard_dup = {}
    n = 0
    with pysam.AlignmentFile(MARK_BAM) as bam:
        for r in bam.fetch(until_eof=True):
            if not r.is_unmapped:
                picard_dup[r.query_name] = int(r.is_duplicate)
                n += 1
                if n % 1_000_000 == 0:
                    print(f"      ... {n:,} reads loaded")
    print(
        f"      {sum(picard_dup.values()):,} marked as dup by Picard (of {len(picard_dup):,})"
    )

    print(f"[3/5] Pass 1: scanning {RAW_BAM} for position density ...")
    pos_density = Counter()
    all_reads_meta = []  # tuples of (qname, chrom, start, end, strand, n_reads_seen)
    n = 0
    n_mapped = 0
    with pysam.AlignmentFile(RAW_BAM) as bam:
        chroms = list(bam.references)
        print(f"      Reference contigs: {len(chroms)}")
        for r in bam.fetch(until_eof=True):
            n += 1
            if args.max_reads and n > args.max_reads:
                break
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            n_mapped += 1
            chrom = r.reference_name
            start = r.reference_start
            pos_density[(chrom, start)] += 1
            # store minimal info needed for pass 2
            all_reads_meta.append(
                (r.query_name, chrom, start, r.reference_end, 1 if r.is_reverse else 0)
            )
            if n % 1_000_000 == 0:
                print(f"      ... scanned {n:,} reads ({n_mapped:,} mapped)")
    print(f"      Total: {n:,} reads scanned, {n_mapped:,} mapped")
    print(f"      Distinct positions: {len(pos_density):,}")

    print(f"[4/5] Pass 2: feature extraction ...")
    rows = []
    read_idx = 0
    with pysam.AlignmentFile(RAW_BAM) as bam:
        for r in bam.fetch(until_eof=True):
            read_idx += 1
            if args.max_reads and read_idx > args.max_reads:
                break
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue

            seq = r.query_sequence or ""
            qual = r.query_qualities
            if qual is None:
                qual = np.array([])
            cig = parse_cigar(r.cigartuples)
            chrom = r.reference_name
            start = r.reference_start

            # local density (lookup only, no range scan)
            d_exact = pos_density.get((chrom, start), 0)

            feat = {
                "qname": r.query_name,
                "chrom": chrom,
                "start": start,
                "end": r.reference_end,
                "strand": 1 if r.is_reverse else 0,
                "mapq": r.mapping_quality,
                "read_len": r.query_length,
                "n_M": cig["M"],
                "n_I": cig["I"],
                "n_D": cig["D"],
                "n_S_left": cig["S"],  # approximation
                "n_S_right": cig["S"],
                "n_softclip": cig["S"],
                "gc_content": gc_content(seq),
                "mean_q": float(np.mean(qual)) if len(qual) > 0 else 0.0,
                "std_q": float(np.std(qual)) if len(qual) > 0 else 0.0,
                "min_q": float(np.min(qual)) if len(qual) > 0 else 0.0,
                "as_score": r.get_tag("AS") if r.has_tag("AS") else 0,
                "nm_mismatch": r.get_tag("nM") if r.has_tag("nM") else 0,
                "n_reads_at_pos": d_exact,
                "umi_label": 0 if r.query_name in umi_kept else 1,
                "picard_pred": picard_dup.get(r.query_name, 0),
            }
            rows.append(feat)

            if len(rows) % 500_000 == 0:
                print(f"      ... {len(rows):,} features extracted")

    print(f"      Total mapped reads with features: {len(rows):,}")

    print(f"[5/5] Saving parquet ...")
    df = pd.DataFrame(rows)
    # encode chrom as categorical for memory
    df["chrom"] = df["chrom"].astype("category")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"      Shape: {df.shape}")
    print(f"      Saved -> {OUT_PATH}")

    # quick label summary
    print(f"\n=== Label summary ===")
    print(
        f"UMI=unique (0): {(df.umi_label == 0).sum():,} ({(df.umi_label == 0).mean() * 100:.1f}%)"
    )
    print(
        f"UMI=dup    (1): {(df.umi_label == 1).sum():,} ({(df.umi_label == 1).mean() * 100:.1f}%)"
    )
    print(
        f"Picard=dup    : {(df.picard_pred == 1).sum():,} ({(df.picard_pred == 1).mean() * 100:.1f}%)"
    )

    # Picard vs UMI agreement
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(df.umi_label, df.picard_pred)
    if cm.shape == (2, 2):
        print(f"\nPicard vs UMI confusion:")
        print(f"               Picard=uniq  Picard=dup")
        print(f"  UMI=unique      {cm[0, 0]:>9,}      {cm[0, 1]:>9,}")
        print(f"  UMI=dup         {cm[1, 0]:>9,}      {cm[1, 1]:>9,}")
        acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
        print(f"  Picard accuracy vs UMI: {acc:.4f}")

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
