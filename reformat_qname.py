#!/usr/bin/env python3
"""Reformat fastq QNAME: collapse Illumina comment to single token, keep UMI suffix.

  Before: @SRR28314028.2 LH00289:8:2273W2LT3:7:1101:8055:1016_CTTCGTTCTGG length=151
  After:  @SRR28314028.2_CTTCGTTCTGG

umi_tools --umi-separator=_ will then correctly parse UMI from QNAME.
"""

import sys, gzip

in_path = sys.argv[1]
out_path = sys.argv[2]

opener = gzip.open if in_path.endswith(".gz") else open
writer = gzip.open if out_path.endswith(".gz") else open

n = 0
with opener(in_path, "rt") as fin, writer(out_path, "wt") as fout:
    for i, line in enumerate(fin):
        if i % 4 == 0:  # header
            # @SRR28314028.2 LH00289:...:1016_CTTCGTTCTGG length=151
            line = line.rstrip("\n")
            if not line.startswith("@"):
                fout.write(line + "\n")
                continue
            parts = line[1:].split(" ", 1)  # split on first space
            spot = parts[0]  # SRR28314028.2
            rest = parts[1] if len(parts) > 1 else ""
            # rest = 'LH00289:...:1016_CTTCGTTCTGG length=151'
            # take everything before ' length=' as comment, then split by '_'
            rest = rest.split(" length=")[0]
            # rest = 'LH00289:8:2273W2LT3:7:1101:8055:1016_CTTCGTTCTGG'
            # the UMI is after the LAST '_'
            if "_" in rest:
                umi = rest.rsplit("_", 1)[1]
                new_header = f"@{spot}_{umi}"
            else:
                new_header = f"@{spot}"
            fout.write(new_header + "\n")
            n += 1
        else:
            fout.write(line)
        if n % 1_000_000 == 0 and n > 0 and i % 4 == 0:
            print(f"  ... {n:,} reads", file=sys.stderr)

print(f"Done. {n:,} reads", file=sys.stderr)
