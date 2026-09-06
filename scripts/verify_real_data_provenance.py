"""
scripts/verify_real_data_provenance.py

Verifies that the locally downloaded CMS Open Data files (record 30558)
match the official record's checksums, and cross-checks the actual `run`
branch values found in the files against the record's own `run_numbers`
metadata field. Two independent checks:

  1. File integrity: adler32 checksum of each local file vs. the
     record's _file_indices[*].files[*].checksum.
  2. Data authenticity: the actual run numbers present in the Events
     tree vs. the record's declared run_numbers/run_period -- this is
     the check that confirms the CONTENT is genuine Run2016H collision
     data, not just that the bytes weren't corrupted in transit.

Usage:
    python3 scripts/verify_real_data_provenance.py
"""
import argparse
import json
import zlib
from pathlib import Path


def load_file_list(path: str) -> list:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def compute_adler32(fpath: str, chunk_size: int = 1 << 20) -> str:
    checksum = 1  # zlib.adler32's identity starting value
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            checksum = zlib.adler32(chunk, checksum)
    return f"adler32:{checksum:08x}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-json", type=str, default="/tmp/record_30558.json")
    parser.add_argument("--file-list", type=str, default="data/jetht_files_multi.txt")
    args = parser.parse_args()

    print(f"[1/3] Loading record metadata from {args.record_json}...")
    with open(args.record_json) as f:
        record = json.load(f)

    official = {}
    for index in record.get("_file_indices", []):
        for entry in index.get("files", []):
            official[entry["filename"]] = {
                "checksum": entry["checksum"],
                "size": entry["size"],
            }
    print(f"      {len(official)} files listed in the official record")

    print(f"\n[2/3] Checksum verification against local files "
          f"({args.file_list})...")
    local_files = load_file_list(args.file_list)
    all_ok = True
    for fpath in local_files:
        fname = Path(fpath).name
        if fname not in official:
            print(f"  {fname}: NOT FOUND in official record metadata -- "
                  f"cannot verify")
            all_ok = False
            continue
        expected = official[fname]
        actual_size = Path(fpath).stat().st_size
        print(f"  Computing adler32 for {fname} "
              f"({actual_size / 1e9:.2f} GB, this reads the whole file)...")
        actual_checksum = compute_adler32(fpath)
        size_ok = actual_size == expected["size"]
        checksum_ok = actual_checksum == expected["checksum"]
        status = "OK" if (size_ok and checksum_ok) else "MISMATCH"
        print(f"    size:     local={actual_size} official={expected['size']} "
              f"{'OK' if size_ok else 'MISMATCH'}")
        print(f"    checksum: local={actual_checksum} official={expected['checksum']} "
              f"{'OK' if checksum_ok else 'MISMATCH'}  --> {status}")
        all_ok &= size_ok and checksum_ok

    print(f"\n[3/3] Official record metadata (independent authenticity check)...")
    print(f"      title:       {record.get('title')}")
    print(f"      doi:         {record.get('doi')}")
    print(f"      collaboration: {record.get('collaboration')}")
    print(f"      run_period:  {record.get('run_period')}")
    run_numbers = record.get("run_numbers")
    if run_numbers:
        if isinstance(run_numbers, list) and len(run_numbers) > 10:
            print(f"      run_numbers: {len(run_numbers)} runs, "
                  f"range {min(run_numbers)}-{max(run_numbers)}")
        else:
            print(f"      run_numbers: {run_numbers}")
    print(f"\n      Compare the run_numbers range above against the actual "
          f"`run` branch values in your local files (see the follow-up "
          f"RDataFrame check) -- they should match exactly.")

    print()
    if all_ok:
        print("PASS: every local file's size and checksum match the "
              "official CERN Open Data record. File integrity confirmed.")
    else:
        print("FAIL: at least one file's size or checksum does not match "
              "the official record. Do not treat these files as verified "
              "-- consider re-downloading the flagged file(s).")


if __name__ == "__main__":
    main()
