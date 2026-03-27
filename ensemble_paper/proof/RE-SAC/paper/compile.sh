#!/usr/bin/env bash
# compile.sh — compiles paper.tex via Windows xelatex.exe (WSL workaround)
# xelatex.exe (Windows binary) cannot write output to WSL /home filesystem.
# Solution: copy sources to /mnt/d/ (Windows NTFS) where it CAN write, compile there, copy PDF back.

SRCDIR="$(cd "$(dirname "$0")" && pwd)"
TMPDIR="/mnt/d/tmp_latex_proof"

echo "[compile] Syncing sources to $TMPDIR ..."
mkdir -p "$TMPDIR"
cp "$SRCDIR"/*.tex "$TMPDIR"/
cp "$SRCDIR"/*.bib "$TMPDIR"/
cp "$SRCDIR"/*.png "$TMPDIR"/ 2>/dev/null || true

cd "$TMPDIR"

echo "[compile] xelatex (pass 1) ..."
xelatex.exe -interaction=nonstopmode paper || true

echo "[compile] bibtex ..."
bibtex.exe paper || true

echo "[compile] xelatex (pass 2) ..."
xelatex.exe -interaction=nonstopmode paper || true

echo "[compile] xelatex (pass 3 — finalise) ..."
xelatex.exe -interaction=nonstopmode paper

echo "[compile] Copying output back ..."
cp "$TMPDIR/paper.pdf"  "$SRCDIR/paper.pdf"
cp "$TMPDIR/paper.log"  "$SRCDIR/paper.log"

echo "[compile] Done — $(grep 'Output written' $TMPDIR/paper.log | tail -1)"
