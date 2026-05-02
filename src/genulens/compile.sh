#!/bin/bash

# ソースファイル名（デフォルト: genprior.c）
SRCFILE=${1:-genprior.c}
OUTNAME=$(basename "$SRCFILE" .c)

# ビルド
gcc -O3 -g -c "$SRCFILE" -I. -I/usr/local/include -std=c99 -o "${OUTNAME}.o"
gcc -O3 -g -o "$OUTNAME" -lm -I. -I/usr/local/include -std=c99 "${OUTNAME}.o" random.o option.o

echo "Build complete: $OUTNAME"

