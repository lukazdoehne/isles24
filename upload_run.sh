#!/bin/bash
#
# Upload a run to remote storage

RUNS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -r|--run)
      shift
      while [[ $# -gt 0 && ! "$1" == -* ]]; do
        RUNS+=("$1"); shift
      done
      ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ ${#RUNS[@]} -eq 0 ]]; then
  echo "Usage: $0 -r/--run <run-name> [run-name ...]"; exit 1
fi

mkdir -p /home/renku/work/data/runs/

for RUN in "${RUNS[@]}"; do
  rsync -avz --progress /home/renku/work/data-local/runs/$RUN/ /home/renku/work/data/runs/$RUN/
done
