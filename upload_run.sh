#!/bin/bash
#
# Upload a run to remote storage
while [[ $# -gt 0 ]]; do
  case $1 in
    -r|--run) RUN="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$RUN" ]]; then
  echo "Usage: $0 -r/--run <run-name>"; exit 1
fi

mkdir -p /home/renku/work/data/runs/
rsync -avz --progress /home/renku/work/data-local/runs/$RUN/ /home/renku/work/data/runs/$RUN/
