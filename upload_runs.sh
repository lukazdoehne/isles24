#!/bin/bash
#
# Upload runs to remote storage
mkdir data/runs/
rsync -avz --progress /home/renku/work/data-local/runs/ /home/renku/work/data/runs/
