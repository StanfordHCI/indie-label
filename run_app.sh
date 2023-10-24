#!/bin/bash

python3 server.py > pylog.txt 2>&1 &
cd indie_label_svelte
host=0.0.0.0 port=5000 npm run dev 2>&1 &




