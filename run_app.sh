#!/bin/bash

cd indie_label_svelte
host=0.0.0.0 port=5000 npm run dev &
cd ..
python3 server.py





