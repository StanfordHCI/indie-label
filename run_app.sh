#!/bin/bash

cd indie_label_svelte
npm start &
cd ..
gunicorn -b 0.0.0.0:5001 -w 4 --limit-request-line 0 server:app





