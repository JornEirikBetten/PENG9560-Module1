#!/bin/bash -x 
python3 -m venv ~/myvenv/env
source ~/myvenv/env/bin/activate
cp requirements.txt ~/myvenv/env/requirements.txt
pip3 install -r ~/myvenv/env/requirements.txt