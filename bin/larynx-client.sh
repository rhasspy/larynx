#!/usr/bin/env bash
voice="$1"
text="$2"

curl -X POST -s \
     -H 'Content-Type: text/plain' \
     --data-binary @- \
     "localhost:15002/api/tts?voice=${voice}" \
     --output -
