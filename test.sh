#!/bin/bash

SERVER_COMMANDS="cd ML_CC && python server.py"

osascript -e 'tell application "Terminal"' \
          -e 'do script "'"$SERVER_COMMANDS"'"' \
          -e 'end tell'

sleep 5
python aimd.py
sleep 5

sleep 5
python gen_dataframes.py 0
sleep 5

sleep 5
python gen_ranges.py 0
sleep 5

sleep 5
python gen_models.py 0
sleep 5

osascript -e 'tell application "Terminal"' \
          -e 'do script "'"$SERVER_COMMANDS"'"' \
          -e 'end tell'

sleep 5
python newreno.py
sleep 5

sleep 5
python gen_dataframes.py 1
sleep 5

sleep 5
python gen_ranges.py 1
sleep 5

sleep 5
python gen_models.py 1
sleep 5

osascript -e 'tell application "Terminal"' \
          -e 'do script "'"$SERVER_COMMANDS"'"' \
          -e 'end tell'

sleep 5
python lp.py 0
sleep 5

osascript -e 'tell application "Terminal"' \
          -e 'do script "'"$SERVER_COMMANDS"'"' \
          -e 'end tell'

sleep 5
python lp.py 1
sleep 5

osascript -e 'tell application "Terminal"' \
          -e 'do script "'"$SERVER_COMMANDS"'"' \
          -e 'end tell'

sleep 5
python rl.py 0
sleep 5

osascript -e 'tell application "Terminal"' \
          -e 'do script "'"$SERVER_COMMANDS"'"' \
          -e 'end tell'

sleep 5
python rl.py 1
sleep 5

sleep 5
python gen_dataframes.py 2
sleep 5

sleep 5
python gen_dataframes.py 3
sleep 5

sleep 5
python gen_dataframes.py 4
sleep 5

sleep 5
python gen_dataframes.py 5
sleep 5

sleep 5
python plots.py
sleep 5