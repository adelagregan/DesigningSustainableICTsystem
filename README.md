# Footprint model for our AI camera traps

This script computes the numbers in the Environmental Impact Assessment of our Sustainable AI report (case study: AI for wildlife monitoring in Borneo).

Run it with:

python3 footprint_model.py


## What it does

All inputs are in the `base` dictionary at the top. Each one has a comment with its source (a bib key from our report) or says "assumed" if it is our own estimate.

For one unit over 3 years, the script:

1. calculates how much energy the unit uses per day (`energy_per_day`)
2. sizes the battery and solar panel from that, with a minimum size for both
3. adds up the manufacturing emissions of the electronics, battery and panels, the training emissions, and the car trips for maintenance (`footprint`)

`mode` sets how the Raspberry Pi is used:
- `gated`: our design; the Pi is off and wakes once a day
- `always_on`: the Pi stays on and waits for the motion sensor
- `continuous`: no motion sensor; the Pi runs inference all the time

## Output

- **Our design:** energy per day, training energy and water, and the footprint split into parts
- **Decision table:** each row changes one of our design decisions and shows the new footprint
- **Sensitivity:** each row changes one assumption to a low and a high value; the last rows combine the two transport assumptions and test sunlight with an always-on Pi

## Changing an assumption

Change the value in `base` and run the script again. The most important one to check is `core_kg`, which is still a placeholder and should match the component table in our report.