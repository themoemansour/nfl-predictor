# NFL Predictor

Terminal-based NFL win probability predictor using rolling play-by-play features.

## Features
- `exact` mode: predict scheduled games by season/week
- `upcoming` mode: predict any matchup using latest available team rolling stats
- ASCII menu mode for non-CLI users

## Requirements
- Python 3.11+
- See `requirements.txt`

## Run
```bash
python main.py --mode upcoming --home KC --away BUF
python main.py --mode exact --season 2023 --week 14 --home DAL --away PHI
python main.py --mode menu
