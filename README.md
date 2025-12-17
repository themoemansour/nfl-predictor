NFL Predictor

NFL Predictor is a terminal-based NFL win probability model built using rolling play-by-play data.
The project is designed to avoid data leakage, support realistic historical evaluation, and provide
future matchup predictions using the most recent available team statistics.

The repository includes both a Python CLI tool and a standalone Windows executable.

------------------------------------------------------------

Overview

The model estimates the probability that the home team wins a matchup by comparing rolling team-level
performance metrics derived from play-by-play data.

Key design principles:
- No forward-looking data leakage
- Rolling feature construction using only prior games
- Clear separation between training, evaluation, and prediction
- Usable via command-line arguments or an interactive ASCII menu
- Distributable as a standalone executable

------------------------------------------------------------

Features

Prediction Modes

Exact mode:
- Predicts scheduled games using season and week
- Requires the matchup to exist and teams to have sufficient prior games

Upcoming mode:
- Predicts any matchup using each team’s latest available rolling statistics
- Suitable for future games and hypothetical matchups

Interface Options

- Command-line interface for scripting and automation
- ASCII menu interface for users who prefer not to use CLI arguments
- Windows executable that does not require Python to be installed

Modeling

- Rolling team features computed from play-by-play data
- Season-reset rolling windows to prevent data leakage
- Logistic regression baseline with probability calibration
- Walk-forward weekly evaluation for realistic in-season performance

------------------------------------------------------------

Usage

Predict an upcoming matchup using the latest available data:
python main.py --mode upcoming --home KC --away BUF

Predict a scheduled game by season and week:
python main.py --mode exact --season 2023 --week 14 --home DAL --away PHI

Launch the interactive ASCII menu:
python main.py --mode menu

If no arguments are provided, the application defaults to menu mode.

------------------------------------------------------------

Windows Executable

A standalone Windows executable is available via GitHub Releases.

To build the executable locally:
python -m PyInstaller --clean --onefile --name NFL_Predictor --collect-all nfl_data_py --collect-all pyarrow main.py

The executable will be created at:
dist/NFL_Predictor.exe

------------------------------------------------------------

Data Sources

- Play-by-play and schedule data provided by nfl_data_py
- Data is downloaded at runtime unless cached locally

------------------------------------------------------------

Evaluation

Model performance is evaluated using:
- ROC AUC
- Accuracy
- Brier score
- Walk-forward weekly evaluation for in-season realism

Evaluation metrics are printed directly to the terminal during execution.

------------------------------------------------------------

Project Structure

.
|-- main.py          Core application logic
|-- README.md        Project documentation
|-- .gitignore       Git exclusions
|-- build/           PyInstaller build artifacts (ignored)
|-- dist/            Executable output (ignored)

------------------------------------------------------------

Future Plans

Planned improvements include:
- Cross-season rolling support for Week 1 predictions
- Preseason team strength priors
- Additional features such as pace, flag rates, and splits
- Model upgrades using tree-based methods with probabilities
- 2025 Season stas
- Local CSV Stats upload
- Multi-platform executable builds

------------------------------------------------------------

Disclaimer

This project is intended for educational and analytical purposes only.
It is not designed for gambling or financial decision-making.

------------------------------------------------------------

License

MIT License
