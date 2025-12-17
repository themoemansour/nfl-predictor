# main.py — NFL win probability model
#
# Binary classifier for NFL game outcomes built on rolling play-by-play features.
# The pipeline is structured to avoid data leakage and to support both
# historical backtesting and forward-looking matchup evaluation.
#
# Example usage:
#   python main.py --mode exact --season 2023 --week 14 --home DAL --away PHI
#   python main.py --mode upcoming --home KC --away BUF
#   python main.py --mode upcoming --home KC --away BUF --asof-season 2023 --asof-week 9
#
# If no CLI arguments are provided, the program starts an interactive ASCII menu.


print(r"""
███╗   ██╗ ███████╗██╗         ██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗ ██████╗ ██████╗
████╗  ██║ ██╔════╝██║         ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
██╔██╗ ██║ █████╗  ██║         ██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║   ██║   ██║██████╔╝
██║╚██╗██║ ██╔══╝  ██║         ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║   ██║   ██║██╔══██╗
██║ ╚████║ ██║     ███████╗    ██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║   ╚██████╔╝██║  ██║
╚═╝  ╚═══╝ ╚═╝     ╚══════╝    ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝

                      WIN PROBABILITY | ROLLING STATS | NO DATA LEAKAGE
                           NFL PREDICTOR - TERMINAL APPLICATION
""")


import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

print("FILE  :", __file__)
print("CWD   :", os.getcwd())
print("PYTHON:", sys.executable)

import numpy as np
import pandas as pd

import nfl_data_py as nfl

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV


# -----------------------------
# Team alias handling
# -----------------------------
TEAM_ALIASES = {
    "KC":  ["KC", "KAN"],
    "JAX": ["JAX", "JAC"],
    "LV":  ["LV", "LVR", "OAK"],
    "OAK": ["OAK", "LV", "LVR"],
    "LAR": ["LAR", "LA", "STL"],
    "LA":  ["LA", "LAR", "LAC"],
    "LAC": ["LAC", "LA", "SD"],
    "SD":  ["SD", "LAC"],
    "WAS": ["WAS", "WSH"],
    "WSH": ["WSH", "WAS"],
}

def normalize_team(team: str, valid_teams: set[str]) -> str:
    if team is None:
        return team
    t = team.strip().upper()
    if t in valid_teams:
        return t
    for cand in TEAM_ALIASES.get(t, [t]):
        if cand in valid_teams:
            return cand
    return t


# -----------------------------
# Feature engineering helpers
# -----------------------------
def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


def _prep_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.copy()
    pbp = pbp[pbp["season"].notna() & pbp["week"].notna()]
    pbp = pbp[pbp["posteam"].notna() & pbp["defteam"].notna()]
    pbp = pbp[pbp["epa"].notna()]

    if "play_type" in pbp.columns:
        pbp = pbp[pbp["play_type"].isin(["run", "pass"])]

    if "qb_spike" in pbp.columns:
        pbp = pbp[pbp["qb_spike"] != 1]
    if "qb_kneel" in pbp.columns:
        pbp = pbp[pbp["qb_kneel"] != 1]
    return pbp


def build_team_week_features(pbp_raw: pd.DataFrame, roll_n: int = 6, min_periods: int = 3) -> tuple[pd.DataFrame, dict]:
    """
    Construct per-team, per-week features and season-scoped rolling aggregates
    that only use information from games played strictly before the given week.
    """
    pbp = _prep_pbp(pbp_raw)

    meta = {"proe_available": False, "proe_cols_used": None}

    pbp["success"] = (pbp["epa"] > 0).astype(int)

    # Explosive plays: pass >= 20 yards, rush >= 10 yards
    if "yards_gained" in pbp.columns and "play_type" in pbp.columns:
        pbp["explosive"] = np.where(
            (pbp["play_type"] == "pass") & (pbp["yards_gained"] >= 20), 1,
            np.where((pbp["play_type"] == "run") & (pbp["yards_gained"] >= 10), 1, 0)
        )
    else:
        pbp["explosive"] = 0

    # Turnovers (interceptions + fumbles lost)
    pbp["turnover"] = 0
    if "interception" in pbp.columns:
        pbp["turnover"] += pbp["interception"].fillna(0).astype(int)
    if "fumble_lost" in pbp.columns:
        pbp["turnover"] += pbp["fumble_lost"].fillna(0).astype(int)

    # Pass rate over expected (PROE), if available
    if _has_cols(pbp, ["pass", "pass_probability"]):
        pbp["proe_play"] = pbp["pass"].fillna(0).astype(float) - pbp["pass_probability"].astype(float)
        meta["proe_available"] = True
        meta["proe_cols_used"] = ["pass", "pass_probability"]
    else:
        pbp["proe_play"] = np.nan

    off_aggs = {
        "off_epa_per_play": ("epa", "mean"),
        "off_plays": ("epa", "size"),
        "off_success_rate": ("success", "mean"),
        "off_explosive_rate": ("explosive", "mean"),
        "off_turnovers": ("turnover", "sum"),
    }
    if meta["proe_available"]:
        off_aggs["off_proe"] = ("proe_play", "mean")

    off = (
        pbp.groupby(["season", "week", "posteam"], as_index=False)
           .agg(**off_aggs)
           .rename(columns={"posteam": "team"})
    )

    def_aggs = {
        "def_epa_allowed_per_play": ("epa", "mean"),
        "def_plays": ("epa", "size"),
        "def_success_allowed_rate": ("success", "mean"),
        "def_explosive_allowed_rate": ("explosive", "mean"),
        "def_takeaways": ("turnover", "sum"),
    }
    if meta["proe_available"]:
        def_aggs["def_proe_allowed"] = ("proe_play", "mean")

    deff = (
        pbp.groupby(["season", "week", "defteam"], as_index=False)
           .agg(**def_aggs)
           .rename(columns={"defteam": "team"})
    )

    team_week = (
        off.merge(deff, on=["season", "week", "team"], how="outer")
           .sort_values(["team", "season", "week"])
           .reset_index(drop=True)
    )

    for c in ["off_plays", "def_plays", "off_turnovers", "def_takeaways"]:
        if c in team_week.columns:
            team_week[c] = team_week[c].fillna(0)

    grp = team_week.groupby(["team", "season"], sort=False)

    def shifted_roll_mean(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(roll_n, min_periods=min_periods).mean()

    roll_cols = [c for c in team_week.columns if c not in ["season", "week", "team"]]
    for col in roll_cols:
        team_week[f"{col}_roll{roll_n}"] = grp[col].transform(shifted_roll_mean)

    keep = ["season", "week", "team"] + [f"{c}_roll{roll_n}" for c in roll_cols]
    return team_week[keep].copy(), meta


def build_model_df(games: pd.DataFrame, team_feats: pd.DataFrame) -> pd.DataFrame:
    df = games.merge(
        team_feats,
        left_on=["season", "week", "home_team"],
        right_on=["season", "week", "team"],
        how="left"
    ).drop(columns=["team"]).rename(columns={
        **{c: "home_" + c for c in team_feats.columns if c not in ["season", "week", "team"]}
    })

    df = df.merge(
        team_feats,
        left_on=["season", "week", "away_team"],
        right_on=["season", "week", "team"],
        how="left"
    ).drop(columns=["team"]).rename(columns={
        **{c: "away_" + c for c in team_feats.columns if c not in ["season", "week", "team"]}
    })
    return df


def make_feature_matrix(df: pd.DataFrame, roll_n: int) -> tuple[pd.DataFrame, pd.Series]:
    y = df["home_win"].astype(int)

    def col(name: str) -> tuple[str, str]:
        return (f"home_{name}_roll{roll_n}", f"away_{name}_roll{roll_n}")

    features = {}

    # Offense: higher better (turnovers inverted)
    for base in ["off_epa_per_play", "off_success_rate", "off_explosive_rate", "off_turnovers", "off_proe"]:
        h, a = col(base)
        if h in df.columns and a in df.columns:
            if base == "off_turnovers":
                features["turnover_adv"] = df[a] - df[h]  # positive => home commits fewer
            else:
                features[f"{base}_diff"] = df[h] - df[a]

    # Defense allowed: lower better => advantage = away - home
    for base in ["def_epa_allowed_per_play", "def_success_allowed_rate", "def_explosive_allowed_rate", "def_proe_allowed"]:
        h, a = col(base)
        if h in df.columns and a in df.columns:
            features[f"{base}_adv"] = df[a] - df[h]

    # Takeaways: higher better
    h, a = col("def_takeaways")
    if h in df.columns and a in df.columns:
        features["takeaways_diff"] = df[h] - df[a]

    # Play counts
    for base in ["off_plays", "def_plays"]:
        h, a = col(base)
        if h in df.columns and a in df.columns:
            features[f"{base}_diff"] = df[h] - df[a]

    features["home_field"] = 1.0
    X = pd.DataFrame(features, index=df.index)
    return X, y


# -----------------------------
# Model training and evaluation
# -----------------------------
def static_train_test_with_calibration(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    """
    Train a logistic regression model with a fixed, time-ordered split and
    apply out-of-sample probability calibration.

    - Training set: seasons <= 2019
    - Calibration set: season == 2020
    - Test set: seasons >= 2021
    """
    train_base = df["season"] <= 2019
    calib = df["season"] == 2020
    test = df["season"] >= 2021

    X_train, y_train = X[train_base], y[train_base]
    X_cal, y_cal = X[calib], y[calib]
    X_test, y_test = X[test], y[test]

    base = LogisticRegression(max_iter=5000)
    base.fit(X_train, y_train)

    cal = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
    cal.fit(X_cal, y_cal)

    pred_proba = cal.predict_proba(X_test)[:, 1]
    pred_label = (pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, pred_proba)
    acc = accuracy_score(y_test, pred_label)
    brier = brier_score_loss(y_test, pred_proba)

    frac_pos, mean_pred = calibration_curve(y_test, pred_proba, n_bins=10, strategy="quantile")
    cal_df = pd.DataFrame({"mean_pred": mean_pred, "frac_pos": frac_pos})

    return cal, {"AUC": auc, "Accuracy": acc, "Brier": brier, "calibration_table": cal_df}


def walk_forward_weekly(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, start_season: int = 2021):
    """
    Perform a simple walk-forward evaluation where each (season, week)
    is predicted using a model trained on all prior weeks.
    """
    mask_test = df["season"] >= start_season
    test_weeks = (
        df.loc[mask_test, ["season", "week"]]
          .drop_duplicates()
          .sort_values(["season", "week"])
          .values.tolist()
    )

    all_probs, all_true = [], []
    for (S, W) in test_weeks:
        train_mask = (df["season"] < S) | ((df["season"] == S) & (df["week"] < W))
        pred_mask = (df["season"] == S) & (df["week"] == W)

        X_train, y_train = X[train_mask], y[train_mask]
        X_pred, y_pred = X[pred_mask], y[pred_mask]

        if len(X_train) < 200 or len(X_pred) == 0:
            continue

        m = LogisticRegression(max_iter=5000)
        m.fit(X_train, y_train)
        probs = m.predict_proba(X_pred)[:, 1]

        all_probs.append(probs)
        all_true.append(y_pred.values)

    if not all_probs:
        return None

    probs = np.concatenate(all_probs)
    true = np.concatenate(all_true)
    return {
        "AUC": roc_auc_score(true, probs),
        "Accuracy": accuracy_score(true, (probs >= 0.5).astype(int)),
        "Brier": brier_score_loss(true, probs),
        "n_games": len(true),
    }


# -----------------------------
# Prediction entry points
# -----------------------------
def predict_exact_game(model, team_feats: pd.DataFrame, roll_n: int, home: str, away: str, season: int, week: int):
    """
    Predict a scheduled regular-season game at (season, week) using the
    precomputed rolling features for the corresponding week.

    If either team has no features for that (season, week), the function
    logs an error and returns without producing a probability.
    """
    valid_teams = set(team_feats["team"].unique())
    home = normalize_team(home, valid_teams)
    away = normalize_team(away, valid_teams)

    rh = team_feats[(team_feats["season"] == season) & (team_feats["week"] == week) & (team_feats["team"] == home)]
    ra = team_feats[(team_feats["season"] == season) & (team_feats["week"] == week) & (team_feats["team"] == away)]

    if rh.empty or ra.empty:
        print(f"\n[ERROR] EXACT mode: missing rolling features for {home} or {away} at season={season}, week={week}.")
        print("        Common reasons: BYE week, matchup not in that week, or too early (week<4).")
        return

    return _predict_from_rows(model, team_feats, roll_n, home, away, rh.iloc[0], ra.iloc[0], context=f"Season {season}, Week {week} (EXACT)")


def predict_upcoming_matchup(model, team_feats: pd.DataFrame, roll_n: int, home: str, away: str, asof_season: int | None, asof_week: int | None):
    """
    Predict a generic matchup using each team’s latest available rolling
    features, optionally constrained to an "as-of" (season, week) cutoff.
    """
    valid_teams = set(team_feats["team"].unique())
    home = normalize_team(home, valid_teams)
    away = normalize_team(away, valid_teams)

    tf = team_feats.copy()

    # Apply as-of constraint
    if asof_season is not None and asof_week is not None:
        tf = tf[(tf["season"] < asof_season) | ((tf["season"] == asof_season) & (tf["week"] <= asof_week))]
        context = f"as-of Season {asof_season}, Week {asof_week} (UPCOMING)"
    else:
        context = "latest available data (UPCOMING)"

    # Latest row per team by (season, week)
    tf = tf.sort_values(["team", "season", "week"])
    latest = tf.groupby("team", as_index=False).tail(1).set_index("team")

    if home not in latest.index or away not in latest.index:
        print(f"\n[ERROR] UPCOMING mode: missing latest features for {home} or {away}.")
        return

    rh = latest.loc[home]
    ra = latest.loc[away]
    return _predict_from_rows(model, team_feats, roll_n, home, away, rh, ra, context=context)


def _predict_from_rows(model, team_feats: pd.DataFrame, roll_n: int, home: str, away: str, rh: pd.Series, ra: pd.Series, context: str):
    temp = pd.DataFrame({"home_win": [0]})
    for c in team_feats.columns:
        if c in ["season", "week", "team"]:
            continue
        temp[f"home_{c}"] = rh.get(c, np.nan)
        temp[f"away_{c}"] = ra.get(c, np.nan)

    X_one, _ = make_feature_matrix(temp, roll_n)
    if X_one.isna().any().any():
        print("\n[ERROR] Prediction failed: NaNs in matchup features (likely too early for rolling stats).")
        return

    prob = model.predict_proba(X_one)[:, 1][0]

    print(f"\n[NFL] {away} @ {home} — {context}")
    print(f"[OK] Home win probability ({home}): {prob:.3f}")

    diffs = X_one.iloc[0].drop(labels=["home_field"], errors="ignore").sort_values(key=lambda s: s.abs(), ascending=False)
    print("\nTop feature edges (home advantage + / away advantage -):")
    for k, v in diffs.head(10).items():
        print(f"  {k:30s} {v:+.4f}")

    return prob


# -----------------------------
# Interactive ASCII menu
# -----------------------------
def ascii_menu(valid_teams: list[str]) -> dict:
    print("\n" + "=" * 70)
    print(" NFL PREDICTOR — ASCII MODE (for CLI-phobic legends) ")
    print("=" * 70)
    print("1) EXACT game (requires season+week that game actually occurs)")
    print("2) UPCOMING matchup (any teams; uses latest rolling stats)")
    print("3) Quit")
    choice = input("\nSelect option [1-3]: ").strip()

    if choice == "3":
        return {"quit": True}

    mode = "exact" if choice == "1" else "upcoming"
    print("\nTeam codes examples:", ", ".join(valid_teams[:12]), "...")
    home = input("Home team code (ex: KC, DAL, PHI): ").strip().upper()
    away = input("Away team code (ex: BUF, SF, MIA): ").strip().upper()

    out = {"mode": mode, "home": home, "away": away, "menu": True}

    if mode == "exact":
        out["season"] = int(input("Season (ex: 2023): ").strip())
        out["week"] = int(input("Week (ex: 14): ").strip())
    else:
        ans = input("Use 'as-of' date? (y/N): ").strip().lower()
        if ans == "y":
            out["asof_season"] = int(input("As-of season (ex: 2023): ").strip())
            out["asof_week"] = int(input("As-of week (ex: 9): ").strip())
        else:
            out["asof_season"] = None
            out["asof_week"] = None

    return out


# -----------------------------
# Command-line interface
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="NFL win probability model (rolling PBP features, no leakage)."
    )
    p.add_argument("--mode", type=str, choices=["exact", "upcoming", "menu"], default=None,
                   help="exact = scheduled game; upcoming = any matchup using latest stats; menu = ASCII UI")
    p.add_argument("--home", type=str, default=None, help="Home team (e.g., KC, DAL, PHI)")
    p.add_argument("--away", type=str, default=None, help="Away team (e.g., BUF, SF, MIA)")
    p.add_argument("--season", type=int, default=None, help="Season (required for exact)")
    p.add_argument("--week", type=int, default=None, help="Week (required for exact)")
    p.add_argument("--asof-season", type=int, default=None, help="As-of season (optional for upcoming)")
    p.add_argument("--asof-week", type=int, default=None, help="As-of week (optional for upcoming)")
    return p.parse_args()


def main():
    args = parse_args()

    # Config
    seasons = list(range(2010, 2024))
    ROLL_N = 6
    MIN_PERIODS = 3

    # Load schedules + labels
    sched = nfl.import_schedules(seasons)
    games = sched[(sched["game_type"] == "REG") & (sched["result"].notna())].copy()
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    games = games[[
        "season", "week", "game_id",
        "home_team", "away_team",
        "home_score", "away_score",
        "home_win"
    ]].copy()

    # Load PBP
    pbp = nfl.import_pbp_data(seasons, downcast=True)

    # Build rolling features
    team_feats, meta = build_team_week_features(pbp, roll_n=ROLL_N, min_periods=MIN_PERIODS)

    # Join for evaluation
    df = build_model_df(games, team_feats)
    required_cols = []
    for c in team_feats.columns:
        if c in ["season", "week", "team"]:
            continue
        required_cols += [f"home_{c}", f"away_{c}"]
    df = df.dropna(subset=required_cols).copy()

    print("\n--- DATASET SUMMARY ---")
    print("Earliest week in modeling df:", int(df["week"].min()))
    print("Games total (after dropna):", len(df))
    print("PROE available:", meta["proe_available"], "| cols:", meta["proe_cols_used"])

    X, y = make_feature_matrix(df, ROLL_N)

    # Train + calibrate (stable)
    print("\n--- STATIC SPLIT + CALIBRATION ---")
    model_cal, static_metrics = static_train_test_with_calibration(df, X, y)
    print("Train: 2010–2019 | Calibrate: 2020 | Test: 2021+")
    print("AUC     :", static_metrics["AUC"])
    print("Accuracy:", static_metrics["Accuracy"])
    print("Brier   :", static_metrics["Brier"])
    print("\nCalibration (10 bins):")
    print(static_metrics["calibration_table"].to_string(index=False))

    print("\n--- WALK-FORWARD (WEEKLY) EVAL ---")
    wf = walk_forward_weekly(df, X, y, start_season=2021)
    if wf is None:
        print("Walk-forward failed (not enough data).")
    else:
        print("Start season: 2021+")
        print("AUC     :", wf["AUC"])
        print("Accuracy:", wf["Accuracy"])
        print("Brier   :", wf["Brier"])
        print("n_games :", wf["n_games"])

    # Determine how to run (CLI vs Menu)
    valid_teams_sorted = sorted(list(set(team_feats["team"].unique())))

    use_menu = False
    if args.mode == "menu":
        use_menu = True
    elif args.mode is None and not (args.home and args.away):
        # double-click / no args => menu
        use_menu = True

    if use_menu:
        m = ascii_menu(valid_teams_sorted)
        if m.get("quit"):
            print("\nBye.")
            return
        mode = m["mode"]
        home = m["home"]
        away = m["away"]
        season = m.get("season")
        week = m.get("week")
        asof_season = m.get("asof_season")
        asof_week = m.get("asof_week")
    else:
        mode = args.mode or "upcoming"
        home = args.home
        away = args.away
        season = args.season
        week = args.week
        asof_season = args.asof_season
        asof_week = args.asof_week

    # Execute
    print("\n--- PREDICTION ---")
    if mode == "exact":
        if not (home and away and season and week):
            print("[ERROR] EXACT mode requires --home --away --season --week")
            return
        predict_exact_game(model_cal, team_feats, ROLL_N, home, away, season, week)

    elif mode == "upcoming":
        if not (home and away):
            print("[ERROR] UPCOMING mode requires --home --away")
            return
        predict_upcoming_matchup(model_cal, team_feats, ROLL_N, home, away, asof_season, asof_week)

    else:
        print("[ERROR] Unknown mode. Use --mode exact | upcoming | menu")
        return


if __name__ == "__main__":
    main()
