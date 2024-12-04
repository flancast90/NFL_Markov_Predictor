from typing import List, Dict


def get_historical_mov(team: str, date: int, games: List[Dict]) -> float:
    """
    Calculate historical margin of victory for a team up to a given date.

    Args:
        team: The name of the selected team being analyzed.
        date: The date before which games will be considered.
        games: A list of all games in consideration.
    Returns:
        float: The mean margin of victory.
    """
    historical_games = []

    for game in games:
        if game["date"] >= date:
            continue

        if game["home_team"] == team:
            mov = game["home_score"] - game["away_score"]
            historical_games.append(mov)
        elif game["away_team"] == team:
            mov = game["away_score"] - game["home_score"]
            historical_games.append(mov)

    if not historical_games:
        return 0.0

    return sum(historical_games) / len(historical_games)


def get_game_observation(home_mov: float, away_mov: float) -> str:
    """Convert historical MOVs to an observation category."""
    mov_diff = home_mov - away_mov

    if mov_diff > 7:
        return "strong_history"
    elif mov_diff > 0:
        return "positive_history"
    elif mov_diff >= -3:
        return "neutral_history"
    elif mov_diff >= -7:
        return "negative_history"
    else:
        return "weak_history"


def get_game_result(home_score: int, away_score: int) -> str:
    """Determine if the game was a home win or away win."""
    return "home_win" if home_score > away_score else "away_win"
