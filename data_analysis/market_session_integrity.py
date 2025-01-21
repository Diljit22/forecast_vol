import logging
import datetime as dt
from typing import Dict, Tuple
from script_generated_files.market_hours import ACTIVE_SESSIONS

logger = logging.getLogger(__name__)
ONE_MIN_MSEC = 60_000


def analyze_sessions(active_sessions: Dict[str, Tuple[int, int]]):
    """
    Analyzes the ACTIVE_SESSIONS dict for:
      1) Dates with a start time != 9:30 AM
      2) Dates with an end time != 1:00 PM or 4:00 PM
      3) Dates with a start time == 1:00 PM
      4) Any missing weekdays between min and max dates
      5) Dates that are Saturday or Sunday
    Returns a dict with these lists:
      {
        "not_930am_start": [...],
        "not_1p_or_4p_end": [...],
        "start_is_1pm": [...],
        "missing_weekdays": [...],
        "weekend_dates": [...]
      }
    """

    # milliseconds -> datetime
    def ms_to_dt(msec: int) -> dt.datetime:
        return dt.datetime.fromtimestamp(msec / 1000)

    not_930am_start = []
    not_1p_or_4p_end = []
    start_is_1pm = []
    weekend_dates = []

    date_to_session = {}
    for date_str, (start_ms, end_ms) in active_sessions.items():
        # parse the "YYYY-MM-DD" key into a date object
        date_obj = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        date_to_session[date_obj] = (start_ms, end_ms)

    all_dates_sorted = sorted(date_to_session.keys())
    if not all_dates_sorted:
        return {
            "not_930am_start": [],
            "not_1p_or_4p_end": [],
            "end_is_1pm": [],
            "missing_weekdays": [],
            "weekend_dates": [],
        }

    min_date = all_dates_sorted[0]
    max_date = all_dates_sorted[-1]

    for date_obj in all_dates_sorted:
        start_ms, end_ms = date_to_session[date_obj]

        start_dt = ms_to_dt(start_ms)
        end_dt = ms_to_dt(end_ms)

        if not (start_dt.hour == 9 and start_dt.minute == 30):
            not_930am_start.append(date_obj.strftime("%Y-%m-%d"))

        if not (
            (end_dt.hour == 13 and end_dt.minute == 0)
            or (end_dt.hour == 16 and end_dt.minute == 0)
        ):
            not_1p_or_4p_end.append(date_obj.strftime("%Y-%m-%d"))

        if end_dt.hour == 13 and end_dt.minute == 0:
            start_is_1pm.append(date_obj.strftime("%Y-%m-%d"))

        if date_obj.weekday() in [5, 6]:  # 5=Sat, 6=Sun
            weekend_dates.append(date_obj.strftime("%Y-%m-%d"))

    missing_weekdays = []
    current_date = min_date
    while current_date <= max_date:
        # If it's Mon-Fri (weekday() in 0..4) and not in dictionary => missing
        if current_date.weekday() < 5 and current_date not in date_to_session:
            missing_weekdays.append(current_date.strftime("%Y-%m-%d"))
        current_date += dt.timedelta(days=1)

    return {
        "not_930am_start": not_930am_start,
        "not_1p_or_4p_end": not_1p_or_4p_end,
        "end_is_1pm": start_is_1pm,
        "missing_weekdays": missing_weekdays,
        "weekend_dates": weekend_dates,
    }


def show_session():
    results = analyze_sessions(ACTIVE_SESSIONS)

    print("Dates that start != 9:30 AM:")
    print(results["not_930am_start"])
    print("Dates that end != 1:00 PM or 4:00 PM:")
    print(results["not_1p_or_4p_end"])
    print("Dates that start == 1:00 PM:")
    print(results["end_is_1pm"])
    print("Missing weekdays between min & max date:")
    print(results["missing_weekdays"])
    print("Dates that are Sat or Sun in the dict:")
    print(results["weekend_dates"])
