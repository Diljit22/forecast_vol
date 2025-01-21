import datetime
import logging
import zoneinfo
from typing import Tuple, Union
from src.utils.file_io import save_dict_to_python_file

# Timezone references
NY_TZ = zoneinfo.ZoneInfo("America/New_York")

logger = logging.getLogger(__name__)


def is_active_trading_day(date_to_check: datetime.date, holidays: dict) -> bool:
    """
    Returns True if weekday (Mon=0..Fri=4) and the date is not in the holidays set.
    """
    year_holidays = holidays.get(date_to_check.year, set())
    return (date_to_check.weekday() < 5) and (date_to_check not in year_holidays)


def get_open_close_msec(date_val: datetime.date, half_days: dict) -> Tuple[int, int]:
    """
    For date_val, returns (open_msec, close_msec) in UTC.

    - If the date is in half_days, market closes at 13:00 NYC time.
    - Otherwise, it closes at 16:00 NYC time.
    """
    year_half_days = half_days.get(date_val.year, set())

    close_hour = 13 if date_val in year_half_days else 16

    open_dt = datetime.datetime(
        date_val.year, date_val.month, date_val.day, 9, 30, tzinfo=NY_TZ
    )
    close_dt = datetime.datetime(
        date_val.year, date_val.month, date_val.day, close_hour, 0, tzinfo=NY_TZ
    )

    open_utc = open_dt.astimezone(datetime.timezone.utc)
    close_utc = close_dt.astimezone(datetime.timezone.utc)

    return int(open_utc.timestamp() * 1000), int(close_utc.timestamp() * 1000)


def save_active_market_ranges(
    start_date: Union[datetime.date, str],
    end_date: Union[datetime.date, str],
    holidays: dict,
    half_days: dict,
    out_path: str,
    ) -> dict[str, tuple[int, int]]:
    """
    Generates a dictionary mapping each active trading day in [start_date, end_date]
    to its (open_msec, close_msec) in UTC.
    """
    # Convert strings to date
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

    day_count = (end_date - start_date).days + 1
    dates = [start_date + datetime.timedelta(days=i) for i in range(day_count)]

    active_ranges = {}
    for date_val in dates:
        if is_active_trading_day(date_val, holidays):
            open_msec, close_msec = get_open_close_msec(date_val, half_days)
            active_ranges[str(date_val)] = (open_msec, close_msec)

    logger.info(f"Generated active market ranges for {len(active_ranges)} days.")
    save_dict_to_python_file(active_ranges, out_path, "ACTIVE_SESSIONS")
    return active_ranges
