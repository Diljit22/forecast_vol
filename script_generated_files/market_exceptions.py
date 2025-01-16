import datetime

HOLIDAYS = {
    2023: {
        datetime.date(2023, 1, 2),
        datetime.date(2023, 1, 16),
        datetime.date(2023, 2, 20),
        datetime.date(2023, 4, 7),
        datetime.date(2023, 5, 29),
        datetime.date(2023, 6, 19),
        datetime.date(2023, 7, 4),
        datetime.date(2023, 9, 4),
        datetime.date(2023, 11, 23),
        datetime.date(2023, 12, 25),
    },
    2024: {
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 15),
        datetime.date(2024, 2, 19),
        datetime.date(2024, 3, 29),
        datetime.date(2024, 5, 27),
        datetime.date(2024, 6, 19),
        datetime.date(2024, 7, 4),
        datetime.date(2024, 9, 2),
        datetime.date(2024, 11, 28),
        datetime.date(2024, 12, 25),
    },
}

HALF_DAYS = {
    2023: {
        datetime.date(2023, 7, 3),
        datetime.date(2023, 11, 24),
    },
    2024: {
        datetime.date(2024, 7, 3),
        datetime.date(2024, 11, 29),
        datetime.date(2024, 12, 24),
    },
}
