from collections import namedtuple
from enum import Enum
from typing import List, Union, Optional

WEEK_DAY_MAPPING = [
    'Mon',
    'Tue',
    'Wed',
    'Thu',
    'Fri',
    'Sat',
    'Sun'
]


class WeekDay(str, Enum):
    Mon = 0
    Tue = 1
    Wed = 2
    Thu = 3
    Fri = 4
    Sat = 5
    Sun = 6


MONTH_MAPPING = [
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec'
]


class Month(str, Enum):
    Jan = 0
    Feb = 1
    Mar = 2
    Apr = 3
    May = 4
    Jun = 5
    Jul = 6
    Aug = 7
    Sep = 8
    Oct = 9
    Nov = 10
    Dec = 11


STEP_UNITS_MAPPING = {
    's': 'second_step',
    'S': 'second',
    'm': 'minute',
    'h': 'hour',
    'w': 'week_day',
    'W': 'week',
    'M': 'month',
    'y': 'year',
    't': 'total_steps'
}

STEP_UNITS_MAPPING_INVERSE = {v: k for k, v in STEP_UNITS_MAPPING.items()}


STEP_UNITS_NORMALIZATION_FACTOR = {
    'second_step': 1,
    'second': 59,
    'minute': 59,
    'hour': 23,
    'week_day': 6,
    'week': 3,
    'month': 11,
    'year': 1,
    'total_steps': 1
}


def week_day_to_str(week_day: int) -> str:
    return WEEK_DAY_MAPPING[week_day]


def str_week_day_to_int(week_day: str) -> int:
    return WeekDay(week_day).value


def month_to_str(month: int) -> str:
    return MONTH_MAPPING[month]


def str_month_to_int(month: str) -> int:
    return Month(month).value


class Step(
    namedtuple(
        'Step',
        'second_step second minute hour week_day week month year total_steps',
        defaults=(0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
):

    def to_dict(self):
        if getattr(self, '_asdict', None) is not None:
            return self._asdict()
        else:
            fields = self._fields
            d = {}
            for field in fields:
                d[field] = getattr(self, field)
            return d

    def __dict__(self):
        return self.to_dict()

    def __eq__(self, other):
        return self.second_step == other.second_step \
               and self.second == other.second \
               and self.minute == other.minute \
               and self.hour == other.hour \
               and self.week_day == other.week_day \
               and self.week == other.week \
               and self.month == other.month \
               and self.year == other.year \
               and self.total_steps == other.total_steps

    def equal_no_total_steps(self, other):
        return self.second_step == other.second_step \
               and self.second == other.second \
               and self.minute == other.minute \
               and self.hour == other.hour \
               and self.week_day == other.week_day \
               and self.week == other.week \
               and self.month == other.month \
               and self.year == other.year

    def full_str(self, filled=True):
        if filled:
            return 'second_step={}-second={}-minute={}-hour={}-week_day={}-week={}-month={}-year={}-total_steps={}' \
                .format(self.second_step, self.second, self.minute, self.hour, WEEK_DAY_MAPPING[self.week_day],
                        self.week, MONTH_MAPPING[self.month], self.year, self.total_steps)
        else:
            return 'second_step={}-second={}-minute={}-hour={}-week_day={}-week={}-month={}-year={}-total_steps={}' \
                .format(self.second_step, self.second, self.minute, self.hour,
                        self.week_day, self.week, self.month, self.year, self.total_steps)

    def to_second(self) -> int:
        _, second, minute, hour, week_day, week, month, year, total_steps = self
        total = second
        total += minute * 60
        total += hour * 60 * 60
        total += week_day * 24 * 60 * 60
        total += week * 7 * 24 * 60 * 60
        total += month * 4 * 7 * 24 * 60 * 60
        total += year * 12 * 4 * 7 * 24 * 60 * 60
        return total

    def to_days(self) -> int:
        _, _, _, _, week_day, week, month, year, total_steps = self
        total = week_day
        total += week * 7
        total += month * 7 * 4
        total += year * 7 * 4 * 12
        return total

    def __str__(self):
        return self.to_str(full_key=False)

    def __repr__(self):
        return str(self)

    def to_str(self, full_key=False, no_total_step: bool = False):
        value = ''
        for key, val in self.to_dict().items():
            if no_total_step:
                if key != 'total_steps':
                    if full_key:
                        value += f'{val}{STEP_UNITS_MAPPING_INVERSE[key]} '
                    elif val > 0:
                        value += f'{val}{STEP_UNITS_MAPPING_INVERSE[key]} '
            else:
                if full_key:
                    value += f'{val}{STEP_UNITS_MAPPING_INVERSE[key]} '
                elif val > 0:
                    value += f'{val}{STEP_UNITS_MAPPING_INVERSE[key]} '
        return value[0: -1]

    def is_weekend(self):
        return self.week_day > 4

    def weeks_difference(self, other: 'Step') -> int:
        return (self.total_steps - other.total_steps) // (60*60*24*7)

    def hours_difference(self, other: 'Step') -> int:
        return (self.total_steps - other.total_steps) // (60*60)

    def to_array(self, normalized: bool = False, units_to_skip: Optional[List[str]] = None) -> List[Union[int, float]]:
        values: List[Union[int, float]] = []
        skip_units: List[str] = units_to_skip if units_to_skip is not None else []
        dict_step = self.to_dict()
        for unit, value in dict_step.items():
            if unit not in skip_units:
                if normalized:
                    values.append(value / STEP_UNITS_NORMALIZATION_FACTOR[unit])
                else:
                    values.append(value)
        return values

    @classmethod
    def from_str(cls, string_value: str, step_per_second=1, split_value=' '):
        parts = string_value.split(split_value)
        kwargs = {}
        for part in parts:
            for key, mapping in STEP_UNITS_MAPPING.items():
                if key in part:
                    value = int(part.replace(key, ''))
                    kwargs[mapping] = value
        if 'total_steps' not in kwargs:
            return populate_step_total_steps(cls(**kwargs), step_per_second)
        else:
            return cls(**kwargs)


def populate_step_total_steps(old_step, step_per_second):
    if old_step is not None:
        second_step, second, minute, hour, week_day, week, month, year, total_steps = old_step
        if second_step + second + minute + hour + week_day + week + month + year != 0 and total_steps == 0:
            total = second_step
            total += second * step_per_second
            total += minute * 60 * step_per_second
            total += hour * 60 * 60 * step_per_second
            total += week_day * 24 * 60 * 60 * step_per_second
            total += week * 7 * 24 * 60 * 60 * step_per_second
            total += month * 4 * 7 * 24 * 60 * 60 * step_per_second
            total += year * 12 * 4 * 7 * 24 * 60 * 60 * step_per_second
            total_steps = total
        return Step(second_step, second, minute, hour, week_day, week, month, year, total_steps)
    else:
        return None
