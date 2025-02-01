import logging
from typing import Optional

from prorl.core.step import Step, populate_step_total_steps


class SimulationEnded(Exception):
    def __init__(self, step: Step = None, *args):
        message = 'Simulation ended'
        if step is not None:
            message += f' at time step: {step}'
        super(SimulationEnded, self).__init__(message, *args)


class TimeStep:

    def __init__(
            self,
            step_per_second=1,
            step_size=1,
            stop_step=-1,
            stop_date: Step = None,
            initial_date: Step = None,
            logger: logging.Logger = None,
            show_log=True,
    ):
        self.step_per_second = step_per_second
        self.step_size = step_size
        self.stop_step = stop_step
        self.stop_date = populate_step_total_steps(stop_date, self.step_per_second)
        self.logger: logging.Logger = logger
        if (stop_step == -1 or stop_step is None) and self.stop_date is not None:
            self.stop_step = self.stop_date.total_steps
        self.initial_date = initial_date
        if self.initial_date is not None:
            self._set_current_step(self.initial_date)
            # self.stop_step += self.current_step.total_steps
        else:
            self.current_step = Step(second_step=0, second=0, minute=0, hour=0,
                                     week_day=0, week=0, month=0, year=0, total_steps=0)
        self.is_last = self._current_is_last_step()
        # if show_log:
        #     self.log(f'Initialized TimeStep at step: {self.current_step}', level=logging.DEBUG)

    def log(self, message: str, level: int = 10, *args, **kwargs):
        if self.logger is not None:
            self.logger.log(level, message, *args, **kwargs)

    @property
    def stop_time_step(self):
        return self.stop_step

    def next(self):
        self._next_step()
        return self.current_step

    def info(self):
        return {
            'step_per_second': self.step_per_second,
            'step_size': self.step_size,
            'stop_step': self.stop_step,
            'stop_date': self.stop_date,
            'initial_date': self.initial_date
        }

    def __str__(self):
        return '<TimeStep {} stop_step={} stop_date={} step_size={} step_per_second={} >' \
            .format(self.current_step, self.stop_step,
                    self.stop_date, self.step_size,
                    self.step_per_second)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self._next_step()
            return self.current_step
        except SimulationEnded:
            raise StopIteration

    def _set_current_step(self, new_step: Step):
        self.current_step = populate_step_total_steps(new_step, self.step_per_second)

    def _current_is_last_step(self):
        if self.stop_step == -1 and self.stop_date is None:
            return False
        if self.stop_step != -1 and self.current_step.total_steps == self.stop_step:
            return True
        return False

    def _next_step_values(self) -> Step:
        second_step, second, minute, hour, week_day, week, month, year, total_steps = self.current_step
        second_step += self.step_size
        total_steps += self.step_size
        if second_step >= self.step_per_second:
            remaining_step, increase_next = increase_step_unit(second_step, self.step_per_second)
            second_step = remaining_step
            second += increase_next
            if second >= 60:
                remaining_step, increase_next = increase_step_unit(second, 60)
                second = remaining_step
                minute += increase_next
                if minute >= 60:
                    remaining_step, increase_next = increase_step_unit(minute, 60)
                    minute = remaining_step
                    hour += increase_next
                    if hour >= 24:
                        remaining_step, increase_next = increase_step_unit(hour, 24)
                        hour = remaining_step
                        week_day += increase_next
                        if week_day >= 7:
                            remaining_step, increase_next = increase_step_unit(week_day, 7)
                            week_day = remaining_step
                            week += increase_next
                            if week >= 4:
                                remaining_step, increase_next = increase_step_unit(week, 4)
                                week = remaining_step
                                month += increase_next
                                if month >= 12:
                                    remaining_step, increase_next = increase_step_unit(month, 12)
                                    month = remaining_step
                                    year += increase_next
        return Step(
            second_step=second_step,
            second=second,
            minute=minute,
            hour=hour,
            week_day=week_day,
            week=week,
            month=month,
            year=year,
            total_steps=total_steps)

    def get_simulated_next_step(self) -> Optional[Step]:
        if self.is_last:
            return None
        else:
            return self._next_step_values()

    def _next_step(self):
        if self.is_last:
            raise SimulationEnded(step=self.current_step)
        else:
            self.current_step = self._next_step_values()
            self.is_last = self._current_is_last_step()


def increase_step_unit(current_value_increased, max_value):
    diff = current_value_increased - max_value
    if diff < 0:
        return 0, 1
    if diff >= max_value:
        increase_next_value = diff // max_value
        remain_value = diff - increase_next_value * max_value
        return remain_value, increase_next_value + 1
    else:
        return diff, 1


time_step_instances = {}


def time_step_factory_get(run_code: str, **config) -> TimeStep:
    global time_step_instances
    if run_code not in time_step_instances:
        time_step = TimeStep(**config)
        time_step_instances[run_code] = time_step
        return time_step
    else:
        return time_step_instances[run_code]


def time_step_factory_reset(run_code: str, recreate=False, new_run_code=None, **config):
    global time_step_instances
    del time_step_instances[run_code]
    if recreate:
        code = run_code if new_run_code is None else new_run_code
        return time_step_factory_get(run_code=code, **config)
