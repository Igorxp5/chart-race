import datetime

from dataclasses import dataclass, field


@dataclass
class Entity:
    key: str
    label: str
    image_path: str | None = None


@dataclass
class Entry:
    entity_key: str
    timestamp: int
    _date: datetime.datetime = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._date = datetime.datetime.fromtimestamp(self.timestamp)

    @property
    def date(self):
        return self._date
