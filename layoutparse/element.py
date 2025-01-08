from typing import Dict
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class Element:
    category: str  # table, figure, chart, heading1, header, footer, caption, paragraph, equation, list, index, footnote
    content: str = ""
    html: str = ""
    markdown: str = ""
    base64_encoding: str = None
    image_filename: str = None
    page: int = None
    id: int = None
    coordinates: list[Dict] = None
    entity: str = ""

    def copy(self):
        return deepcopy(self)
