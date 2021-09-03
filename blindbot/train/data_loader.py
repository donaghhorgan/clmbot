from __future__ import annotations

import pathlib
from typing import Dict

from pydantic.dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


def load_documents(path: pathlib.Path, pattern: str) -> Dict[str, str]:
    documents = {}
    for file in path.glob(pattern):
        title = file.stem
        with file.open() as fp:
            documents[title] = fp.read()
    return documents


@dataclass(frozen=True)
class DataLoader:
    path: pathlib.Path
    pattern: str
    sep: str

    def __call__(self) -> str:
        documents = load_documents(self.path, self.pattern)
        logger.info(f"Loaded {len(documents)} documents from {self.path}")
        return self.sep.join(list(documents.values()))
