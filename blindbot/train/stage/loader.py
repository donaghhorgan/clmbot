from __future__ import annotations

import pathlib
from typing import Dict, List

from pydantic.dataclasses import dataclass


def load_documents(path, pattern) -> Dict[str, str]:
    documents = {}
    for file in path.glob(pattern):
        title = file.stem
        with file.open() as fp:
            documents[title] = fp.read()
    return documents


@dataclass(frozen=True)
class LoaderConfig:
    path: pathlib.Path
    pattern: str = "*.txt"


@dataclass(frozen=True)
class Loader:
    path: pathlib.Path
    pattern: str

    def __call__(self) -> List[str]:
        documents = load_documents(self.path, self.pattern)
        return list(documents.values())

    @classmethod
    def from_config(cls, config: LoaderConfig) -> Loader:
        return Loader(path=config.path, pattern=config.pattern)
