"""Progress bars for the long steps (run_s3, compare_maps, extract_triples, enrich_citations).

Every long function takes `progress`:

    None (default)   a progress bar: a widget in Jupyter and Colab, a text bar in a terminal
    False            nothing
    a function       progress(stage, fraction), called with a short stage text and 0..1
"""
from __future__ import annotations

import logging
import sys
from typing import Callable, Optional, Union

ProgressCallback = Callable[[str, float], None]
log = logging.getLogger("s3bib")


class Bar:
    """One bar from 0 to 100%; the text next to it is the current stage."""

    def __init__(self, label: str = "S3"):
        self.label = label
        self.bar = None
        self.last = ""

    def __call__(self, stage: str, fraction: float) -> None:
        log.info("%s (%.0f%%)", stage, 100 * fraction)
        try:
            from tqdm.auto import tqdm
        except ImportError:  # plain lines, one per stage
            key = stage.split("(")[0].strip()
            if key != self.last:
                print(f"{100 * fraction:3.0f}%  {stage}", file=sys.stderr, flush=True)
                self.last = key
            return
        if self.bar is None:
            self.bar = tqdm(total=100, desc=self.label, leave=True, dynamic_ncols=True,
                            bar_format="{desc} {percentage:3.0f}%|{bar}| {elapsed}")
        n = int(round(100 * max(0.0, min(1.0, float(fraction)))))
        if n > self.bar.n:
            self.bar.update(n - self.bar.n)
        self.bar.set_description_str(f"{self.label} \u00b7 {str(stage)[:48]:<48}", refresh=True)
        if fraction >= 1:
            self.close()

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None


def resolve(progress: Union[None, bool, ProgressCallback], label: str = "S3") -> ProgressCallback:
    """The callback to use for a `progress` argument (see the module docstring)."""
    if progress is False:
        return lambda s, p: log.info("%s (%.0f%%)", s, 100 * p)
    if callable(progress):
        return progress
    return Bar(label)
