"""Configuration files and run logs for the data and test scripts.

Every script reads its section of a YAML file in configs/ (data.yaml or test.yaml). Top-level values
that are not sections are shared by all sections. Command-line options override the file:

    python scripts/make_hpr_testset.py                          # configs/data.yaml, section hpr
    python scripts/make_hpr_testset.py --gamma 2 --seed 1       # override two values
    python scripts/make_hpr_testset.py --config my_data.yaml    # another file

The resolved configuration and everything the script prints are written, with timestamps, to the
console and to outputs/logs/<script>.log (appended, one block per run).
"""
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs/logs"


def read(name):
    """Raw contents of configs/<name>.yaml."""
    return yaml.safe_load(open(ROOT / "configs" / f"{name}.yaml"))


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() or p.exists() else ROOT / p


class _ToLog:
    """File-like object that sends each printed line to a logger."""

    def __init__(self, logger, level):
        self.logger, self.level, self.buf = logger, level, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def start_log(name):
    """Log to the console and to outputs/logs/<name>.log; print() goes through the logger."""
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.__stdout__)
    file = logging.FileHandler(LOGS / f"{name}.log", mode="a")
    for h in (console, file):
        h.setFormatter(fmt)
        logger.addHandler(h)
    sys.stdout = _ToLog(logger, logging.INFO)

    def excepthook(*exc):
        logger.error("run failed", exc_info=exc)

    sys.excepthook = excepthook
    return logger


def load(ap, config, section):
    """Parse the command line and merge it into `section` of `config` (a path relative to the project).

    Options of `ap` must use default=None; a value given on the command line replaces the one in the file.
    Returns the configuration as a namespace; the run is logged as described above.
    """
    ap.add_argument("--config", default=config, help=f"configuration file (default {config})")
    args = ap.parse_args()
    path = _resolve(args.config)
    full = yaml.safe_load(open(path)) or {}
    cfg = {k: v for k, v in full.items() if not isinstance(v, dict)}   # shared settings
    cfg.update(full.get(section) or {})
    for k, v in vars(args).items():
        if k == "config":
            continue
        if v is not None:
            cfg[k] = v
        elif k not in cfg:
            ap.error(f"'{k}' is missing in section '{section}' of {path}")

    name = Path(sys.argv[0]).stem
    log = start_log(name)
    log.info("=" * 70)
    log.info("%s  (config: %s, section: %s)", name, path.relative_to(ROOT) if path.is_relative_to(ROOT) else path, section)
    for line in yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None, width=110).splitlines():
        log.info("  %s", line)
    log.info("-" * 70)
    return SimpleNamespace(**cfg)

