"""Projects: a folder with the tracks as FLAC files plus ``project.json``.

    my-song/
      project.json
      tracks/vocals.flac, tracks/drums.flac, ...

Tracks are stored at original pitch and tempo; the transposition and the
speed are settings that are re-applied on load. The FLAC files are ordinary audio files, usable on their own.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .audio_io import AudioLoadError, load_audio_scaled, write_flac
from .fsutil import staged_dir, write_text_atomic

PROJECT_FILE = "project.json"
TRACKS_DIR = "tracks"
FORMAT = "backing-track-generator-project"
VERSION = 1


class ProjectError(Exception):
    pass


@dataclass
class TrackSettings:
    name: str
    mute: bool = False
    solo: bool = False
    gain: float = 1.0


@dataclass
class ProjectSettings:
    """Everything in project.json except the per-file details."""

    sample_rate: int
    tracks: list[TrackSettings]
    source_name: str = ""  # original audio file name, for display
    source_path: str | None = None  # where it was, if known (for re-separation)
    separated: bool = False
    model: str | None = None
    transpose: int = 0
    speed: int = 100  # percent of the original tempo
    loop: tuple[float, float] | None = None
    loop_enabled: bool = False
    position: float = 0.0
    # Beat grid as detected plus the user's corrections:
    # {"beats": [...], "downbeats": [...], "corrections": {...}}; None = not detected yet.
    beats: dict | None = None
    metronome: dict | None = None  # {"mute": bool, "solo": bool, "gain": float}
    extra: dict = field(default_factory=dict)  # unknown keys, preserved on re-save


def project_file(path: str | Path) -> Path:
    """Accept a project folder or its project.json; return the project.json path."""
    path = Path(path)
    return path / PROJECT_FILE if path.is_dir() else path


def is_project(path: str | Path) -> bool:
    p = project_file(path)
    return p.name == PROJECT_FILE and p.is_file()


def _safe_filename(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^\w.-]+", "_", name).strip("._") or "track"
    candidate, i = base, 2
    while candidate.lower() in taken:
        candidate, i = f"{base}_{i}", i + 1
    taken.add(candidate.lower())
    return f"{candidate}.flac"


def _settings_json(settings: ProjectSettings, files: list[tuple[str, float]]) -> str:
    doc = {
        **settings.extra,
        "format": FORMAT,
        "version": VERSION,
        "source": {"name": settings.source_name, "path": settings.source_path},
        "separated": settings.separated,
        "model": settings.model,
        "sample_rate": settings.sample_rate,
        "transpose": settings.transpose,
        "speed": settings.speed,
        "loop": None if settings.loop is None else {
            "start": settings.loop[0], "end": settings.loop[1], "enabled": settings.loop_enabled},
        "position": settings.position,
        "beats": settings.beats,
        "metronome": settings.metronome,
        "tracks": [
            {**asdict(t), "file": f"{TRACKS_DIR}/{fname}", "file_gain": file_gain}
            for t, (fname, file_gain) in zip(settings.tracks, files)
        ],
    }
    return json.dumps(doc, indent=2) + "\n"


def save_project(directory: str | Path, settings: ProjectSettings, datas: list[np.ndarray] | None) -> Path:
    """Save to ``directory`` (created or replaced).

    With ``datas`` the whole project is (re)written atomically. With
    ``datas=None`` only the settings are updated and the existing track files
    are kept; the track names must match those already saved.
    """
    directory = Path(directory)
    if datas is None:
        existing = _read_json(directory / PROJECT_FILE)
        files = [(Path(t["file"]).name, float(t.get("file_gain", 1.0))) for t in existing["tracks"]]
        if [t["name"] for t in existing["tracks"]] != [t.name for t in settings.tracks]:
            raise ProjectError("Saved tracks differ from the current ones; the tracks must be re-saved.")
        write_text_atomic(directory / PROJECT_FILE, _settings_json(settings, files))
        return directory

    if len(datas) != len(settings.tracks):
        raise ValueError("save_project: one array per track is required")
    if directory.exists() and not is_project(directory) and any(directory.iterdir()):
        raise ProjectError(f"{directory} exists and is not a project folder; choose an empty or new folder.")
    taken: set[str] = set()
    names = [_safe_filename(t.name, taken) for t in settings.tracks]
    with staged_dir(directory) as tmp:
        (tmp / TRACKS_DIR).mkdir()
        with ThreadPoolExecutor(max_workers=max(len(datas), 1)) as pool:
            gains = list(pool.map(lambda nd: write_flac(tmp / TRACKS_DIR / nd[0], nd[1], settings.sample_rate),
                                  zip(names, datas)))
        (tmp / PROJECT_FILE).write_text(_settings_json(settings, list(zip(names, gains))), encoding="utf-8")
    return directory


def _read_json(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ProjectError(f"No project found at {path}") from None
    except (OSError, ValueError) as e:
        raise ProjectError(f"Cannot read {path}: {e}") from e
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise ProjectError(f"{path} is not a backing track project.")
    if doc.get("version", 0) > VERSION:
        raise ProjectError(f"{path} was saved by a newer version of the app (format {doc['version']}).")
    return doc


_KNOWN_KEYS = {"format", "version", "source", "separated", "model", "sample_rate", "transpose", "speed", "loop",
               "position", "tracks", "beats", "metronome"}


def track_files(path: str | Path) -> tuple[list[str], list[float]]:
    """Paths and FLAC gains of a saved project's tracks (for beat detection)."""
    pfile = project_file(path)
    doc = _read_json(pfile)
    entries = doc.get("tracks") or []
    return ([str(pfile.parent / t["file"]) for t in entries], [float(t.get("file_gain", 1.0)) for t in entries])


def load_project(path: str | Path) -> tuple[ProjectSettings, list[np.ndarray], Path]:
    """Load a project (folder or project.json). Returns settings, the
    original-pitch track audio, and the project folder."""
    pfile = project_file(path)
    doc = _read_json(pfile)
    directory = pfile.parent
    try:
        sr = int(doc["sample_rate"])
        entries = doc["tracks"]
        tracks = [TrackSettings(name=str(t["name"]), mute=bool(t.get("mute", False)),
                                solo=bool(t.get("solo", False)), gain=float(t.get("gain", 1.0)))
                  for t in entries]
        loop = doc.get("loop")
        source = doc.get("source") or {}
        settings = ProjectSettings(
            sample_rate=sr,
            tracks=tracks,
            source_name=str(source.get("name") or ""),
            source_path=source.get("path"),
            separated=bool(doc.get("separated", False)),
            model=doc.get("model"),
            transpose=int(doc.get("transpose", 0)),
            speed=int(doc.get("speed", 100)),
            loop=None if not loop else (float(loop["start"]), float(loop["end"])),
            loop_enabled=bool(loop and loop.get("enabled", False)),
            position=float(doc.get("position", 0.0)),
            beats=doc.get("beats") if isinstance(doc.get("beats"), dict) else None,
            metronome=doc.get("metronome") if isinstance(doc.get("metronome"), dict) else None,
            extra={k: v for k, v in doc.items() if k not in _KNOWN_KEYS},
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ProjectError(f"{pfile} is malformed: {e}") from e
    if not tracks:
        raise ProjectError(f"{pfile} has no tracks.")

    def load(entry: dict) -> np.ndarray:
        f = directory / entry["file"]
        try:
            data, file_sr = load_audio_scaled(f, float(entry.get("file_gain", 1.0)))
        except AudioLoadError as e:
            raise ProjectError(f"Missing or unreadable track file: {f}") from e
        if file_sr != sr:
            raise ProjectError(f"{f} is {file_sr} Hz, project is {sr} Hz.")
        return data

    with ThreadPoolExecutor(max_workers=len(entries)) as pool:
        datas = list(pool.map(load, entries))
    return settings, datas, directory
