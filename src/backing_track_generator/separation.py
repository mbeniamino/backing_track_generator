"""Source separation with Demucs, plus an on-disk cache of separated stems.

This module has no Qt dependency. ``separate()`` runs in-process (it is what the
worker process calls); ``run_worker()`` is the entry point of that process.
Heavy imports (torch, demucs) happen lazily so the GUI starts fast.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_io import load_audio, load_audio_scaled, write_flac
from . import paths
from .fsutil import staged_dir

DEFAULT_MODEL = "htdemucs_6s"  # vocals, drums, bass, guitar, piano, other
MANIFEST = "manifest.json"
MANIFEST_VERSION = 1

# Display order (top to bottom); unknown stems go after these.
STEM_ORDER = ["vocals", "guitar", "piano", "other", "bass", "drums"]

# progress callback: (fraction in [0, 1] or None for indeterminate, message)
ProgressFn = Callable[[float | None, str], None]


class SeparationError(Exception):
    pass


@dataclass
class Stem:
    name: str
    data: np.ndarray  # float32 (frames, 2)


# ----- cache ---------------------------------------------------------------

def cache_root() -> Path:
    return paths.cache_dir() / "stems"


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_dir_for(path: str | Path, model: str = DEFAULT_MODEL, root: Path | None = None) -> Path:
    return (root or cache_root()) / f"{file_digest(path)[:32]}-{model}"


def is_cached(stem_dir: Path) -> bool:
    return (stem_dir / MANIFEST).is_file()


def write_stems(stem_dir: Path, stems: list[Stem], sample_rate: int, source: str, model: str,
                beats=None) -> None:
    """Write stems as 24-bit FLAC plus a manifest (and the beat grid, if
    given), atomically."""
    with staged_dir(stem_dir) as tmp:
        if beats is not None:
            from .beats import BEATS_FILE, write_beats_file

            write_beats_file(tmp / BEATS_FILE, beats)
        entries = []
        for stem in stems:
            fname = f"{stem.name}.flac"
            gain = write_flac(tmp / fname, stem.data, sample_rate)
            entries.append({"name": stem.name, "file": fname, "gain": gain})
        manifest = {
            "version": MANIFEST_VERSION,
            "model": model,
            "source": source,
            "sample_rate": sample_rate,
            "stems": entries,
        }
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_stems(stem_dir: Path) -> tuple[list[Stem], int]:
    """Load cached stems, sorted in display order."""
    manifest = json.loads((stem_dir / MANIFEST).read_text(encoding="utf-8"))
    try:
        os.utime(stem_dir / MANIFEST)  # record "last used" for cache management
    except OSError:
        pass
    stems = []
    sr = manifest["sample_rate"]
    for entry in manifest["stems"]:
        data, file_sr = load_audio_scaled(stem_dir / entry["file"], entry.get("gain", 1.0))
        if file_sr != sr:
            raise SeparationError(f"Sample rate mismatch in {entry['file']}")
        stems.append(Stem(entry["name"], data))
    return sort_stems(stems), sr


def sort_stems(stems: list[Stem]) -> list[Stem]:
    def key(s: Stem) -> tuple[int, str]:
        return (STEM_ORDER.index(s.name) if s.name in STEM_ORDER else len(STEM_ORDER), s.name)

    return sorted(stems, key=key)


# ----- separation ----------------------------------------------------------

def _progress_fraction(d: dict, audio_length: int, segment_length: int, shifts: int) -> float:
    """Overall progress from a demucs ``apply_model`` callback dict.

    The dict carries model/shift indices and the current segment offset, but
    not the audio length or shift count, so those are passed in.
    """
    models = max(d.get("models", 1), 1)
    shifts = max(shifts, 1)
    done = d.get("segment_offset", 0)
    if d.get("state") == "end":
        done += segment_length
    seg = min(done / max(audio_length, 1), 1.0)
    inner = (d.get("shift_idx", 0) + seg) / shifts
    return min((d.get("model_idx_in_bag", 0) + inner) / models, 1.0)


def separate(
    input_path: str | Path,
    stem_dir: Path,
    model_name: str = DEFAULT_MODEL,
    progress: ProgressFn | None = None,
    shifts: int = 1,
) -> Path:
    """Separate ``input_path`` into stems written to ``stem_dir``."""
    report = progress or (lambda f, m: None)
    report(None, "Loading audio…")
    mix, sr = load_audio(input_path)
    mix_sr = sr

    report(None, "Loading model…")
    import torch
    from demucs.apply import apply_model

    model = load_model(model_name, on_download=lambda: report(None, "Downloading model (first use only)…"))
    model.eval()

    wav = torch.from_numpy(mix.T.copy())  # (2, frames)
    if sr != model.samplerate:
        import julius

        wav = julius.resample_frac(wav, sr, model.samplerate)
        sr = model.samplerate

    # Same normalisation as demucs' own CLI.
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std

    segment_s = float(getattr(model, "segment", None) or model.models[0].segment)
    segment_length = int(segment_s * sr)
    # With shifts, demucs pads the input by up to 0.5 s.
    audio_length = wav.shape[-1] + (int(0.5 * sr) if shifts else 0)
    best = 0.0

    def on_progress(d: dict) -> None:
        nonlocal best
        frac = _progress_fraction(d, audio_length, segment_length, shifts)
        if frac > best + 0.005:
            best = frac
            report(frac, "Separating…")

    report(0.0, "Separating…")
    with torch.inference_mode():
        out = apply_model(model, wav[None], shifts=shifts, split=True, overlap=0.25,
                          progress=False, device="cpu", callback=on_progress)[0]
    # Undo the normalisation. The mix's DC offset is shared out between the
    # stems (demucs' CLI adds it to every stem) so that they still sum to the mix.
    out = out * std + mean / len(model.sources)

    stems = [Stem(name, np.ascontiguousarray(out[i].numpy().T, dtype=np.float32))
             for i, name in enumerate(model.sources)]

    # Beats come from the original mix; a failure here must not lose the stems.
    report(None, "Detecting beats…")
    from .beats import BeatError, detect_beats

    try:
        grid = detect_beats(mix, mix_sr)
    except BeatError:
        grid = None
    report(1.0, "Writing stems…")
    write_stems(stem_dir, stems, sr, source=Path(input_path).name, model=model_name, beats=grid)
    return stem_dir


# ----- model loading ---------------------------------------------------------

def load_model(model_name: str, on_download: Callable[[], None] | None = None):
    """Load a Demucs model, from the local Hugging Face cache when possible.

    The first attempt runs in Hugging Face offline mode: no network request at
    all (Demucs would otherwise contact the Hub on every load just to check for
    updates, and the Hub answers anonymous requests with a warning). Only when
    the model isn't cached yet is it downloaded, via Demucs' normal loader.
    """
    from demucs.hf import get_hf_model
    from demucs.pretrained import get_model
    from huggingface_hub import constants as hf_constants
    from huggingface_hub.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # no "unauthenticated requests" warning on download
    previous = hf_constants.HF_HUB_OFFLINE
    hf_constants.HF_HUB_OFFLINE = True  # read at request time by huggingface_hub
    try:
        return get_hf_model(model_name)
    except Exception:  # not cached (LocalEntryNotFoundError), or a broken cache
        pass
    finally:
        hf_constants.HF_HUB_OFFLINE = previous

    if on_download:
        on_download()
    try:
        return get_model(model_name)
    except Exception as e:  # network errors, unknown model, ...
        raise SeparationError(f"Could not load model {model_name!r}: {e}") from e


# ----- worker process --------------------------------------------------------

def run_beats_worker(audio_paths: list[str], gains: list[float], out_file: str, queue) -> None:
    """Subprocess entry point: detect beats on the sum of ``audio_paths``
    (e.g. a song's stems) and write the grid to ``out_file``. Same messages
    as ``run_worker``."""
    try:
        from .audio_io import load_audio_scaled
        from .beats import detect_beats, write_beats_file

        queue.put(("progress", None, "Detecting beats…"))
        mix, sr = None, None
        for path, gain in zip(audio_paths, gains):
            data, file_sr = load_audio_scaled(path, gain)
            if mix is None:
                mix, sr = data.copy(), file_sr
            else:
                n = min(len(mix), len(data))
                mix[:n] += data[:n]
        if mix is None:
            raise SeparationError("No audio to analyse")
        write_beats_file(Path(out_file), detect_beats(mix, sr))
        queue.put(("done", out_file))
    except Exception as e:  # report everything to the GUI
        queue.put(("error", f"{type(e).__name__}: {e}"))


def stem_files(stem_dir: Path) -> tuple[list[str], list[float]]:
    """Paths and gains of a cached separation's stems (for the beats worker)."""
    manifest = json.loads((stem_dir / MANIFEST).read_text(encoding="utf-8"))
    return ([str(stem_dir / e["file"]) for e in manifest["stems"]],
            [float(e.get("gain", 1.0)) for e in manifest["stems"]])


def run_worker(input_path: str, stem_dir: str, model_name: str, queue) -> None:
    """Entry point of the separation subprocess. Posts messages to ``queue``:
    ("progress", fraction|None, message), ("done", stem_dir), ("error", message)."""
    try:
        separate(input_path, Path(stem_dir), model_name,
                 progress=lambda f, m: queue.put(("progress", f, m)))
        queue.put(("done", stem_dir))
    except Exception as e:  # report everything to the GUI
        queue.put(("error", f"{type(e).__name__}: {e}"))
