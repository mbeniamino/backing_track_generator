"""Source separation with Demucs, plus an on-disk cache of separated stems.

This module has no Qt dependency. ``separate()`` runs in-process (it is what the
worker process calls); ``run_worker()`` is the entry point of that process.
Heavy imports (torch, demucs) happen lazily so the GUI starts fast.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .audio_io import load_audio

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
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "backing-track-generator" / "stems"


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


def write_stems(stem_dir: Path, stems: list[Stem], sample_rate: int, source: str, model: str) -> None:
    """Write stems as 24-bit FLAC plus a manifest, atomically (temp dir + rename).

    FLAC is integer-based, so a stem that peaks above 1.0 is scaled down and the
    inverse gain is stored in the manifest to restore it exactly on load.
    """
    stem_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=stem_dir.parent))
    try:
        entries = []
        for stem in stems:
            peak = float(np.abs(stem.data).max(initial=0.0))
            gain = peak / 0.999 if peak > 0.999 else 1.0
            fname = f"{stem.name}.flac"
            sf.write(tmp / fname, stem.data / gain, sample_rate, subtype="PCM_24")
            entries.append({"name": stem.name, "file": fname, "gain": gain})
        manifest = {
            "version": MANIFEST_VERSION,
            "model": model,
            "source": source,
            "sample_rate": sample_rate,
            "stems": entries,
        }
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2))
        if stem_dir.exists():
            shutil.rmtree(stem_dir)
        tmp.rename(stem_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def load_stems(stem_dir: Path) -> tuple[list[Stem], int]:
    """Load cached stems, sorted in display order."""
    manifest = json.loads((stem_dir / MANIFEST).read_text())
    stems = []
    sr = manifest["sample_rate"]
    for entry in manifest["stems"]:
        data, file_sr = load_audio(stem_dir / entry["file"])
        if file_sr != sr:
            raise SeparationError(f"Sample rate mismatch in {entry['file']}")
        if entry.get("gain", 1.0) != 1.0:
            data *= np.float32(entry["gain"])
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

    report(None, "Loading model (downloaded on first use)…")
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    try:
        model = get_model(model_name)
    except Exception as e:  # network errors, unknown model, ...
        raise SeparationError(f"Could not load model {model_name!r}: {e}") from e
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

    report(1.0, "Writing stems…")
    stems = [Stem(name, np.ascontiguousarray(out[i].numpy().T, dtype=np.float32))
             for i, name in enumerate(model.sources)]
    write_stems(stem_dir, stems, sr, source=Path(input_path).name, model=model_name)
    return stem_dir


# ----- worker process --------------------------------------------------------

def run_worker(input_path: str, stem_dir: str, model_name: str, queue) -> None:
    """Entry point of the separation subprocess. Posts messages to ``queue``:
    ("progress", fraction|None, message), ("done", stem_dir), ("error", message)."""
    try:
        separate(input_path, Path(stem_dir), model_name,
                 progress=lambda f, m: queue.put(("progress", f, m)))
        queue.put(("done", stem_dir))
    except Exception as e:  # report everything to the GUI
        queue.put(("error", f"{type(e).__name__}: {e}"))
