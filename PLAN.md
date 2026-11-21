# Backing Track Generator — Implementation Plan

See `specs` for the feature list.

## Decisions
- **GUI:** PySide6 (Qt) desktop app, all Python.
- **Separation:** Demucs `htdemucs_6s` (local, free, MIT) → vocals, drums, bass, guitar, piano, other.
  Runs in a worker process with progress reporting; results cached per input file (hash).
  Fallback: `audio-separator` package.
- **Playback:** `sounddevice` (PortAudio) output stream; numpy mixing in the callback.
- **Transpose:** offline, high-quality Rubber Band (`pyrubberband`) per stem, rendered in parallel
  in the background; old audio keeps playing until the new one is ready, then hot-swapped.
  Cached per semitone value (LRU-capped for memory).
- **I/O:** `soundfile` + ffmpeg for decoding mp3/m4a/etc.
- **Environment:** `uv`, Python 3.12 venv (torch/Demucs on 3.14 is risky).

## Architecture
```
app/
  main.py
  separation.py     # Demucs wrapper, worker process, progress → UI
  engine.py         # audio stream, mixer, playhead, loop, mute/solo (thread-safe state)
  transpose.py      # rubberband per stem, cache by semitone
  project.py        # save/load: <name>/stems/*.flac + project.json
  render.py         # mix audible stems → WAV/FLAC
  ui/
    main_window.py  # menus, transport (play/stop, transpose spinbox, loop toggle)
    timeline.py     # shared ruler, playhead, loop region (drag to set A/B)
    track_lane.py   # waveform (precomputed peaks) + M / S buttons
```

### Rules
- Audible tracks: if any track is soloed → soloed tracks only; else → all non-muted tracks.
- Click anywhere on the timeline/lanes seeks; drag on the ruler sets loop A/B.
- Loop wrap handled sample-accurately inside the audio callback.
- Engine consumes a generic list of tracks, so a generated metronome track can be added later.

### project.json (draft)
```json
{
  "version": 1,
  "source_file": "song.mp3",
  "sample_rate": 44100,
  "transpose": 0,
  "loop": {"enabled": false, "start": 0.0, "end": 0.0},
  "tracks": [{"name": "vocals", "file": "stems/vocals.flac", "mute": false, "solo": false, "gain": 1.0}]
}
```

## Milestones
1. ✅ Skeleton: open file, play/stop, click-to-seek, one waveform.
2. ✅ Separation: Demucs in a background process with status-bar progress + cancel; stem cache in ~/.cache/backing-track-generator/stems (keyed by file hash + model).
3. ✅ Multi-track lanes on a shared timeline + Mute/Solo (Ctrl+click = exclusive solo, 10 ms click-free gain ramps, inaudible lanes greyed).
4. ✅ Loop: drag on ruler (edges adjustable), L toggles, [ ] set at playhead; 10 ms equal-power crossfade at wrap and on seek-while-playing.
5. ✅ Transpose ±12 st: Rubber Band R3 (formant-preserving for vocals), parallel per stem, debounced, LRU cache of 2 versions, crossfaded hot-swap.
6. ✅ Projects: folder with project.json + tracks/*.flac (original pitch); Save/Save As/Open, dirty tracking + prompt, settings-only re-save, atomic writes.
7. ✅ Render: audible tracks at current transposition (whole song or loop) to WAV/FLAC/MP3/Ogg; scaled instead of clipped.
8. (Future) Tempo/beat tracking (`beat_this` / `madmom`) → metronome track, count-in intro.

## Risks
- torch/Demucs compatibility → pinned Python 3.12 venv.
- Memory: ~500 MB for 6 stereo stems × 4 min in float32; each cached transposition adds about as much → cap the cache.
- CPU-only separation: ~1–3 min per song on this machine; must be cached and non-blocking.
