# Backing Track Generator

A desktop app that turns a song into a backing track: it splits the audio into
instrument/voice tracks, lets you mute, solo, loop and transpose them, and
saves the result as a project or renders it to a single audio file.

- **Separation** with [Demucs](https://github.com/facebookresearch/demucs)
  (`htdemucs_6s`: vocals, guitar, piano, bass, drums, other), running locally
  on the CPU. The model is downloaded on first use.
- **Transposition** by ±12 semitones without changing tempo, with
  [Rubber Band](https://breakfastquay.com/rubberband/) (formant-preserving for vocals).

## Requirements

- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 and the Python dependencies)
- System programs/libraries:
  - `rubberband` command-line tool, for transposition
  - `ffmpeg`, to open formats libsndfile can't read (m4a, aac, …)
  - PortAudio, for playback (usually already present)

  On Arch Linux: `pacman -S rubberband ffmpeg portaudio`

The first `uv sync` downloads PyTorch (CPU-only build, ~1 GB).

## Running

```sh
uv run backing-track-generator                 # empty window
uv run backing-track-generator song.mp3        # open a song (separation starts automatically)
uv run backing-track-generator my-project/     # open a saved project
```

## Using it

| Action | How |
|---|---|
| Open a song / project | Ctrl+O / Ctrl+Shift+O, or drop it on the window |
| Separate into tracks | automatic on open; Tracks → Separate (Ctrl+E) if cancelled |
| Play / pause, stop | Space, Home (Stop returns to the loop start while looping) |
| Jump | click on a waveform or the ruler (drag on a waveform to scrub) |
| Mute / solo a track | M / S buttons; Ctrl+click S solos only that track; Ctrl+R resets |
| Track volume | slider in the track header (−∞ to +6 dB); double-click resets to 0 dB |
| Set a loop | drag on the ruler; drag its edges to adjust; `[` / `]` set start / end at the playhead (if the playhead is outside the new loop, it jumps to the loop start) |
| Toggle the loop | L |
| Transpose | toolbar − / + buttons, `+` / `-` keys, `0` to reset |
| Save project | Ctrl+S / Ctrl+Shift+S |
| Render to one file | File → Render mix (Ctrl+Shift+E) or Render loop region; WAV, FLAC, MP3 or Ogg |

While looping, Play always starts inside the loop.

Rendering contains exactly what you hear (mute/solo, volume and transposition). If the
mix would clip, it is lowered as a whole instead (the status bar says by how much).

## Files

- **Projects** are folders: `project.json` (settings) plus `tracks/*.flac`
  (24-bit, original pitch; usable on their own).
- **Separated stems** are cached in `~/.cache/backing-track-generator/stems/`
  (about 100–150 MB per song), so reopening a song is instant. Tracks → Manage
  cache lists them (size, last used) and deletes the ones you no longer need.
  The cache has a size limit (5 GB by default, changeable there, or "No limit"):
  after each separation, the songs not used for the longest time are removed
  until it fits; the open song is never removed.
- **Settings** are stored in `~/.config/backing-track-generator/settings.ini`.
- **The separation model** lives in the Hugging Face cache (`~/.cache/huggingface/`).
  Tracks → Check for separation model updates downloads a newer version on request;
  Manage cache removes the older versions.

## Development

```sh
uv run pytest                 # fast tests (headless)
RUN_SLOW=1 uv run pytest      # also runs Demucs end to end
```

See `PLAN.md` for the design and milestones.
