# Backing Track Generator

A desktop app that turns a song into a backing track: it splits the audio into
instrument/voice tracks, lets you mute, solo, loop, transpose them and change their speed, and
saves the result as a project or renders it to a single audio file.

- **Separation** with [Demucs](https://github.com/facebookresearch/demucs)
  (`htdemucs_6s`: vocals, guitar, piano, bass, drums, other), running locally
  on the CPU. The model is downloaded on first use.
- **Transposition** by ±12 semitones without changing tempo, and **speed**
  from 50 % to 150 % without changing pitch, with
  [Rubber Band](https://breakfastquay.com/rubberband/) (formant-preserving for vocals).
- **Metronome and count-in** from beats detected with
  [beat_this](https://github.com/CPJKU/beat_this): the clicks follow the song's
  actual beats, so tempo drift and tempo changes are tracked.

## Requirements

- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 and the Python dependencies)
- System programs/libraries:
  - `rubberband` command-line tool, for transposition and speed changes
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
| Go to the beginning | Ctrl+Home (Play from there counts in when Count-in is on) |
| Jump | click on a waveform or the ruler (drag on a waveform to scrub) |
| Mute / solo a track | M / S buttons; Ctrl+click S solos only that track; Ctrl+R resets |
| Track volume | slider in the track header (−∞ to +6 dB); double-click resets to 0 dB |
| Set a loop | drag on the ruler; drag its edges to adjust; `[` / `]` set start / end at the playhead (if the playhead is outside the new loop, it jumps to the loop start) |
| Toggle the loop | L |
| Transpose | toolbar − / + buttons, `+` / `-` keys, `0` to reset |
| Speed (tempo) | toolbar − / + buttons in 5 % steps, `.` / `,` keys, Ctrl+0 to reset; the ruler, loop and beats stay in song time, and the metronome and count-in follow the new tempo |
| Metronome | the top track, muted when a song is opened: unmute it to hear the click (mute / solo / volume like the others); accented on beat 1 |
| Count-in | metronome button in the toolbar or C: one bar of clicks before playback starts, at the tempo where you start |
| Fix the beats | Beats menu: double / halve tempo (Ctrl+Shift+↑/↓), bar start one beat earlier / later (Ctrl+Shift+←/→), clicks on the off-beat, reset |
| Save project | Ctrl+S / Ctrl+Shift+S |
| Render to one file | File → Render mix (Ctrl+Shift+E) or Render loop region; WAV, FLAC, MP3 or Ogg |

While looping, Play always starts inside the loop.

Rendering contains exactly what you hear (mute/solo, volume, transposition and speed). If the
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
- **Beats** are detected once per song (a few seconds) and stored with the cached
  stems (`beats.json`) and in projects, together with your corrections. The
  beat model is downloaded on first use into `~/.cache/torch/hub/checkpoints/`.
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

## License

MIT — see [LICENSE](LICENSE).
