# Changelog

Each `## <version>` section is published as the notes of that version's GitHub
Release (see the `Windows` workflow): write it before tagging `v<version>`.

## 0.2.0

### What's new

- **Zoom and scroll the timeline:** the mouse wheel zooms around the pointer; middle-drag, Shift+wheel or a horizontal wheel scrolls; Z (View › Show the whole song) zooms back out. While playing, the view follows the playhead until you scroll away from it.
- **Bars on the timeline:** numbered bars under the time ruler, and bar lines on the waveforms (beat lines too when zoomed in). Click a bar number to jump to the start of that bar; drag across the bar numbers to loop whole bars.
- **Clicking outside the loop clears it** and jumps to where you clicked.

### Fixes

- Rendered mixes, saved projects and cached stems were created readable only by their owner (`-rw-------`); they now get the usual default permissions.

## 0.1.0

First release.
