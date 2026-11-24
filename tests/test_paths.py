from pathlib import Path

from backing_track_generator import fsutil, paths


def test_xdg_variables_win_everywhere(monkeypatch, tmp_path):
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "c"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "k"))
    assert paths.config_dir() == tmp_path / "c" / paths.APP_DIR
    assert paths.cache_dir() == tmp_path / "k" / paths.APP_DIR


def test_windows_uses_appdata(monkeypatch, tmp_path):
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    assert paths.config_dir() == tmp_path / "Roaming" / paths.APP_DIR
    assert paths.cache_dir() == tmp_path / "Local" / paths.APP_DIR


def test_linux_defaults_and_ignores_appdata(monkeypatch, tmp_path):
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert paths.config_dir() == Path.home() / ".config" / paths.APP_DIR
    assert paths.cache_dir() == Path.home() / ".cache" / paths.APP_DIR


def test_replace_retries_transient_lock(monkeypatch, tmp_path):
    src, dst = tmp_path / "a", tmp_path / "b"
    src.write_text("new", encoding="utf-8")
    dst.write_text("old", encoding="utf-8")
    real, calls = fsutil.os.replace, []

    def flaky(a, b):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("locked")
        real(a, b)

    monkeypatch.setattr(fsutil, "_RETRY_DELAYS", (0, 0, 0))
    monkeypatch.setattr(fsutil.os, "replace", flaky)
    monkeypatch.setattr(fsutil.time, "sleep", lambda s: None)
    fsutil.replace(src, dst)
    assert dst.read_text(encoding="utf-8") == "new" and len(calls) == 3
