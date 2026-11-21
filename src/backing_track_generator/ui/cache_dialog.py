"""Tracks > Manage cache: see and delete cached separations and old model versions."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from .. import cache
from ..cache import format_size
from ..separation import DEFAULT_MODEL, cache_root
from . import theme
from .settings import CACHE_LIMIT_CHOICES, AppSettings

_PATH_ROLE = Qt.ItemDataRole.UserRole


class CacheDialog(QDialog):
    def __init__(self, parent=None, current_dir: Path | None = None, separation_running: bool = False,
                 settings: AppSettings | None = None):
        """``current_dir``: stem folder of the song that is open (marked in the
        list). ``separation_running``: temporary folders may be in use, so they
        aren't offered for deletion."""
        super().__init__(parent)
        self.setWindowTitle("Manage cache")
        self.resize(760, 520)
        self.current_dir = current_dir
        self.separation_running = separation_running
        self.settings = settings or AppSettings()
        self.freed = 0

        dim = f"color: {theme.TEXT_DIM.name()};"
        title = QLabel("Separated songs")
        title.setStyleSheet("font-size: 13pt; font-weight: 700;")
        explain = QLabel("Separating a song takes a while, so its tracks are kept here and reopening it is "
                         "instant. Deleting only frees space: the song is separated again next time it is "
                         "opened. Saved projects have their own copy of the tracks and are not affected.")
        explain.setWordWrap(True)
        explain.setStyleSheet(dim)
        self.summary = QLabel()

        self.limit = QComboBox()
        current = self.settings.cache_limit
        choices = list(CACHE_LIMIT_CHOICES)
        if current not in choices:  # a value set by hand in settings.ini
            choices.append(current)
            choices.sort(key=lambda v: -1 if v is None else v)
        for value in choices:
            self.limit.addItem("No limit" if value is None else format_size(value), value)
        self.limit.setCurrentIndex(choices.index(current))
        self.limit.currentIndexChanged.connect(self._on_limit_changed)
        limit_hint = QLabel("When a new song is separated and the cache is over the limit, the songs "
                            "not used for the longest time are removed (never the open song).")
        limit_hint.setWordWrap(True)
        limit_hint.setStyleSheet(dim)
        limit_row = QHBoxLayout()
        limit_row.addWidget(QLabel("Size limit:"))
        limit_row.addWidget(self.limit)
        limit_row.addWidget(limit_hint, 1)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Song", "Model", "Size", "Last used"])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSortingEnabled(False)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setStyleSheet(
            f"QTreeWidget {{ background: {theme.LANE_BG.name()}; alternate-background-color: {theme.PANEL.name()};"
            f" border: 1px solid {theme.BORDER.name()}; border-radius: 6px; }}"
            f"QTreeWidget::item {{ padding: 5px 2px; }}"
            f"QTreeWidget::item:selected {{ background: {theme.rgba(theme.ACCENT, 90)}; }}"
            f"QHeaderView::section {{ background: {theme.PANEL.name()}; color: {theme.TEXT_DIM.name()};"
            f" border: none; border-bottom: 1px solid {theme.BORDER.name()}; padding: 4px 6px; }}")
        self.tree.itemSelectionChanged.connect(self._update_buttons)

        self.delete_selected = QPushButton("Delete selected")
        self.delete_selected.clicked.connect(self._on_delete_selected)
        self.delete_all = QPushButton("Delete all")
        self.delete_all.clicked.connect(self._on_delete_all)
        open_folder = QPushButton("Open folder")
        open_folder.clicked.connect(self._open_folder)
        row = QHBoxLayout()
        row.addWidget(self.delete_selected)
        row.addWidget(self.delete_all)
        row.addStretch()
        row.addWidget(open_folder)

        model_title = QLabel("Separation model")
        model_title.setStyleSheet("font-size: 13pt; font-weight: 700; margin-top: 10px;")
        self.model_label = QLabel()
        self.model_label.setWordWrap(True)
        self.remove_old_models = QPushButton("Remove older versions")
        self.remove_old_models.clicked.connect(self._on_remove_old_models)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_label, 1)
        model_row.addWidget(self.remove_old_models)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        close = buttons.button(QDialogButtonBox.StandardButton.Close)
        close.setIcon(QIcon())
        # Enter must never delete anything: only Close may be the default button.
        for b in (self.delete_selected, self.delete_all, open_folder, self.remove_old_models):
            b.setAutoDefault(False)
            b.setDefault(False)
        close.setDefault(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        for w in (title, explain, self.summary):
            layout.addWidget(w)
        layout.addLayout(limit_row)
        layout.addWidget(self.tree)
        layout.addLayout(row)
        layout.addWidget(model_title)
        layout.addLayout(model_row)
        layout.addWidget(buttons)
        self.refresh()

    # ----- content ---------------------------------------------------------------

    def refresh(self) -> None:
        self.songs = cache.list_cached_songs()
        self.leftovers = [] if self.separation_running else cache.leftovers()
        self.tree.clear()
        for song in self.songs:
            is_open = self.current_dir is not None and song.path == self.current_dir
            item = QTreeWidgetItem([
                song.source + ("   (open now)" if is_open else ""),
                song.model,
                format_size(song.size),
                song.last_used.strftime("%Y-%m-%d %H:%M"),
            ])
            item.setData(0, _PATH_ROLE, str(song.path))
            item.setTextAlignment(2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            item.setToolTip(0, str(song.path))
            self.tree.addTopLevelItem(item)
        total = sum(s.size for s in self.songs)
        extra = sum(cache.dir_size(p) for p in self.leftovers)
        text = f"{len(self.songs)} song{'s' if len(self.songs) != 1 else ''}, {format_size(total)}"
        if self.leftovers:
            n = len(self.leftovers)
            text += (f"  ·  plus {n} folder{'s' if n != 1 else ''} ({format_size(extra)}) "
                     "left over from interrupted separations")
        self.summary.setText(text)

        try:
            m = cache.model_cache_info(DEFAULT_MODEL)
        except Exception as e:  # unreadable HF cache: show, don't fail
            m = None
            self.model_label.setText(f"Could not read the model cache: {e}")
        if m is not None:
            if m.current_revision is None and not m.old_revisions:
                self.model_label.setText(f"{DEFAULT_MODEL}: not downloaded yet.")
            else:
                text = (f"{DEFAULT_MODEL}: current version {(m.current_revision or '?')[:8]} "
                        f"({format_size(m.current_size)})")
                if m.old_revisions:
                    n = len(m.old_revisions)
                    text += f"; {n} older version{'s' if n != 1 else ''} ({format_size(m.old_size)})"
                else:
                    text += "; no older versions"
                self.model_label.setText(text)
        self.remove_old_models.setEnabled(bool(m and m.old_revisions))
        self._update_buttons()

    def _update_buttons(self) -> None:
        self.delete_selected.setEnabled(bool(self.tree.selectedItems()))
        self.delete_all.setEnabled(bool(self.songs or self.leftovers))

    # ----- actions ---------------------------------------------------------------

    def _confirm(self, text: str) -> bool:
        return QMessageBox.question(self, "Delete from cache", text) == QMessageBox.StandardButton.Yes

    def _on_delete_selected(self) -> None:
        paths = [Path(i.data(0, _PATH_ROLE)) for i in self.tree.selectedItems()]
        size = sum(s.size for s in self.songs if s.path in paths)
        if paths and self._confirm(f"Delete the cached tracks of {len(paths)} song(s) ({format_size(size)})?"):
            self.freed += cache.delete_paths(paths)
            self.refresh()

    def _on_delete_all(self) -> None:
        paths = [s.path for s in self.songs] + self.leftovers
        size = sum(s.size for s in self.songs) + sum(cache.dir_size(p) for p in self.leftovers)
        if paths and self._confirm(f"Delete all cached separations ({format_size(size)})?"):
            self.freed += cache.delete_paths(paths)
            self.refresh()

    def _on_limit_changed(self, _index: int) -> None:
        new = self.limit.currentData()
        keep = {self.current_dir} if self.current_dir is not None else set()
        remove = cache.plan_limit(new, keep)
        if remove:
            names = "\n".join(f"  • {s.source}  ({format_size(s.size)})" for s in remove[:12])
            if len(remove) > 12:
                names += f"\n  … and {len(remove) - 12} more"
            size = format_size(sum(s.size for s in remove))
            ok = QMessageBox.question(
                self, "Cache size limit",
                f"With a limit of {format_size(new)}, {len(remove)} song(s) not used recently "
                f"({size}) will be removed now:\n\n{names}\n\nContinue?")
            if ok != QMessageBox.StandardButton.Yes:
                previous = self.settings.cache_limit
                self.limit.blockSignals(True)
                self.limit.setCurrentIndex(self.limit.findData(previous))
                self.limit.blockSignals(False)
                return
        self.settings.cache_limit = new
        if remove:
            self.freed += cache.delete_paths([s.path for s in remove])
            self.refresh()

    def _on_remove_old_models(self) -> None:
        if self._confirm("Remove the older versions of the separation model? The current one is kept."):
            try:
                self.freed += cache.delete_old_model_revisions(DEFAULT_MODEL)
            except Exception as e:
                QMessageBox.warning(self, "Manage cache", f"Could not remove old model versions: {e}")
            self.refresh()

    def _open_folder(self) -> None:
        root = cache_root()
        root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))
