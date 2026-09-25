"""Native (AppKit) Settings window for TinkAgent — themed via theme.py.

Every control delegates to TinkAgentApp setter methods, which own logic and
persistence. Must be used on the main thread.
"""
from __future__ import annotations

import os

import objc
from AppKit import (
    NSApplication, NSScrollView, NSBezelBorder, NSMakeRect, NSButton,
    NSControlStateValueOn, NSControlStateValueOff, NSButtonTypeSwitch,
)
from Foundation import NSObject, NSBundle

from . import audio, theme
from .transcribe import BACKEND_ORDER, BACKEND_LABELS

W = 440                  # window width; height computed in _build
PW, PH = 420, 460        # separate app-picker window
ROW_H = 26


_APP_DIRS = (
    "/Applications",
    "/Applications/Utilities",
    "/System/Applications",
    "/System/Applications/Utilities",
    os.path.expanduser("~/Applications"),
)


def _iter_app_paths():
    """Yield .app bundle paths from the standard install locations (top-level only)."""
    for base in _APP_DIRS:
        try:
            entries = os.listdir(base)
        except OSError:
            continue
        for entry in entries:
            if entry.endswith(".app"):
                yield os.path.join(base, entry)


def _installed_apps():
    """(bundle_id, name) for installed .app bundles, name-sorted, deduped by id."""
    out = {}
    for path in _iter_app_paths():
        try:
            bundle = NSBundle.bundleWithPath_(path)
            bid = bundle.bundleIdentifier() if bundle is not None else None
            if not bid:
                continue
            name = os.path.splitext(os.path.basename(path))[0]
            out.setdefault(str(bid), name)
        except Exception:  # noqa: BLE001
            continue
    return sorted(out.items(), key=lambda kv: kv[1].lower())


class SettingsController(NSObject):
    def initWithApp_(self, app):
        self = objc.super(SettingsController, self).init()
        if self is None:
            return None
        self.app = app
        self._build()
        self._build_picker()
        return self

    # --- build ---
    def _build(self):
        H = 556
        win, c = theme.window("TINK Agent", W, H)
        win.setTitle_("TINK Agent")
        self.window = win
        m = 18                       # outer margin
        cw = W - 2 * m               # card width
        y = m

        # General
        gh = theme.section_header("General"); gh.setFrame_(NSMakeRect(m + 13, y, 200, 16))
        c.addSubview_(gh); y += 22
        gcard = theme.card(m, y, cw, 5 * 44 + 2)
        gbody = gcard.contentView()
        r1, self.sw_enabled = theme.switch_row(
            "Enabled", "Fire actions & type text", self.app.engine.enabled,
            self._enabled_cb, cw, 0)
        r2, self.sw_dictation = theme.switch_row(
            "Push-to-talk dictation", "Per-app shortcuts in config.json",
            self.app.config.dictation_enabled, self._dictation_cb, cw, 44)
        r3, self.sw_login = theme.switch_row(
            "Start at login", None, False, self._login_cb, cw, 88)
        r4, self.sw_log = theme.switch_row(
            "Log activity to file", "Transcripts, app, actions",
            self.app.config.log_activity, self._log_cb, cw, 132)
        for r in (r1, r2, r3, r4):
            gbody.addSubview_(r)
        gbody.addSubview_(theme.hairline(15, 44, cw - 15))
        gbody.addSubview_(theme.hairline(15, 88, cw - 15))
        gbody.addSubview_(theme.hairline(15, 132, cw - 15))
        gbody.addSubview_(theme.hairline(15, 176, cw - 15))
        self.btn_edit_actions = theme.accent_button(
            "Edit Button Actions…", self._edit_actions, (cw - 210) / 2.0, 183, 210, 30)
        gbody.addSubview_(self.btn_edit_actions)
        c.addSubview_(gcard); y += 5 * 44 + 2 + 18

        # Input
        ih = theme.section_header("Input"); ih.setFrame_(NSMakeRect(m + 13, y, 200, 16))
        c.addSubview_(ih); y += 22
        icard = theme.card(m, y, cw, 3 * 44 + 2)
        ibody = icard.contentView()
        br, self.popup_backend = theme.popup_row("Transcription", cw, 0)
        self.popup_backend.addItemsWithTitles_(
            [BACKEND_LABELS[k] for k in BACKEND_ORDER])
        self._wire_popup(self.popup_backend, self._backend_cb)
        ibody.addSubview_(br)
        dr, self.popup_device = theme.popup_row("Audio source", cw, 44, trailing_w=74)
        self._wire_popup(self.popup_device, self._device_cb)
        self.btn_refresh = theme.plain_button("Refresh", self._refresh_click,
                                              cw - 15 - 70, 9, 70, 26)
        dr.addSubview_(self.btn_refresh)
        ibody.addSubview_(dr)
        ar, self.popup_apps = theme.popup_row("Allowed apps", cw, 88)
        # Allowed apps is a button, not a real popup: hide the popup, add a button.
        self.popup_apps.setHidden_(True)
        self.btn_apps = theme.plain_button("", self._open_apps, 130, 9, cw - 145, 26)
        ar.addSubview_(self.btn_apps)
        ibody.addSubview_(ar)
        ibody.addSubview_(theme.hairline(15, 44, cw - 15))
        ibody.addSubview_(theme.hairline(15, 88, cw - 15))
        c.addSubview_(icard); y += 3 * 44 + 2 + 14

        self.btn_setup = theme.accent_button("Set Up TINK…", self._setup,
                                             (W - 160) / 2.0, y, 160, 32)
        c.addSubview_(self.btn_setup); y += 44

        c.addSubview_(theme.hairline(m + 4, y, cw - 8)); y += 12
        self.lbl_status = theme.label("", size=11, color=theme.SUBTLE, wrap=True)
        self.lbl_status.setFrame_(NSMakeRect(m + 4, y, 240, 34)); c.addSubview_(self.lbl_status)
        c.addSubview_(theme.link_button("Open config file",
                                        lambda: self.app.open_config(None),
                                        W - m - 130, y, 130))
        c.addSubview_(theme.link_button("Open activity log",
                                        lambda: self.app.open_log(None),
                                        W - m - 130, y + 20, 130))

    def _wire_popup(self, popup, cb):
        t = theme._Target.alloc().initWithCb_(
            lambda: cb(popup.titleOfSelectedItem()))
        popup.setTarget_(t)
        popup.setAction_("fire:")
        theme._retain(popup, t)

    # --- show / refresh ---
    def show(self):
        self.refresh()
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)

    def refresh(self):
        on, off = NSControlStateValueOn, NSControlStateValueOff
        self.sw_enabled.setState_(on if self.app.engine.enabled else off)
        self.sw_dictation.setState_(on if self.app.config.dictation_enabled else off)
        from . import launchagent
        self.sw_login.setState_(on if launchagent.is_enabled() else off)
        self.sw_log.setState_(on if self.app.config.log_activity else off)
        backend = self.app.config.stt_backend or "macwhisper"
        if backend in BACKEND_ORDER:
            self.popup_backend.selectItemAtIndex_(BACKEND_ORDER.index(backend))
        self._refresh_devices()
        n = len(self.app.config.target_apps or [])
        self.btn_apps.setTitle_(f"{n} apps…" if n else "Any app")
        if self.app.config.onboarding_done:
            theme.set_secondary(self.btn_setup, True, "TINK is set up · Re-run")
        else:
            theme.set_secondary(self.btn_setup, False, "Set Up TINK…")
        self.lbl_status.setStringValue_(
            f"Last heard · {self.app._last_heard}\n"
            f"Last action · {self.app._last_action}")

    def _refresh_devices(self):
        try:
            connected = audio.input_device_names()
        except Exception:  # noqa: BLE001
            connected = []
        items, idx = audio.device_menu_items(self.app.config.device_name, connected)
        self.popup_device.removeAllItems()
        if items:
            self.popup_device.addItemsWithTitles_(items)
        if 0 <= idx < len(items):
            self.popup_device.selectItemAtIndex_(idx)

    def _refresh_click(self):
        # Show a brief "Refreshing…" state, then do the (blocking) PortAudio
        # re-scan on the next runloop tick so the label actually paints first.
        theme.set_plain_title(self.btn_refresh, "Refreshing…")
        self.btn_refresh.setEnabled_(False)
        self.performSelector_withObject_afterDelay_("doRescan:", None, 0.05)

    def doRescan_(self, _):
        try:
            self.app.rescan_devices()
            self._refresh_devices()
        finally:
            theme.set_plain_title(self.btn_refresh, "Refresh")
            self.btn_refresh.setEnabled_(True)

    # --- callbacks ---
    def _enabled_cb(self, on): self.app.set_enabled(on)

    def _dictation_cb(self, on): self.app.set_dictation_enabled(on)

    def _login_cb(self, on):
        if not self.app.set_start_at_login(on):
            self.refresh()

    def _log_cb(self, on): self.app.set_log_activity(on)

    def _backend_cb(self, title):
        for k in BACKEND_ORDER:
            if BACKEND_LABELS[k] == title:
                self.app.set_stt_backend(k); return

    def _device_cb(self, title):
        if not title:
            return
        name = audio.device_name_from_title(title)
        if name is not None:
            self.app.set_device_name(name)
            self._refresh_devices()

    def _open_apps(self): self.openApps_(None)

    def _setup(self): self.app.open_onboarding(None)

    def _edit_actions(self): self.app.open_button_actions(None)

    # --- picker ---
    def _build_picker(self):
        """Separate window holding the scrollable installed-apps checklist."""
        win, c = theme.window("Allowed Apps", PW, PH)
        self.picker = win
        intro = theme.label(
            "Only act when one of these apps is frontmost. None checked means "
            "every app is allowed.", size=12.5, color=theme.MUTED, wrap=True)
        intro.setFrame_(NSMakeRect(20, 16, PW - 40, 34)); c.addSubview_(intro)
        self.scroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(20, 60, PW - 40, PH - 60 - 60))
        self.scroll.setHasVerticalScroller_(True)
        self.scroll.setBorderType_(NSBezelBorder)
        self.scroll.setDrawsBackground_(True)
        self.scroll.setBackgroundColor_(theme.CARD)
        self.scroll.setDocumentView_(
            theme.Flipped.alloc().initWithFrame_(NSMakeRect(0, 0, PW - 60, 10)))
        c.addSubview_(self.scroll)
        c.addSubview_(theme.plain_button("Refresh", self._refresh_apps,
                                         20, PH - 46, 100, 30))
        c.addSubview_(theme.accent_button("Done", self._close_picker,
                                          PW - 20 - 100, PH - 46, 100, 30))

    def _refresh_apps(self): self._rebuild_app_list()
    def _close_picker(self): self.picker.orderOut_(None)

    def _rebuild_app_list(self):
        selected = list(self.app.config.target_apps or [])
        apps = _installed_apps()
        installed_ids = {bid for bid, _ in apps}
        for bid in selected:
            if bid not in installed_ids:
                apps.append((bid, f"{bid} (not installed)"))
        apps.sort(key=lambda kv: kv[1].lower())

        width = PW - 60
        height = max(int(self.scroll.contentSize().height), len(apps) * ROW_H + 4)
        doc = theme.Flipped.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
        self._app_targets = []
        for i, (bid, name) in enumerate(apps):
            cb = NSButton.alloc().initWithFrame_(
                NSMakeRect(8, i * ROW_H + 2, width - 16, 22))
            cb.setButtonType_(NSButtonTypeSwitch)
            cb.setAttributedTitle_(theme._attr_title(
                name, theme.font(13, "medium"), theme.GREEN))
            cb.setState_(NSControlStateValueOn if bid in selected
                         else NSControlStateValueOff)
            t = theme._Target.alloc().initWithCb_(self._make_toggle_cb(bid, cb))
            cb.setTarget_(t); cb.setAction_("fire:")
            self._app_targets.append(t)
            doc.addSubview_(cb)
        self.scroll.setDocumentView_(doc)

    def _make_toggle_cb(self, bid, cb):
        def cb_fn():
            self.app.toggle_target(bid, cb.state() == NSControlStateValueOn)
            n = len(self.app.config.target_apps or [])
            self.btn_apps.setTitle_(f"{n} apps…" if n else "Any app")
        return cb_fn

    def openApps_(self, sender):
        self._rebuild_app_list()
        self.picker.center()
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self.picker.makeKeyAndOrderFront_(None)
