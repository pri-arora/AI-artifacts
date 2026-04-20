from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

matplotlib.use("TkAgg")

AMBER = "#EF9F27"
CYAN  = "#00E5FF"
DU = 0.2097
DV = 0.105
GRID_SIZE = 256
DT = 1.0
STEPS_PER_FRAME = 20
DX = 1.0

# (F, k, v_seed_half) — v_seed_half controls the initial seed square half-width.
# Mitosis uses a tiny 6×6 seed so the single spot visibly divides rather than
# immediately flooding the grid with parallel spots like the Spots preset does.
PRESETS: dict[str, tuple[float, float, int]] = {
    "Spots":   (0.035, 0.065, 10),
    "Stripes": (0.040, 0.060, 10),  # was (0.060, 0.058) — wrong zone; 0.040/0.060 = labyrinthine
    "Holes":   (0.039, 0.058, 10),
    "Mitosis": (0.028, 0.053,  3),  # small seed → single spot that self-replicates
    "Chaos":   (0.026, 0.051, 10),
}

F_RANGE = (0.010, 0.080)
K_RANGE = (0.040, 0.070)


def _validate_f(f: float) -> None:
    if not (F_RANGE[0] <= f <= F_RANGE[1]):
        raise ValueError(f"F={f:.4f} out of range [{F_RANGE[0]}, {F_RANGE[1]}]")


def _validate_k(k: float) -> None:
    if not (K_RANGE[0] <= k <= K_RANGE[1]):
        raise ValueError(f"k={k:.4f} out of range [{K_RANGE[0]}, {K_RANGE[1]}]")


def _make_initial_grid(half: int = 10) -> tuple[np.ndarray, np.ndarray]:
    u = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    v = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    rng = np.random.default_rng(42)
    cx, cy = GRID_SIZE // 2, GRID_SIZE // 2
    s = half * 2
    sl = (slice(cx - half, cx + half), slice(cy - half, cy + half))
    u[sl] = 0.50 + rng.uniform(-0.05, 0.05, (s, s))
    v[sl] = 0.25 + rng.uniform(-0.05, 0.05, (s, s))
    return u, v


@dataclass
class GrayScottSimulation:
    f: float
    k: float
    seed_half: int = 10
    u: np.ndarray = field(init=False)
    v: np.ndarray = field(init=False)
    frame: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        _validate_f(self.f)
        _validate_k(self.k)
        stability_limit = DX ** 2 / (4.0 * max(DU, DV))
        assert DT < stability_limit, (
            f"dt={DT} violates stability: must be < {stability_limit:.4f}"
        )
        self.reset()

    def reset(self) -> None:
        self.u, self.v = _make_initial_grid(self.seed_half)
        self.frame = 0

    def set_f(self, f: float) -> None:
        _validate_f(f)
        self.f = f

    def set_k(self, k: float) -> None:
        _validate_k(k)
        self.k = k

    def _laplacian(self, a: np.ndarray) -> np.ndarray:
        return (
            np.roll(a,  1, axis=0) + np.roll(a, -1, axis=0)
            + np.roll(a,  1, axis=1) + np.roll(a, -1, axis=1)
            - 4.0 * a
        )

    def step(self) -> None:
        for _ in range(STEPS_PER_FRAME):
            u, v = self.u, self.v
            uvv  = u * v * v
            self.u = np.clip(u + DT * (DU * self._laplacian(u) - uvv + self.f * (1.0 - u)), 0.0, 1.0)
            self.v = np.clip(v + DT * (DV * self._laplacian(v) + uvv - (self.f + self.k) * v), 0.0, 1.0)
        self.frame += 1

    @property
    def elapsed_time(self) -> float:
        return self.frame * DT * STEPS_PER_FRAME


class MorphogenesisApp:
    def __init__(self) -> None:
        plt.style.use("dark_background")
        f0, k0, h0 = PRESETS["Spots"]
        self.sim = GrayScottSimulation(f=f0, k=k0, seed_half=h0)
        self.paused = False

        self.fig = plt.figure(figsize=(9, 7), facecolor="black",
                               num="Turing Morphogenesis / Gray-Scott")
        self.fig.patch.set_facecolor("black")
        self.fig.suptitle("Turing Morphogenesis / Gray-Scott",
                          color=AMBER, fontsize=13, y=0.98)

        outer = gridspec.GridSpec(
            2, 1, figure=self.fig,
            height_ratios=[3, 1], hspace=0.08,
            left=0.04, right=0.97, top=0.94, bottom=0.03,
        )

        # Single composite panel: u → red, v → blue, mixed → magenta
        self.ax_sim: plt.Axes = self.fig.add_subplot(outer[0])
        self.ax_sim.set_xticks([]); self.ax_sim.set_yticks([])
        subtitle = "red = activator u   |   blue = inhibitor v   |   magenta = overlap"
        self.ax_sim.set_title(subtitle, color="#aaaaaa", fontsize=9, pad=4)

        self.im_composite = self.ax_sim.imshow(
            self._make_rgb(),
            interpolation="bilinear", origin="lower",
        )

        # Controls area
        ctrl_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], hspace=0.6)

        slider_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=ctrl_gs[0], hspace=1.2)
        ax_f = self.fig.add_subplot(slider_gs[0])
        ax_k = self.fig.add_subplot(slider_gs[1])
        for ax in (ax_f, ax_k):
            ax.set_facecolor("#1a1a1a")

        self.slider_f = Slider(ax_f, "F", F_RANGE[0], F_RANGE[1],
                                valinit=self.sim.f, valstep=0.001, color=AMBER)
        self.slider_k = Slider(ax_k, "k", K_RANGE[0], K_RANGE[1],
                                valinit=self.sim.k, valstep=0.001, color=AMBER)
        for sl in (self.slider_f, self.slider_k):
            sl.label.set_color("white")
            sl.valtext.set_color(AMBER)
            sl.valtext.set_fontfamily("monospace")

        btn_gs = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=ctrl_gs[1], wspace=0.3)
        preset_names = list(PRESETS.keys())
        self.preset_btns: list[Button] = []
        for i, name in enumerate(preset_names):
            ax_btn = self.fig.add_subplot(btn_gs[i])
            btn = Button(ax_btn, name, color="#2a2a2a", hovercolor=AMBER)
            btn.label.set_color(AMBER); btn.label.set_fontsize(8)
            self.preset_btns.append(btn)

        ax_reset = self.fig.add_subplot(btn_gs[5])
        ax_pause = self.fig.add_subplot(btn_gs[6])
        self.btn_reset = Button(ax_reset, "Reset", color="#2a2a2a", hovercolor=AMBER)
        self.btn_pause = Button(ax_pause, "Pause", color="#2a2a2a", hovercolor=AMBER)
        for b in (self.btn_reset, self.btn_pause):
            b.label.set_color(AMBER); b.label.set_fontsize(8)

        ax_stats = self.fig.add_subplot(ctrl_gs[2])
        ax_stats.set_facecolor("black"); ax_stats.set_xticks([]); ax_stats.set_yticks([])
        for spine in ax_stats.spines.values():
            spine.set_visible(False)
        self.stats_text = ax_stats.text(
            0.5, 0.5, self._stats_str(),
            ha="center", va="center",
            color=AMBER, fontfamily="monospace", fontsize=9,
            transform=ax_stats.transAxes,
        )

        # Wire up events
        self.slider_f.on_changed(self._on_f_changed)
        self.slider_k.on_changed(self._on_k_changed)
        for i, btn in enumerate(self.preset_btns):
            name = preset_names[i]
            btn.on_clicked(lambda _, n=name: self._on_preset(n))
        self.btn_reset.on_clicked(lambda _: self._on_reset())
        self.btn_pause.on_clicked(lambda _: self._on_pause())

        self.anim = FuncAnimation(
            self.fig, self._update,
            interval=int(1000 / 30),
            blit=False, cache_frame_data=False,
        )

    def _make_rgb(self) -> np.ndarray:
        r = np.clip(self.sim.u, 0.0, 1.0)
        b = np.clip(self.sim.v / 0.5, 0.0, 1.0)  # v rarely exceeds 0.5, stretch to full range
        g = np.zeros_like(r)
        return np.stack([r, g, b], axis=-1)

    def _stats_str(self) -> str:
        u, v = self.sim.u, self.sim.v
        return (
            f"Frame: {self.sim.frame:6d}   "
            f"Sim time: {self.sim.elapsed_time:8.1f}   "
            f"u  mean/max: {u.mean():.2f} / {u.max():.2f}   "
            f"v  mean/max: {v.mean():.2f} / {v.max():.2f}"
        )

    def _update(self, _frame: int) -> None:
        if not self.paused:
            self.sim.step()
        self.im_composite.set_data(self._make_rgb())
        self.stats_text.set_text(self._stats_str())

    def _on_f_changed(self, val: float) -> None:
        try:
            self.sim.set_f(round(val, 3))
            self.sim.reset()
        except ValueError:
            pass

    def _on_k_changed(self, val: float) -> None:
        try:
            self.sim.set_k(round(val, 3))
            self.sim.reset()
        except ValueError:
            pass

    def _on_preset(self, name: str) -> None:
        f, k, h = PRESETS[name]
        self.sim.set_f(f); self.sim.set_k(k); self.sim.seed_half = h
        self.sim.reset()
        self.slider_f.set_val(f)
        self.slider_k.set_val(k)

    def _on_reset(self) -> None:
        self.sim.reset()

    def _on_pause(self) -> None:
        self.paused = not self.paused
        self.btn_pause.label.set_text("Resume" if self.paused else "Pause")

    def run(self) -> None:
        plt.show()


def main() -> None:
    app = MorphogenesisApp()
    app.run()


if __name__ == "__main__":
    main()
