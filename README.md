These are a collection of vibe-coded projects for a history of science course concerning AI. Very cool course :)

---

# Turing Morphogenesis / Gray-Scott Visualizer

An interactive real-time simulator of Gray-Scott reaction-diffusion dynamics — the mathematical model behind Turing patterns such as animal-coat spots, skin stripes, and cellular mitosis. The application renders the activator field `u` live using matplotlib's `inferno` colormap, with sliders, presets, and pause/reset controls for interactive parameter exploration. Built in pure NumPy with no per-cell Python loops.

## Install & Run

```bash
pip install -r requirements.txt
python morphogenesis.py
```

> On macOS you may need `pip install pyobjc-framework-Cocoa` if TkAgg is unavailable; alternatively set `matplotlib.use("MacOSX")` at the top of `morphogenesis.py`.

## Parameter Presets

| Preset  | F     | k     | Pattern                          |
|---------|-------|-------|----------------------------------|
| Spots   | 0.035 | 0.065 | Isolated circular spots          |
| Stripes | 0.060 | 0.058 | Labyrinthine stripe networks     |
| Holes   | 0.039 | 0.058 | Inverted spots (holes in medium) |
| Mitosis | 0.028 | 0.053 | Self-replicating spot division   |
| Chaos   | 0.026 | 0.051 | Unstable, chaotic wave fronts    |

## Turing Instability

A Turing instability (diffusion-driven instability) arises in a two-component reaction-diffusion system when a uniform steady state that is stable in the absence of diffusion becomes unstable once diffusion is added. The key condition is asymmetric diffusivity: the inhibitor (v) must diffuse significantly faster than the activator (u), so that local self-activation of u outpaces its own suppression at short range while long-range inhibition quenches growth elsewhere. In the Gray-Scott model, feed rate F controls how quickly the substrate u is replenished, and kill rate k controls how quickly the product v is removed; together they select which spatial wavelength of perturbation grows fastest, determining whether the resulting steady pattern takes the form of spots, stripes, holes, or more complex transient structures.
