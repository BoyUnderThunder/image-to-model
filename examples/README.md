# Examples

## Built-in samples

No photograph needed — the package generates its own test subjects:

```bash
python examples/generate_examples.py --out examples/output
```

This reconstructs all three samples, writes the exported models, and saves a
contact sheet showing each source image beside a four-view turntable of its
model.

The samples are chosen to exercise different paths through the pipeline:

| Sample | What it tests |
|---|---|
| `ball` | A lit sphere on a plain backdrop — real shading cues, segmentation by border colour |
| `star` | A flat cut-out with a real alpha channel — segmentation takes the alpha path, and there is no true depth to recover |
| `rocket` | Several parts at different depths on a **gradient** background, which a single global background colour gets badly wrong |

## Your own image

```bash
image-to-model photo.jpg -o out/ --preview --viewer
```

Best results come from a photo where the subject is:

- roughly centred and fully in frame;
- on a plain or simple background (or a PNG that already has an alpha channel);
- evenly lit, without hard shadows falling across the silhouette.

If the background is busy, install `rembg` for learned matting:

```bash
pip install -e '.[matting]'
```

If the subject is not being found, `--debug-dir out/debug` writes the mask and
depth map so you can see what the pipeline actually saw.

## Getting it into Roblox Studio

Every Roblox-target run writes a `ROBLOX_IMPORT.md` next to the model with the
current steps. In short: **Avatar → 3D Importer**, pick the `.obj`, keep the
`_texture.png` beside it so the importer finds it, check the triangle count in
the preview, and import.
