# image-to-model

Turn a single photograph into a textured, watertight 3D model — built for getting
props into **Roblox Studio**.

```bash
pip install -e .
image-to-model photo.jpg -o out/
```

That writes an OBJ, a GLB, a baked 1024×1024 PNG texture, an import guide and a
Luau spawn snippet — with the mesh already under Roblox's triangle limit, in
studs, Y-up, and UV-mapped.

```
Target        : Roblox (Roblox Studio MeshPart, imported through the 3D Importer.)
Backend       : depth
Triangles     : 10,000  (budget 10,000, limit 21,000)
Watertight    : True
Size          : 2.54 x 4.00 x 2.23 studs
Texture       : 1024x1024
Target checks : all passed
```

---

## Why the Roblox bits matter

Roblox rejects or mangles meshes that ignore its constraints, so the pipeline
treats them as hard requirements rather than suggestions:

| Constraint | What the tool does |
|---|---|
| 21,000 triangles per single import (10,000 for batch imports and avatar meshes) | Decimates to a budget of 10,000 by default, and refuses to call a mesh valid above the hard limit |
| Per-vertex colour is **dropped on import** | Always bakes a real UV atlas and PNG for Roblox targets — vertex colour alone would import grey |
| Textures capped at 1024×1024 | Bakes at the cap and clamps any larger request |
| Y-up, measured in studs | Exports Y-up and scales the model to 4 studs by default |
| The 3D Importer reads OBJ, FBX and glTF | Writes OBJ (with MTL + PNG) and GLB |

Every finished model is checked against the target profile, and the exit code is
non-zero if anything would block the import.

## Quickstart

```bash
pip install -e .                      # numpy + pillow only
pip install -e '.[ai]'                # adds learned depth (torch + transformers)
pip install -e '.[matting]'           # adds rembg for cluttered backgrounds
```

```bash
image-to-model info                             # what's available in your install
image-to-model demo --sample rocket -o out/     # no photo needed
image-to-model photo.jpg -o out/ --preview --viewer

image-to-model photo.jpg -o chair.glb --target game
image-to-model logo.png  -o sign.obj  --preset relief
image-to-model photo.jpg -o part.stl  --target print
```

`--viewer` writes a single self-contained HTML file that spins the model in a
browser — worth a look before spending time in Studio.

## How it works

```
photo ─► segment ─► depth ─► mesh ─► refine ─► texture ─► export
```

1. **Segment.** Isolate the subject: a real alpha channel if the image has one,
   `rembg` if installed, otherwise a classical background model built from the
   image border. That fallback compares each pixel against the background colour
   of its own row and column, so a gradient backdrop doesn't fool it.
2. **Depth.** Predict relative depth with Depth Anything V2 (or any transformers
   depth model). Without `torch` installed it falls back to *silhouette
   inflation*: push each pixel out in proportion to its distance from the
   outline, which turns a flat shape into a rounded solid. For logos, sprites and
   decals that's often the better choice anyway.
3. **Mesh.** Sample the solid into a scalar field and extract one continuous
   isosurface. Depth is scaled to the subject's own silhouette, so a round
   outline comes back round and a thin one stays thin.
4. **Refine.** Taubin-smooth (which removes stair-stepping without the steady
   shrinking a plain Laplacian causes), drop noise islands, then simplify with
   quadric error metric edge collapse down to the polygon budget.
5. **Texture.** Bake a two-tile atlas — the subject on top, a darkened copy
   underneath for the back — and assign UVs by projecting each vertex back
   through the camera that made it.
6. **Export.** OBJ + MTL + PNG, GLB, PLY or STL, converting axes and units for
   the target.

### Details worth knowing

**Geometry is low-poly, detail lives in the texture.** The mesh is built a few
times over the budget and then decimated, rather than built at the budget. QEM
spends triangles on creases and silhouettes and thins out flat areas, so a
10,000-triangle prop with a 1024² texture reads far better than a uniformly
coarse mesh — which is exactly how Roblox assets are normally authored.

**The solid is extracted from a volume, not stitched from two sheets.** The
obvious construction — a front height field, a back one, and a seam sewing their
boundaries together — carries three problems: a visible join right around the
silhouette, a minimum wall thickness needed to stop front and back vertices
coinciding (which would make that join non-manifold), and a hard front/back
split of every vertex that shows up as a ragged line in the texture. Sampling

    f(x, y, z) = H(x, y) - |z - M(x, y)|

on a grid and extracting where it crosses zero has none of them: the surface
wraps around the silhouette because that is simply where the field changes sign.
`--mesher sheets` keeps the old construction, which is faster.

**Extraction handles cells the surface crosses twice.** Plain Surface Nets puts
one vertex in each crossed cell, so a cell the surface passes through twice gets
a single vertex standing in for two sheets and every edge into it carries four
faces. That is not exotic — 88 of the 256 corner patterns have their inside
corners split into more than one group — so each cell emits one vertex *per
group* and each quad takes the one its own corner belongs to. Checked against a
sphere (Euler 2, volume within 0.5%), a torus (Euler 0, so the handle is real)
and two separate blobs in one grid.

**Depth follows the silhouette, not the bounding box.** How deep the model gets
is set by the subject's *inradius* — the radius of the largest disc that fits
inside its outline. Inflating a disc of radius R by that much gives a hemisphere
of height R, so front and back together span 2R: the disc's own diameter, i.e. a
sphere. The obvious alternative, making depth a fixed fraction of the bounding
box, gives every subject the same thickness regardless of shape — a ball comes
out 45% too flat and a thin rocket 71% too fat. That single choice is the
difference between a recognisable prop and a misshapen blob. `--relief-mode
fraction` restores the old behaviour when you want a specific depth.

**Inflation uses a true hemisphere profile.** For a disc the distance transform
is `d = R - r`, so a hemisphere's height `sqrt(R² - r²)` becomes
`R·sqrt(1 - (1 - d/R)²)`. The tempting shortcut `(d/R)^0.5` runs up to 17% low
through the mid-radius and inflates discs into pointed bicones instead of domes.

**Lifting is orthographic.** Perspective divergence shrinks a point's lateral
offset as it comes towards the camera, so a surface bulging forward also narrows
— turning a reconstructed sphere into a teardrop. That narrowing is only correct
if the relief is true metric depth; monocular depth is relative and the relief is
a prior, so the silhouette is the widest cross-section and lateral position
should not move with height. `--projection perspective` is there for when a
metric depth model makes it meaningful.

**The silhouette is rounded, not cornered.** The relief tapers to the outline
along a quarter-circle, so the surface arrives at the silhouette tangent to the
view direction — which is what a silhouette on a smooth object actually is. The
obvious alternative, a smoothstep, has zero slope there: it leaves the outline
flat and lets the side wall drop away at a right angle, which reads as a hard
pointed lens rather than a solid. `--rim-profile` exposes both.

**Depth is filtered with an edge-preserving bilateral, not a blur.** Depth
models are noisy per-pixel and that noise becomes surface roughness, but a plain
Gaussian is indiscriminate: it smooths across depth discontinuities just as
happily as along a flat face, rounding off the very edges that carry the shape.
Filtering guided by the photograph's luminance stops smoothing where the image
shows an edge. On a synthetic step edge it removes as much noise as the
Gaussian while keeping the step intact.

**UV seams split vertices, so "watertight" is judged after welding.** Giving the
front and back tiles their own texture coordinates means duplicating the
vertices along the join, or else triangles spanning both tiles interpolate their
V coordinate through the gap between them and smear a band across the silhouette.
That duplication breaks index-level watertightness while leaving the solid
geometrically closed, so closure is checked once coincident vertices are merged.
Every textured asset has seams; this is the check that matches what "is it a
solid" actually means.

## Getting better results

Roughly in order of how much difference they make.

**1. Install the learned extras.** This is the single biggest jump and it is one
command. Without `torch` the pipeline falls back to inflating the silhouette,
which invents plausible roundness rather than measuring real depth:

```bash
pip install -e '.[ai,matting]'
```

`ai` swaps in Depth Anything V2 for actual monocular depth; `matting` adds
`rembg`, which handles backgrounds the classical border model cannot. Run
`image-to-model info` to confirm both are active.

**2. Feed it a better photo.** More than any flag:

- subject centred, fully in frame, filling a good part of it;
- plain, evenly lit background — or a PNG that already has an alpha channel;
- diffuse light, no hard shadows crossing the silhouette;
- shoot slightly off-axis rather than dead-on, so there is depth variation to find.

**3. Spend the triangles you are allowed.** The default budget of 10,000 is the
batch-import-safe number. A prop imported on its own may use the full 21,000:

```bash
image-to-model photo.jpg -o out/ --preset roblox-detail
```

That also raises the working resolution to 768, which sharpens both the
silhouette and the baked texture.

**4. Tune the shape.** `--relief-scale` controls how far the front bulges
(raise it for rounded objects, lower it for flat ones); `--thickness` sets how
deep the back goes, down to `0` for a flat-backed relief. If the subject is not
being found, `--debug-dir out/debug` writes the mask and depth map so you can
see what the pipeline actually saw.

**5. For genuine 360° geometry, change the backend.** No amount of tuning gets
past the ceiling below — a feed-forward image-to-3D model is the real answer.
See [Custom backends](#custom-backends).

## What this can and cannot do

A single photograph contains no information about the back of an object. This
produces a **rounded relief closed into a solid** — the front is reconstructed
from predicted depth, the back is a mirrored, darkened estimate. That is the
honest ceiling for single-view reconstruction, and it is a good fit for props,
signage, decals and background scenery.

It is *not* a full 360° reconstruction. Expect:

- a visible seam ridge where the front and back surfaces meet;
- an invented back surface, not a measured one;
- poor results on subjects with deep concavities, or on cluttered backgrounds
  without `rembg` installed;
- flat art (logos, sprites) to work well, since there is little true depth to
  miss.

For genuine 360° geometry you want a feed-forward image-to-3D model — see
[Custom backends](#custom-backends).

## Python API

```python
from image_to_model import reconstruct

result = reconstruct("photo.jpg", target="roblox")
print(result.summary())
print(result.validation.ok, result.mesh.n_faces)

result.save("out/")                        # target's preferred formats
result.save("out/thing.glb")               # one specific file
result.save_debug("out/debug")             # mask, depth map, texture
```

`result.mesh` is a plain NumPy `Mesh` with `vertices`, `faces`, `uvs`,
`vertex_colors` and `vertex_normals`, plus geometry helpers:

```python
mesh.is_watertight()          # every edge shared by exactly two faces
mesh.euler_characteristic()   # 2 for a closed genus-0 surface
mesh.volume()                 # signed, so the sign reveals inverted winding
mesh.stats()
```

## Targets

| Target | Budget | Hard limit | Texture | Units | Up |
|---|---|---|---|---|---|
| `roblox` | 10,000 | 21,000 | 1024 | studs | Y |
| `roblox-avatar` | 4,000 | 10,000 | 1024 | studs | Y |
| `game` | 50,000 | — | 2048 | metres | Y |
| `print` | 300,000 | — | 4096 | millimetres | Z |
| `generic` | 200,000 | — | 4096 | metres | Y |

Presets: `fast`, `balanced`, `detailed`, `relief`, `roblox-prop`,
`roblox-detail` (full 21k budget), `roblox-accessory`.

## Useful options

```
--target-faces N        triangle budget (0 = target default, negative = no decimation)
--size-units N          longest axis, in the target's units
--mesher NAME           volume (seamless, default) | sheets (faster)
--volume-resolution N   grid cells across the subject; 0 derives from the budget
--relief-mode NAME      inradius (follows the outline, default) | fraction
--relief-scale N        relief multiplier; 1.0 is the correct inflation
--thickness N           back depth relative to the front; 0 gives a flat back
--projection NAME       orthographic (default) | perspective
--min-thickness N       thinnest silhouette edge, as a fraction of width
--open-back             leave the model open instead of closing it
--working-resolution N  image size used for depth and texture (default 512)
--depth-model NAME      auto | heuristic | depth-anything | dpt | any HF model id
--segmentation NAME     auto | alpha | rembg | border | none
--rim-profile NAME      fillet (rounded, default) | smoothstep | linear
--depth-filter NAME     bilateral (edge-preserving, default) | gaussian | none
--preview / --viewer    turntable PNG / self-contained WebGL page
--debug-dir DIR         dump the mask, depth map and baked texture
```

## Custom backends

The reconstruction step is pluggable. Register a backend to swap in a
feed-forward image-to-3D model (TripoSR, Stable Fast 3D, Hunyuan3D) and keep the
segmentation, budgeting, texture baking, export and Roblox validation around it:

```python
from image_to_model.backends import ReconstructionBackend, BackendOutput, register_backend

class TripoSRBackend(ReconstructionBackend):
    name = "triposr"

    def reconstruct(self, subject, config):
        mesh = my_model(subject.image, subject.mask)   # -> image_to_model.types.Mesh
        return BackendOutput(mesh=mesh)

register_backend(TripoSRBackend)
```

Return the mesh Y-up with +Z towards the camera and unscaled; the pipeline
handles units, axes and validation. Then use it with `--backend triposr`.

## Development

```bash
pip install -e '.[dev]'
pytest -q
ruff check src tests
```

The test suite covers mesh invariants, all four exporters (the GLB is parsed
back and its accessors walked, since an invalid glTF still looks like a
plausible file), segmentation accuracy against known-area shapes, decimation
topology preservation, and the CLI end to end.

## License

MIT — see [LICENSE](LICENSE).
