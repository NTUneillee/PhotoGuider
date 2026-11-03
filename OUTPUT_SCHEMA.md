# Output schema

Both public commands write UTF-8 `result.json`. All numeric fields are finite JSON
numbers; pixel coordinates and measurements are floating point unless identified as
integers below. Optional values are JSON `null`, not the string `"null"`.

| Field | Type and meaning |
|---|---|
| `input` | string: absolute input image path |
| `mode` | string: `fast` or `full` |
| `aspect` | string or null: original requested CLI ratio, e.g. `"16:9"`; null means free-form. By default the returned `box` has exactly this ratio (within 0.1 %). With `--no-strict-aspect` the box follows the benchmark protocol, which clips each side at the image border; compute its ratio from `box`. |
| `box` | array of four numbers: `[x1,y1,x2,y2]`, in upright (EXIF-applied) input-photo pixels; for panoramas, in `working_view.png` pixels |
| `scores` | object with integer `S`, `C`, `A`, `I` in [0,100]: calibrated Subject, Composition, Aesthetics, Integrity indicators; not probabilities |
| `geometry` | object defined below, computed live from the current subject mask and saliency map |
| `explanation` | string with generated paragraph when `--explain` succeeds; otherwise null. Requested generation failures exit nonzero. |
| `panorama` | null for photos; camera object defined below for panoramas |
| `versions` | object: `code` is the git commit string or `"unknown"`; `adapter_sha` is the adapter file's lowercase 64-character SHA-256 string if installed, otherwise null |

Pixel coordinates have origin at the upper left, x increasing right and y increasing
down. Boxes use left/top inclusive and right/bottom exclusive bounds, with
`0 <= x1 < x2 <= W` and `0 <= y1 < y2 <= H`. Public photo boxes are reported to
one decimal place after mapping from a working copy (long side at most 2048 pixels)
to the original input. Raster crops truncate the unrounded mapped coordinates with
`int()` before PIL cropping; the reported rounded box can therefore differ slightly
from the raster boundary. Inputs carrying an EXIF orientation tag are first turned upright,
so `box` and `crop.jpg` are in the orientation a photo viewer shows; the stored pixel grid of a
rotated file is not used.

`geometry.mask_ok` is boolean. False produces exactly `{"mask_ok": false}`.
True adds the following dimensionless floating-point measurements:

| Field | Precise definition |
|---|---|
| `capture` | sum of subject-mask values within the crop / (sum over the entire input mask + 1e-9); fraction of total subject mass retained |
| `lost` | `1 - capture`; fraction of subject mass outside the crop |
| `cx`, `cy` | mask-weighted centroid inside the cropped mask, normalized by `max(crop_mask_width-1,1)` and `max(crop_mask_height-1,1)` respectively; (0,0) is crop upper left and (1,1) lower right |
| `ssr` | mean subject-mask value within the cropped mask; subject-to-crop area fraction for a binary mask, mean coverage for a soft mask |
| `sym_h_diff` | absolute difference between left-half and right-half **saliency** mass, divided by `max(left+right,1e-9)`; 0 means balanced mass and values approach 1 for one-sided mass. This is not a mask overlap or mirror-similarity metric. |

Mask values are in [0,1]. Geometry uses the unrounded box on the internal working
image, scaled to mask resolution: lower indices are truncated, upper indices use
`ceil`, and indices are clipped to the map bounds with at least one pixel per axis.
`mask_ok` requires full-mask mean > 0.005, more than four crop-mask pixels, and
crop-mask mass > 1e-6. For saliency symmetry, the crop uses the same map-index
conversion; with `mid = width // 2`, halves are columns `[0:mid]` and `[mid:2*mid]`
(the last column of an odd-width crop is omitted). Missing or fewer than four
saliency pixels, or zero half-width, returns 0. All these measurements are in [0,1]
up to floating-point roundoff.

For panoramas, `panorama` contains:

| Field | Type, units and coordinate frame |
|---|---|
| `pan` | number, degrees, signed atan2 range [-180,180]; 0 faces the panorama centre column, positive pan moves toward increasing panorama columns |
| `tilt` | number, degrees in [-90,90]; 0 is the equator, positive tilt looks up |
| `fov` | number, degrees: **horizontal** pinhole field of view used for the final native render |
| `width`, `height` | integers: actual `final.png` pixel dimensions |
| `aspect` | number: actual raster `width / height` |
| `hotspots` | array of objects `{pan,tilt,score}`: coarse pan in [0,360), tilt in degrees, and raw uncalibrated composition score (not a 0–100 indicator); the first is the shooting direction |
| `direction` | `"scan"` when the shot is taken from this run's strongest hotspot, `"study"` when `--hdr auto` used the direction the user study shot this panorama from (`--scan-direction` turns this off) |
| `exposure` | object: `{"source": "hdr", "gain": number}` when the working view and `final.png` were rendered from the linear HDR (`--hdr`) with that per-view exposure gain, otherwise `{"source": "panorama"}` |

Camera axes are x right, y up, z forward, with zero roll and rotation
`Ry(pan) @ Rx(-tilt)`. Pan -180 and +180 identify the panorama seam; hotspot pan
270 is equivalent to final pan -90. Final pan/tilt/fov are reported to three decimal
places, hotspot angles to two and hotspot scores to four. Rendering uses the
unrounded camera values, so JSON reconstruction is accurate only to that precision.
The renderer uses focal length `(width/2)/tan(fov/2)` and principal point
`((width-1)/2,(height-1)/2)`; vertical FOV is
`2*atan(tan(horizontal_fov/2)*height/width)`.

The coarse scan renders 1024×1024 square views at horizontal FOV 40°. The working
view is 1600×1067 at horizontal FOV 60°. Final render dimensions have long side
1920; the short side is rounded from the working crop's width/height ratio. The
final camera points along the working crop centre ray; its horizontal FOV is the
angle between the working crop's left and right edge rays at crop mid-height.

Scores, geometry and the explanation are computed on the crop selected inside the 60° working view. `final.png` is the native re-render at camera parameters derived from that crop, preserving its centre ray and horizontal angular width.
