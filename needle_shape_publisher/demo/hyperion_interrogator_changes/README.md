# hyperion_interrogator_changes

Reference implementation files for the `ros2_hyperion_interrogator` repo changes
required by `sim_needle.launch.py` (sim_level=1 shape-driven simulation).

## What goes where

| File here | Destination in ros2_hyperion_interrogator |
|-----------|-------------------------------------------|
| `sim_fbg_publisher.py` | `hyperion_interrogator/sim_fbg_publisher.py` (new) |
| `hyperion_demo.py` | `hyperion_interrogator/hyperion_demo.py` (replace) |
| `hyperion_demo.launch.py` | `launch/hyperion_demo.launch.py` (replace) |

## Steps to apply

```bash
# From the ros2_hyperion_interrogator workspace root:

git checkout -b linear_optim

# Copy files
cp /path/to/ros2_needle_shape_publisher/needle_shape_publisher/demo/hyperion_interrogator_changes/sim_fbg_publisher.py \
   hyperion_interrogator/sim_fbg_publisher.py

cp /path/to/ros2_needle_shape_publisher/needle_shape_publisher/demo/hyperion_interrogator_changes/hyperion_demo.py \
   hyperion_interrogator/hyperion_demo.py

cp /path/to/ros2_needle_shape_publisher/needle_shape_publisher/demo/hyperion_interrogator_changes/hyperion_demo.launch.py \
   launch/hyperion_demo.launch.py

git add hyperion_interrogator/sim_fbg_publisher.py \
        hyperion_interrogator/hyperion_demo.py \
        launch/hyperion_demo.launch.py
git commit -m "feat: sim_level=1 shape-driven FBG wavelength generation"
git push origin linear_optim
```

## What changed and why

### `sim_fbg_publisher.py` (new)

Self-contained utility implementing the shape → curvature → Δλ pipeline:

1. Load a YAML/JSON polyline file.
2. Parametrise by arc length.
3. Compute unit tangents via Savitzky-Golay (or `np.gradient` if scipy
   unavailable).
4. Build a rotation-minimising parallel-transport frame (conceptual baseline:
   `piecewise_exp_test.py` in `rban01/needle_shape_sensing_original`).
5. Compute κx, κy in the material frame.
6. Interpolate at each active-area (AA) sensor location.
7. Apply pseudo-inverse calibration matrices:
   `Δλ_k = pinv(C_k) @ [κx_k, κy_k]`

### `hyperion_demo.py` (modified)

New ROS2 parameters declared in `connect()`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sim.shape_file` | string | `''` | Path to YAML/JSON shape file |
| `sim.insertion_depth` | double | `100.0` | Insertion depth (mm) |

When `sim.shape_file` is non-empty **and** the FBG needle calibration is
loaded, `publish_peaks()` replaces `interrogator.peaks` with shape-derived
wavelengths (`ref + Δλ`) before delegating to the parent implementation.
This means the `sensor/processed` topic publishes exactly Δλ – the shifts
that encode the target curvature – while leaving `sensor/raw` intact.

When `sim.shape_file` is empty the original demo behaviour (base wavelengths
`1540 + 10*i + ch`) is preserved.

### `hyperion_demo.launch.py` (modified)

Two new launch arguments forwarded to the node:

```
sim_shape_file     (default: '')
sim_insertion_depth (default: '100.0')
```

`sim_needle.launch.py` (in `ros2_needle_shape_publisher`) passes these when
`sim_level=1`.

## Testing

After applying the changes:

```bash
# In a sourced ROS2 workspace:
ros2 launch needle_shape_publisher sim_needle.launch.py \
    needleParamFile:=3CH-4AA-0005_needle_params_2022-01-26_Jig-Calibration_best_weights.json

# Override shape file:
ros2 launch needle_shape_publisher sim_needle.launch.py \
    sim_shape_file:=/path/to/custom_shape.yaml \
    sim_insertion_depth:=120.0
```

To verify: `ros2 topic echo /needle/sensor/processed` should show non-zero
wavelength shifts that correspond to the curvature of the loaded shape.

## Unit tests for shape_to_wavelength

The roundtrip test in `ros2_needle_shape_publisher` verifies the
curvature → Δλ → curvature cycle:

```bash
cd ros2_needle_shape_publisher/needle_shape_publisher
PYTHONPATH=. python3 -m pytest test/test_shape_to_wavelength.py -v
```
