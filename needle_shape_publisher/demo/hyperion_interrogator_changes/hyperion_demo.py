import rclpy
from rclpy.node import Node

from rcl_interfaces.msg import SetParametersResult

import numpy as np
from collections import namedtuple

from .hyperion_talker import HyperionPublisher
from .sim_fbg_publisher import load_shape_file, shape_to_wavelength_shifts
from socket import gaierror

# interrogator named tuple
DemoHyperionInterrogator = namedtuple( 'Interrogator', [ 'is_ready', 'peaks', 'channel_count' ] )


class HyperionDemo( HyperionPublisher ):
    # PARAMETER NAMES
    param_names = HyperionPublisher.param_names.copy()
    param_names[ 'num_chs' ] = 'demo.num_channels'
    param_names[ 'num_aa' ] = 'demo.num_active_areas'
    # sim_level=1 shape-driven parameters
    param_names[ 'sim_shape_file' ] = 'sim.shape_file'
    param_names[ 'sim_insertion_depth' ] = 'sim.insertion_depth'

    def __init__( self, name='HyperionDemo', num_chs=3, num_aa=4 ):
        self.num_chs = num_chs
        self.num_aa = num_aa

        # Sim-mode state (populated in connect() after parameters are read)
        self._sim_shape_file = ''
        self._sim_insertion_depth = 100.0
        self._sim_shape_points = None   # cached Nx3 array

        super().__init__( name=name )

    # __init__

    def connect( self ) -> bool:
        # ---- FBG channel / AA parameters ----
        self.declare_parameter( HyperionDemo.param_names[ 'num_chs' ], self.num_chs )
        self.declare_parameter( HyperionDemo.param_names[ 'num_aa' ], self.num_aa )
        self.num_chs = self.get_parameter(
            HyperionDemo.param_names[ 'num_chs' ]
        ).get_parameter_value().integer_value
        self.num_aa = self.get_parameter(
            HyperionDemo.param_names[ 'num_aa' ]
        ).get_parameter_value().integer_value

        self.get_logger().info(
            f"Demo Interrogator config: CHs = {self.num_chs} | AAs = {self.num_aa}"
        )

        # ---- Sim-level=1 shape-driven parameters ----
        self.declare_parameter( HyperionDemo.param_names[ 'sim_shape_file' ], '' )
        self.declare_parameter(
            HyperionDemo.param_names[ 'sim_insertion_depth' ], 100.0
        )
        self._sim_shape_file = self.get_parameter(
            HyperionDemo.param_names[ 'sim_shape_file' ]
        ).get_parameter_value().string_value
        self._sim_insertion_depth = self.get_parameter(
            HyperionDemo.param_names[ 'sim_insertion_depth' ]
        ).get_parameter_value().double_value

        # Pre-load shape polyline if a file was specified
        if self._sim_shape_file:
            self._load_sim_shape()

        # ---- Base (reference) wavelengths for demo mode ----
        self.base_wavelengths = { }
        for i in range( 1, self.num_chs + 1 ):
            self.base_wavelengths[ i ] = 1540 + 10 * np.arange( self.num_aa ) + i

        # for

        # Guard: seed ref_wavelengths with base_wavelengths so that
        # process_signals() has valid reference values even before the
        # sensor/calibrate service is called.  setdefault leaves any
        # already-calibrated channel untouched.
        for ch, wl in self.base_wavelengths.items():
            self.ref_wavelengths.setdefault( ch, wl.copy() )

        # for

        """ Connect demo override """
        if (self.num_chs > 0) and (self.num_aa > 0):
            # configure demo interrogator
            self.interrogator = DemoHyperionInterrogator(
                is_ready=True, peaks=self.base_wavelengths,
                channel_count=self.num_chs
            )
            self.is_connected = self.interrogator.is_ready
            self.get_logger().info( "Connected to demo hyperion interrogator" )
            self.get_logger().info( "Connected to IP: {}".format( self.ip_address ) )

        # if

        else:
            self.interrogator = DemoHyperionInterrogator(
                is_ready=False, peaks=self.base_wavelengths,
                channel_count=self.num_chs
            )
            self.is_connected = self.interrogator.is_ready

        # else

        return self.is_connected

    # connect

    # ------------------------------------------------------------------
    # Sim-mode helpers
    # ------------------------------------------------------------------

    def _load_sim_shape( self ):
        """(Re-)Load the shape polyline from *self._sim_shape_file*."""
        try:
            self._sim_shape_points = load_shape_file( self._sim_shape_file )
            self.get_logger().info(
                f"Loaded sim shape from '{self._sim_shape_file}' "
                f"({len(self._sim_shape_points)} points, "
                f"arc-length ≈ "
                f"{np.sum(np.linalg.norm(np.diff(self._sim_shape_points, axis=0), axis=1)):.1f} mm)"
            )
        except Exception as exc:
            self.get_logger().warning(
                f"Could not load sim shape file '{self._sim_shape_file}': {exc}"
            )
            self._sim_shape_points = None

    # _load_sim_shape

    def _get_cal_matrices_from_needle( self ):
        """Return calibration matrices from the loaded FBG-needle object.

        Falls back to ``None`` if the needle was not loaded.  The returned
        dict maps ``float(location_mm_from_tip)`` → ``np.ndarray (2, num_chs)``.
        """
        if self.fbgneedle is None:
            return None
        try:
            return { float( k ): np.asarray( v, dtype=float )
                     for k, v in self.fbgneedle.cal_matrices.items() }
        except AttributeError:
            return None

    # _get_cal_matrices_from_needle

    def _get_sensor_locs_from_tip( self ):
        """Return sensor active-area locations (mm from tip), in AA order."""
        if self.fbgneedle is None:
            return None
        try:
            locs = self.fbgneedle.sensor_location_tip  # ndarray, locs from tip
            return list( float( l ) for l in locs )
        except AttributeError:
            return None

    # _get_sensor_locs_from_tip

    def _compute_shape_driven_peaks( self ) -> dict:
        """Compute raw wavelength peaks that encode the target shape curvature.

        The returned dict ``{ch: ndarray(num_aa)}`` contains *absolute*
        wavelengths (ref + Δλ).  When ``process_signals()`` subtracts the
        reference wavelengths it yields exactly the target Δλ that corresponds
        to the desired curvature.
        """
        cal_matrices = self._get_cal_matrices_from_needle()
        sensor_locs = self._get_sensor_locs_from_tip()

        if cal_matrices is None or sensor_locs is None:
            self.get_logger().warning(
                "Needle calibration not loaded; "
                "cannot compute shape-driven peaks. "
                "Ensure needleParamFile is set correctly."
            )
            return None

        delta_lambda = shape_to_wavelength_shifts(
            self._sim_shape_points,
            cal_matrices,
            sensor_locs,
            self.num_chs,
            insertion_depth=self._sim_insertion_depth,
        )

        # Build peaks = ref_wavelengths + Δλ
        shape_peaks = { }
        for ch in range( 1, self.num_chs + 1 ):
            ref = self.ref_wavelengths.get(
                ch, self.base_wavelengths.get( ch, np.zeros( self.num_aa ) )
            )
            ref = np.asarray( ref, dtype=float )
            dl = delta_lambda.get( ch, np.zeros( self.num_aa ) )
            shape_peaks[ ch ] = ref + dl

        return shape_peaks

    # _compute_shape_driven_peaks

    # ------------------------------------------------------------------
    # Overridden publish_peaks
    # ------------------------------------------------------------------

    def publish_peaks( self ):
        """Publish peaks; injects shape-driven wavelengths when sim_shape_file
        is set (sim_level=1 shape-driven mode).

        If a valid shape file has been supplied the ``interrogator.peaks``
        are replaced each cycle with values derived from the target shape
        curvature before the parent ``publish_peaks()`` processes and
        publishes them.  All other sim levels keep existing behaviour.
        """
        if (
            self._sim_shape_file
            and self._sim_shape_points is not None
            and self.is_connected
        ):
            try:
                shape_peaks = self._compute_shape_driven_peaks()
                if shape_peaks is not None:
                    self.interrogator = DemoHyperionInterrogator(
                        is_ready=True,
                        peaks=shape_peaks,
                        channel_count=self.num_chs,
                    )
            except Exception as exc:
                self.get_logger().warning(
                    f"Shape-driven peak computation failed: {exc}"
                )

        # if: shape-driven mode

        # Delegate actual publishing to the parent implementation
        super().publish_peaks()

    # publish_peaks

    def parameter_callback( self, params ):
        """Extend parameter callback – also handle sim parameters."""
        # values that should not be changed
        fixed_params = [
            HyperionDemo.param_names[ 'num_chs' ],
            HyperionDemo.param_names[ 'num_aa' ],
        ]
        if any( [ fix_p in params for fix_p in fixed_params ] ):
            return SetParametersResult( successful=False )

        # Handle mutable sim parameters
        for param in params:
            if param.name == HyperionDemo.param_names[ 'sim_shape_file' ]:
                self._sim_shape_file = param.get_parameter_value().string_value
                if self._sim_shape_file:
                    self._load_sim_shape()
                else:
                    self._sim_shape_points = None

            elif param.name == HyperionDemo.param_names[ 'sim_insertion_depth' ]:
                self._sim_insertion_depth = (
                    param.get_parameter_value().double_value
                )

        # for

        return super().parameter_callback( params )

    # parameter_callback

    def ref_wl_service( self, request, response ):
        """Service to get the reference wavelength."""
        if self.is_connected:
            self.get_logger().info(
                f"Starting to recalibrate the sensors wavelengths "
                f"for {self.num_samples} samples."
            )

            data = { }
            counter = 0
            error_counter = 0
            max_errors = 5
            while counter < self.num_samples:
                try:
                    signal = self.parse_peaks( self.interrogator.peaks )

                    for ch_num, peaks in signal.items():
                        if ch_num not in data.keys():
                            data[ ch_num ] = peaks
                        else:
                            data[ ch_num ] += peaks

                    # for

                    counter += 1
                    error_counter = 0

                # try
                except gaierror:
                    error_counter += 1
                    if error_counter > max_errors:
                        response.success = False
                        response.message = "Timeout interrogation occurred."
                    # if
                    continue

                # except
            # while

            for ch_num, agg_peaks in data.items():
                self.ref_wavelengths[ ch_num ] = agg_peaks / self.num_samples
            # for

            self.update_fbgneedle()

            response.success = True
            self.get_logger().info( "Recalibration successful" )
            self.get_logger().info(
                "Reference wavelengths: {}".format(
                    list( self.ref_wavelengths.values() )
                )
            )

        # if
        else:
            response.success = False
            response.message = (
                "Interrogator was not able to gather peaks. "
                "Check connection and IP address."
            )

        # else

        return response

    # ref_wl_service


# class: HyperionDemo

def main( args=None ):
    # Arguments
    num_chs = 3
    num_aa = 4

    rclpy.init( args=args )

    hyperion_demo = HyperionDemo( num_chs=num_chs, num_aa=num_aa )

    try:
        rclpy.spin( hyperion_demo )

    except KeyboardInterrupt:
        pass

    hyperion_demo.get_logger().info( 'Shutting down...' )
    hyperion_demo.destroy_node()
    rclpy.shutdown()


# main

if __name__ == "__main__":
    main()

# if: main
