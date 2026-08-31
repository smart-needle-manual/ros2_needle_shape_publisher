####Change was for prior modifications necessary for manual/automatic inputs in place of robot dependence.
####Edit is for new optimization method. First change occurred in linear_optim. See number of commits ahead of main to determine most recent branch.

import numpy as np
# ROS2 packages
import rclpy
from rclpy import Parameter
from rclpy.logging import LoggingSeverity
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading

# messages
from geometry_msgs.msg import PoseArray
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from std_msgs.msg import Float64MultiArray, Header, Float64

from std_srvs.srv import Trigger

from needle_shape_publisher_interfaces.srv import (
    GetPoseFromPoseArray,
    GetPoseArray,
    UpdateShapeType,
)

# needle shape sensing package
from needle_shape_sensing.intrinsics import SHAPETYPE as NEEDLESHAPETYPE

# LIM_SO3 (0x40) is the LIM implementation.
# current package
from . import utilities
from .sensorized_shape_sensing_needle import NeedleNode


class ShapeSensingNeedleNode( NeedleNode ):
    """Needle to handle shape-sensing applications"""

    # - optimization options
    PARAM_OPTIMIZER        = ".".join( [ NeedleNode.PARAM_NEEDLE, 'optimizer' ] )
    PARAM_R_INIT = ".".join([NeedleNode.PARAM_NEEDLE, 'R_init'])

    # needle pose parameters
    # R_NEEDLEPOSE = geometry.rotx( -np.pi / 2 )  # +z-axis -> +y-axis
    # R_NEEDLEPOSE = np.array( [ [ -1, 0, 0 ],
    #                            [ 0, 0, 1 ],
    #                            [ 0, 1, 0 ] ] )
    # The needle frame is assumed to be the world frame, and the stage z-axis is
    # assumed to be aligned with the needle insertion axis, so no rotation is needed.

    def __init__( self, name="ShapeSensingNeedle" ):
        super().__init__( name )
        ####Edit: FIXME: Note -- Keep kc_i/winit_i for now, publish add. params if need be
        self.get_logger().set_level(LoggingSeverity.DEBUG)

        ####Change
        
        self._required_inputs = {'curvatures', 'insertion_depth', 'R_init'}

        # Which input slots have been received at least once
        self._received = set()   # grows as: 'curvatures', 'insertion_depth'; 'R_init' pre-seeded from parameter
        self._inputs_dirty = False  # still a bool; set by any input cb, cleared after optimizer
        self._cached_pmat  = None
        self._cached_Rmat  = None

        # Callback groups: timer gets its own MutuallyExclusive group so the optimizer
        # cannot block incoming subscriber callbacks. Subscriptions share a Reentrant
        # group so curvature and pose messages are processed concurrently.
        self._timer_cbg = MutuallyExclusiveCallbackGroup()
        self._sub_cbg   = ReentrantCallbackGroup()

        self._curvature_lock = threading.Lock()
        # no latch needed — curvature source is a launch-time topic remap

        self.get_logger().info(f"Required inputs: {self._required_inputs}")

        # Declare and apply the shape type parameter.
        # A value of -1 (the default) means "keep whatever the needle parameter
        # file loaded"; any non-negative integer is interpreted as a
        # needle_shape_sensing.intrinsics.SHAPETYPE value.
        shape_type_int = self.declare_parameter(
            self.PARAM_NEEDLESHAPE,
            value=-1,
            descriptor=ParameterDescriptor(
                name=self.PARAM_NEEDLESHAPE,
                type=Parameter.Type.INTEGER.value,
                description=(
                    'Shape type for ShapeSensingNeedleNode '
                    '(integer value of needle_shape_sensing.intrinsics.SHAPETYPE). '
                    '-1 means use the default from the needle parameter file.'
                ),
            ),
        ).get_parameter_value().integer_value
        if shape_type_int >= 0:
            try:
                self.ss_needle.update_shapetype(NEEDLESHAPETYPE(shape_type_int))
                self.get_logger().info(f"Shape type set to: {self.ss_needle.current_shapetype}")
            except ValueError:
                self.get_logger().warning(
                    f"Unknown shape_type value {shape_type_int}; keeping default "
                    f"{self.ss_needle.current_shapetype}."
                )
        else:
            self.get_logger().info(f"Using default shape type: {self.ss_needle.current_shapetype}")
        ####End Change

        # R_init: initial needle orientation (row-major 3×3, 9 floats). Default = eye(3).
        # Set at runtime with:
        #   ros2 param set /needle needle.R_init "[1,0,0, 0,1,0, 0,0,1]"
        PARAM_R_INIT = ".".join([NeedleNode.PARAM_NEEDLE, 'R_init'])
        r_init_flat = self.declare_parameter(
            PARAM_R_INIT,
            value=[1.,0.,0., 0.,1.,0., 0.,0.,1.],
            descriptor=ParameterDescriptor(
                type=Parameter.Type.DOUBLE_ARRAY.value,
                description='Initial needle orientation as a row-major 3×3 rotation matrix (9 floats).',
            ),
        ).get_parameter_value().double_array_value
        self._R_init = np.array(r_init_flat, dtype=float).reshape(3, 3)
        self._received.add('R_init')   # eye(3) default is always valid; unblock startup
        self.add_on_set_parameters_callback(self._on_set_parameters) 

        # configure shape-sensing needle

        self.ss_needle.current_depth      = 0
        self._depth_lock = threading.Lock()
        self.air_gap                      = 0  # the length of the gap in the air from the tissue
        self.ss_needle.current_curvatures = np.zeros( (2, self.ss_needle.num_activeAreas), dtype=float )

        # configure current needle pose parameters - insertion depth mod ds, theta rotation (rads)
        self._R_init = np.eye(3)   # initial orientation; updated by sub_R_init_callback
        self._R_init_lock = threading.Lock()

        # create publishers
        self.pub_shape = self.create_publisher( PoseArray, 'state/current_shape', 1 )
        
        # create subscriptions
        self.sub_curvatures = self.create_subscription(
            Float64MultiArray,
            'state/curvatures',
            self.sub_curvatures_callback,
            10,
            callback_group=self._sub_cbg,
        )

        self.sub_insertion_depth = self.create_subscription(
            Float64,
            'state/insertion_depth',
            self.sub_insertion_depth_callback,
            10,
            callback_group=self._sub_cbg,
        )

        # services
        self.srv_needleshape_querypt = self.create_service(
             GetPoseFromPoseArray,
             "current_shape/query_point",
             self.srv_needleshape_querypt_callback,
         )
        self.srv_needleshape_query = self.create_service(
             GetPoseArray,
             "current_shape/query",
             self.srv_needleshape_query_callback,
         )
        self.srv_update_shapetype = self.create_service(
            UpdateShapeType,
            "shapetype/update",
            self.srv_update_shapetype_callback,
        )

        # create timers
        ####Change
        self.pub_shape_timer = self.create_timer(
            0.05, self.publish_shape,
            callback_group=self._timer_cbg,  # isolated: optimizer never blocks subs
        )
        ####End Change

    # __init__

    ####Edit: FIXME: May need to use setters in shell script to make life easier
    def _mark_received(self, slot: str):
        """Mark an input slot as received and flag inputs as dirty."""
        self._received.add(slot)
        self._inputs_dirty = True

    @property
    def insertion_depth( self ):
        return self.ss_needle.current_depth

    # property: insertion_depth

    @insertion_depth.setter
    def insertion_depth( self, depth ):
        self.ss_needle.current_depth = depth

    # insertion_depth setter

    def get_needleshape( self ):
        """ Get the current needle shape"""
        # TODO: incorporate rotation while inserted into tissue
        ####Change
        self.get_logger().debug(f"Current shapetype: {self.ss_needle.current_shapetype}")
        self.get_logger().debug(f"num_ActiveAreas: {self.ss_needle.num_activeAreas}")
        ####End Change

        with self._R_init_lock:
            R_init = self._R_init.copy()

        if (self.ss_needle.current_shapetype & NEEDLESHAPETYPE.LIM_SO3) == NEEDLESHAPETYPE.LIM_SO3:
            pmat, Rmat = self.ss_needle.get_needle_shape(R_init=R_init)
        else:
            self.get_logger().error(
                f"Shape type {self.ss_needle.current_shapetype} is not supported. "
                f"Only PIECEWISE_EXP (LIM) is valid in this branch."
            )
            return None, None

        if (pmat is None) and (Rmat is None):
            return pmat, Rmat

        # For shape-driven simulation, the model can return only the inserted
        # section (relative to full needle length). In that case append the
        # remaining straight segment so publishing remains over full length.
        #
        # For regular data-driven/non-simulation operation, the model already
        # returns full-length shape, so dL <= 0 and this block is a no-op.
        L_needle = utilities.calculate_needle_length( pmat )
        dL = self.ss_needle.length - L_needle
        if dL > 0:
            pmat_straight = np.zeros( (1, 3), dtype=pmat.dtype )
            if dL > self.ss_needle.ds:
                # generate straight needle length in ds increments
                L_straight = np.arange(
                    0,
                    (dL // self.ss_needle.ds + 1) * self.ss_needle.ds,
                    self.ss_needle.ds,
                )

                # generate straight needle shape
                pmat_straight = np.zeros( (len( L_straight ), 3), dtype=pmat.dtype )
                pmat_straight[:, 2] = L_straight

            # if
            else:  # less than ds increment
                pmat_straight = np.zeros( (2, 3), dtype=pmat.dtype )
                pmat_straight[-1, 2] = dL

            # else
            Rmat_straight = np.tile(
                np.eye( 3, dtype=Rmat.dtype ),
                (pmat_straight.shape[0], 1, 1),
            )

            # update the needle shapes to move coordinate frames
            pmat = pmat @ Rmat_straight[-1].T + pmat_straight[-1:]
            Rmat = Rmat_straight[-1:] @ Rmat

            # append to the current pmat and Rmat
            pmat = np.concatenate(
                (
                    pmat_straight,
                    pmat[1:],
                ),
                axis=0,
            )
            Rmat = np.concatenate(
                (
                    Rmat_straight,
                    Rmat[1:],
                ),
                axis=0,
            )

        return pmat, Rmat

    # get_needleshape

    def publish_shape( self ):
        """ Publish the 3D needle shape"""
        ####Change
        if not self._required_inputs.issubset(self._received):
            missing = self._required_inputs - self._received
            self.get_logger().debug(f"Waiting for inputs: {missing}")
            return
        ####End Change

        # Skip optimizer when inputs have not changed since last compute.
        # Publish the cached shape so downstream (ShapeCall) continues to receive
        # updates at 20 Hz even during idle periods, without re-running the optimizer.
        if not self._inputs_dirty:
            if self._cached_pmat is not None and self._cached_Rmat is not None:
                header    = Header( stamp=self.get_clock().now().to_msg(), frame_id='needle' )
                msg_shape = utilities.poses2msg( self._cached_pmat, self._cached_Rmat, header=header )
                self.pub_shape.publish( msg_shape )
            return

        ####Change
        try:
            self.get_logger().debug("Calling get_needleshape()")
            pmat, Rmat = self.get_needleshape()
            self.get_logger().debug("get_needleshape() returned successfully")
        except Exception as e:
            self.get_logger().error(f"Error during get_needleshape(): {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return

        #pmat, Rmat = self.get_needleshape()
        ####End Change

        # Determine whether the current shape type uses kappa_c / w_init.
        # For PIECEWISE_EXP, current_kc may be a scalar (not iterable) and
        # kc/winit are not used, so skip the cache update to avoid
        # overwriting the cached arrays with an incompatible value.
        # check to make sure messages are not None

        if pmat is None or Rmat is None:
            ####Change
            self.get_logger().warn( f"pmat or Rmat is None: pmat={pmat}, Rmat={Rmat}" )
            self.get_logger().warn( f"Current shapetype: {self.ss_needle.current_shapetype}" )
            ####End Change
            return

        # if

        # Cache result and clear dirty flag now that optimizer has run successfully.
        self._cached_pmat  = pmat
        self._cached_Rmat  = Rmat
        self._inputs_dirty = False

        # needle shape length
        needle_L = np.linalg.norm( np.diff( pmat, axis=0 ), 2, 1 ).sum()
        self.get_logger().debug(
                f"Needle L: {self.ss_needle.length} | Needle Shape L: {needle_L} | Current Depth: {self.ss_needle.current_depth}" )

        # generate pose message
        header    = Header( stamp=self.get_clock().now().to_msg(), frame_id='needle' )
        msg_shape = utilities.poses2msg( pmat, Rmat, header=header )

        self.get_logger().debug( f"Needle Shapes: {pmat.shape}, {Rmat.shape}, {len( msg_shape.poses )}" )

        # publish the messages
        ####Change
        self.get_logger().debug(f"Publishing shape with {len(msg_shape.poses)} poses")
        self.get_logger().debug("About to publish to /needle/state/current_shape")
        ####End Change
        self.pub_shape.publish( msg_shape )
        ####Change
        self.get_logger().debug(f"Shape poses: {[ (p.position.x, p.position.y, p.position.z) for p in msg_shape.poses ]}")

    # publish_shape

    async def publish_shape_async(self):
        """ Function to asynchronously publish the needle shape """
        # TODO
        pass

    # publish_shape_async

    def sub_curvatures_callback( self, msg: Float64MultiArray ):
        """ Subscription to needle sensor curvatures """
        with self._curvature_lock:
            self.ss_needle.current_curvatures = np.reshape(msg.data, (2, -1), order='F')
            self._mark_received('curvatures')

        self.get_logger().debug(f"Curvatures X: {self.ss_needle.current_curvatures[0]}")
        self.get_logger().debug(f"Curvatures Y: {self.ss_needle.current_curvatures[1]}")

        if not self.ss_needle.is_calibrated:
            self.ss_needle.ref_wavelengths = np.ones_like( self.ss_needle.ref_wavelengths )

        # if

    # sub_curvatures_callback

    def _on_set_parameters(self, params):
        """Handle dynamic parameter updates."""
        for p in params:
            if p.name == self.PARAM_R_INIT and p.type_ == Parameter.Type.DOUBLE_ARRAY:
                if len(p.value) != 9:
                    self.get_logger().error(
                        f"needle.R_init must have 9 elements (got {len(p.value)}); ignoring."
                    )
                    return SetParametersResult(successful=False,
                                               reason="R_init must be 9 floats (row-major 3×3)")
                with self._R_init_lock:
                    self._R_init = np.array(p.value, dtype=float).reshape(3, 3)
                self._mark_received('R_init')
                self.get_logger().info(f"R_init updated:\n{self._R_init}")
        return SetParametersResult(successful=True)
    def sub_insertion_depth_callback(self, msg: Float64):
        with self._depth_lock:
            self.insertion_depth = msg.data
        self._mark_received('insertion_depth')

    def srv_needleshape_query_callback(self, req: GetPoseArray.Request, res: GetPoseArray.Response):
        """ Query the current needle shape """
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='needle')
        pmat, Rmat = self.get_needleshape()

        if pmat is None or Rmat is None:
            res.success = False
            return res

        # if

        msg_pose = utilities.poses2msg(pmat, Rmat, header=header)

        res.success    = True
        res.pose_array = msg_pose

        return res


    # srv_query_needle_shape_callback

    def srv_needleshape_querypt_callback(self, req: GetPoseFromPoseArray.Request, res: GetPoseFromPoseArray.Response):
        """ Query the current needle shape """
        pmat, Rmat = self.get_needleshape()

        if pmat is None or Rmat is None:
            res.success = False
            return res

        # if

        idx = req.index
        msg_pose = utilities.pose2msg(pmat[idx], Rmat[idx])

        res.success = True
        res.x       = msg_pose.position.x
        res.y       = msg_pose.position.y
        res.z       = msg_pose.position.z
        res.qx      = msg_pose.orientation.x
        res.qy      = msg_pose.orientation.y
        res.qz      = msg_pose.orientation.z
        res.qw      = msg_pose.orientation.w

        return res


    # srv_query_needle_shape_callback

    def srv_update_shapetype_callback(self, req: UpdateShapeType.Request, res: UpdateShapeType.Response):
        """ Service to dynamically update the needle shape type at runtime.

            Call example:
              ros2 service call /needle/shapetype/update \\
                needle_shape_publisher_interfaces/srv/UpdateShapeType \\
                "{shape_type: 1}"
        """
        shape_type_int = req.shape_type
        try:
            new_shapetype = NEEDLESHAPETYPE(shape_type_int)
        except ValueError:
            res.success = False
            valid_values = ", ".join(
                f"{m.name}={m.value}" for m in NEEDLESHAPETYPE
            )
            res.message = (
                f"Unknown shape_type value {shape_type_int}. "
                f"Valid values: {valid_values}."
            )
            self.get_logger().error(res.message)
            return res

        old_shapetype = self.ss_needle.current_shapetype
        success = self.ss_needle.update_shapetype(new_shapetype)
        if not success:
            res.success = False
            res.message = (
                f"Shape type {new_shapetype} is not supported by the optimizer "
                f"(only PIECEWISE_EXP / LIM is valid in this branch)."
            )
            self.get_logger().error(res.message)
            return res
 
        # Keep the ROS parameter in sync so `ros2 param get` reflects reality.
        self.set_parameters([
            Parameter(self.PARAM_NEEDLESHAPE, Parameter.Type.INTEGER, shape_type_int)
        ])

        res.success = True
        res.message = (
            f"Shape type updated from {old_shapetype} to {self.ss_needle.current_shapetype}."
        )
        self.get_logger().info(res.message)
        return res

    # srv_update_shapetype_callback

# class: ShapeSensingNeedleNode

def main( args=None ):
    rclpy.init( args=args )
    # Node must be instantiated first so that manual_mode is available as an instance attribute
    ssneedle_node = ShapeSensingNeedleNode()
    ####Change
    ####End Change

    try:
        executor = MultiThreadedExecutor()
        executor.add_node( ssneedle_node )
        executor.spin()

    except KeyboardInterrupt:
        pass

    # clean-up
    ssneedle_node.destroy_node()
    rclpy.shutdown()


# main

if __name__ == "__main__":
    main()

# if __main__
