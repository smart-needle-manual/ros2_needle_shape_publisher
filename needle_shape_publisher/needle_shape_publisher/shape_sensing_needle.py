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
from geometry_msgs.msg import PoseArray, Point, PoseStamped
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import Float64MultiArray, Header, Float64

from std_srvs.srv import Trigger

from needle_shape_publisher_interfaces.srv import (
    GetPoseFromPoseArray,
    GetPoseArray,
    UpdateShapeType,
)

# needle shape sensing package
from needle_shape_sensing.intrinsics import SHAPETYPE as NEEDLESHAPETYPE, AirDeflection

####Edit: FIXME: SHAPETYPE needs to include LIM, and eventually a separate launch file

# current package
from . import utilities
from .frame_update import (
    insertion_point_from_stage_pose,
    stage_pose_translation,
    transform_shape,
)
from .sensorized_shape_sensing_needle import NeedleNode


class ShapeSensingNeedleNode( NeedleNode ):
    """Needle to handle shape-sensing applications"""

    # - optimization options
    PARAM_OPTIMIZER        = ".".join( [ NeedleNode.PARAM_NEEDLE, 'optimizer' ] )
    ####Edit: FIXME: Join parameters need a routing switch based on model/shapetype
    PARAM_UPDATE_ORNT_AIR  = ".".join( [ PARAM_OPTIMIZER, 'update_orientation_with_airgap'] )
    PARAM_INITIAL_INSERTION_POINT = ".".join( [ NeedleNode.PARAM_NEEDLE, 'initial_insertion_point' ] )

    # needle pose parameters
    # R_NEEDLEPOSE = geometry.rotx( -np.pi / 2 )  # +z-axis -> +y-axis
    # R_NEEDLEPOSE = np.array( [ [ -1, 0, 0 ],
    #                            [ 0, 0, 1 ],
    #                            [ 0, 1, 0 ] ] )
    # The needle frame is assumed to be the world frame, and the stage z-axis is
    # assumed to be aligned with the needle insertion axis, so no rotation is needed.
    R_NEEDLEPOSE = np.eye(3)

    def __init__( self, name="ShapeSensingNeedle" ):
        super().__init__( name )
        ####Edit: FIXME: Note -- Keep kc_i/winit_i for now, publish add. params if need be
        self.get_logger().set_level(LoggingSeverity.DEBUG)

        ####Change
        self.manual_mode = self.declare_parameter(
            'needle.manual_mode',
            value= False
        ).get_parameter_value().bool_value

        self.entrypoint_received = False
        self.needlepose_received = False
        self.curvatures_received = False

        # Dirty flag: set True by sub_curvatures_callback and sub_needlepose_callback.
        # publish_shape skips get_needleshape() (the optimizer) when inputs have not
        # changed since the last successful compute, and serves cached poses instead.
        self._inputs_dirty = False
        self._cached_pmat  = None
        self._cached_Rmat  = None

        # Callback groups: timer gets its own MutuallyExclusive group so the optimizer
        # cannot block incoming subscriber callbacks. Subscriptions share a Reentrant
        # group so curvature and pose messages are processed concurrently.
        self._timer_cbg = MutuallyExclusiveCallbackGroup()
        self._sub_cbg   = ReentrantCallbackGroup()

        self._curvature_lock     = threading.Lock()
        self._use_ext_curvatures = False  # latched True by service call from Slicer

        self.get_logger().info(f"Manual mode: {self.manual_mode}")

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

        # configure shape-sensing needle
        self.ss_needle._update_orientation_needle_airgap       = self.declare_parameter(
            self.PARAM_UPDATE_ORNT_AIR,
            value=True,
            descriptor=ParameterDescriptor(type=Parameter.Type.BOOL.value),
        ).get_parameter_value().bool_value

        ####Edit: FIXME: self.ss_needle.optimizer needs to account for the new method, and eventually such optimizer checks may need to be removed. Parameters like maxiter--are they correct? Maxi[...]

        self.ss_needle.current_depth      = 0
        self.air_gap                      = 0  # the length of the gap in the air from the tissue
        self.ss_needle.current_curvatures = np.zeros( (2, self.ss_needle.num_activeAreas), dtype=float )

        # Initialise insertion point from a ROS parameter so the sim (and any
        # launch file) can set a valid value without needing a subscriber message
        # on /needle/state/skin_entry.  Defaults to the origin [0, 0, 0].
        pd_init_ins_pt = ParameterDescriptor(
            name=self.PARAM_INITIAL_INSERTION_POINT,
            type=Parameter.Type.DOUBLE_ARRAY.value,
            description="Initial skin-entry insertion point [x, y, z] in mm (world frame).",
        )
        init_insertion_point = self.declare_parameter(
            pd_init_ins_pt.name,
            descriptor=pd_init_ins_pt,
            value=[ 0.0, 0.0, 0.0 ],
        ).get_parameter_value().double_array_value
        self.ss_needle.insertion_point = np.array( list( init_insertion_point ) )

        # configure current needle pose parameters
        self.current_needle_pose = (np.zeros( 3 ), self.R_NEEDLEPOSE)

        # - look-up table of (insertion depth (mod ds), theta rotation (rads))
        self.history_needle_pose = np.reshape([ 0, 0 ], (-1, 1))

        # create publishers
        self.pub_shape = self.create_publisher( PoseArray, 'state/current_shape', 1 )
        self.pub_depth = self.create_publisher( Float64, 'state/insertion_depth', 1 )

        # create subscriptions
        self.sub_curvatures = self.create_subscription(
            Float64MultiArray,
            'state/curvatures',
            self.sub_curvatures_callback,
            10,
            callback_group=self._sub_cbg,
        )
        self.sub_curvatures_ext = self.create_subscription(
            Float64MultiArray,
            'state/curvatures_in',
            self.sub_curvatures_ext_callback,
            10,
            callback_group=self._sub_cbg,
        )
        self.sub_entrypoint = self.create_subscription(
            Point,
            'state/skin_entry',
            self.sub_entrypoint_callback,
            10,
            callback_group=self._sub_cbg,
        )
        self.sub_needlepose = self.create_subscription(
            PoseStamped,
            '/stage/state/needle_pose',
            self.sub_needlepose_callback,
            10,
            callback_group=self._sub_cbg,
        )

        # services
        self.sub_needleshape_querypt = self.create_service(
            GetPoseFromPoseArray,
            "current_shape/query_point",
            self.srv_needleshape_querypt_callback,
        )
        self.sub_needleshape_querypt = self.create_service(
            GetPoseArray,
            "current_shape/query",
            self.srv_needleshape_query_callback,
        )
        self.srv_update_shapetype = self.create_service(
            UpdateShapeType,
            "shapetype/update",
            self.srv_update_shapetype_callback,
        )
        self.srv_use_ext_curvatures = self.create_service(
            Trigger,
            'curvatures/use_external',
            self.srv_use_ext_curvatures_callback,
        )
        self.srv_use_fbg_curvatures = self.create_service(
            Trigger,
            'curvatures/use_fbg',
            self.srv_use_fbg_curvatures_callback,
        )

        # create timers
        ####Change
        self.pub_shape_timer = self.create_timer(
            0.05, self.publish_shape,
            callback_group=self._timer_cbg,  # isolated: optimizer never blocks subs
        )
        # NOTE: In `needle.manual_mode`, `publish_shape()` will only compute/publish
        # once BOTH `/needle/state/skin_entry` (geometry_msgs/msg/Point) and
        # `/stage/state/needle_pose` (geometry_msgs/msg/PoseStamped) have been received.
        #
        # For manual teleop/testing, you typically want to publish these two topics
        # continuously (or at least back-to-back) and include `header.frame_id` in
        # the PoseStamped. Example continuous publishers:
        #
        #   ros2 topic pub /needle/state/skin_entry geometry_msgs/msg/Point "{x: 0.0, y: 0.0, z: 0.0}"
        #   ros2 topic pub /stage/state/needle_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: needle}, pose: {position: {x: 0.0, y: 0.0, z: 0.05}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"
        #
        # Using `--once` is fine for single-shot updates, but for robust operation
        # it is often better to continuously publish both inputs.
        ####End Change

    # __init__

    ####Edit: FIXME: May need to use setters in shell script to make life easier

    @property
    def insertion_depth( self ):
        return self.ss_needle.current_depth

    # property: insertion_depth

    @insertion_depth.setter
    def insertion_depth( self, depth ):
        self.ss_needle.current_depth = depth

    # insertion_depth setter

    @property
    def needle_guide_exit_pt(self):
        return stage_pose_translation(self.current_needle_pose[0])

    # needle_guide_exit_pt

    def __transform( self, pmat: np.ndarray, Rmat: np.ndarray ):
        """ Transforms the needle pose of an N-D array using the current needle pose

            :param pmat: numpy array of N x 3 size.
            :param Rmat: numpy array of orientations of size N x 3 x 3

            :returns: pmat transformed by current needle pose, Rmat transformed by current needle pose

        """

        current_p, current_R = self.current_needle_pose

        return transform_shape( pmat, Rmat, current_p, current_R )

    # __transform

    def get_needleshape( self ):
        """ Get the current needle shape"""
        # TODO: incorporate rotation while inserted into tissue
        ####Change
        self.get_logger().debug(f"Current shapetype: {self.ss_needle.current_shapetype}")
        self.get_logger().debug(f"num_ActiveAreas: {self.ss_needle.num_activeAreas}")
        ####End Change

        ####Edit: FIXME: elif self.ss_needle.current_shapetype & NEEDLESHAPETYPE.LIM == NEEDLESHAPETYPE.LIM # inverse strain optim + linear interp
        if (self.ss_needle.current_shapetype & NEEDLESHAPETYPE.PIECEWISE_EXP) == NEEDLESHAPETYPE.PIECEWISE_EXP:
            pmat, Rmat = self.ss_needle.get_needle_shape()
        else:
            self.get_logger().error( f"Needle shape type: {self.ss_needle.current_shapetype} is not implemented." )
            self.get_logger().error( f"Resorting to shape type: {NEEDLESHAPETYPE.SINGLEBEND_SINGLELAYER}." )
            self.ss_needle.update_shapetype( NEEDLESHAPETYPE.SINGLEBEND_SINGLELAYER )
            pmat, Rmat = None, None  # pop out of the loop and redo

        # else

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

        pmat, Rmat = self.__transform(pmat, Rmat)

        return pmat, Rmat

    # get_needleshape

    def publish_shape( self ):
        """ Publish the 3D needle shape"""
        ####Change
        if self.manual_mode:
            if not (self.entrypoint_received and self.needlepose_received and self.curvatures_received):
                self.get_logger().debug(
                    "Manual mode active --- waiting for entry point, needle pose, and curvature data"
                )
                return
            self.get_logger().debug("Manual data received - computing shape...")
        else:
            if not (self.needlepose_received and self.curvatures_received):
                self.get_logger().debug(
                    "Waiting for needle pose and curvature data before publishing shape"
                )
                return
        ####End Change

        # Skip optimizer when inputs have not changed since last compute.
        # Publish the cached shape so downstream (ShapeCall) continues to receive
        # updates at 20 Hz even during idle periods, without re-running the optimizer.
        if not self._inputs_dirty:
            if self._cached_pmat is not None and self._cached_Rmat is not None:
                header    = Header( stamp=self.get_clock().now().to_msg(), frame_id='needle' )
                msg_shape = utilities.poses2msg( self._cached_pmat, self._cached_Rmat, header=header )
                msg_depth = Float64( data=float( self.insertion_depth ) )
                self.pub_shape.publish( msg_shape )
                self.pub_depth.publish( msg_depth )
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
        is_piecewise_exp = (
            (self.ss_needle.current_shapetype & NEEDLESHAPETYPE.PIECEWISE_EXP)
            == NEEDLESHAPETYPE.PIECEWISE_EXP
        )
        if not is_piecewise_exp:
            self.kc_i     = self.ss_needle.current_kc
            self.w_init_i = self.ss_needle.current_winit

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

        msg_depth = Float64(data=float(self.insertion_depth))

        self.get_logger().debug( f"Needle Shapes: {pmat.shape}, {Rmat.shape}, {len( msg_shape.poses )}" )

        # publish the messages
        ####Change
        self.get_logger().debug(f"Publishing shape with {len(msg_shape.poses)} poses")
        self.get_logger().debug("About to publish to /needle/state/current_shape")
        ####End Change
        self.pub_shape.publish( msg_shape )
        ####Change
        self.get_logger().debug(f"Shape poses: {[ (p.position.x, p.position.y, p.position.z) for p in msg_shape.poses ]}")
        ####End Change
        self.pub_depth.publish( msg_depth )

        # kappa_c and w_init are not applicable for PIECEWISE_EXP; skip those
        # topics to avoid a TypeError when current_kc is a scalar.
        if is_piecewise_exp:
            self.get_logger().debug(
                "PIECEWISE_EXP shape type: skipping state/kappac and state/winit publish "
                "(kappa_c and w_init are not applicable for this shape type)."
            )
        else:
            msg_kc    = Float64MultiArray( data=self.kc_i )
            msg_winit = Float64MultiArray( data=self.w_init_i.tolist() )
            self.pub_kc.publish( msg_kc )
            self.pub_winit.publish( msg_winit )
            self.get_logger().debug(
                "Published needle shape, kappa_c and w_init on topics: "
                f"{self.pub_shape.topic},{self.pub_kc.topic},{self.pub_winit.topic}"
            )

        ####Change
        if self.manual_mode:
            self.entrypoint_received = False
            self.needlepose_received = False
            self.get_logger().debug("Shape published — waiting for next manual input.")
        ####End Change

    # publish_shape

    async def publish_shape_async(self):
        """ Function to asynchronously publish the needle shape """
        # TODO
        pass

    # publish_shape_async

    def sub_curvatures_callback( self, msg: Float64MultiArray ):
        """ Subscription to needle sensor curvatures """
        with self._curvature_lock:
            if self._use_ext_curvatures:
                return
            curvatures = np.reshape( msg.data, (2, -1), order='F' )
            self.ss_needle.current_curvatures = curvatures
            self.curvatures_received = True
            self._inputs_dirty = True

        self.get_logger().debug(f"Curvatures X: {self.ss_needle.current_curvatures[0]}")
        self.get_logger().debug(f"Curvatures Y: {self.ss_needle.current_curvatures[1]}")

        if not self.ss_needle.is_calibrated:
            self.ss_needle.ref_wavelengths = np.ones_like( self.ss_needle.ref_wavelengths )

        # if

    # sub_curvatures_callback
    def sub_curvatures_ext_callback( self, msg: Float64MultiArray ):
        """ External curvature feed (e.g. Slicer). Active only while latched on via service. """
        with self._curvature_lock:
            if not self._use_ext_curvatures:
                return
            curvatures = np.reshape( msg.data, (2, -1), order='F' )
            self.ss_needle.current_curvatures = curvatures
            self.curvatures_received = True
            self._inputs_dirty = True

        self.get_logger().debug(f"[ext] Curvatures X: {self.ss_needle.current_curvatures[0]}")
        self.get_logger().debug(f"[ext] Curvatures Y: {self.ss_needle.current_curvatures[1]}")

        if not self.ss_needle.is_calibrated:
            self.ss_needle.ref_wavelengths = np.ones_like( self.ss_needle.ref_wavelengths )

    # sub_curvatures_ext_callback

    def srv_use_ext_curvatures_callback( self, request, response ):
        """ Latch onto external curvature feed. Call once from Slicer when ready. """
        with self._curvature_lock:
            self._use_ext_curvatures = True
        response.success = True
        response.message = "Curvature source: external (state/curvatures_in)"
        self.get_logger().info( response.message )
        return response

    # srv_use_ext_curvatures_callback

    def srv_use_fbg_curvatures_callback( self, request, response ):
        """ Revert to FBG pipeline curvatures. """
        with self._curvature_lock:
            self._use_ext_curvatures = False
        response.success = True
        response.message = "Curvature source: FBG pipeline (state/curvatures)"
        self.get_logger().info( response.message )
        return response

    # srv_use_fbg_curvatures_callback


    def sub_entrypoint_callback( self, msg: Point ):
        """ Subscription to entrypoint topic """
        # skin_entry is assumed to be in the needle/world frame (needle frame == world frame)
        insertion_point = np.array( [ msg.x, msg.y, msg.z ] )

        # The stage pose carries lateral guide offsets in x/y and insertion depth
        # in z.  Keep the z component in the needle frame so the downstream shape
        # model can use it as the air-gap / entry depth along the insertion axis.
        self.ss_needle.insertion_point = insertion_point_from_stage_pose(
            insertion_point,
            self.current_needle_pose[0],
        )

        ####Change
        self.entrypoint_received = True
        self.get_logger().debug("Received entrypoint data (manual mode flag updated).")
        ####End Change

        ####Change level 2
        #if self.entrypoint_received and self.needlepose_received:
        #    try:
        #        self.manual_input_timer.cancel()
        #        self.get_logger().info("Manual inputs complete - canceled manual input timer")
        #    except Exception:
        #        pass
        ####End Change

        self.get_logger().debug(f"Current insertion point rel. to needle base = {self.ss_needle.insertion_point}")

    # sub_entrypoint_callback

    def sub_needlepose_callback( self, msg: PoseStamped ):
        """ Subscription to entrypoint topic """
        self.current_needle_pose      = list( utilities.msg2pose( msg.pose ) )
        self.current_needle_pose[ 0 ] = self.current_needle_pose[ 0 ]
        self.get_logger().debug( f"NeedlePoseCB: pose[0]: {self.current_needle_pose[ 0 ]}" )
        self.get_logger().debug( f"NeedlePoseCB: pose[1]: {self.current_needle_pose[ 1 ]}" )

        self.current_needle_pose[ 1 ] = self.current_needle_pose[ 1 ] @ self.R_NEEDLEPOSE  # update current needle pose

        # update the insertion depth along the z-axis (stage z-axis == insertion axis).
        # Assumes stage z=0 when the needle tip is exactly at the skin surface,
        # so depth = needle_base_z - skin_entry_z gives depth into tissue.
        self.insertion_depth = max(
            0,
            min(
                self.current_needle_pose[0][2] - self.ss_needle.insertion_point[2], # z-axis
                self.ss_needle.length
            )
        )

        ####Change
        self.needlepose_received = True
        self.get_logger().debug("Received needle pose data (manual mode flag updated).")
        self._inputs_dirty = True
        ####End Change

        self.get_logger().debug( f"Current insertion depth: {self.insertion_depth}" )

        # update the history of orientations (NOT USED YET)
        # Uses raw stage z (not depth into tissue); valid only when stage z=0 at skin contact.
        depth_ds = msg.pose.position.z - msg.pose.position.z % self.ss_needle.ds
        theta    = msg.pose.orientation.z
        if np.any( self.history_needle_pose[ 0 ] == depth_ds ):  # check if we already have this value
            idx = np.argwhere( self.history_needle_pose[ 0 ] == depth_ds ).ravel()
            self.history_needle_pose[ 1, idx ] = theta

        # if

        else:  # add a new value
            np.hstack( (self.history_needle_pose, [ [ depth_ds ], [ theta ] ]) )

        # else
        ####Change level 2
        #if self.entrypoint_received and self.needlepose_received:
        #    try:
        #        self.manual_input_timer.cancel()
        #        self.get_logger().info("Manual inputs complete - canceled manual input timer")
        #    except Exception:
        #        pass
        ####End Change

    # sub_needlepose_callback

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
        self.ss_needle.update_shapetype(new_shapetype)

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
    if ssneedle_node.manual_mode:
        ssneedle_node.get_logger().info(
            """Manual Mode Instructions (copy/paste):
    bash -lc '
      ros2 topic pub /needle/state/skin_entry geometry_msgs/msg/Point "{x: 0.0, y: 0.0, z: 0.0}" &
      ros2 topic pub /stage/state/needle_pose geometry_msgs/msg/PoseStamped "{
        header: {frame_id: needle},
        pose: {
          position: {x: 0.0, y: 0.0, z: 75.0},
          orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
        }
      }"
    '
    """
        )
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
