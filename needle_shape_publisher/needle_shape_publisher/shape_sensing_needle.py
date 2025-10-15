import numpy as np
# ROS2 packages
import rclpy
from rclpy import Parameter
# messages
from geometry_msgs.msg import PoseArray, Point, PoseStamped
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import Float64MultiArray, Header, Float64
from std_srvs.srv import Trigger

from needle_shape_publisher_interfaces.srv import (
    GetPoseFromPoseArray,
    GetPoseArray,
)

# needle shape sensing package
from needle_shape_sensing.intrinsics import SHAPETYPE as NEEDLESHAPETYPE, AirDeflection

# current package
from . import utilities
from .sensorized_shape_sensing_needle import NeedleNode


class ShapeSensingNeedleNode( NeedleNode ):
    """Needle to handle shape-sensing applications"""

    # - optimization options
    PARAM_OPTIMIZER        = ".".join( [ NeedleNode.PARAM_NEEDLE, 'optimizer' ] )
    PARAM_KCINIT           = ".".join( [ PARAM_OPTIMIZER, 'initial_kappa_c' ] ) # initial kappa_c for optimization
    PARAM_WINIT            = ".".join( [ PARAM_OPTIMIZER, 'initial_w_init' ] )  # initial omega_init for optimization
    PARAM_OPTIM_MAXITER    = ".".join( [ PARAM_OPTIMIZER, 'max_iterations' ] )
    PARAM_UPDATE_ORNT_AIR  = ".".join( [ PARAM_OPTIMIZER, 'update_orientation_with_airgap'] )
    PARAM_OPTIM_MAXITER_LB = 2

    # needle pose parameters
    # R_NEEDLEPOSE = geometry.rotx( -np.pi / 2 )  # +z-axis -> +y-axis
    # R_NEEDLEPOSE = np.array( [ [ -1, 0, 0 ],
    #                            [ 0, 0, 1 ],
    #                            [ 0, 1, 0 ] ] )
    R_NEEDLEPOSE = np.eye(3)

    def __init__( self, name="ShapeSensingNeedle" ):
        super().__init__( name )

        # declare ang get parameters
        self.kc_i     = np.array( [ 0.0005 ] )
        self.w_init_i = np.array( [ self.kc_i[ 0 ], 0.0, 0.0 ] )

        ####Change
        self.manual_mode = self.declare_parameter(
            'needle.manual_mode',
            value= 'false'
        ).get_parameter_value().bool_value

        self.entrypoint_received = false
        self.needlepose_received = false
        ####End Change

        pd_optim_maxiter = ParameterDescriptor( name=self.PARAM_OPTIM_MAXITER, type=Parameter.Type.INTEGER.value,
                                                description="Maximum iterations for convergence" )
        optim_maxiter = self.declare_parameter( pd_optim_maxiter.name, value=15,
                                                descriptor=pd_optim_maxiter ).get_parameter_value().integer_value

        # configure shape-sensing needle
        self.ss_needle._update_orientation_needle_airgap       = self.declare_parameter(
            self.PARAM_UPDATE_ORNT_AIR,
            value=True,
            descriptor=ParameterDescriptor(type=Parameter.Type.BOOL.value),
        ).get_parameter_value().bool_value
        self.ss_needle.optimizer.options[ 'options' ]          = { 'maxiter': optim_maxiter }
        self.ss_needle.optimizer.options[ 'w_init_bounds' ][2] = [ -1e-3, 1e-3 ]
        
        self.ss_needle.ref_wavelengths    = np.ones_like( self.ss_needle.ref_wavelengths )
        self.ss_needle.current_depth      = 0
        self.air_gap                      = 0  # the length of the gap in the air from the tissue
        self.ss_needle.current_curvatures = np.zeros( (2, self.ss_needle.num_activeAreas), dtype=float )

        # configure current needle pose parameters
        self.current_needle_pose = (np.zeros( 3 ), self.R_NEEDLEPOSE)

        # - look-up table of (insertion depth (mod ds), theta rotation (rads))
        self.history_needle_pose = np.reshape([ 0, 0 ], (-1, 1))

        # create publishers
        self.pub_kc    = self.create_publisher( Float64MultiArray, 'state/kappac', 1 )
        self.pub_winit = self.create_publisher( Float64MultiArray, 'state/winit', 1 )
        self.pub_shape = self.create_publisher( PoseArray, 'state/current_shape', 1 )
        self.pub_depth = self.create_publisher( Float64, 'state/insertion_depth', 1 )

        # create subscriptions
        self.sub_curvatures = self.create_subscription(
            Float64MultiArray,
            'state/curvatures',
            self.sub_curvatures_callback,
            10
        )
        self.sub_entrypoint = self.create_subscription(
            Point,
            'state/skin_entry',
            self.sub_entrypoint_callback,
            10
        )
        self.sub_needlepose = self.create_subscription(
            PoseStamped,
            '/stage/state/needle_pose',
            self.sub_needlepose_callback,
            10
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

        # create timers
        self.pub_shape_timer = self.create_timer( 0.05, self.publish_shape )

    # __init__

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
        return self.current_needle_pose[0] * [1, 1, 1]

    # needle_guide_exit_pt

    def __transform( self, pmat: np.ndarray, Rmat: np.ndarray ):
        """ Transforms the needle pose of an N-D array using the current needle pose

            :param pmat: numpy array of N x 3 size.
            :param Rmat: numpy array of orientations of size N x 3 x 3

            :returns: pmat transformed by current needle pose, Rmat transformed by current needle pose

        """

        current_p, current_R = self.current_needle_pose

        # self.get_logger().debug(
        #         f"__transform: pmat: {pmat.shape}, Rmat: {Rmat.shape}, p: {current_p.shape}, R:{current_R.shape}" )

        # rigid body transform the current needle pose
        pmat_tf, Rmat_tf = None, None
        if pmat is not None:
            # update needle origin to the insertion point (in the needle frame)
            pmat_tf = pmat @ current_R.T + current_p.reshape( 1, -1 )

        # if

        if Rmat is not None:
            Rmat_tf = np.einsum( 'jk, ikl -> ijl', current_R, Rmat )

        return pmat_tf, Rmat_tf

    # __transform

    def get_needleshape( self ):
        """ Get the current needle shape"""
        # TODO: incorporate rotation while inserted into tissue
        if self.ss_needle.current_shapetype & NEEDLESHAPETYPE.SINGLEBEND_SINGLELAYER == NEEDLESHAPETYPE.SINGLEBEND_SINGLELAYER:  # single layer
            pmat, Rmat = self.ss_needle.get_needle_shape( self.kc_i[ 0 ], self.w_init_i )

        elif self.ss_needle.current_shapetype & NEEDLESHAPETYPE.SINGLEBEND_DOUBLELAYER == 0x02:  # 2 layers
            kc1 = self.kc_i[0]
            kc2 = kc1
            if len( self.kc_i ) >= 2:
                kc2 = self.kc_i[1]

            pmat, Rmat = self.ss_needle.get_needle_shape( kc1, kc2, self.w_init_i )

        # elif

        elif self.ss_needle.current_shapetype == NEEDLESHAPETYPE.CONSTANT_CURVATURE:
            pmat, Rmat = self.ss_needle.get_needle_shape()

        else:
            self.get_logger().error( f"Needle shape type: {self.ss_needle.current_shapetype} is not implemented." )
            self.get_logger().error( f"Resorting to shape type: {NEEDLESHAPETYPE.SINGLEBEND_SINGLELAYER}." )
            self.ss_needle.update_shapetype( NEEDLESHAPETYPE.SINGLEBEND_SINGLELAYER )
            pmat, Rmat = None, None  # pop out of the loop and redo

        # else

        if (pmat is None) and (Rmat is None):
            return pmat, Rmat

        # generate the straight length section
        L_needle      = utilities.calculate_needle_length(pmat)
        dL            = self.ss_needle.length - L_needle
        pmat_straight = np.zeros((1, 3), dtype=pmat.dtype)
        if dL > self.ss_needle.ds:
            # generate straight needle length in ds increments
            L_straight = np.arange( 0, (dL // self.ss_needle.ds + 1) * self.ss_needle.ds, self.ss_needle.ds )

            # generate straight needle shape
            pmat_straight       = np.zeros( (len(L_straight), 3), dtype=pmat.dtype )
            pmat_straight[:, 2] = L_straight

        # if
        elif dL > 0:  # less than ds increment
            pmat_straight        = np.zeros((2, 3), dtype=pmat.dtype)
            pmat_straight[-1, 2] = dL

        # elif
        Rmat_straight = np.tile(
            np.eye(3, dtype=Rmat.dtype),
            (pmat_straight.shape[0], 1, 1)
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
            if not (self.entrypoint_received and self.needlepose_received):
                self.get_logger().debug("Manual mode active --- waiting for entry point and needle pose data")
                return
        ####End Change
        
        pmat, Rmat = self.get_needleshape()

        # update initial kappa_c values
        self.kc_i     = self.ss_needle.current_kc
        self.w_init_i = self.ss_needle.current_winit

        # check to make sure messages are not None

        if pmat is None or Rmat is None:
            # self.get_logger().warn( f"pmat or Rmat is None: pmat={pmat}, Rmat={Rmat}" )
            # self.get_logger().warn( f"Current shapetype: {self.ss_needle.current_shapetype}" )
            return

        # if

        # needle shape length
        needle_L = np.linalg.norm( np.diff( pmat, axis=0 ), 2, 1 ).sum()
        self.get_logger().debug(
                f"Needle L: {self.ss_needle.length} | Needle Shape L: {needle_L} | Current Depth: {self.ss_needle.current_depth}" )

        # generate pose message
        header    = Header( stamp=self.get_clock().now().to_msg(), frame_id='needle' )
        msg_shape = utilities.poses2msg( pmat, Rmat, header=header )

        # generate kappa_c and w_init message
        msg_kc    = Float64MultiArray( data=self.kc_i )
        msg_winit = Float64MultiArray( data=self.w_init_i.tolist() )
        msg_depth = Float64(data=float(self.insertion_depth))

        self.get_logger().debug( f"Needle Shapes: {pmat.shape}, {Rmat.shape}, {len( msg_shape.poses )}" )

        # publish the messages
        self.pub_shape.publish( msg_shape )
        self.pub_kc.publish( msg_kc )
        self.pub_winit.publish( msg_winit )
        self.pub_depth.publish( msg_depth )
        self.get_logger().debug(
            "Published needle shape, kappa_c and w_init on topics: "
            f"{self.pub_shape.topic},{self.pub_kc.topic},{self.pub_winit.topic}"
        )

    # publish_shape

    async def publish_shape_async(self):
        """ Function to asynchronously publish the needle shape """
        # TODO
        pass

    # publish_shape_async

    def sub_curvatures_callback( self, msg: Float64MultiArray ):
        """ Subscription to needle sensor curvatures """
        # grab the current curvatures
        curvatures = np.reshape( msg.data, (2, -1), order='F' )

        self.get_logger().debug(f"Curvatures X: {curvatures[0]}")
        self.get_logger().debug(f"Curvatures Y: {curvatures[1]}")

        # update the curvatures
        self.ss_needle.current_curvatures = curvatures

        if not self.ss_needle.is_calibrated:
            self.ss_needle.ref_wavelengths = np.ones_like( self.ss_needle.ref_wavelengths )

        # if

    # sub_curvatures_callback

    def sub_entrypoint_callback( self, msg: Point ):
        """ Subscription to entrypoint topic """
        insertion_point = np.array( [ msg.x, msg.y, msg.z ] )  # assume it is in the

        # update the insertion point relative to the initial base of the insertion point
        self.ss_needle.insertion_point = (
            insertion_point
            - self.needle_guide_exit_pt * [1, 1, 0]
        )

        ####Change
        self.entrypoint_received = True
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

        # update the insertion depth (y-coordinate is the insertion depth)
        self.insertion_depth = max(
            0,
            min(
                self.current_needle_pose[0][2] - self.ss_needle.insertion_point[2], # z-axis
                self.ss_needle.length
            )
        )

        ####Change
        self.needlepose_received = True
        ####End Change
        
        self.get_logger().debug( f"Current insertion depth: {self.insertion_depth}" )

        # update the history of orientations (NOT USED YET)
        depth_ds = msg.pose.position.z - msg.pose.position.z % self.ss_needle.ds
        theta    = msg.pose.orientation.z
        if np.any( self.history_needle_pose[ 0 ] == depth_ds ):  # check if we already have this value
            idx = np.argwhere( self.history_needle_pose[ 0 ] == depth_ds ).ravel()
            self.history_needle_pose[ 1, idx ] = theta

        # if
        else:  # add a new value
            np.hstack( (self.history_needle_pose, [ [ depth_ds ], [ theta ] ]) )

        # else

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


# class: ShapeSensingNeedleNode

def main( args=None ):
    rclpy.init( args=args )

    ssneedle_node = ShapeSensingNeedleNode()

    try:
        rclpy.spin( ssneedle_node )

    except KeyboardInterrupt:
        pass

    # clean-up
    ssneedle_node.destroy_node()
    rclpy.shutdown()


# main

if __name__ == "__main__":
    main()

# if __main__
