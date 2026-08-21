import numpy as np
from typing import List
# ROS2 packages
import rclpy
from rclpy.parameter import Parameter
# messages
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult

# custom package
from .sensorized_shape_sensing_needle import NeedleNode
from . import utilities


class SensorizedNeedleNode( NeedleNode ):
    """Needle to handle sensor calibration/curvatures"""

    def __init__( self, name="SensorizedNeedle" ):
        super().__init__( name )

        # declare and get parameters

        # create publishers
        self.pub_curvatures = self.create_publisher( Float64MultiArray, 'state/curvatures', 10 )

        # create timers
        self.pub_curvatures_timer = self.create_timer( 0.01, self.publish_curvatures )

        # set parameter callback function
        self.add_on_set_parameters_callback( self.parameters_callback )

    # __init__

    def parameters_callback( self, parameters: List[ Parameter ] ):
        """ Parameter set calllbacks"""
        PASS
        # for

        return SetParametersResult( successful=successful, reason="\n".join( reasons ) )

    # parameters_callback

    def publish_curvatures( self ):
        """ Publish the curvatures of the shape-sensing needle"""
        # current_curvatures are N x 2 ( columns are: x,  y ) -> ravel('F') -> (X_AA1, X_AA2, ..., Y_AA1, Y_AA2,...)
        curvatures = self.ss_needle.current_curvatures.ravel( order='F' )

        self.get_logger().debug(f"Curvatures: {self.ss_needle.current_curvatures}")
        itemsize = self.ss_needle.current_curvatures.dtype.itemsize
        dimx = MultiArrayDimension(
                label="x", stride=itemsize,
                size=self.ss_needle.current_curvatures.shape[ 0 ] * itemsize )
        dimy = MultiArrayDimension(
                label="y", stride=itemsize,
                size=self.ss_needle.current_curvatures.shape[ 0 ] * itemsize )

        msg = Float64MultiArray(
                data=curvatures.tolist(), 
                layout=MultiArrayLayout( dim=[ dimx, dimy ] ) )

        self.pub_curvatures.publish( msg )

    # publish_curvatures

# class: SensorizedNeedleNode

def main( args=None ):
    rclpy.init( args=args )

    sensorized_needle_node = SensorizedNeedleNode()

    try:
        rclpy.spin( sensorized_needle_node )

    except KeyboardInterrupt:
        pass

    # clean-up
    sensorized_needle_node.destroy_node()
    rclpy.shutdown()


# main

if __name__ == "__main__":
    main()

# if __main__
