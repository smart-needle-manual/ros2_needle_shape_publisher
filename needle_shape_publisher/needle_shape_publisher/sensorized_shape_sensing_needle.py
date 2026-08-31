import numpy as np
from typing import List
# ROS2 packages
import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.parameter import Parameter

# messages
from std_msgs.msg import Header, Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from geometry_msgs.msg import Point, Pose, PoseArray
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult

# services

# custom package
from needle_shape_sensing.intrinsics import SHAPETYPE as NEEDLESHAPETYPE
from needle_shape_sensing.shape_sensing import ShapeSensingFBGNeedle

class NeedleNode(Node):
    # PARAMETER NAMES
    PARAM_NEEDLE = "needle"

    # - needle parameters
    PARAM_NEEDLELENGTH = ".".join( [ PARAM_NEEDLE, "length" ] )  # needle length
    PARAM_AAS = ".".join( [ PARAM_NEEDLE, "activeAreas" ] )  # needle number of active areas
    PARAM_SLOCS = ".".join( [ PARAM_AAS, 'locations' ] )  # needle AA locations from tip of the needle
    PARAM_NEEDLESHAPE = ".".join( (PARAM_NEEDLE, "shape_type") )  # needle shape type
    PARAM_NEEDLE_FILE = ".".join( [PARAM_NEEDLE, "param_file"] )   # path to needle JSON

    def __init__(self, name='Needle'):
        super().__init__(name)
        # Load needle from JSON file (required parameter)
        needle_file = self.declare_parameter(
            self.PARAM_NEEDLE_FILE, value=""
        ).get_parameter_value().string_value

        if not needle_file:
            self.get_logger().fatal(
                f"Parameter '{self.PARAM_NEEDLE_FILE}' is required. "
                "Provide the path to a needle parameter JSON file."
            )
            raise RuntimeError(f"Missing required parameter: {self.PARAM_NEEDLE_FILE}")

        self.get_logger().info(f"Loading needle parameters from: {needle_file}")
        self.ss_needle: ShapeSensingFBGNeedle = ShapeSensingFBGNeedle.load_json(needle_file)
        self.get_logger().info(
            f"length={self.ss_needle.length} mm, "
            f"activeAreas={self.ss_needle.num_activeAreas}"
        )


        # set (read-only) needle parameters
        pd_ndllen = ParameterDescriptor( name=self.PARAM_NEEDLELENGTH, type=Parameter.Type.DOUBLE.value,
                                         description="The length of the needle.", read_only=True )
        pd_numaas = ParameterDescriptor( name=self.PARAM_AAS, type=Parameter.Type.INTEGER.value,
                                         description="Number of Activa Areas in the FBG-sensorized needle",
                                         read_only=True )
        pd_slocs = ParameterDescriptor( name=self.PARAM_SLOCS, type=Parameter.Type.DOUBLE_ARRAY.value,
                                        description="Location of the active areas in (mm) from the tip of the needle",
                                        read_only=True )

        # - declarations
        self.declare_parameter( self.PARAM_NEEDLELENGTH, descriptor=pd_ndllen, value=self.ss_needle.length )
        self.declare_parameter( self.PARAM_AAS, descriptor=pd_numaas, value=self.ss_needle.num_activeAreas )
        self.declare_parameter( self.PARAM_SLOCS, descriptor=pd_slocs, value=self.ss_needle.sensor_location_tip.tolist() )

        # Declare a log_level parameter for runtime verbosity control
        log_level_str = self.declare_parameter(
            "log_level", "DEBUG"
        ).get_parameter_value().string_value
        log_level = getattr(
            rclpy.logging.LoggingSeverity,
            log_level_str.upper(),
            rclpy.logging.LoggingSeverity.DEBUG,
        )
        self.get_logger().set_level(log_level)

    # __init__

    def destroy_node( self ) -> bool:
        """ Destroy the node override"""
        self.get_logger().info( "Shutting down..." )
        retval = super().destroy_node()
        self.get_logger().info( "Shut down complete." )
        return retval

    # destroy_node

# class: NeedleNode

def main( args=None ):
    rclpy.init( args=args )

    ssneedle_node = NeedleNode()

    try:
        rclpy.spin( ssneedle_node )

    except KeyboardInterrupt:
        pass

    # clean up
    ssneedle_node.destroy_node()
    rclpy.shutdown()


# main


if __name__ == "__main__":
    main()

# if: main
