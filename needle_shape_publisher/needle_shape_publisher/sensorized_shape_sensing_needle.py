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
from needle_shape_sensing import geometry
from needle_shape_sensing.intrinsics import SHAPETYPE as NEEDLESHAPETYPE
from needle_shape_sensing.shape_sensing import ShapeSensingFBGNeedle

class NeedleNode(Node):
    # PARAMETER NAMES
    PARAM_NEEDLE = "needle"

    # - needle parameters
    PARAM_NEEDLEPARAMFILE = ".".join( [ PARAM_NEEDLE, "paramFile" ] )  # needle parameter json file
    PARAM_NEEDLESN = ".".join( [ PARAM_NEEDLE, "serialNumber" ] )  # needle serial number
    PARAM_NEEDLELENGTH = ".".join( [ PARAM_NEEDLE, "length" ] )  # needle length
    PARAM_CHS = ".".join( [ PARAM_NEEDLE, "channels" ] )  # needle number of channels
    PARAM_AAS = ".".join( [ PARAM_NEEDLE, "activeAreas" ] )  # needle number of active areas
    PARAM_AAWEIGHTS = ".".join( [ PARAM_AAS, "weights" ] )  # needle AA reliability weightings
    PARAM_SLOCS = ".".join( [ PARAM_AAS, 'locations' ] )  # needle AA locations from tip of the needle
    PARAM_NEEDLESHAPE = ".".join( (PARAM_NEEDLE, "shape_type") )  # needle shape type

    def __init__(self, name='Needle'):
        super().__init__(name)
        # declare and get parameters
        pd_needleparam = ParameterDescriptor( name=self.PARAM_NEEDLEPARAMFILE, type=Parameter.Type.STRING.value,
                                              description='needle parameter json file.', read_only=True )
        needleparam_file = self.declare_parameter( pd_needleparam.name, descriptor=pd_needleparam ). \
            get_parameter_value().string_value

        # get shape-sensing FBG Needle object
        try:
            self.ss_needle = ShapeSensingFBGNeedle.load_json( needleparam_file )

            self.get_logger().info( "Successfully loaded FBG Needle: \n" + str( self.ss_needle ) )

        # try
        except Exception as e:
            self.get_logger().error( f"{needleparam_file} is not a valid needle parameter file." )
            raise e

        # except

        # set (read-only) needle parameters
        pd_ndllen = ParameterDescriptor( name=self.PARAM_NEEDLELENGTH, type=Parameter.Type.DOUBLE.value,
                                         description="The length of the needle.", read_only=True )
        pd_ndlsn = ParameterDescriptor( name=self.PARAM_NEEDLESN, type=Parameter.Type.STRING.value,
                                        description="The serial number of the shape-sensing needle",
                                        read_only=True )
        pd_numchs = ParameterDescriptor( name=self.PARAM_CHS, type=Parameter.Type.INTEGER.value,
                                         description="Number of Channels in the FBG-sensorized needle",
                                         read_only=True )
        pd_numaas = ParameterDescriptor( name=self.PARAM_AAS, type=Parameter.Type.INTEGER.value,
                                         description="Number of Activa Areas in the FBG-sensorized needle",
                                         read_only=True )
        pd_slocs = ParameterDescriptor( name=self.PARAM_SLOCS, type=Parameter.Type.DOUBLE_ARRAY.value,
                                        description="Location of the active areas in (mm) from the tip of the needle",
                                        read_only=True )
        # - AA reliability weightings
        if len( self.ss_needle.weights ) > 0:
            w = list( self.ss_needle.weights.values() )
        else:
            w = (np.ones( self.ss_needle.num_activeAreas ) / self.ss_needle.num_activeAreas).tolist()

        pd_aawgts = ParameterDescriptor( name=self.PARAM_AAWEIGHTS, type=Parameter.Type.DOUBLE_ARRAY.value,
                                         description="Active Area reliability weightings", read_only=True )

        # - declarations
        self.declare_parameter( self.PARAM_NEEDLESN, descriptor=pd_ndlsn, value=self.ss_needle.serial_number )
        self.declare_parameter( self.PARAM_NEEDLELENGTH, descriptor=pd_ndllen, value=self.ss_needle.length )
        self.declare_parameter( self.PARAM_CHS, descriptor=pd_numchs, value=self.ss_needle.num_channels )
        self.declare_parameter( self.PARAM_AAS, descriptor=pd_numaas, value=self.ss_needle.num_activeAreas )
        self.declare_parameter( self.PARAM_SLOCS, descriptor=pd_slocs, value=self.ss_needle.sensor_location_tip.tolist() )
        self.declare_parameter( self.PARAM_AAWEIGHTS, descriptor=pd_aawgts, value=w )

        # Declare a log_level parameter for runtime verbosity control
        log_level_str = self.declare_parameter(
            "log_level", "INFO"
        ).get_parameter_value().string_value
        log_level = getattr(
            rclpy.logging.LoggingSeverity,
            log_level_str.upper(),
            rclpy.logging.LoggingSeverity.INFO,
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

    ssneedle_node = SensorizedShapeSensingNeedleNode()

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
