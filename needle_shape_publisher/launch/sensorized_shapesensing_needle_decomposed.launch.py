from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    # arguments
    arg_needleparam = DeclareLaunchArgument( 'needleParamFile',
                                             description="The shape-sensing needle parameter json file." )

    ####Change
    arg_shape_type = DeclareLaunchArgument(
        'shape_type',
        default_value='-1',
        description=(
            'Shape type used by ShapeSensingNeedleNode '
            '(integer value of needle_shape_sensing.intrinsics.SHAPETYPE; '
            '-1 means use the value from the needle parameter file).'
        )
    )
    ####End Change

    # Nodes
    # node_sensorizedneedle is deprecated - node_ssnneedle will eceive the curvatures in ravel-order directly, back from tip to base.
    node_ssneedle = Node(
            package='needle_shape_publisher',
            namespace='needle',
            executable='shapesensing_needle',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'needle.param_file'                               : LaunchConfiguration( 'needleParamFile' ),
                ####Change
                'needle.shape_type'                              : LaunchConfiguration('shape_type'),
                ####End Change
                # Skin-entry will be viewed only from the lens of insertion depth in the absence of a stator.
                # Any new inserted args should be overriden on the command line with:
                #   ros2 param set /needle/shapesensing_needle \
                #       {name} [value]
            }],
        )

    # add to launch description
    ld.add_action( arg_needleparam )
    ld.add_action( arg_shape_type )

    ld.add_action( node_ssneedle )

    return ld

# generate_launch_description
