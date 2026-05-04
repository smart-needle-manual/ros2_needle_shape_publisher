from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    # arguments
    arg_needleparam = DeclareLaunchArgument( 'needleParamFile',
                                             description="The shape-sensing needle parameter json file." )
    arg_numsignals = DeclareLaunchArgument( 'numSignals', description="The number of FBG signals to collect.",
                                            default_value="200" )
    arg_optim_maxiter = DeclareLaunchArgument( 'optimMaxIterations', default_value="15",
                                               description="The maximum number of iterations for needle shape optimizer." )
    arg_temp_compensate = DeclareLaunchArgument( 'tempCompensate', default_value="True",
                                               description="Whether to perform temperature compensation or not." )
    
    arg_optim_update_ornt_airgap = DeclareLaunchArgument(
        'optimNeedleUpdateOrientationAirGap',
        default_value="True",
        description="Whether to update the needle's tissue orientation based on estimated air gap orientation",
    )

    ####Change
    arg_manual_mode = DeclareLaunchArgument(
        'manual_mode',
        default_value='false',
        description='Enable manual trigger mode for ShapeSensingNeedle node'
    )
    ####End Change

    # Nodes
    node_sensorizedneedle = Node(
            package='needle_shape_publisher',
            namespace='needle',
            executable='sensorized_needle',
            output='screen',
            emulate_tty=True,
            parameters=[ {
                    'needle.paramFile'                   : LaunchConfiguration( 'needleParamFile' ),
                    'needle.sensor.numberSignals'        : LaunchConfiguration( 'numSignals' ),
                    'needle.sensor.temperatureCompensate': LaunchConfiguration( 'tempCompensate' ),
                    } ]
            )
    node_ssneedle = Node(
            package='needle_shape_publisher',
            namespace='needle',
            executable='shapesensing_needle',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'needle.paramFile'                               : LaunchConfiguration( 'needleParamFile' ),
                'needle.optimizer.max_iterations'                : LaunchConfiguration( 'optimMaxIterations' ),
                'needle.optimizer.update_orientation_with_airgap': LaunchConfiguration( 'optimNeedleUpdateOrientationAirGap' ),
                ####Change
                'needle.manual_mode'                             : LaunchConfiguration('manual_mode'),
                ####End Change
            }],
        )

    # add to launch description
    ld.add_action( arg_needleparam )
    ld.add_action( arg_numsignals )
    ld.add_action( arg_optim_maxiter )
    ld.add_action( arg_temp_compensate )
    ld.add_action( arg_optim_update_ornt_airgap )
    ld.add_action( arg_manual_mode )

    ld.add_action( node_sensorizedneedle )
    ld.add_action( node_ssneedle )

    return ld

# generate_launch_description
