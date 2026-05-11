import sys
import os
import json
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription, actions, conditions
from launch.substitutions.launch_configuration import LaunchConfiguration
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import (
    PythonExpression,
    TextSubstitution,
    PathJoinSubstitution,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource

pkg_hyperion_interrogator = get_package_share_directory('hyperion_interrogator')
pkg_needle_shape_publisher = get_package_share_directory('needle_shape_publisher')


# Determine numChs and numAAs from needleParamFile
def determineCHsAAs(needleParamFile: str):
    """Determine the number of channels and active areas available."""
    with open(needleParamFile, 'r') as paramFile:
        params = json.load(paramFile)
    numChs = params['# channels']
    numAAs = params['# active areas']
    return numChs, numAAs


def generate_launch_description():
    ld = LaunchDescription()

    # Set numChs and numAAs
    numCHs, numAAs = 3, 4
    for arg in sys.argv:
        if arg.startswith('needleParamFile:='):
            needleParamFile = arg.split(':=')[1]
            numCHs, numAAs = determineCHsAAs(needleParamFile)

    # Arguments
    arg_simlevel = DeclareLaunchArgument(
        'sim_level',
        default_value='1',
        description='Simulation level: 1 - shape-driven virtual sensors, 2 - real sensors'
    )
    arg_params = DeclareLaunchArgument(
        'needleParamFile',
        default_value='3CH-4AA-0005_needle_params_2022-01-26_Jig-Calibration_best_weights.json',
        description='The shape-sensing needle parameter json file'
    )
    arg_interrIP = DeclareLaunchArgument(
        'interrogatorIP',
        default_value='10.0.0.55',
        description='Interrogator IP'
    )
    arg_manual_mode = DeclareLaunchArgument(
        'manual_mode',
        default_value='false',
        description='Enable manual trigger mode for ShapeSensingNeedleNode'
    )
    arg_shape_type = DeclareLaunchArgument(
        'shape_type',
        default_value='-1',
        description=(
            'Shape type used by ShapeSensingNeedleNode '
            '(integer value of needle_shape_sensing.intrinsics.SHAPETYPE; '
            '-1 means use the value from the needle parameter file).'
        )
    )
    # Shape file for sim_level=1 shape-driven simulation
    arg_sim_shape_file = DeclareLaunchArgument(
        'sim_shape_file',
        default_value=PathJoinSubstitution([
            pkg_needle_shape_publisher, 'needle_data', 'sim_shape_default.yaml'
        ]),
        description=(
            'Path to YAML/JSON shape file used by sim_level=1 to generate '
            'shape-driven FBG wavelength shifts. '
            'Format: {shape: [[x0,y0,z0], [x1,y1,z1], ...]} (mm).'
        )
    )
    # Insertion depth used when computing curvatures from the shape file
    arg_sim_insertion_depth = DeclareLaunchArgument(
        'sim_insertion_depth',
        default_value='100.0',
        description=(
            'Current insertion depth (mm) used together with the shape file '
            'to locate sensor active areas along the polyline.'
        )
    )

    num_signals_to_collect = 50

    # Needle shape publisher
    ld_needlepub = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_needle_shape_publisher, 'needle.launch.py')),
        launch_arguments={
            'needleParamFile': LaunchConfiguration(arg_params.name),
            'numSignals': TextSubstitution(text=str(num_signals_to_collect)),
            'optimNeedleUpdateOrientationAirGap': TextSubstitution(text='False'),
            'manual_mode': LaunchConfiguration('manual_mode'),
            'shape_type': LaunchConfiguration('shape_type'),
            # Disable temperature compensation: demo signals have no temperature
            # channel, so enabling it zeros out the processed wavelength shifts.
            'tempCompensate': TextSubstitution(text='False'),
        }.items()
    )

    # Hyperion Interrogator – shape-driven virtual demo (sim_level=1)
    ld_hyperiondemo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_hyperion_interrogator, 'hyperion_demo.launch.py')),
        condition=conditions.IfCondition(
            PythonExpression([LaunchConfiguration('sim_level'), ' == 1'])
        ),
        launch_arguments={
            'ip': LaunchConfiguration('interrogatorIP'),
            'numCH': TextSubstitution(text=str(numCHs)),
            'numAA': TextSubstitution(text=str(numAAs)),
            'numSamples': TextSubstitution(text=str(num_signals_to_collect)),
            'needleParamFile': PathJoinSubstitution([
                pkg_needle_shape_publisher, 'needle_data',
                LaunchConfiguration(arg_params.name)
            ]),
            # New sim_level=1 shape-driven parameters
            'sim_shape_file': LaunchConfiguration('sim_shape_file'),
            'sim_insertion_depth': LaunchConfiguration('sim_insertion_depth'),
        }.items()
    )

    ld_hyperionstream = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_hyperion_interrogator, 'hyperion_streamer.launch.py')),
        condition=conditions.IfCondition(
            PythonExpression([LaunchConfiguration('sim_level'), ' == 2'])
        ),
        launch_arguments={
            'ip': LaunchConfiguration('interrogatorIP'),
            'numSamples': TextSubstitution(text=str(num_signals_to_collect)),
            'needleParamFile': PathJoinSubstitution([
                pkg_needle_shape_publisher, 'needle_data',
                LaunchConfiguration(arg_params.name)
            ]),
        }.items()
    )

    # Add to launch description
    ld.add_action(arg_simlevel)
    ld.add_action(arg_params)
    ld.add_action(arg_interrIP)
    ld.add_action(arg_manual_mode)
    ld.add_action(arg_shape_type)
    ld.add_action(arg_sim_shape_file)
    ld.add_action(arg_sim_insertion_depth)

    ld.add_action(ld_needlepub)
    ld.add_action(ld_hyperiondemo)
    ld.add_action(ld_hyperionstream)

    # ---------------------------------------------------------------------------
    # Sim-only helpers: publish the two topics that ShapeSensingNeedleNode needs
    # ---------------------------------------------------------------------------

    # Continuously publish /stage/state/needle_pose so that the node computes a
    # non-zero insertion depth.  In this convention pose.position.z encodes the
    # insertion depth state; the publisher keeps the shape in the guide frame and
    # does not re-apply this z value as an additional world-frame translation.
    needle_pose_pub = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--rate', '10',
            '/stage/state/needle_pose',
            'geometry_msgs/msg/PoseStamped',
            [
                '{"header": {"frame_id": "needle"}, '
                '"pose": {"position": {"x": 0.0, "y": 0.0, "z": ',
                LaunchConfiguration('sim_insertion_depth'),
                '}, "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}}',
            ],
        ],
        output='screen',
    )

    # Auto-call the sensor calibrate service so that /needle/sensor/processed
    # starts flowing without manual intervention.  A short delay ensures the
    # hyperion_demo node is fully up before the call is made.
    # NOTE: the service name 'sensor/calibrate' must match the name registered
    #       in HyperionPublisher.  Verify against the hyperion_interrogator
    #       package if a different name is used.
    calibrate_service_call = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'service', 'call',
                    '/needle/sensor/calibrate',
                    'std_srvs/srv/Trigger',
                    '{}',
                ],
                output='screen',
            ),
        ],
    )

    ld.add_action(needle_pose_pub)
    ld.add_action(calibrate_service_call)

    return ld
