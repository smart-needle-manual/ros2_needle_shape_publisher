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
from launch_ros.actions import Node

# The control flow will be self-contained. Everything happens in ros2_needle_shape_publisher/needle_shape_publisher.
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
    # There will only be simulated readings in this branch.
    # json needs to be written and updated in accordance with active area placements.
    arg_params = DeclareLaunchArgument(
        'needleParamFile',
        default_value='3CH-4AA-0005_needle_params_2022-01-26_Jig-Calibration_best_weights.json',
        description='The shape-sensing needle parameter json file'
    )
    # An interrogator IP is no longer needed as the computer is completely ineffectual in simulation.
    # manual_mode is deprecated. Topics will be published to directly.
    # shape_type will be corrected TO-DO: auto-assign to piecewise_exp shape type
    arg_shape_type = DeclareLaunchArgument(
        'shape_type',
        default_value='-1',
        description=(
            'Shape type used by ShapeSensingNeedleNode '
            '(integer value of needle_shape_sensing.intrinsics.SHAPETYPE; '
            '-1 means use the value from the needle parameter file).'
        )
    )
    # 3D Slicer will provide the needed topics directly following post-processing.
    
    # num_signals_to_collect will be phased out; topics will be bagged until info needed provided, then the publisher will trigger. Else, at speed, info stored is not info calculated.

    # Needle shape publisher
    # Temperature compensation is deprecated for simulation.
    ld_needlepub = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_needle_shape_publisher, 'needle.launch.py')),
        launch_arguments={
            'needleParamFile': LaunchConfiguration(arg_params.name),
            'optimNeedleUpdateOrientationAirGap': TextSubstitution(text='False'),
            'shape_type': LaunchConfiguration('shape_type'),
        }.items()
    )

    # Hyperion Interrogator is deprecated for simulation.

    # Add to launch description
    ld.add_action(arg_params)
    ld.add_action(arg_shape_type)

    ld.add_action(ld_needlepub)

    return ld
