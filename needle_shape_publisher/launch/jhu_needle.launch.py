import sys
import os
import json
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription, actions, conditions
from launch.substitutions.launch_configuration import LaunchConfiguration
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import PythonExpression, LocalSubstitution, TextSubstitution, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource

pkg_hyperion_interrogator  = get_package_share_directory('hyperion_interrogator')
pkg_needle_shape_publisher = get_package_share_directory('needle_shape_publisher')

# Determine numChs and numAAs from needleParamFile
def determineCHsAAs(needleParamFile: str):
    """ Determine the number of channels and active areas available """
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
        default_value ='2',
        description = 'Simulation level: 1 - virtual sensors (demo), 2 - real sensors'
    )
    arg_params = DeclareLaunchArgument(
        'needleParamFile',
        default_value = '3CH-4AA-0005_needle_params_2022-01-26_Jig-Calibration_best_weights.json',
        description = 'The shape-sensing needle parameter json file' 
    )                                 
    arg_interrIP = DeclareLaunchArgument(
        'interrogatorIP', 
        default_value = '10.0.0.55',
        description = "Interrogator IP" 
    )

    ####Change
    arg_manual_mode = DeclareLaunchArgument(
        'manual_mode',
        default_value = 'false',
        description = 'Enable manual trigger mode for ShapeSensingNeedleNode'
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
    ####End Change
    
    num_signals_to_collect = 50

    # Needle shape publisher
    ld_needlepub = IncludeLaunchDescription( # needle shape publisher
        PythonLaunchDescriptionSource(
        os.path.join(pkg_needle_shape_publisher, 'needle.launch.py')),
        launch_arguments = {
            'needleParamFile'                   : LaunchConfiguration(arg_params.name),
            'numSignals'                        : TextSubstitution(text=str(num_signals_to_collect)),
            'optimNeedleUpdateOrientationAirGap': TextSubstitution(text="False"),
            ####Change
            'manual_mode'                : LaunchConfiguration('manual_mode'),
            'shape_type'                 : LaunchConfiguration('shape_type'),
            ####End Change
        }.items()
    )

    # Hyperion Interrogator
    ld_hyperiondemo = IncludeLaunchDescription( # Virtual demo (sim_level_needle = 1)
        PythonLaunchDescriptionSource(os.path.join(pkg_hyperion_interrogator, 'hyperion_demo.launch.py')),
        condition = conditions.IfCondition(
            PythonExpression([LaunchConfiguration('sim_level'), " == 1"])
        ),
        launch_arguments = {
            'ip'             : LaunchConfiguration('interrogatorIP'),
            'numCH'          : TextSubstitution(text=str(numCHs)), 
            'numAA'          : TextSubstitution(text=str(numAAs)),
            'numSamples'     : TextSubstitution(text=str(num_signals_to_collect)),
            'needleParamFile': PathJoinSubstitution([pkg_needle_shape_publisher, 'needle_data', LaunchConfiguration(arg_params.name)]),
        }.items()
    )

    ld_hyperionstream = IncludeLaunchDescription( # Real hardware (sim_level_needle = 2)
        # PythonLaunchDescriptionSource(os.path.join(pkg_hyperion_interrogator, 'hyperion_talker.launch.py')),
        PythonLaunchDescriptionSource(os.path.join(pkg_hyperion_interrogator, 'hyperion_streamer.launch.py')),
        condition=conditions.IfCondition(
            PythonExpression([LaunchConfiguration('sim_level'), " == 2"])
        ),
        launch_arguments = {
            'ip'             : LaunchConfiguration('interrogatorIP'),
            'numSamples'     : TextSubstitution(text=str(num_signals_to_collect)),
            'needleParamFile': PathJoinSubstitution([pkg_needle_shape_publisher, 'needle_data', LaunchConfiguration(arg_params.name)]),
        }.items()
    )

    # Add to launch description
    ld.add_action(arg_simlevel)
    ld.add_action(arg_params)
    ld.add_action(arg_interrIP)
    ld.add_action(arg_manual_mode)
    ld.add_action(arg_shape_type)
   
    ld.add_action(ld_needlepub)
    ld.add_action(ld_hyperiondemo)
    ld.add_action(ld_hyperionstream)    

    return ld
