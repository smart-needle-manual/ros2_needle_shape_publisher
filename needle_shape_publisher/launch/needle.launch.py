import sys, os
import json
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource

pkg_needle_shape_publisher = get_package_share_directory('needle_shape_publisher')

def determineAAs(needleParamFile: str):
    """ Determine the number of channels and active areas available """
    with open(needleParamFile, 'r') as paramFile:
        params = json.load(paramFile) 

    # number of channels is completely unnecessary.
    numAAs = params['# active areas']

    return numAAs

# determineCHsAAs

def generate_launch_description():
    ld = LaunchDescription()

    # determine #chs and numAAs
    numAAs = None
    # TO-DO: Default needs correction based on number and placement of active areas.
    default_needleparam_file = "3CH-4AA-0005_needle_params_2022-01-26_Jig-Calibration_best_weights.json"
    for arg in sys.argv:
        if arg.startswith("needleParamFile:="):
            needleParamFile = arg.split(":=")[1]
            break
        # if        
    # for 

    if numAAs is None: # just in-case using default value
        needleParamFile = default_needleparam_file

    # numAAs = determineAAs(os.path.join(pkg_needle_shape_publisher, "needle_data", needleParamFile))

    # arguments
    arg_params = DeclareLaunchArgument( 'needleParamFile',
                                        default_value=default_needleparam_file,
                                        description="The shape-sensing needle parameter json file." )

    # arg_num_signals is deprecated.

    # arg_optim_maxiter is deprecated.

    # arg_temp_compensate is deprecated.
    
    arg_optim_update_ornt_airgap = DeclareLaunchArgument(
        'optimNeedleUpdateOrientationAirGap',
        default_value="True",
        description="Whether to update the needle's tissue orientation based on estimated air gap orientation",
    )

    ####Change
    # arg_manual_mode is deprecated.
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

    # included launch arguments
    ld_needlepub = IncludeLaunchDescription( # needle shape publisher
        PythonLaunchDescriptionSource(
        os.path.join(pkg_needle_shape_publisher, 'sensorized_shapesensing_needle_decomposed.launch.py')),
        launch_arguments = {
            'needleParamFile'                   : PathJoinSubstitution([pkg_needle_shape_publisher, 'needle_data', LaunchConfiguration('needleParamFile')]),
            'optimNeedleUpdateOrientationAirGap': LaunchConfiguration('optimNeedleUpdateOrientationAirGap'),
            ####Change
            'shape_type'                 : LaunchConfiguration('shape_type'),
            ####End Change
        }.items()
    )
    # configure launch description
    ld.add_action(arg_params)
    ld.add_action(arg_optim_update_ornt_airgap)
    ld.add_action(arg_shape_type)

    ld.add_action(ld_needlepub)

    return ld

# generate_launch_descrtiption
