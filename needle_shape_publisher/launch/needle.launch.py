import sys, os
import json
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource

pkg_needle_shape_publisher = get_package_share_directory('needle_shape_publisher')

def determineCHsAAs(needleParamFile: str):
    """ Determine the number of channels and active areas available """
    with open(needleParamFile, 'r') as paramFile:
        params = json.load(paramFile) 

    # with
    numChs = params['# channels']
    numAAs = params['# active areas']

    return numChs, numAAs

# determineCHsAAs

def generate_launch_description():
    ld = LaunchDescription()

    # determine #chs and numAAs
    numCHs, numAAs = None, None
    default_needleparam_file = "3CH-4AA-0005_needle_params_2022-01-26_Jig-Calibration_best_weights.json"
    for arg in sys.argv:
        if arg.startswith("needleParamFile:="):
            needleParamFile = arg.split(":=")[1]
            break
        # if        
    # for 

    if numCHs is None and numAAs is None: # just in-case using default value
        needleParamFile = default_needleparam_file

    # numCHs, numAAs = determineCHsAAs(os.path.join(pkg_needle_shape_publisher, "needle_data", needleParamFile))

    # arguments
    arg_params = DeclareLaunchArgument( 'needleParamFile',
                                        default_value=default_needleparam_file,
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
        description='Enable manual trigger mode for ShapeSensingNeedleNode'
    )
    ####End Change

    # included launch arguments
    ld_needlepub = IncludeLaunchDescription( # needle shape publisher
        PythonLaunchDescriptionSource(
        os.path.join(pkg_needle_shape_publisher, 'sensorized_shapesensing_needle_decomposed.launch.py')),
        launch_arguments = {
            'needleParamFile'                   : PathJoinSubstitution([pkg_needle_shape_publisher, 'needle_data', LaunchConfiguration('needleParamFile')]),
            'numSignals'                        : LaunchConfiguration('numSignals'),
            'optimMaxIterations'                : LaunchConfiguration('optimMaxIterations'),
            'tempCompensate'                    : LaunchConfiguration('tempCompensate'),
            'optimNeedleUpdateOrientationAirGap': LaunchConfiguration('optimNeedleUpdateOrientationAirGap'),
            ####Change
            'manual_mode'                : LaunchConfiguration('manual_mode'),
            ####End Change
        }.items()
    )
    # configure launch description
    ld.add_action(arg_params)
    ld.add_action(arg_numsignals)
    ld.add_action(arg_optim_maxiter)
    ld.add_action(arg_temp_compensate)
    ld.add_action(arg_optim_update_ornt_airgap)
    ld.add_action(arg_manual_mode)

    ld.add_action(ld_needlepub)

    return ld

# generate_launch_descrtiption
