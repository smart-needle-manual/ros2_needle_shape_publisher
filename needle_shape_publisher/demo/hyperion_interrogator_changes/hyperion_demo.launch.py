from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    # arguments
    hyperion_ip_arg = DeclareLaunchArgument(
            'ip',
            default_value='10.0.0.55'
    )
    num_samples_arg = DeclareLaunchArgument(
            'numSamples',
            default_value='200'
    )
    needle_paramfile_arg = DeclareLaunchArgument(
            'needleParamFile',
            default_value='',
            description="needle parameter JSON file for loading the persistent reference wavelengths",
    )
    demo_num_chs_arg = DeclareLaunchArgument( 'numCH', default_value="3" )
    demo_num_aa_arg = DeclareLaunchArgument( 'numAA', default_value="4" )

    # sim_level=1 shape-driven arguments
    sim_shape_file_arg = DeclareLaunchArgument(
            'sim_shape_file',
            default_value='',
            description=(
                "Path to YAML/JSON shape file used for sim_level=1 shape-driven "
                "FBG wavelength generation.  Leave empty to use the original "
                "base-wavelength demo behaviour."
            ),
    )
    sim_insertion_depth_arg = DeclareLaunchArgument(
            'sim_insertion_depth',
            default_value='100.0',
            description=(
                "Current insertion depth (mm) used to locate sensor active "
                "areas along the shape polyline when sim_shape_file is set."
            ),
    )

    # Nodes
    hyperion_node = Node(
            package='hyperion_interrogator',
            namespace='needle',
            executable='hyperion_demo',
            output='screen',
            emulate_tty=True,
            parameters=[ {
                    "interrogator.ip_address"  : LaunchConfiguration( 'ip' ),
                    "sensor.num_samples"       : LaunchConfiguration( 'numSamples' ),
                    "fbg_needle.path"          : LaunchConfiguration( 'needleParamFile' ),
                    "demo.num_channels"        : LaunchConfiguration( "numCH" ),
                    "demo.num_active_areas"    : LaunchConfiguration( "numAA" ),
                    # sim_level=1 shape-driven parameters
                    "sim.shape_file"           : LaunchConfiguration( "sim_shape_file" ),
                    "sim.insertion_depth"      : LaunchConfiguration( "sim_insertion_depth" ),
            } ]
    )

    # add to launch description
    ld.add_action( hyperion_ip_arg )
    ld.add_action( num_samples_arg )
    ld.add_action( needle_paramfile_arg )
    ld.add_action( demo_num_chs_arg )
    ld.add_action( demo_num_aa_arg )
    ld.add_action( sim_shape_file_arg )
    ld.add_action( sim_insertion_depth_arg )
    ld.add_action( hyperion_node )

    return ld

# generate_launch_description
