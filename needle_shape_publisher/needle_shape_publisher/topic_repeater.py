"""topic_repeater.py

A generic ROS 2 node that continuously re-publishes the last received message
on a given topic.

Any publisher (including a one-shot ``ros2 topic pub --once``) that sends a
message to the topic will update the value that is continuously broadcast.
This allows the "live default" for helper topics such as
``/stage/state/needle_pose`` and ``/needle/state/skin_entry`` to be overridden
at runtime without restarting the launch file.

ROS 2 parameters
----------------
topic : str
    The topic name to subscribe to and publish on (required).
msg_type : str
    The fully-qualified message type string, e.g.
    ``'geometry_msgs/msg/PoseStamped'`` (required).
rate_hz : float
    Re-publish rate in Hz (default: 10.0).
default_msg : str
    JSON/YAML representation of the initial message to publish before any
    external message has been received (default: ``'{}'``, i.e. all-zeros).
"""

import json

import rclpy
from rclpy.node import Node
from rosidl_runtime_py.utilities import get_message
from rosidl_runtime_py import set_message_fields


class TopicRepeater(Node):
    """Continuously re-publishes the last received message on a topic.

    The node subscribes and publishes on the same topic.  When any other node
    (or a manual ``ros2 topic pub``) sends a message, the new value is cached
    and becomes the value that is re-broadcast on every timer tick.
    """

    def __init__(self):
        super().__init__('topic_repeater')

        # Declare parameters
        self.declare_parameter('topic', '')
        self.declare_parameter('msg_type', '')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('default_msg', '{}')

        topic = self.get_parameter('topic').get_parameter_value().string_value
        msg_type_str = self.get_parameter('msg_type').get_parameter_value().string_value
        rate_hz = self.get_parameter('rate_hz').get_parameter_value().double_value
        default_msg_str = self.get_parameter('default_msg').get_parameter_value().string_value

        if not topic:
            raise ValueError('Parameter "topic" must be set to a non-empty topic name.')
        if not msg_type_str:
            raise ValueError('Parameter "msg_type" must be set, e.g. "geometry_msgs/msg/PoseStamped".')

        # Resolve the message class dynamically
        MsgType = get_message(msg_type_str)

        # Build the default message
        self._latest_msg = MsgType()
        if default_msg_str and default_msg_str != '{}':
            try:
                default_fields = json.loads(default_msg_str)
                if default_fields:
                    set_message_fields(self._latest_msg, default_fields)
            except Exception as exc:
                self.get_logger().warn(
                    f'Could not parse default_msg "{default_msg_str}": {exc}. '
                    'Using zero-initialised message.'
                )

        # Publisher and subscriber on the same topic.
        # The subscriber receiving its own published messages is benign:
        # it simply re-caches the same value.
        qos = 10
        self._pub = self.create_publisher(MsgType, topic, qos)
        self._sub = self.create_subscription(MsgType, topic, self._callback, qos)

        # Timer to continuously republish the cached message
        self._timer = self.create_timer(1.0 / rate_hz, self._publish)

        self.get_logger().info(
            f'TopicRepeater ready: topic={topic!r}  type={msg_type_str}  rate={rate_hz} Hz'
        )

    def _callback(self, msg):
        """Cache the latest message from any publisher on the topic."""
        self._latest_msg = msg

    def _publish(self):
        """Re-publish the cached message."""
        self._pub.publish(self._latest_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TopicRepeater()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
