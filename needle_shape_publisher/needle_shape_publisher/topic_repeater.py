"""topic_repeater.py

A generic ROS 2 node that continuously re-publishes the last received message
on a given topic.

ROS 2 parameters
----------------
topic : str
    The topic name to publish on (required).
input_topic : str
    The topic name to subscribe to (default: same as topic).
msg_type : str
    The fully-qualified message type string, e.g.
    ``'geometry_msgs/msg/PoseStamped'`` (required).
rate_hz : float
    Re-publish rate in Hz (default: 10.0).
default_msg : str
    JSON/YAML representation of the initial message to publish before any
    external message has been received (default: ``'{}'``, i.e. all-zeros).
wait_for_input : bool
    If True, do not publish anything until the first real message arrives on
    input_topic. Ignores default_msg. Use this when a separate one-shot seed
    publisher handles the initial value and the repeater should only take over
    once a live external publisher (e.g. Slicer) has sent its first message.
"""

import json

import rclpy
from rclpy.node import Node
from rosidl_runtime_py.utilities import get_message
from rosidl_runtime_py import set_message_fields


class TopicRepeater(Node):
    """Continuously re-publishes the last received message on a topic.

    Subscribes on input_topic, publishes on topic. When they differ the node
    never receives its own output, eliminating the self-feedback loop that
    causes alternating values when a competing publisher is also present.

    When wait_for_input=True the repeater stays completely silent until the
    first message arrives on input_topic, then starts broadcasting at rate_hz.
    """

    def __init__(self):
        super().__init__('topic_repeater')

        # Declare parameters
        self.declare_parameter('topic', '')
        self.declare_parameter('input_topic', '')
        self.declare_parameter('msg_type', '')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('default_msg', '{}')
        self.declare_parameter('wait_for_input', False)

        topic           = self.get_parameter('topic').get_parameter_value().string_value
        input_topic     = self.get_parameter('input_topic').get_parameter_value().string_value
        msg_type_str    = self.get_parameter('msg_type').get_parameter_value().string_value
        rate_hz         = self.get_parameter('rate_hz').get_parameter_value().double_value
        default_msg_str = self.get_parameter('default_msg').get_parameter_value().string_value
        self._wait_for_input = self.get_parameter('wait_for_input').get_parameter_value().bool_value

        if not topic:
            raise ValueError('Parameter "topic" must be set to a non-empty topic name.')
        if not msg_type_str:
            raise ValueError('Parameter "msg_type" must be set, e.g. "geometry_msgs/msg/PoseStamped".')

        # If no separate input_topic given, fall back to topic (old behaviour).
        if not input_topic:
            input_topic = topic

        self._rate_hz = rate_hz
        self._input_received = False

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

        qos = 10
        self._pub = self.create_publisher(MsgType, topic, qos)
        self._sub = self.create_subscription(MsgType, input_topic, self._callback, qos)

        # When wait_for_input=True the timer is created only after the first
        # real message arrives — see _callback. Otherwise start immediately.
        self._timer = None
        if not self._wait_for_input:
            self._timer = self.create_timer(1.0 / rate_hz, self._publish)

        self.get_logger().info(
            f'TopicRepeater ready: sub={input_topic!r}  pub={topic!r}  '
            f'type={msg_type_str}  rate={rate_hz} Hz  '
            f'wait_for_input={self._wait_for_input}'
        )

    def _callback(self, msg):
        """Cache the latest message received on input_topic."""
        self._latest_msg = msg
        if self._wait_for_input and not self._input_received:
            self._input_received = True
            self._timer = self.create_timer(1.0 / self._rate_hz, self._publish)
            self.get_logger().info(
                'TopicRepeater: first input received — starting broadcast.'
            )

    def _publish(self):
        """Re-publish the cached message on topic."""
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
