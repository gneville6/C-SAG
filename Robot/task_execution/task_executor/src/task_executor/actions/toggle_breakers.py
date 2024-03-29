#!/usr/bin/env python
# The toggle_breakers action in a task plan

import rospy

from task_executor.abstract_step import AbstractStep

from power_msgs.msg import BreakerState
from power_msgs.srv import BreakerCommand


class ToggleBreakersAction(AbstractStep):
    """
    Enable or disable the breakers through the services
    :const:`ARM_BREAKER_SERVICE_NAME`, :const:`BASE_BREAKER_SERVICE_NAME`, or
    :const:`GRIPPER_BREAKER_SERVICE_NAME`.

    .. note::

        The breaker services, and therefore this action, are not available in
        simulation
    """

    ARM_BREAKER_SERVICE_NAME = "/arm_breaker"
    BASE_BREAKER_SERVICE_NAME = "/base_breaker"
    GRIPPER_BREAKER_SERVICE_NAME = "/gripper_breaker"

    SIMULATION_PARAM = "/use_sim_time"

    def init(self, name):
        self.name = name

        # Service to communicate with the breakers
        self._arm_breaker_srv = rospy.ServiceProxy(
            ToggleBreakersAction.ARM_BREAKER_SERVICE_NAME,
            BreakerCommand
        )
        self._base_breaker_srv = rospy.ServiceProxy(
            ToggleBreakersAction.BASE_BREAKER_SERVICE_NAME,
            BreakerCommand
        )
        self._gripper_breaker_srv = rospy.ServiceProxy(
            ToggleBreakersAction.GRIPPER_BREAKER_SERVICE_NAME,
            BreakerCommand
        )

        # Check that this is enabled only on the real robot
        self._simulation = rospy.get_param(ToggleBreakersAction.SIMULATION_PARAM, False)

        # Initialize our connections to the robot driver
        rospy.loginfo("Connecting to robot driver (if not in simulation)...")
        if not self._simulation:
            self._arm_breaker_srv.wait_for_service()
            self._base_breaker_srv.wait_for_service()
            self._gripper_breaker_srv.wait_for_service()
        rospy.loginfo("...robot driver connected")

    def run(self, enable_base=True, enable_arm=True, enable_gripper=True):
        """
        The run function for this step

        Args:
            enable_base (bool) : if ``True``, enable the base breaker; else disable
                it
            enable_arm (bool) : if ``True``, enable the arm breaker; else disable it
            enable_gripper (bool) : if ``True``, enable the gripper breaker; else
                disable it

        .. seealso::

            :meth:`task_executor.abstract_step.AbstractStep.run`
        """
        rospy.loginfo("Action {}: Base - {}, Arm - {}, Gripper - {}"
                      .format(self.name, enable_base, enable_arm, enable_gripper))

        # For each of the breakers, set the compliance accordingly and test the
        # returned setting
        if not self._simulation:
            self._validate_response(
                self._gripper_breaker_srv(enable=enable_gripper),
                BreakerState.STATE_ENABLED if enable_gripper else BreakerState.STATE_DISABLED
            )
            self.notify_service_called(ToggleBreakersAction.GRIPPER_BREAKER_SERVICE_NAME)
            self._validate_response(
                self._arm_breaker_srv(enable=enable_arm),
                BreakerState.STATE_ENABLED if enable_arm else BreakerState.STATE_DISABLED
            )
            self.notify_service_called(ToggleBreakersAction.ARM_BREAKER_SERVICE_NAME)
            self._validate_response(
                self._base_breaker_srv(enable=enable_base),
                BreakerState.STATE_ENABLED if enable_base else BreakerState.STATE_DISABLED
            )
            self.notify_service_called(ToggleBreakersAction.BASE_BREAKER_SERVICE_NAME)

        yield self.set_succeeded()

    def stop(self):
        # This action cannot be stopped
        pass

    def _validate_response(self, response, expected_value):
        assert response.status.state == expected_value
