from robot_method import RobotMethod
import ev3dev.ev3 as ev3
from time import sleep
import threading as t

class Robot:

    LIFT_DOWN_POS = 0
    LIFT_DRIVE_POS = 900
    LIFT_UP_POS = 5500
    LIFT_STATE_UP = 'lift_up'
    LIFT_STATE_DOWN = 'lift_down'
    LIFT_STATE_DRIVING = 'lift_driving'
    LIFT_STATE_UNKNOWN = 'lift_unknown'


    GRIPPER_OPEN_POS = 0
    GRIPPER_CLOSE_POS = 2000
    GRIPPER_STATE_OPEN = 'gripper_open'
    GRIPPER_STATE_CLOSE = 'gripper_close'

    APPROACH_TIME = 2500
    APPROACH_SPEED = 400

    def diagnostic(self):
                assert self.lift.connected, "Lift is not connected"
                assert self.tire_left.connected, "Left tire is not connected"
                assert self.tire_right.connected, "Right tire is not connected"
                assert self.grip.connected, "Grip is not connected"
           #     assert self.color_sensor.connected, "Color Sensor is not connected"

    def __init__(self):

        self.lift = ev3.LargeMotor('outC')
        self.tire_left = ev3.LargeMotor('outA')
        self.tire_right = ev3.LargeMotor('outB')
        self.grip = ev3.MediumMotor('outD')

        print('Diagnostics')
        self.diagnostic()

        print('Reset everything')
        self.RESET()
        print(self.lift.state)

        print('initialize lift')
        self._lift_state = self._init_lift()

        print('initialize gripper')
        self._gripper_state = self._init_gripper()

    def _init_lift(self):
        self.lift.run_timed(time_sp=500, speed_sp=-500)
        self.lift.wait_while('running')
        self.lift.run_timed(time_sp=500, speed_sp=500)
        self.lift.wait_while('running')
        self._run_lift_to_abs(Robot.LIFT_DRIVE_POS)
        return Robot.LIFT_STATE_DRIVING

    def _init_gripper(self):
        self.grip.run_timed(time_sp=500, speed_sp=500)
        self.grip.wait_while('running')
        self.grip.run_timed(time_sp=500, speed_sp=-500)
        self.grip.wait_while('running')
        return Robot.GRIPPER_STATE_OPEN

    def _run_lift_to_abs(self, abs_pos, stop_action=''):
        self.lift.run_to_abs_pos(position_sp=-abs_pos, stop_action=stop_action)
        self.lift.wait_while('running')

    @RobotMethod.robot_method(input_types=[])
    def lower_lift(self):
        self._run_lift_to_abs(Robot.LIFT_DOWN_POS)
        self._lift_state = Robot.LIFT_STATE_DOWN

    @RobotMethod.robot_method(input_types=[])
    def raise_lift_driving(self):
        self._run_lift_to_abs(Robot.LIFT_DRIVE_POS, stop_action='brake')
        self._lift_state = Robot.LIFT_STATE_DRIVING

    @RobotMethod.robot_method(input_types=[])
    def raise_lift(self):
        self._run_lift_to_abs(Robot.LIFT_UP_POS, stop_action='brake')
        self._lift_state = Robot.LIFT_STATE_UP

    @RobotMethod.robot_method(input_types=['int'])
    def raise_lift_arbitrary(self, absolute_position):
        self._run_lift_to_abs(absolute_position, stop_action='brake')
        self._lift_state = Robot.LIFT_STATE_UNKNOWN

    @RobotMethod.robot_method(input_types=['int', 'int'])
    def move_lift(self, time, speed):
        self.lift.run_timed(time_sp=time, speed_sp=speed)

    def _run_gripper_to_abs_pos(self, abs_pos, stop_action=''):
        self.grip.run_to_abs_pos(position_sp=abs_pos, stop_action=stop_action)
        self.grip.wait_while('running')

    @RobotMethod.robot_method(input_types=[])
    def open_gripper(self):
        self._run_gripper_to_abs_pos(Robot.GRIPPER_OPEN_POS)
        self._gripper_state = Robot.GRIPPER_OPEN_POS

    @RobotMethod.robot_method(input_types=[])
    def close_gripper(self):
        self._run_gripper_to_abs_pos(Robot.GRIPPER_CLOSE_POS, stop_action='brake')
        self._gripper_state = Robot.GRIPPER_CLOSE_POS

    def shut_down(self):
        self._stop_all()
        self.grip.run_to_abs_pos(position_sp=Robot.GRIPPER_OPEN_POS)
        self._gripper_state = Robot.GRIPPER_STATE_OPEN
        self.lift.run_to_abs_pos(position_sp=Robot.LIFT_DOWN_POS)
        self._lift_state = Robot.LIFT_STATE_DOWN


    def _turn_left(self, speed, time=30000):
        self.tire_left.run_timed(time_sp=time, speed_sp=speed)
        self.tire_right.run_timed(time_sp=time, speed_sp=-speed)

    def _turn_right(self, speed, time=30000):
        self.tire_left.run_timed(time_sp=time, speed_sp=-speed)
        self.tire_right.run_timed(time_sp=time, speed_sp=speed)

    def _turn(self, degrees):
     #Todo Calibrate better
        factor = 1000.0 / 90.0
        time_sp = factor * degrees
        if degrees > 0:
            self.tire_left.run_timed(time_sp=time_sp, speed_sp=500)
            self.tire_right.run_timed(time_sp=time_sp, speed_sp=-500)
        else:
            self.tire_left.run_timed(time_sp=-time_sp, speed_sp=-500)
            self.tire_right.run_timed(time_sp=-time_sp, speed_sp=500)
        self.tire_left.wait_while('running')
        self.tire_right.wait_while('running')

    @RobotMethod.robot_method(input_types='float')
    def turn_left(self, degrees):
        self._turn(degrees)

    @RobotMethod.robot_method(input_types='float')
    def turn_right(self, degrees):
        self._turn(-degrees)

    def _move_straight(self, speed, time=120000):
        self.tire_left.run_timed(time_sp=time, speed_sp=-speed)
        self.tire_right.run_timed(time_sp=time, speed_sp=-speed)
        self.tire_left.wait_while('running')
        self.tire_right.wait_while('running')

    def _move_straight_unblocking(self, speed, time=120000):
        self.tire_left.run_timed(time_sp=time, speed_sp=-speed)
        self.tire_right.run_timed(time_sp=time, speed_sp=-speed)
  
    @RobotMethod.robot_method(input_types=['int', 'int'])
    def move_straight(self, speed, time):
        self._move_straight(speed=speed, time=time)

    @RobotMethod.robot_method(input_types='int')
    def approach_brick(self):
        self._move_straight(speed=Robot.APPROACH_SPEED, time=Robot.APPROACH_TIME)

    def approach_box(self):
        self.move_straight(speed=Robot.APPROACH_SPEED, time=Robot.APPROACH_TIME)

    @RobotMethod.robot_method(input_types=[])
    def move_forever(self):
        self._stop_all()
        self._move_straight_unblocking(speed=150, time=120000)

    @RobotMethod.robot_method(input_types=[])
    def turn_left_forever(self):
        self._stop_all()
        self._turn_left(speed=75, time=120000)

    @RobotMethod.robot_method(input_types=[])
    def turn_right_forever(self):
        self._stop_all()
        self._turn_right(speed=75, time=120000)

    def RESET(self):
        self.lift.reset()
        self.tire_left.reset()
        self.tire_right.reset()
        self.grip.reset()

    def _stop_all(self):
        self.tire_right.stop()
        self.tire_left.stop()
    
    @RobotMethod.robot_method(input_types=[])
    def stop_motion(self):
        self._stop_all()
        self.tire_right.wait_while('running')
        self.tire_left.wait_while('running')
