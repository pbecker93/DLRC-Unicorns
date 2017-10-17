from slave import Slave
import ev3dev.ev3 as ev3
from time import sleep
from threading import Thread

class Robot:
    def diagnostic(self):
        """
        Check if everything is still connected
        """

        # Check actuators
        for name, actuator in self.act_dict.items():
            assert(actuator.connected), '{} is not connected'.format(name)

        # Check sensors
        for name, sensor in self.sen_dict.items():
            assert(sensor.connected), '{} is not connected'.format(name)

    def __init__(self):
        """
        Start the bot
        """

        # Setup the actuators
        self.act_dict = {
            'lift': ev3.LargeMotor('outA')
        }

        # Setup the sensors
        self.sen_dict = {
            'color': ev3.ColorSensor('in4')
        }
        self.sen_dict['color'].mode = 'RGB-RAW'

        # Set the settings
        self.DetectBrick = False
        self.diagnostic()

    def turn(self, degrees):
        """
        It turns the MediumMotor a certain number of degrees, for now it is a test function
        """
        degrees = int(degrees)
        if bool(self.act_dict['lift'].connected):
            self.act_dict['lift'].run_timed(time_sp=300, speed_sp=degrees, stop_action='brake')
            print('It is working')
        else:
            print('Lift is not connected, cant do anything, pls connect lift')

    def colorQ(self):
        """
        Are we detecting anything
        """
        color = self.sen_dict['color'].value
        return color(0), color(1), color(2)

    def RESET(self):
        """
        Reset the robot ...
        """
        pass

    def drive_to_block(self):
        def listen_to_sensor():
            cs = self.sen_dict['color']
            while True:
                if cs.value(0) > 25 or cs.value(1) > 25 or cs.value(2) > 25:
                    self.act_dict['lift'].stop()
                    break

        t = Thread(target=listen_to_sensor)
        t.start()

        self.act_dict['lift'].run_timed(time_sp=30000, speed_sp=500)