#s demo shows how to remote control an Explor3r robot
#
# Red buttons control left motor, blue buttons control right motor.
# Leds are used to indicate movement direction.

from time import sleep
from ev3dev.ev3 import *

# Connect two large motors on output ports B and C
lmotor = LargeMotor('outA')
rmotor = MediumMotor('outB')

# Check that the motors are actually connected
assert lmotor.connected
assert rmotor.connected

#rmotor.run_timed(time_sp=500, speed_sp=50)
#rmotor.wait_while('running')
#rmotor.run_timed(time_sp=500, speed_sp=-50)
#rmotor.wait_while('running')


lmotor.run_timed(time_sp=4000, speed_sp=-1000, stop_action='brake')
lmotor.wait_while('running')

#rmotor.run_timed(time_sp=500, speed_sp=50)
#rmotor.wait_while('running')
#rmotor.run_timed(time_sp=500, speed_sp=-50)
#rmotor.wait_while('running')
sleep(1)
lmotor.run_timed(time_sp=4000, speed_sp=1000)

#lmotor.run_timed(time_sp=3000, speed_sp=1000)
