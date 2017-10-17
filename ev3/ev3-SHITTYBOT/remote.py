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

# Connect remote control
rc = RemoteControl(); assert rc.connected

# Initialize button handler
# button = Button()   # not working so disabled

# Turn leds off
Leds.all_off()

def roll(motor, led_group, direction, speed):
    """
    Generate remote control event handler. It rolls given motor into given
    direction (1 for forward, -1 for backward). When motor rolls forward, the
    given led group flashes green, when backward -- red. When motor stops, the
    leds are turned off.

    The on_press function has signature required by RemoteControl class.
    It takes boolean state parameter; True when button is pressed, False
    otherwise.
    """
    def on_press(state):
        if state:
            # Roll when button is pressed
            motor.run_forever(speed_sp=speed*direction)
            Leds.set_color(led_group, direction > 0 and Leds.GREEN or Leds.RED)
        else:
            # Stop otherwise
            motor.stop(stop_action='brake')
            Leds.set(led_group, brightness_pct=0)

    return on_press

# Assign event handler to each of the remote buttons
rc.on_red_up    = roll(lmotor, Leds.LEFT,   1, 1000)
rc.on_red_down  = roll(lmotor, Leds.LEFT,  -1, 1000)
rc.on_blue_up   = roll(rmotor, Leds.RIGHT,  1, 90)
rc.on_blue_down = roll(rmotor, Leds.RIGHT, -1, 90)

# Enter event processing loop
#while not button.any():   #not working so commented out
while True:   #replaces previous line so use Ctrl-C to exit
    rc.process()
    sleep(0.01)
    
# Press Ctrl-C to exit
