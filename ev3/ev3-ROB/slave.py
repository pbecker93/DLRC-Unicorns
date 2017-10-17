import paho.mqtt.client as mqtt
from robot import Robot
import config


class Slave(mqtt.Client):

    to_master = '2master'
    to_slave = '2slave'

    supported_types = {
        'str' : str,
        'string' : str,
        'int': int,
        'float': float,
        'ack': ''
    }


    def __init__(self):

    # Setup the client
        print('Connecting ... should be good if we have the correct port and IP')
        super().__init__()
        self.connect(config.BROKER_IP, config.BROKER_PORT, keepalive=60)

        # Getting the robot started :)
        print('Setting up the robot')
        self.rob = Robot()

        # Set the functions for when stuff happens
     #   self.on_connect = self.on_connect
     #   self.on_message = self.on_message

    # Wait for messages
        self.loop_forever(timeout=10)

    def on_connect(self, client, userdata, flags, rc):
        """
        What we once we connect
        """
        print('we are connected for sure')
        self.subscribe(Slave.to_slave)

    def on_message(self, client, userdata, msg):
        """
        When we get a message, it is assumed this is a command for the robot

        We assume the message is structured as follows:
        'function we call| parameters of the function seperated by comma's

        for example:
        'run_forward|speed=100, time=300'
        """

        # Get the command and its arguments
        payload = msg.payload.decode()

        if 'shut_down' in payload:
            self.rob.shut_down()
            self.disconnect()
            self.loop_stop()
        else:
            try:
                split_payload = msg.payload.decode().split('(')


                func_name = split_payload[0]
                attr_list = split_payload[1].split(',')

                attr_list[-1] = attr_list[-1].replace(')', '')
                if attr_list[0] == '':
                    attr_list = []
                print(func_name, *attr_list)
                func = Robot.__getattribute__(self.rob, func_name)
                res = func(*attr_list)
                print('publishing', res)
                self.publish(Slave.to_master, res)
            except Exception as e:
                print('Exception', str(e))
                self.publish(Slave.to_master, str(e) + ',Exception')

if __name__ == '__main__':
    client = Slave()
