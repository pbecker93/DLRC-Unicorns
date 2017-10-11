import time
from threading import Lock

import paho.mqtt.client as mqtt

class Master(mqtt.Client):

    to_slave = '2slave'
    to_master = '2master'


    def __init__(self, broker_ip, broker_port):
        super().__init__()
        self.supported_types = {
            'str': str,
            'string': str,
            'int': int,
            'float': float,
            'ack': lambda x: None,
            'Exception': self._handle_slave_exception
        }

        print('Connecting ... should be good if we have the correct port and IP')
        self.connect(broker_ip, broker_port, keepalive=60)
        self.loop_start()
        self.wait_for_resp_lock = Lock()
        self.ret_value = None
        time.sleep(1)

    def on_connect(self, client, userdata, flags, rc):
        """
        What we once we connect
        """
        print('we are connected for sure')
        self.subscribe(Master.to_master)

    def on_message(self, client, userdata, msg):

        msg = msg.payload.decode()
        print(msg)
        value, dtype = msg.split(',')
        if value is not 'None':
            self.ret_value = self.supported_types[dtype](value)
        else:
            self.ret_value = None
        if self.wait_for_resp_lock.locked():
            self.wait_for_resp_lock.release()

    def blocking_call(self, topic, payload):
        print('publish')
        self.wait_for_resp_lock.acquire()
        self.publish(topic=topic, payload=payload)
        self.wait_for_resp_lock.acquire()
        self.wait_for_resp_lock.release()
        return self.ret_value

    @property
    def loop_threat(self):
        assert self._thread is not None, 'Loop not yet started'
        return self._thread

    def _handle_slave_exception(self, exception_as_string):
        print("Slave Exception", exception_as_string)

