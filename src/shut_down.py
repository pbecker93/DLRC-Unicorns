from mqtt_master.master import Master

#FREIMAN
#BROKER_IP = "192.168.179.40" 

#DLRC
BROKER_IP = "10.250.144.181"
BROKER_PORT = 1883

mqtt_master = Master(BROKER_IP, BROKER_PORT)
res = mqtt_master.publish(Master.to_slave, 'shut_down')
mqtt_master.loop_threat.join()

