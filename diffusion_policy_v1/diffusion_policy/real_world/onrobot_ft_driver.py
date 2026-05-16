import socket
import struct
import signal
import sys
import json
import time
from threading import Thread

class OnRobotFTDriver:
    kDriverName = "OnRobotFTDriver"
    kSetSoftwareBias = 0x0010
    kZeroBias = 0x0001
    kResetBias = 0x0000
    kCommand = 0x0002
    kNumSamples = 0

    def __init__(self, ip_address, port):
        self.m_ip_address = ip_address
        self.m_port = port
        self.m_socket_handle = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Default scaling values for OnRobotFTSensor
        counts_per_force = 10000
        counts_per_torque = 100000

        self.m_force_scale = 1.0 / counts_per_force
        self.m_torque_scale = 1.0 / counts_per_torque

    def connect(self):
        try:
            self.m_socket_handle.connect((self.m_ip_address, self.m_port))
            return True
        except socket.error as e:
            print(f"Socket error: {e}")
            return False

    def OffsetBias(self):
        request = bytearray(8)
        struct.pack_into('!H', request, 0, 0x1234)  # standard header
        struct.pack_into('!H', request, 2, self.kSetSoftwareBias)  # Set or reset zero offset
        struct.pack_into('!I', request, 4, self.kZeroBias)  # Zero the sensor to current static bias

        try:
            self.m_socket_handle.send(request)

            # No response expected

            return True

        except socket.error as e:
            print(f"Socket error: {e}")
            return False

    def ResetBias(self):
        request = bytearray(8)
        struct.pack_into('!H', request, 0, 0x1234)  # standard header
        struct.pack_into('!H', request, 2, self.kSetSoftwareBias)  # Set or reset zero offset
        struct.pack_into('!I', request, 4, self.kResetBias)  # Unset bias

        try:
            self.m_socket_handle.send(request)

            # No response expected

            return True

        except socket.error as e:
            print(f"Socket error: {e}")
            return False

    def Read(self, readings):
        request = bytearray(8)
        struct.pack_into('!H', request, 0, 0x1234)  # standard header
        struct.pack_into('!H', request, 2, self.kCommand)  # Will fill buffer to length kNumSamples
        struct.pack_into('!I', request, 4, self.kNumSamples)  # If kNumSamples is 0, will read continuously

        try:
            self.m_socket_handle.send(request)

            response = self.m_socket_handle.recv(36)  # output is always 36 bytes

            # Byte order: expects big-endian, struct.unpack should force the result
            # 4 bytes: sequence number, 4 bytes sample, 4 bytes status
            # Then 3x 4 bytes for force and 3x 4 bytes for torque

            resp = struct.unpack('!3I6i', response)

            readings.set(
                resp[3] * self.m_force_scale,
                resp[4] * self.m_force_scale,
                resp[5] * self.m_force_scale,
                resp[6] * self.m_torque_scale,
                resp[7] * self.m_torque_scale,
                resp[8] * self.m_torque_scale
            )

            return True

        except socket.error as e:
            print(f"Socket error: {e}")
            return False


class ForceTorqueValue:
    def set(self, fx, fy, fz, tx, ty, tz):
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.tx = tx
        self.ty = ty
        self.tz = tz


class MsgChannel:
    def __init__(self, port, is_server=False, address="tcp://127.0.0.1"):
        self.port = port
        self.is_server = is_server
        self.address = address
        self.is_open = True

    def send(self, message):
        print(f"Sending message: {json.dumps(message)}")

    def recv(self, timeout):
        time.sleep(timeout / 1000.0)
        return None

    def close(self):
        self.is_open = False

    def is_open(self):
        return self.is_open


class FTDriver:
    def __init__(self, port, sensor):
        self.port = port
        self.sensor = sensor
        self.channel = MsgChannel(port, False, "tcp://127.0.0.1")

    def run(self):
        if self.sensor.connect():
            while self.channel.is_open:
                readings = ForceTorqueValue()
                if self.sensor.Read(readings):
                    data = {
                        "force": [readings.fx, readings.fy, readings.fz],
                        "torque": [readings.tx, readings.ty, readings.tz]
                    }
                    self.channel.send(data)
                time.sleep(0.1)

    def close(self):
        self.channel.close()


g_driver = None

def my_handler(signum, frame):
    global g_driver
    if g_driver:
        g_driver.close()


def main(args):
    global g_driver

    if '-h' in args:
        print("-id <OnRobot FT device hardware id (ip address)>")
        print("-p <msg channel port number>")
        print("-d [optional, defaults to 49152] <port upon which to communicate with the OnRobot FT device>")
        return

    ip_address = "192.168.7.12"
    port = 46152
    device_port = 49152

    if port == 0 or ip_address == "":
        print("Expected a message port and ip address in the input parameters")
        return 1

    print(f"Starting OnRobot FT Driver with the following params \n \t MsgChannel Port: {port} \n \t Device IP: {ip_address} \n \t Device Port: {device_port} \n")

    signal.signal(signal.SIGINT, my_handler)
    signal.signal(signal.SIGTERM, my_handler)

    sensor = OnRobotFTDriver(ip_address, device_port)

    try:
        g_driver = FTDriver(port, sensor)
        g_driver.run()
    except Exception as e:
        print(f"Exception: {e}")

    del g_driver

def test():
    ip_address = "192.168.7.12"
    port = 46152
    device_port = 49152

    if port == 0 or ip_address == "":
        print("Expected a message port and ip address in the input parameters")
        return 1

    print(f"Starting OnRobot FT Driver with the following params \n \t MsgChannel Port: {port} \n \t Device IP: {ip_address} \n \t Device Port: {device_port} \n")

    sensor = OnRobotFTDriver(ip_address, device_port)

    try:
        for i in range(100):
            if sensor.connect():
                readings = ForceTorqueValue()
                if sensor.Read(readings):
                    data = {
                        "force": [readings.fx, readings.fy, readings.fz],
                        "torque": [readings.tx, readings.ty, readings.tz]
                    }
                    print(data)
                    time.sleep(0.1)
    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    # main(sys.argv)
    test()
