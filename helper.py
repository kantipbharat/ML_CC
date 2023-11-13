from collections import deque
import errno
import math
import numpy as np
import os
import pandas as pd
import pickle
import random
import socket
import struct
import sys
import threading
import time
import traceback
import zlib

HOST = '127.0.0.1'
PORT = 8080
ADDR = (HOST, PORT)
TIMEOUT = 1

DATA = 0
ACK = 1
SYN = 2
SYN_ACK = 3
FIN = 4
FIN_ACK = 5

TS_SIZE = 8
MAX_TRANSMIT = 2

RUNTIME = 10
CSV_NAME = 'out.csv'
PKL_NAME = 'model.sav'

class Packet:
    PACK_FORM = '!IIIdd'
    PACK_SIZE = struct.calcsize(PACK_FORM)

    def __init__(self, num, idx, typ, send_time=0.0, recv_time=0.0):
        self.num = num
        self.idx = idx
        self.typ = typ
        self.send_time = send_time
        self.recv_time = recv_time
        self.chksum = zlib.crc32(struct.pack(self.PACK_FORM, self.num, self.idx, self.typ, self.send_time, self.recv_time)) & 0xffffffff
    
    def to_bytes(self):
        packet = struct.pack(self.PACK_FORM, self.num, self.idx, self.typ, self.send_time, self.recv_time)
        return packet + struct.pack('!I', self.chksum)

    @classmethod
    def from_bytes(cls, data):
        packet = data[:cls.PACK_SIZE]
        num, idx, typ, send_time, recv_time = struct.unpack(cls.PACK_FORM, packet)
        orig_chksum = struct.unpack('!I', data[cls.PACK_SIZE:])[0]
        calc_chksum = zlib.crc32(packet) & 0xffffffff
        if orig_chksum != calc_chksum:
            raise ValueError("Checksums do not match. Packet was corrupted!")
        return cls(num, idx, typ, send_time, recv_time)

PACKET_SIZE = Packet.PACK_SIZE + 4