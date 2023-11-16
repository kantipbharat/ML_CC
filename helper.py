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
TIMEOUT = 2

DATA = 0
ACK = 1
SYN = 2
SYN_ACK = 3
FIN = 4
FIN_ACK = 5

TS_SIZE = 8
MAX_TRANSMIT = 2

RUNTIME = 20 #5000

VERSION_MAP = {'0':'aimd', '1':'newreno', '2':'lp-aimd', '3':'lp-newreno', '4':'rl-aimd', '5':'rl-newreno'}

COLUMNS = ['num', 'idx', 'cwnd', 'cwnd_order']
COLUMNS += ['ewma_inter_send', 'min_inter_send']
COLUMNS += ['ts_inter_send_' + str(i + 1) for i in range(TS_SIZE)]
COLUMNS += ['ts_ratio_inter_send_' + str(i + 1) for i in range(TS_SIZE)]
COLUMNS += ['ewma_inter_arr', 'min_inter_arr']
COLUMNS += ['ts_inter_arr_' + str(i + 1) for i in range(TS_SIZE)]
COLUMNS += ['ts_ratio_inter_arr_' + str(i + 1) for i in range(TS_SIZE)]
COLUMNS += ['min_rtt'] + ['ts_rtt_' + str(i + 1) for i in range(TS_SIZE)]
COLUMNS += ['ts_ratio_rtt_' + str(i + 1) for i in range(TS_SIZE)]
COLUMNS += ['recvd']
COLUMNS += ['ssthresh', 'throughput', 'max_throughput', 'loss_rate', 'overall_loss_rate', 'delay']
COLUMNS += ['ratio_inter_send', 'ratio_inter_arr', 'ratio_rtt']

def recv_packet_func(sock, exp_typ):
    try:
        data = sock.recv(PACKET_SIZE)
        if not data: raise Exception("No information received. Terminating...")
        if len(data) > PACKET_SIZE: raise Exception("Too many bytes received. Terminating...")
        if len(data) < PACKET_SIZE: raise Exception("Too few bytes received. Terminating...")
        recv_packet = Packet.from_bytes(data)
        if not recv_packet: return None
        if recv_packet.typ not in exp_typ: raise Exception("Expected packet type " + str(exp_typ) + " but received " + str(recv_packet.typ) + ". Terminating...")
        return recv_packet
    except socket.timeout: print("The receive function timed out!")
    except Exception as err: print("The following error occured before terminating the connection: " + str(err))


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