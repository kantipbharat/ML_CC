from helper import *

cli_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cli_socket.connect(ADDR)

start_time = 0.0

try:
    cli_socket.sendall(Packet(0, 0, SYN).to_bytes())
    
    data = cli_socket.recv(PACKET_SIZE)
    if not data: raise Exception("No information received. Terminating...")
    if len(data) > PACKET_SIZE: raise Exception("Too many bytes received. Terminating...")
    if len(data) < PACKET_SIZE: raise Exception("Too few bytes received. Terminating...")
    recv_packet = Packet.from_bytes(data)
    if recv_packet.typ != SYN_ACK:
        raise Exception("Did not receive a SYN_ACK packet to establish the connection. Terminating...")

    data = cli_socket.recv(PACKET_SIZE)
    if not data: raise Exception("No information received. Terminating...")
    if len(data) > PACKET_SIZE: raise Exception("Too many bytes received. Terminating...")
    if len(data) < PACKET_SIZE: raise Exception("Too few bytes received. Terminating...")
    recv_packet = Packet.from_bytes(data)
    if recv_packet.typ != SYN:
        raise Exception("Did not receive a SYN packet to establish the connection. Terminating...")
    
    start_time = time.time()
    cli_socket.sendall(Packet(0, 0, SYN_ACK, recv_time=start_time).to_bytes())
    print("4-way handshake to establish connection was successful.\n")
except KeyboardInterrupt:
    print("Client received a keyboard interrupt before establishing connection. Terminating...")
    cli_socket.close()
    exit(0)
except Exception as err:
    print("The following error occured before establishing connection: " + str(err))
    cli_socket.close()
    traceback.print_exc()
    exit(1)