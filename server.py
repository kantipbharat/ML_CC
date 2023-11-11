from helper import *

serv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serv_socket.bind(ADDR); serv_socket.listen(5)
print("Listening on " + str(HOST) + ":" + str(PORT))
cli_socket, address = serv_socket.accept()
print("Client connected from " + str(address[0]) + ":" + str(address[1]))

try:
    data = cli_socket.recv(PACKET_SIZE)
    if not data: raise Exception("No information received. Terminating...")
    if len(data) > PACKET_SIZE: raise Exception("Too many bytes received. Terminating...")
    if len(data) < PACKET_SIZE: raise Exception("Too few bytes received. Terminating...")
    recv_packet = Packet.from_bytes(data)
    if recv_packet.typ != SYN:
        raise Exception("Did not receive a SYN packet to establish the connection. Terminating...")

    cli_socket.send(Packet(0, 0, SYN_ACK).to_bytes())
    cli_socket.send(Packet(0, 0, SYN).to_bytes())

    data = cli_socket.recv(PACKET_SIZE)
    if not data: raise Exception("No information received. Terminating...")
    if len(data) > PACKET_SIZE: raise Exception("Too many bytes received. Terminating...")
    if len(data) < PACKET_SIZE: raise Exception("Too few bytes received. Terminating...")
    recv_packet = Packet.from_bytes(data)
    if recv_packet.typ != SYN_ACK:
        raise Exception("Did not receive a SYN_ACK packet to establish the connection. Terminating...")

    start_time = recv_packet.recv_time
    print("4-way handshake to establish connection was successful.\n")
except KeyboardInterrupt:
    print("Server received a keyboard interrupt before establishing connection. Terminating...")
    cli_socket.close()
    serv_socket.close()
    exit(0)
except Exception as err:
    print("The following error occured before establishing connection: " + str(err))
    cli_socket.close()
    serv_socket.close()
    traceback.print_exc()
    exit(1)

cli_socket.close()
serv_socket.close()