from helper import *

serv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serv_socket.bind(ADDR); serv_socket.listen(5)

print("Listening on " + str(HOST) + ":" + str(PORT))
cli_socket, address = serv_socket.accept()
print("Client connected from " + str(address[0]) + ":" + str(address[1]))

try:
    recv_packet_func(cli_socket, [SYN])
    cli_socket.send(Packet(0, 0, SYN_ACK).to_bytes())
    cli_socket.send(Packet(0, 0, SYN).to_bytes())
    recv_packet = recv_packet_func(cli_socket, [SYN_ACK])
    start_time = recv_packet.recv_time
    print("4-way handshake to establish connection was successful.")
except KeyboardInterrupt:
    print("Server received a keyboard interrupt before establishing connection. Terminating...")
    cli_socket.close(); serv_socket.close(); exit(0)
except Exception as err:
    print("The following error occured before establishing connection: " + str(err))
    traceback.print_exc()
    cli_socket.close(); serv_socket.close(); exit(1)

recv_packets = 1

try:
    while True:
        if recv_packets % 10000 == 0:
            print("Received " + str(recv_packets) + " packets.")
        recv_packet = recv_packet_func(cli_socket,[FIN, DATA])
        recv_time = time.time() - start_time; recv_packets += 1
        if recv_packet.typ == FIN: break
        elif recv_packet.typ == DATA: cli_socket.send(Packet(recv_packet.num, recv_packet.idx, ACK, send_time=recv_packet.send_time, recv_time=recv_time).to_bytes())
        else: raise Exception("Received an unexpected packet type: " + str(recv_packet.typ))
except KeyboardInterrupt: print("Server received a keyboard interrupt. Terminating...")
except socket.error as err:
    if err.errno == errno.ECONNRESET: print("Client has forcefully disconnected from server. Terminating...")
    else: print("The following error occured: " + str(err)); traceback.print_exc()
except Exception as err: print("The following error occured: " + str(err)); traceback.print_exc()
finally: 
    cli_socket.send(Packet(0, 0, FIN_ACK).to_bytes())
    cli_socket.send(Packet(0, 0, FIN).to_bytes())
    recv_packet_func(cli_socket, [FIN_ACK])
    print("4-way handshake to terminate connection was successful!")
    print("Closing the sockets."); cli_socket.close(); serv_socket.close()