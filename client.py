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
    if recv_packet.typ != SYN_ACK: raise Exception("Did not receive a SYN_ACK packet to establish the connection. Terminating...")

    data = cli_socket.recv(PACKET_SIZE)
    if not data: raise Exception("No information received. Terminating...")
    if len(data) > PACKET_SIZE: raise Exception("Too many bytes received. Terminating...")
    if len(data) < PACKET_SIZE: raise Exception("Too few bytes received. Terminating...")
    recv_packet = Packet.from_bytes(data)
    if recv_packet.typ != SYN: raise Exception("Did not receive a SYN packet to establish the connection. Terminating...")
    
    start_time = time.time()
    print(start_time)
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

running = True
pending_acks = {}
curr_packet = 1
last_packet = 0
sent_packets = 0
lost_packets = 0
send_lock = threading.Lock()

cwnd = 1.0
ssthresh = 64
cwnd_order = 1
last_ack = 0

def send_packs():
    global running, pending_acks, curr_packet, last_packet, sent_packets, lost_packets, cwnd, cwnd_order, last_ack
    try:
        while running:
            if curr_packet > last_ack + math.floor(cwnd):
                continue
            idx = random.randint(1000, 10000)
            send_time = time.time() - start_time
            cli_socket.sendall(Packet(curr_packet, idx, DATA).to_bytes())
            with send_lock:
                cwnd_order += 1
                if cwnd_order > cwnd:
                    cwnd_order = 1
                loss_rate = 0
                if sent_packets != 0:
                    loss_rate = float(lost_packets) / float(sent_packets)
                pending_acks[curr_packet] = {'cwnd_order', cwnd_order}
                sent_packets += 1; last_packet = curr_packet; curr_packet += 1
            if sent_packets % 10000 == 0:
                print("Sent " + str(sent_packets) + " packets.")
    except Exception as err:
        print("The following error occured while sending packets: " + str(err))
        traceback.print_exc()
        return

def recv_acks():
    global running, pending_acks, cwnd, cwnd_order, last_ack
    try:
        while running or pending_acks:
            data = cli_socket.recv(PACKET_SIZE)
            ack_recv_time = time.time() - start_time
            if not data: raise Exception("No information received. Terminating...")
            if len(data) > PACKET_SIZE: raise Exception("Too many bytes received. Terminating...")
            if len(data) < PACKET_SIZE: raise Exception("Too few bytes received. Terminating...")
            recv_packet = Packet.from_bytes(data)
            if recv_packet.typ != ACK: raise Exception("The received packet is not a regular ACK packet. Terminating...")
            with send_lock:
                if recv_packet.num in pending_acks:
                    del pending_acks[recv_packet.num]
                    last_ack += 1
                    if cwnd < ssthresh: cwnd += 1.0
                    else: cwnd += 1.0 / cwnd
                    if cwnd_order > cwnd: cwnd_order = 1
    except Exception as err:
        print("The following error occured while receiving packets: " + str(err))
        traceback.print_exc()
        return

def retransmit():
    global running, pending_acks, last_packet, sent_packets, lost_packets, cwnd
    try:
        while running or pending_acks:
            time.sleep(TIMEOUT)
            with send_lock:
                if pending_acks: sshtresh = max(cwnd / 2, 2); cwnd = 1
                for num in pending_acks.keys():
                    if num <= last_packet:
                        lost_packets += 1
                        idx = random.randint(1000, 10000)
                        send_time = time.time() - start_time
                        print("Lost packet " + str(num) + ". Attempting to retransmit.")
                        cli_socket.sendall(Packet(num, idx, DATA).to_bytes())
                        loss_rate = 0
                        if sent_packets == 0:
                            loss_rate = float(lost_packets) / float(sent_packets)
    except Exception as err:
        print("The following error occured while retransmitting packets: " + str(err))
        traceback.print_exc()
        return

send_thread = threading.Thread(target=send_packs, daemon=True)
recv_thread = threading.Thread(target=recv_acks, daemon=True)
retransmit_thread = threading.Thread(target=retransmit, daemon=True)

send_thread.start()
recv_thread.start()
retransmit_thread.start()

try:
    while running: time.sleep(0.1)
except KeyboardInterrupt:
    print("Client received a keyboard interrupt. Terminating..."); running = False
except Exception as err:
    print("The following error occured: " + str(err)); running = False
finally:
    print("Stopping the client...")

    try:
        send_thread.join()
        recv_thread.join()
        retransmit_thread.join()

        cli_socket.sendall(Packet(0, 0, FIN).to_bytes())

        data = cli_socket.recv(PACKET_SIZE)
        if not data: raise Exception("No information received. Terminating...")
        if len(data) > PACKET_SIZE: raise Exception("Too many bytes received. Terminating...")
        if len(data) < PACKET_SIZE: raise Exception("Too few bytes received. Terminating...")
        recv_packet = Packet.from_bytes(data)
        if recv_packet.typ != FIN_ACK: raise Exception("Did not receive a FIN_ACK packet to terminate the connection. Terminating...")
        
        data = cli_socket.recv(PACKET_SIZE)
        if not data: raise Exception("No information received. Terminating...")
        if len(data) > PACKET_SIZE: raise Exception("Too many bytes received. Terminating...")
        if len(data) < PACKET_SIZE: raise Exception("Too few bytes received. Terminating...")
        recv_packet = Packet.from_bytes(data)
        if recv_packet.typ != FIN: raise Exception("Did not receive a FIN packet to establish the connection. Terminating...")

        cli_socket.sendall(Packet(0, 0, FIN_ACK).to_bytes())
        print("4-way handshake to terminate connection was successful.\n")
    except KeyboardInterrupt:
        print("Client received a keyboard interrupt before terminating the connection. Terminating...")
    except Exception as err:
        print("The following error occured before terminating the connection: " + str(err)); traceback.print_exc()
    finally:
        if cli_socket: cli_socket.close()
        if sent_packets > 0:
            loss_rate = round((lost_packets / sent_packets) * 100, 8)
            print("Packet loss rate was: " + str(loss_rate) + "%")