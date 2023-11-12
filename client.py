from helper import *

cli_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM); cli_socket.connect(ADDR); start_time = 0.0

try:
    cli_socket.send(Packet(0, 0, SYN).to_bytes())
    
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
    cli_socket.send(Packet(0, 0, SYN_ACK, recv_time=start_time).to_bytes())
    print("4-way handshake to establish connection was successful.")
except KeyboardInterrupt:
    print("Client received a keyboard interrupt before establishing connection. Terminating...")
    cli_socket.close(); exit(0)
except Exception as err:
    print("The following error occured before establishing connection: " + str(err))
    cli_socket.close(); traceback.print_exc(); exit(1)

running = True; pending_acks = {}; send_lock = threading.Lock()
curr_packet = 1; last_packet = 0; sent_packets = 0; lost_packets = 0
cwnd = 1.0; ssthresh = 64; cwnd_order = 1; last_ack = 0

# throughput
latency = np.NaN; rtt = np.NaN; min_rtt = sys.maxsize
prev_send = np.NaN; curr_send = np.NaN; inter_send = np.NaN; min_inter_send = sys.maxsize
prev_arr = np.NaN; curr_arr = np.NaN; inter_arr = np.NaN; min_inter_arr = sys.maxsize
ratio_inter_send = np.NaN; ratio_inter_arr = np.NaN; ratio_rtt = np.NaN

ts_inter_send = [np.NaN] * (TS_SIZE + 1); ts_inter_arr = [np.NaN] * (TS_SIZE + 1); ts_rtt = [np.NaN] * (TS_SIZE + 1)
ts_ratio_inter_send = [np.NaN] * (TS_SIZE + 1); ts_ratio_inter_arr = [np.NaN] * (TS_SIZE + 1); ts_ratio_rtt = [np.NaN] * (TS_SIZE + 1)
ewma_inter_send = np.NaN; ewma_inter_arr = np.NaN; ewma_rtt = np.NaN

timestamps = deque()
window_size = 1
throughput = 0.0
reward = 0.0

alpha = 0.2
beta = 0.2
gamma = 0.2
delta = 0.2

columns = ['num', 'idx', 'cwnd', 'cwnd_order', 'ssthresh', 'send_time']
columns += ['latency', 'rtt', 'loss_rate']
columns += ['inter_send', 'min_inter_send', 'ewma_inter_send']
columns += ['ts_inter_send_' + str(i + 1) for i in range(TS_SIZE)]
columns += ['ratio_inter_send'] + ['ts_ratio_inter_send_' + str(i + 1) for i in range(TS_SIZE)]
columns += ['inter_arr', 'min_inter_arr', 'ewma_inter_arr']
columns += ['ts_inter_arr_' + str(i + 1) for i in range(TS_SIZE)]
columns += ['ratio_inter_arr'] + ['ts_ratio_inter_arr_' + str(i + 1) for i in range(TS_SIZE)]
columns += ['rtt', 'min_rtt', 'ewma_rtt']
columns += ['ts_rtt_' + str(i + 1) for i in range(TS_SIZE)]
columns += ['ratio_rtt'] + ['ts_ratio_rtt_' + str(i + 1) for i in range(TS_SIZE)]
columns += ['throughput', 'reward', 'recvd']
print(len(columns))
print()
df = pd.DataFrame(columns=columns)

def send_packs():
    global running, pending_acks, send_lock
    global curr_packet, last_packet, sent_packets, lost_packets
    global cwnd, ssthresh, cwnd_order, last_ack
    global latency, rtt, min_rtt
    global prev_send, curr_send, inter_send, min_inter_send
    global prev_arr, curr_arr, inter_arr, min_inter_arr
    global ratio_inter_send, ratio_inter_arr, ratio_rtt
    global ts_inter_send, ts_inter_arr, ts_rtt
    global ts_ratio_inter_send, ts_ratio_inter_arr, ts_ratio_rtt
    global ewma_inter_send, ewma_inter_arr, ewma_rtt
    global timestamps, window_size, throughput, reward

    try:
        while running:
            if curr_packet > last_ack + math.floor(cwnd): continue
            idx = random.randint(1000, 9999)
            send_time = time.time() - start_time
            cli_socket.send(Packet(curr_packet, idx, DATA, send_time=send_time).to_bytes())
            with send_lock:
                pending_acks[curr_packet] = {'cwnd_order':cwnd_order}

                prev_send = curr_send; curr_send = send_time; inter_send = curr_send - prev_send
                if inter_send < min_inter_send: min_inter_send = inter_send

                loss_rate = 0.0
                if sent_packets != 0: loss_rate = float(lost_packets) / float(sent_packets)
                if min_inter_send != 0: ratio_inter_send = inter_send / min_inter_send

                ts_inter_send = ts_inter_send[1:TS_SIZE + 1] + [inter_send]
                ts_ratio_inter_send = ts_ratio_inter_send[1:TS_SIZE + 1] + [ratio_inter_send]

                if not np.isnan(ewma_inter_send): ewma_inter_send = (inter_send * alpha) + (ewma_inter_send * (1 - alpha))
                if np.isnan(ewma_inter_send) and not np.isnan(inter_send): ewma_inter_send = inter_send

                throughput = len(timestamps) / window_size
                reward = (beta * throughput) - (gamma * latency) - (delta * loss_rate)

                row = [curr_packet, idx, cwnd, cwnd_order, ssthresh, send_time]
                row += [latency, rtt, loss_rate]
                row += [inter_send, min_inter_send, ewma_inter_send]
                row += [ts_inter_send[i] - ts_inter_send[0] for i in range(1, TS_SIZE + 1)] 
                row += [ratio_inter_send] + [ts_ratio_inter_send[i] - ts_ratio_inter_send[0] for i in range(1, TS_SIZE + 1)]
                row += [inter_arr, min_inter_arr, ewma_inter_arr] 
                row += [ts_inter_arr[i] - ts_inter_arr[0] for i in range(1, TS_SIZE + 1)]
                row += [ratio_inter_arr] + [ts_ratio_inter_arr[i] - ts_ratio_inter_arr[0] for i in range(1, TS_SIZE + 1)]
                row += [rtt, min_rtt, ewma_rtt] + [ts_rtt[i] - ts_rtt[0] for i in range(1, TS_SIZE + 1)]
                row += [ratio_rtt] + [ts_ratio_rtt[i] - ts_ratio_rtt[0] for i in range(1, TS_SIZE + 1)]
                row += [throughput, reward]
                df.loc[len(df)] = row + [-1]

                cwnd_order += 1
                if cwnd_order > cwnd: cwnd_order = 1
                sent_packets += 1; last_packet = curr_packet; curr_packet += 1
            if sent_packets % 100000 == 0: print("Sent " + str(sent_packets) + " packets.")
    except Exception as err:
        print("The following error occured while sending packets: " + str(err))
        traceback.print_exc(); return

def recv_acks():
    global running, pending_acks, send_lock
    global curr_packet, last_packet, sent_packets, lost_packets
    global cwnd, ssthresh, cwnd_order, last_ack
    global latency, rtt, min_rtt
    global prev_send, curr_send, inter_send, min_inter_send
    global prev_arr, curr_arr, inter_arr, min_inter_arr
    global ratio_inter_send, ratio_inter_arr, ratio_rtt
    global ts_inter_send, ts_inter_arr, ts_rtt
    global ts_ratio_inter_send, ts_ratio_inter_arr, ts_ratio_rtt
    global ewma_inter_send, ewma_inter_arr, ewma_rtt
    global timestamps, window_size, throughput, reward

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
                df.loc[(df['num'] == recv_packet.num) & (df['idx'] == recv_packet.idx), 'recvd'] = 1
                timestamps.append(ack_recv_time); current_time = time.time() - start_time
                while timestamps and current_time - timestamps[0] > window_size: timestamps.popleft()
                if recv_packet.num in pending_acks:
                    del pending_acks[recv_packet.num]; last_ack += 1

                    prev_arr = curr_arr; curr_arr = ack_recv_time; inter_arr = curr_arr - prev_arr
                    if inter_arr < min_inter_arr: min_inter_arr = inter_arr

                    latency = recv_packet.recv_time - recv_packet.send_time
                    rtt = ack_recv_time - recv_packet.send_time
                    if rtt < min_rtt: min_rtt = rtt
                    if min_inter_arr != 0: ratio_inter_arr = inter_arr / min_inter_arr
                    if min_rtt != 0: ratio_rtt = float(rtt) / float(min_rtt)

                    ts_inter_arr = ts_inter_arr[1:TS_SIZE + 1] + [inter_arr]
                    ts_rtt = ts_rtt[1:TS_SIZE + 1] + [rtt]
                    ts_ratio_inter_arr = ts_ratio_inter_arr[1:TS_SIZE + 1] + [ratio_inter_arr]
                    ts_ratio_rtt = ts_ratio_rtt[1:TS_SIZE + 1] + [ratio_rtt]

                    if not np.isnan(ewma_inter_arr): ewma_inter_arr = (inter_arr * alpha) + (ewma_inter_arr * (1 - alpha))
                    if np.isnan(ewma_inter_arr) and not np.isnan(inter_arr): ewma_inter_arr = inter_arr
                    if not np.isnan(ewma_rtt): ewma_rtt = (rtt * alpha) + (rtt * (1 - alpha))
                    if np.isnan(ewma_rtt) and not np.isnan(rtt): ewma_rtt = rtt

                    if cwnd < ssthresh: cwnd += 1.0
                    else: cwnd += 1.0 / cwnd
                    if cwnd_order > cwnd: cwnd_order = 1
    except Exception as err:
        print("The following error occured while receiving packets: " + str(err))
        traceback.print_exc()
        return

def retransmit():
    global running, pending_acks, send_lock
    global curr_packet, last_packet, sent_packets, lost_packets
    global cwnd, ssthresh, cwnd_order, last_ack
    global latency, rtt, min_rtt
    global prev_send, curr_send, inter_send, min_inter_send
    global prev_arr, curr_arr, inter_arr, min_inter_arr
    global ratio_inter_send, ratio_inter_arr, ratio_rtt
    global ts_inter_send, ts_inter_arr, ts_rtt
    global ts_ratio_inter_send, ts_ratio_inter_arr, ts_ratio_rtt
    global ewma_inter_send, ewma_inter_arr, ewma_rtt
    global timestamps, window_size, throughput, reward

    try:
        while running or pending_acks:
            time.sleep(TIMEOUT)
            with send_lock:
                if pending_acks: 
                    ssthresh = max(cwnd / 2, 2); cwnd = 1
                for num in pending_acks.keys():
                    if num <= last_packet:
                        print('Lost packet ' + str(num) + '. Attempting to retransmit.')
                        idx = random.randint(1000, 9999)
                        send_time = time.time() - start_time
                        cli_socket.send(Packet(num, idx, DATA, send_time=send_time).to_bytes())

                        prev_send = curr_send; curr_send = send_time; inter_send = curr_send - prev_send
                        if inter_send < min_inter_send: min_inter_send = inter_send

                        loss_rate = 0.0
                        if sent_packets != 0: loss_rate = float(lost_packets) / float(sent_packets)
                        if min_inter_send != 0: ratio_inter_send = inter_send / min_inter_send

                        ts_inter_send = ts_inter_send[1:TS_SIZE + 1] + [inter_send]
                        ts_ratio_inter_send = ts_ratio_inter_send[1:TS_SIZE + 1] + [ratio_inter_send]

                        if not np.isnan(ewma_inter_send): ewma_inter_send = (inter_send * alpha) + (ewma_inter_send * (1 - alpha))
                        if np.isnan(ewma_inter_send) and not np.isnan(inter_send): ewma_inter_send = inter_send

                        throughput = len(timestamps) / window_size
                        reward = (beta * throughput) - (gamma * latency) - (delta * loss_rate)

                        row = [num, idx, cwnd, pending_acks[num]['cwnd_order'], ssthresh, send_time]
                        row += [latency, rtt, loss_rate]
                        row += [inter_send, min_inter_send, ewma_inter_send]
                        row += [ts_inter_send[i] - ts_inter_send[0] for i in range(1, TS_SIZE + 1)] 
                        row += [ratio_inter_send] + [ts_ratio_inter_send[i] - ts_ratio_inter_send[0] for i in range(1, TS_SIZE + 1)]
                        row += [inter_arr, min_inter_arr, ewma_inter_arr]
                        row += [ts_inter_arr[i] - ts_inter_arr[0] for i in range(1, TS_SIZE + 1)]
                        row += [ratio_inter_arr] + [ts_ratio_inter_arr[i] - ts_ratio_inter_arr[0] for i in range(1, TS_SIZE + 1)]
                        row += [rtt, min_rtt, ewma_rtt] + [ts_rtt[i] - ts_rtt[0] for i in range(1, TS_SIZE + 1)]
                        row += [ratio_rtt] + [ts_ratio_rtt[i] - ts_ratio_rtt[0] for i in range(1, TS_SIZE + 1)]
                        row += [throughput, reward]
                        df.loc[len(df)] = row + [-1]

                        sent_packets += 1; lost_packets += 1
    except Exception as err:
        print("The following error occured while retransmitting packets: " + str(err))
        traceback.print_exc(); return

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

        cli_socket.send(Packet(0, 0, FIN).to_bytes())

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

        cli_socket.send(Packet(0, 0, FIN_ACK).to_bytes())
        print("4-way handshake to terminate connection was successful.")
    except KeyboardInterrupt: print("Client received a keyboard interrupt before terminating the connection. Terminating...")
    except Exception as err: print("The following error occured before terminating the connection: " + str(err)); traceback.print_exc()
    finally:
        if cli_socket: cli_socket.close()

        df = df.astype({"num":"int", "idx":"int", "cwnd_order":"int", "recvd":"int"})
        df = df.dropna()
        df.to_csv('out.csv')

        if sent_packets > 0:
            loss_rate = round((lost_packets / sent_packets) * 100, 8)
            print("Packet loss rate was: " + str(loss_rate) + "%")