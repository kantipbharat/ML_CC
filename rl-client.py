from helper import *

if len(sys.argv) == 1:
    print("Must include version!"); exit(1)

cli_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM); cli_socket.connect(ADDR); start_time = 0.0

try:
    cli_socket.send(Packet(0, 0, SYN).to_bytes())
    recv_packet_func(cli_socket, [SYN_ACK]); recv_packet_func(cli_socket, [SYN])
    start_time = time.time(); cli_socket.send(Packet(0, 0, SYN_ACK, recv_time=start_time).to_bytes())
    print("4-way handshake to establish connection was successful.")
except KeyboardInterrupt:
    print("Client received a keyboard interrupt before establishing connection. Terminating...")
    cli_socket.close(); exit(0)
except Exception as err:
    print("The following error occured before establishing connection: " + str(err))
    cli_socket.close(); traceback.print_exc(); exit(1)

running = True; pending_acks = {}; send_lock = threading.Lock()
curr_packet = 1; last_packet = 0; sent_packets = 0; lost_packets = 0

MSS = 1.0; cwnd = MSS; ssthresh = 64.0; cwnd_order = 1; last_ack = 0 # initial variables for congestion control
duplicate_acks = 0; recovery_phase = False; high_ack = 0
ewma_smoothing_factor = 0.7 # smoothing factor for EWMA calculations

prev_send, curr_send, inter_send, ratio_inter_send, ewma_inter_send = [np.NaN] * 5
prev_arr, curr_arr, inter_arr, ratio_inter_arr, ewma_inter_arr = [np.NaN] * 5
rtt, ratio_rtt = [np.NaN] * 2; min_inter_send, min_inter_arr, min_rtt = [sys.maxsize] * 3

ts_inter_send, ts_inter_arr, ts_rtt = [[np.NaN] * (TS_SIZE + 1)] * 3
ts_ratio_inter_send, ts_ratio_inter_arr, ts_ratio_rtt = [[np.NaN] * (TS_SIZE + 1)] * 3

throughput, max_throughput, loss_rate = ([0.0] * 3); delay = np.NaN
throughput_timestamps, loss_rate_timestamps = [deque()] * 2
delay_factor = 0.01; loss_rate_factor = 0.1; interval = 1
prev_utility, curr_utility, diff_utility, reward = [0.0] * 4

feature_vector = [ewma_inter_send, ewma_inter_arr, ratio_rtt, ssthresh, cwnd]
actions = [-1, 0, 1, 3]
l = 10; Q_table = np.zeros((l, l, l, l, l, len(actions)))

columns = ['num', 'idx', 'cwnd', 'cwnd_order']
columns += ['ewma_inter_send', 'min_inter_send']
columns += ['ts_inter_send_' + str(i + 1) for i in range(TS_SIZE)]
columns += ['ts_ratio_inter_send_' + str(i + 1) for i in range(TS_SIZE)]
columns += ['ewma_inter_arr', 'min_inter_arr']
columns += ['ts_inter_arr_' + str(i + 1) for i in range(TS_SIZE)]
columns += ['ts_ratio_inter_arr_' + str(i + 1) for i in range(TS_SIZE)]
columns += ['min_rtt'] + ['ts_rtt_' + str(i + 1) for i in range(TS_SIZE)]
columns += ['ts_ratio_rtt_' + str(i + 1) for i in range(TS_SIZE)]
columns += ['recvd']
columns += ['ssthresh', 'throughput', 'max_throughput', 'loss_rate', 'overall_loss_rate', 'delay']
columns += ['ratio_inter_send', 'ratio_inter_arr', 'ratio_rtt']

df = pd.DataFrame(columns=columns)
df = df.astype({"num":"int", "idx":"int", "cwnd_order":"int", "recvd":"int"})

csv_name = 'rl.csv'
if os.path.exists(csv_name): os.remove(csv_name)

version = sys.argv[1]
version_name = ''
if version == '0': version_name += 'aimd'
elif version == '1': version_name += 'newreno'
elif version == '2': version_name += 'lp'
elif version == '3': version_name += 'rl'
else: 
    print("Invaid version!"); exit(1)

ranges_name = 'objects/' + version_name + '.pkl'
ranges = pickle.load(open(ranges_name, 'rb'))

print(ranges)
exit(0)

row = [np.NaN] * len(columns)

def send_packs():
    global running, pending_acks, send_lock, df
    global curr_packet, last_packet, sent_packets, lost_packets
    global MSS, cwnd, ssthresh, cwnd_order, last_ack, ewma_smoothing_factor
    global duplicate_acks, recovery_phase, high_ack
    global prev_send, curr_send, inter_send, ratio_inter_send, ewma_inter_send
    global prev_arr, curr_arr, inter_arr, ratio_inter_arr, ewma_inter_arr
    global rtt, ratio_rtt, min_inter_send, min_inter_arr, min_rtt
    global ts_inter_send, ts_inter_arr, ts_rtt
    global ts_ratio_inter_send, ts_ratio_inter_arr, ts_ratio_rtt
    global throughput, max_throughput, loss_rate, delay
    global throughput_timestamps, loss_rate_timestamps
    global delay_factor, loss_rate_factor, interval
    global prev_utility, curr_utility, diff_utility, reward

    try:
        while running:
            if curr_packet > last_ack + math.floor(cwnd): continue
            idx = random.randint(1000, 9999); send_time = time.time() - start_time
            cli_socket.send(Packet(curr_packet, idx, DATA, send_time=send_time).to_bytes())

            with send_lock:
                pending_acks[curr_packet] = cwnd_order

                prev_send = curr_send; curr_send = send_time; inter_send = curr_send - prev_send
                if inter_send < min_inter_send: min_inter_send = inter_send
                if min_inter_send != 0: ratio_inter_send = inter_send / min_inter_send
                ts_inter_send = ts_inter_send[1:TS_SIZE + 1] + [inter_send]
                ts_ratio_inter_send = ts_ratio_inter_send[1:TS_SIZE + 1] + [ratio_inter_send]
                if not np.isnan(ewma_inter_send): ewma_inter_send = (inter_send * ewma_smoothing_factor) + (ewma_inter_send * (1 - ewma_smoothing_factor))
                if np.isnan(ewma_inter_send) and not np.isnan(inter_send): ewma_inter_send = inter_send

                throughput = len(throughput_timestamps) / interval
                if throughput > max_throughput: max_throughput = throughput
                loss_rate = len(loss_rate_timestamps) / interval; delay = rtt - min_rtt
 
                throughput_utility = 0; delay_utility = 0; loss_rate_utility = 0
                if max_throughput != 0: np.log(throughput / max_throughput)
                if delay != 0 and delay != np.NaN: delay_factor * np.log(delay)
                if loss_rate != 0: loss_rate_factor * np.log(loss_rate)
                utility = throughput_utility - delay_utility + loss_rate_utility
                prev_utility = curr_utility; curr_utility = utility; diff_utility = curr_utility - prev_utility

                if diff_utility >= 1: reward = 10
                elif diff_utility < 1 and diff_utility >= 0: reward = 2
                elif diff_utility < 0 and diff_utility >= -1: reward = -2
                elif diff_utility < -1: reward = -10

                feature_vector = [ewma_inter_send, ewma_inter_arr, ratio_rtt, ssthresh, cwnd]

                row = [curr_packet, idx, cwnd, cwnd_order]
                row += [ewma_inter_send, min_inter_send]
                row += [ts_inter_send[i] - ts_inter_send[0] for i in range(1, TS_SIZE + 1)] 
                row += [ts_ratio_inter_send[i] - ts_ratio_inter_send[0] for i in range(1, TS_SIZE + 1)]
                row += [ewma_inter_arr, min_inter_arr]
                row += [ts_inter_arr[i] - ts_inter_arr[0] for i in range(1, TS_SIZE + 1)] 
                row += [ts_ratio_inter_arr[i] - ts_ratio_inter_arr[0] for i in range(1, TS_SIZE + 1)]
                row += [min_rtt] + [ts_rtt[i] - ts_rtt[0] for i in range(1, TS_SIZE + 1)] 
                row += [ts_ratio_rtt[i] - ts_ratio_rtt[0] for i in range(1, TS_SIZE + 1)]
                row += [0] + [ssthresh, throughput, max_throughput, loss_rate, overall_loss_rate, delay]
                row += [ratio_inter_send, ratio_inter_arr, ratio_rtt]
                df.loc[len(df)] = row

                cwnd_order += 1; sent_packets += 1; last_packet = curr_packet; curr_packet += 1
                if cwnd_order > cwnd: cwnd_order = 1
                if sent_packets % 10000 == 0: print("Sent " + str(sent_packets) + " packets.")

                if len(df) < 5000: continue
                df = df.dropna()
                df.to_csv(csv_name, mode='a', header=not os.path.exists(csv_name))
                df = df.iloc[0:0]
                
    except Exception as err:
        print("The following error occured while sending packets: " + str(err))
        traceback.print_exc(); return

def recv_acks():
    global running, pending_acks, send_lock, df
    global curr_packet, last_packet, sent_packets, lost_packets
    global MSS, cwnd, ssthresh, cwnd_order, last_ack, ewma_smoothing_factor
    global duplicate_acks, recovery_phase, high_ack
    global prev_send, curr_send, inter_send, ratio_inter_send, ewma_inter_send
    global prev_arr, curr_arr, inter_arr, ratio_inter_arr, ewma_inter_arr
    global rtt, ratio_rtt, min_inter_send, min_inter_arr, min_rtt
    global ts_inter_send, ts_inter_arr, ts_rtt
    global ts_ratio_inter_send, ts_ratio_inter_arr, ts_ratio_rtt
    global throughput, max_throughput, loss_rate, delay
    global throughput_timestamps, loss_rate_timestamps
    global delay_factor, loss_rate_factor, interval
    global prev_utility, curr_utility, diff_utility, reward

    try:
        while running or pending_acks:
            recv_packet = recv_packet_func(cli_socket, [ACK])
            ack_recv_time = time.time() - start_time
            with send_lock:
                df.loc[(df['num'] == recv_packet.num) & (df['idx'] == recv_packet.idx), 'recvd'] = 1
                current_time = time.time() - start_time; throughput_timestamps.append(ack_recv_time)
                while throughput_timestamps and current_time - throughput_timestamps[0] > interval: throughput_timestamps.popleft()

                if recv_packet.num not in pending_acks:
                    duplicate_acks += 1
                    if duplicate_acks == 3 and not recovery_phase:
                        ssthresh = max(cwnd / 2, 2); cwnd = ssthresh + (3 * MSS)
                        recovery_phase = True
                        high_ack = last_ack
                    elif recovery_phase: cwnd += MSS
                    continue

                del pending_acks[recv_packet.num]; last_ack += 1

                prev_arr = curr_arr; curr_arr = ack_recv_time; inter_arr = curr_arr - prev_arr
                if inter_arr < min_inter_arr: min_inter_arr = inter_arr
                if min_inter_arr != 0: ratio_inter_arr = inter_arr / min_inter_arr
                ts_inter_arr = ts_inter_arr[1:TS_SIZE + 1] + [inter_arr]
                ts_ratio_inter_arr = ts_ratio_inter_arr[1:TS_SIZE + 1] + [ratio_inter_arr]
                if not np.isnan(ewma_inter_arr): ewma_inter_arr = (inter_arr * ewma_smoothing_factor) + (ewma_inter_arr * (1 - ewma_smoothing_factor))
                if np.isnan(ewma_inter_arr) and not np.isnan(inter_arr): ewma_inter_arr = inter_arr

                rtt = ack_recv_time - recv_packet.send_time
                if rtt < min_rtt: min_rtt = rtt
                if min_rtt != 0: ratio_rtt = float(rtt) / float(min_rtt)
                ts_rtt = ts_rtt[1:TS_SIZE + 1] + [rtt]
                ts_ratio_rtt = ts_ratio_rtt[1:TS_SIZE + 1] + [ratio_rtt]

                if recovery_phase:
                    if recv_packet.num >= high_ack:
                        cwnd = ssthresh; recovery_phase = False
                    else: cwnd = max(ssthresh + ((duplicate_acks - 3) * MSS), MSS)
                    duplicate_acks = 0
                    continue

                if cwnd < ssthresh: cwnd += MSS
                else: cwnd += MSS * (MSS / cwnd)
                if cwnd_order > cwnd: cwnd_order = 1
                duplicate_acks = 0

    except Exception as err:
        print("The following error occured while receiving packets: " + str(err))
        traceback.print_exc(); return

def retransmit():
    global running, pending_acks, send_lock, df
    global curr_packet, last_packet, sent_packets, lost_packets
    global MSS, cwnd, ssthresh, cwnd_order, last_ack, ewma_smoothing_factor
    global duplicate_acks, recovery_phase, high_ack
    global prev_send, curr_send, inter_send, ratio_inter_send, ewma_inter_send
    global prev_arr, curr_arr, inter_arr, ratio_inter_arr, ewma_inter_arr
    global rtt, ratio_rtt, min_inter_send, min_inter_arr, min_rtt
    global ts_inter_send, ts_inter_arr, ts_rtt
    global ts_ratio_inter_send, ts_ratio_inter_arr, ts_ratio_rtt
    global throughput, max_throughput, loss_rate, delay
    global throughput_timestamps, loss_rate_timestamps
    global delay_factor, loss_rate_factor, interval
    global prev_utility, curr_utility, diff_utility, reward

    try:
        while running or pending_acks:
            time.sleep(TIMEOUT)
            with send_lock:
                if pending_acks:
                    ssthresh = max(cwnd / 2, 2); cwnd = 1
                    duplicate_acks = 0; recovery_phase = True; high_ack = last_ack
                for num in pending_acks.keys():
                    if num > last_packet: continue
                    if len(df[df['num'] == num]) >= MAX_TRANSMIT:
                        del pending_acks[num]; continue
                    
                    current_time = time.time() - start_time; loss_rate_timestamps.append(current_time)
                    while loss_rate_timestamps and current_time - loss_rate_timestamps[0] > interval: loss_rate_timestamps.popleft()

                    print('Lost packet ' + str(num) + '. Attempting to retransmit.')
                    idx = random.randint(1000, 9999); send_time = time.time() - start_time
                    cli_socket.send(Packet(num, idx, DATA, send_time=send_time).to_bytes())

                    prev_send = curr_send; curr_send = send_time; inter_send = curr_send - prev_send
                    if inter_send < min_inter_send: min_inter_send = inter_send
                    if min_inter_send != 0: ratio_inter_send = inter_send / min_inter_send
                    ts_inter_send = ts_inter_send[1:TS_SIZE + 1] + [inter_send]
                    ts_ratio_inter_send = ts_ratio_inter_send[1:TS_SIZE + 1] + [ratio_inter_send]
                    if not np.isnan(ewma_inter_send): ewma_inter_send = (inter_send * ewma_smoothing_factor) + (ewma_inter_send * (1 - ewma_smoothing_factor))
                    if np.isnan(ewma_inter_send) and not np.isnan(inter_send): ewma_inter_send = inter_send

                    throughput = len(throughput_timestamps) / interval
                    if throughput > max_throughput: max_throughput = throughput
                    loss_rate = len(loss_rate_timestamps) / interval; delay = rtt - min_rtt

                    throughput_utility = 0; delay_utility = 0; loss_rate_utility = 0
                    if max_throughput != 0: np.log(throughput / max_throughput)
                    if delay != 0 and delay != np.NaN: delay_factor * np.log(delay)
                    if loss_rate != 0: loss_rate_factor * np.log(loss_rate)
                    utility = throughput_utility - delay_utility + loss_rate_utility
                    prev_utility = curr_utility; curr_utility = utility; diff_utility = curr_utility - prev_utility

                    if diff_utility >= 1: reward = 10
                    elif diff_utility < 1 and diff_utility >= 0: reward = 2
                    elif diff_utility < 0 and diff_utility >= -1: reward = -2
                    elif diff_utility < -1: reward = -10

                    feature_vector = [ewma_inter_send, ewma_inter_arr, ratio_rtt, ssthresh, cwnd]

                    row = [num, idx, cwnd, pending_acks[num]]
                    row += [ewma_inter_send, min_inter_send]
                    row += [ts_inter_send[i] - ts_inter_send[0] for i in range(1, TS_SIZE + 1)] 
                    row += [ts_ratio_inter_send[i] - ts_ratio_inter_send[0] for i in range(1, TS_SIZE + 1)]
                    row += [ewma_inter_arr, min_inter_arr]
                    row += [ts_inter_arr[i] - ts_inter_arr[0] for i in range(1, TS_SIZE + 1)] 
                    row += [ts_ratio_inter_arr[i] - ts_ratio_inter_arr[0] for i in range(1, TS_SIZE + 1)]
                    row += [min_rtt] + [ts_rtt[i] - ts_rtt[0] for i in range(1, TS_SIZE + 1)] 
                    row += [0] + [ssthresh, throughput, max_throughput, loss_rate, overall_loss_rate, delay]
                    row += [ratio_inter_send, ratio_inter_arr, ratio_rtt]
                    df.loc[len(df)] = row

                    sent_packets += 1; lost_packets += 1

                    if len(df) < 5000: continue
                    df = df.dropna()
                    df.to_csv(csv_name, mode='a', header=not os.path.exists(csv_name))
                    df = df.iloc[0:0]

    except Exception as err:
        print("The following error occured while retransmitting packets: " + str(err))
        traceback.print_exc(); return

send_thread = threading.Thread(target=send_packs, daemon=True)
recv_thread = threading.Thread(target=recv_acks, daemon=True)
retransmit_thread = threading.Thread(target=retransmit, daemon=True)
send_thread.start(); recv_thread.start(); retransmit_thread.start()

try:
    while running: 
        time.sleep(0.1)
        if time.time() - start_time > RUNTIME:
            running = False
except KeyboardInterrupt: print("Client received a keyboard interrupt. Terminating..."); running = False
except Exception as err: print("The following error occured: " + str(err)); running = False
finally:
    print("Stopping the client...")

    try:
        send_thread.join(); recv_thread.join(); retransmit_thread.join()
        cli_socket.send(Packet(0, 0, FIN).to_bytes())
        recv_packet_func(cli_socket, [FIN_ACK]); recv_packet_func(cli_socket, [FIN])
        cli_socket.send(Packet(0, 0, FIN_ACK).to_bytes())
        print("4-way handshake to terminate connection was successful.")
    except KeyboardInterrupt: print("Client received a keyboard interrupt before terminating the connection. Terminating...")
    except Exception as err: print("The following error occured before terminating the connection: " + str(err)); traceback.print_exc()
    finally:
        if cli_socket: cli_socket.close()
        
        df = df.dropna(); df.to_csv(csv_name, mode='a', header=not os.path.exists(csv_name))

        if sent_packets > 0:
            overall_loss_rate = round((lost_packets / sent_packets) * 100, 8)
            print("Packet loss rate was: " + str(overall_loss_rate) + "%")