from helper import *

if len(sys.argv) == 1:
    print("Must include version!"); exit(1)

cli_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM); cli_socket.connect(ADDR); start_time = 0.0
cli_socket.settimeout(5)

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

throughput, max_throughput, loss_rate, overall_loss_rate = ([0.0] * 4); delay = np.NaN
throughput_timestamps, loss_rate_timestamps = [deque()] * 2
delay_factor = 0.01; loss_rate_factor = 0.1; interval = 1
prev_utility, curr_utility, diff_utility, reward = [0.0] * 4

state_cols = ['ewma_inter_send', 'ewma_inter_arr', 'ratio_rtt', 'ssthresh', 'cwnd']
state_vector = [ewma_inter_send, ewma_inter_arr, ratio_rtt, ssthresh, cwnd]
state_bin_vector = [0] * 5; actions = [-1, 0, 1, 3]
l = 10; discount_factor = 0.4; epsilon = 0.1; cwnd_increase = 1.0
Q_table = np.zeros((l, l, l, l, l, len(actions)))
history = [{'state':state_bin_vector, 'i':0, 'n-2':0.0, 'n-1':0.0}] * 2

df = pd.DataFrame(columns=COLUMNS)
df = df.astype({"num":"int", "idx":"int", "cwnd_order":"int", "recvd":"int"})

csv_name = 'datasets/rl.csv'
if os.path.exists(csv_name): os.remove(csv_name)

version = sys.argv[1]; version_name = ''
if version in VERSION_MAP.keys(): version_name = VERSION_MAP[version]
else:
    print("Invaid version!"); exit(1)

ranges_name = 'objects/' + version_name + '.pkl'
ranges = pickle.load(open(ranges_name, 'rb'))

def find_bin(col, val):
    global ranges, l
    min_val, max_val = ranges[col]
    bin_size = (max_val - min_val) / l
    if val < min_val: return 0
    elif val > max_val: return l - 1
    else: 
        bin_val = int((val - min_val) // bin_size)
        if bin_val < 0: bin_val == 0
        if bin_val >= l - 1: bin_val = l - 1
        return bin_val

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
    global throughput, max_throughput, loss_rate, overall_loss_rate, delay
    global throughput_timestamps, loss_rate_timestamps
    global delay_factor, loss_rate_factor, interval
    global prev_utility, curr_utility, diff_utility, reward
    global state_cols, state_vector, actions, discount_factor, epsilon, cwnd_increase, Q_table, history

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
                if sent_packets != 0: overall_loss_rate = lost_packets / sent_packets

                row = [curr_packet, idx, cwnd, cwnd_order]
                row += [ewma_inter_send, min_inter_send]
                row += [ts_inter_send[i] - ts_inter_send[0] for i in range(1, TS_SIZE + 1)] 
                row += [ts_ratio_inter_send[i] - ts_ratio_inter_send[0] for i in range(1, TS_SIZE + 1)]
                row += [ewma_inter_arr, min_inter_arr]
                row += [ts_inter_arr[i] - ts_inter_arr[0] for i in range(1, TS_SIZE + 1)] 
                row += [ts_ratio_inter_arr[i] - ts_ratio_inter_arr[0] for i in range(1, TS_SIZE + 1)]
                row += [min_rtt] + [ts_rtt[i] - ts_rtt[0] for i in range(1, TS_SIZE + 1)] 
                row += [ts_ratio_rtt[i] - ts_ratio_rtt[0] for i in range(1, TS_SIZE + 1)]
                input_row = [ssthresh, throughput, max_throughput, loss_rate, overall_loss_rate, delay]
                input_row += [ratio_inter_send, ratio_inter_arr, ratio_rtt]
                df.loc[len(df)] = row + [0] + input_row

                cwnd_order += 1; sent_packets += 1; last_packet = curr_packet; curr_packet += 1
                if cwnd_order > cwnd: cwnd_order = 1
                if sent_packets % 10000 == 0: print("Sent " + str(sent_packets) + " packets.")

                if len(df) < 5000: continue
                df.to_csv(csv_name, mode='a', header=not os.path.exists(csv_name)); df = df.iloc[0:0]
                
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
    global throughput, max_throughput, loss_rate, overall_loss_rate, delay
    global throughput_timestamps, loss_rate_timestamps
    global delay_factor, loss_rate_factor, interval
    global prev_utility, curr_utility, diff_utility, reward
    global state_cols, state_vector, actions, discount_factor, epsilon, cwnd_increase, Q_table, history

    try:
        while running or pending_acks:
            recv_packet = recv_packet_func(cli_socket, [ACK])
            ack_recv_time = time.time() - start_time
            with send_lock:
                df.loc[(df['num'] == recv_packet.num) & (df['idx'] == recv_packet.idx), 'recvd'] = 1
                current_time = time.time() - start_time; throughput_timestamps.append(ack_recv_time)
                while throughput_timestamps and current_time - throughput_timestamps[0] > interval: throughput_timestamps.popleft()

                if recv_packet.num not in pending_acks: continue
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

                if cwnd < ssthresh: cwnd += MSS
                else: cwnd += max(cwnd_increase / cwnd, MSS)
                if cwnd_order > cwnd: cwnd_order = 1

                state_vector = [ewma_inter_send, ewma_inter_arr, ratio_rtt, ssthresh, cwnd]
                if np.any(np.isnan(state_vector)): continue

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

                state_bin_vector = [find_bin(state_cols[i], state_vector[i]) for i in range(len(state_vector))]
                Qs = Q_table[tuple(state_bin_vector)]
                if random.choices([0, 1], weights=(epsilon, 1 - epsilon))[0] == 0: i = random.randint(0, len(actions) - 1)
                else:
                    maxes = [i for i, val in enumerate(Qs) if val == max(Qs)]; i = maxes[0]
                    if len(maxes) > 1: i = random.choices(maxes)[0]
                cwnd_increase = actions[i]

                new_Q_one = (1 - ewma_smoothing_factor) * history[0]['n-2']
                new_Q_two = ewma_smoothing_factor * (reward + (discount_factor * history[1]['n-1']))
                Q_table[tuple(history[0]['state'])][history[0]['i']] = new_Q_one + new_Q_two
                history[0] = history[1]; history[0]['n-1'] = Q_table[tuple(history[0]['state'])][history[0]['i']]
                history[1]['state'] = state_bin_vector; history[1]['i'] = i; history[1]['n-2'] = Qs[i]

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
    global throughput, max_throughput, loss_rate, overall_loss_rate, delay
    global throughput_timestamps, loss_rate_timestamps
    global delay_factor, loss_rate_factor, interval
    global prev_utility, curr_utility, diff_utility, reward
    global state_cols, state_vector, actions, discount_factor, epsilon, cwnd_increase, Q_table, history

    try:
        while running or pending_acks:
            time.sleep(TIMEOUT)
            with send_lock:
                if pending_acks: ssthresh = max(cwnd / 2, MSS * 2); cwnd = MSS
                too_many_duplicates = []
                for num in pending_acks.keys():
                    if num > last_packet: continue
                    if len(df[df['num'] == num]) >= MAX_TRANSMIT:
                        too_many_duplicates.append(num)
                        continue
                    
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
                    if sent_packets != 0: overall_loss_rate = lost_packets / sent_packets

                    row = [num, idx, cwnd, pending_acks[num]]
                    row += [ewma_inter_send, min_inter_send]
                    row += [ts_inter_send[i] - ts_inter_send[0] for i in range(1, TS_SIZE + 1)] 
                    row += [ts_ratio_inter_send[i] - ts_ratio_inter_send[0] for i in range(1, TS_SIZE + 1)]
                    row += [ewma_inter_arr, min_inter_arr]
                    row += [ts_inter_arr[i] - ts_inter_arr[0] for i in range(1, TS_SIZE + 1)] 
                    row += [ts_ratio_inter_arr[i] - ts_ratio_inter_arr[0] for i in range(1, TS_SIZE + 1)]
                    row += [min_rtt] + [ts_rtt[i] - ts_rtt[0] for i in range(1, TS_SIZE + 1)] 
                    row += [ts_ratio_rtt[i] - ts_ratio_rtt[0] for i in range(1, TS_SIZE + 1)]
                    input_row = [ssthresh, throughput, max_throughput, loss_rate, overall_loss_rate, delay]
                    input_row += [ratio_inter_send, ratio_inter_arr, ratio_rtt]
                    df.loc[len(df)] = row + [0] + input_row

                    sent_packets += 1; lost_packets += 1

                    if len(df) < 5000: continue
                    df.to_csv(csv_name, mode='a', header=not os.path.exists(csv_name)); df = df.iloc[0:0]
                
                for num in too_many_duplicates: del pending_acks[num]

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
        
        df.to_csv(csv_name, mode='a', header=not os.path.exists(csv_name))

        if sent_packets > 0:
            overall_loss_rate = round((lost_packets / sent_packets) * 100, 8)
            print("Packet loss rate was: " + str(overall_loss_rate) + "%")