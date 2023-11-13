from helper import *

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

cwnd = 1.0; ssthresh = 64; window_size = 1; lpthresh = 0.6 # hyperparameters
ewma_smoothing_factor = 0.2; throughput_factor = 0.2; latency_factor = 0.2; loss_rate_factor = 0.2 # hyperparameter

running = True; pending_acks = {}; send_lock = threading.Lock()
curr_packet = 1; last_packet = 0; sent_packets = 0; lost_packets = 0
cwnd_order = 1; last_ack = 0

rtt = np.NaN; min_rtt = sys.maxsize; ratio_rtt = np.NaN
prev_send = np.NaN; curr_send = np.NaN; inter_send = np.NaN; min_inter_send = sys.maxsize; ratio_inter_send = np.NaN
prev_arr = np.NaN; curr_arr = np.NaN; inter_arr = np.NaN; min_inter_arr = sys.maxsize; ratio_inter_arr = np.NaN

ts_inter_send = [np.NaN] * (TS_SIZE + 1); ts_inter_arr = [np.NaN] * (TS_SIZE + 1); ts_rtt = [np.NaN] * (TS_SIZE + 1)
ts_ratio_inter_send = [np.NaN] * (TS_SIZE + 1); ts_ratio_inter_arr = [np.NaN] * (TS_SIZE + 1); ts_ratio_rtt = [np.NaN] * (TS_SIZE + 1)
ewma_inter_send = np.NaN; ewma_inter_arr = np.NaN; ewma_rtt = np.NaN

throughput_timestamps = deque(); loss_rate_timestamps = deque()
throughput = 0.0; max_throughput = 0.0; latency = np.NaN; loss_rate = 0.0; reward = 0.0
delay = np.NaN; delay_factor = 0.2; prev_utility = 0.0; curr_utility = 0.0; diff_utility = 0.0

columns = ['num', 'idx', 'cwnd', 'cwnd_order', 'ssthresh', 'send_time']
columns += ['latency', 'rtt', 'loss_rate', 'overall_loss_rate']
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

row = [np.NaN] * len(columns)
df = pd.DataFrame(columns=columns)
df = df.astype({"num":"int", "idx":"int", "cwnd_order":"int", "recvd":"int"})
model = pickle.load(open(PKL_NAME, 'rb'))
if os.path.exists(CSV_NAME): os.remove(CSV_NAME)
#levels = 10

def send_packs():
    global cwnd, ssthresh, window_size, lpthresh, running, pending_acks, send_lock
    global ewma_smoothing_factor, throughput_factor, latency_factor, loss_rate_factor
    global curr_packet, last_packet, sent_packets, lost_packets, cwnd_order, last_ack, rtt, min_rtt, ratio_rtt
    global prev_send, curr_send, inter_send, min_inter_send, ratio_inter_send
    global prev_arr, curr_arr, inter_arr, min_inter_arr, ratio_inter_arr
    global ts_inter_send, ts_inter_arr, ts_rtt, ts_ratio_inter_send, ts_ratio_inter_arr, ts_ratio_rtt
    global ewma_inter_send, ewma_inter_arr, ewma_rtt, throughput_timestamps, loss_rate_timestamps
    global throughput, max_throughput, latency, loss_rate, reward, delay, delay_factor, prev_utility, curr_utility, diff_utility, model, df, row

    try:
        while running:
            if curr_packet > last_ack + math.floor(cwnd): continue
            idx = random.randint(1000, 9999)
            send_time = time.time() - start_time

            if not np.any(np.isnan(row[2:])):
                probs = model.predict_proba(np.array(row[2:]).reshape(1, -1))[0]
                if probs[len(probs) - 1] < lpthresh: cwnd = max(cwnd - 1, 1)
                if cwnd_order > cwnd: cwnd_order = 1

            cli_socket.send(Packet(curr_packet, idx, DATA, send_time=send_time).to_bytes())

            with send_lock:
                pending_acks[curr_packet] = {'cwnd_order':cwnd_order}

                prev_send = curr_send; curr_send = send_time; inter_send = curr_send - prev_send
                if inter_send < min_inter_send: min_inter_send = inter_send

                overall_loss_rate = 0.0
                if sent_packets != 0: overall_loss_rate = float(lost_packets) / float(sent_packets)
                if min_inter_send != 0: ratio_inter_send = inter_send / min_inter_send

                ts_inter_send = ts_inter_send[1:TS_SIZE + 1] + [inter_send]
                ts_ratio_inter_send = ts_ratio_inter_send[1:TS_SIZE + 1] + [ratio_inter_send]

                if not np.isnan(ewma_inter_send): ewma_inter_send = (inter_send * ewma_smoothing_factor) + (ewma_inter_send * (1 - ewma_smoothing_factor))
                if np.isnan(ewma_inter_send) and not np.isnan(inter_send): ewma_inter_send = inter_send

                throughput = len(throughput_timestamps) / window_size
                if throughput > max_throughput: max_throughput = throughput
                loss_rate = len(loss_rate_timestamps) / window_size; delay = rtt - min_rtt

                throughput_utility = 0; delay_utility = 0; loss_rate_utility = 0
                if max_throughput != 0: np.log(throughput / max_throughput)
                if delay != 0: delay_factor * np.log(delay)
                if loss_rate != 0: loss_rate_factor * np.log(loss_rate)
                utility = throughput_utility - delay_utility + loss_rate_utility
                prev_utility = curr_utility; curr_utility = utility; diff_utility = curr_utility - prev_utility

                if diff_utility >= 1: reward = 10
                elif diff_utility < 1 and diff_utility >= 0: reward = 2
                elif diff_utility < 0 and diff_utility >= -1: reward = -2
                elif diff_utility < -1: reward = -10

                row = [curr_packet, idx, cwnd, cwnd_order, ssthresh, send_time]
                row += [latency, rtt, loss_rate, overall_loss_rate] + [inter_send, min_inter_send, ewma_inter_send]
                row += [ts_inter_send[i] - ts_inter_send[0] for i in range(1, TS_SIZE + 1)] 
                row += [ratio_inter_send] + [ts_ratio_inter_send[i] - ts_ratio_inter_send[0] for i in range(1, TS_SIZE + 1)]
                row += [inter_arr, min_inter_arr, ewma_inter_arr] + [ts_inter_arr[i] - ts_inter_arr[0] for i in range(1, TS_SIZE + 1)]
                row += [ratio_inter_arr] + [ts_ratio_inter_arr[i] - ts_ratio_inter_arr[0] for i in range(1, TS_SIZE + 1)]
                row += [rtt, min_rtt, ewma_rtt] + [ts_rtt[i] - ts_rtt[0] for i in range(1, TS_SIZE + 1)]
                row += [ratio_rtt] + [ts_ratio_rtt[i] - ts_ratio_rtt[0] for i in range(1, TS_SIZE + 1)] + [throughput, reward]
                df.loc[len(df)] = row + [0]

                if len(df) > 5000:
                    df = df.dropna()
                    df.to_csv(CSV_NAME, mode='a', header=not os.path.exists(CSV_NAME))
                    df = df.iloc[0:0]
                
                cwnd_order += 1
                if cwnd_order > cwnd: cwnd_order = 1
                sent_packets += 1; last_packet = curr_packet; curr_packet += 1

            if sent_packets % 10000 == 0: print("Sent " + str(sent_packets) + " packets.")
    except Exception as err:
        print("The following error occured while sending packets: " + str(err))
        traceback.print_exc(); return

def recv_acks():
    global cwnd, ssthresh, window_size, lpthresh, running, pending_acks, send_lock
    global ewma_smoothing_factor, throughput_factor, latency_factor, loss_rate_factor
    global curr_packet, last_packet, sent_packets, lost_packets, cwnd_order, last_ack, rtt, min_rtt, ratio_rtt
    global prev_send, curr_send, inter_send, min_inter_send, ratio_inter_send
    global prev_arr, curr_arr, inter_arr, min_inter_arr, ratio_inter_arr
    global ts_inter_send, ts_inter_arr, ts_rtt, ts_ratio_inter_send, ts_ratio_inter_arr, ts_ratio_rtt
    global ewma_inter_send, ewma_inter_arr, ewma_rtt, throughput_timestamps, loss_rate_timestamps
    global throughput, max_throughput, latency, loss_rate, reward, delay, delay_factor, prev_utility, curr_utility, diff_utility, model, df, row

    try:
        while running or pending_acks:
            recv_packet = recv_packet_func(cli_socket, [ACK])
            ack_recv_time = time.time() - start_time
            with send_lock:
                df.loc[(df['num'] == recv_packet.num) & (df['idx'] == recv_packet.idx), 'recvd'] = 1
                throughput_timestamps.append(ack_recv_time); current_time = time.time() - start_time
                while throughput_timestamps and current_time - throughput_timestamps[0] > window_size: throughput_timestamps.popleft()
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

                    if not np.isnan(ewma_inter_arr): ewma_inter_arr = (inter_arr * ewma_smoothing_factor) + (ewma_inter_arr * (1 - ewma_smoothing_factor))
                    if np.isnan(ewma_inter_arr) and not np.isnan(inter_arr): ewma_inter_arr = inter_arr
                    if not np.isnan(ewma_rtt): ewma_rtt = (rtt * ewma_smoothing_factor) + (rtt * (1 - ewma_smoothing_factor))
                    if np.isnan(ewma_rtt) and not np.isnan(rtt): ewma_rtt = rtt

                    if cwnd < ssthresh: cwnd += 1.0
                    else: cwnd += 1.0 / cwnd
                    if cwnd_order > cwnd: cwnd_order = 1
    except Exception as err:
        print("The following error occured while receiving packets: " + str(err))
        traceback.print_exc(); return

def retransmit():
    global cwnd, ssthresh, window_size, lpthresh, running, pending_acks, send_lock
    global ewma_smoothing_factor, throughput_factor, latency_factor, loss_rate_factor
    global curr_packet, last_packet, sent_packets, lost_packets, cwnd_order, last_ack, rtt, min_rtt, ratio_rtt
    global prev_send, curr_send, inter_send, min_inter_send, ratio_inter_send
    global prev_arr, curr_arr, inter_arr, min_inter_arr, ratio_inter_arr
    global ts_inter_send, ts_inter_arr, ts_rtt, ts_ratio_inter_send, ts_ratio_inter_arr, ts_ratio_rtt
    global ewma_inter_send, ewma_inter_arr, ewma_rtt, throughput_timestamps, loss_rate_timestamps
    global throughput, max_throughput, latency, loss_rate, reward, delay, delay_factor, prev_utility, curr_utility, diff_utility, model, df, row

    try:
        while running or pending_acks:
            time.sleep(TIMEOUT)
            with send_lock:
                if pending_acks: ssthresh = max(cwnd / 2, 2); cwnd = 1
                for num in pending_acks.keys():
                    if num <= last_packet:
                        if len(df[df['num'] == num]) >= MAX_TRANSMIT:
                            del pending_acks[num]; continue

                        current_time = time.time() - start_time; loss_rate_timestamps.append(current_time)
                        while loss_rate_timestamps and current_time - loss_rate_timestamps[0] > window_size: loss_rate_timestamps.popleft()

                        print('Lost packet ' + str(num) + '. Attempting to retransmit.')
                        idx = random.randint(1000, 9999)
                        send_time = time.time() - start_time

                        prev_send = curr_send; curr_send = send_time; inter_send = curr_send - prev_send
                        if inter_send < min_inter_send: min_inter_send = inter_send

                        overall_loss_rate = 0.0
                        if sent_packets != 0: overall_loss_rate = float(lost_packets) / float(sent_packets)
                        if min_inter_send != 0: ratio_inter_send = inter_send / min_inter_send

                        ts_inter_send = ts_inter_send[1:TS_SIZE + 1] + [inter_send]
                        ts_ratio_inter_send = ts_ratio_inter_send[1:TS_SIZE + 1] + [ratio_inter_send]

                        if not np.isnan(ewma_inter_send): ewma_inter_send = (inter_send * ewma_smoothing_factor) + (ewma_inter_send * (1 - ewma_smoothing_factor))
                        if np.isnan(ewma_inter_send) and not np.isnan(inter_send): ewma_inter_send = inter_send

                        throughput = len(throughput_timestamps) / window_size
                        if throughput > max_throughput: max_throughput = throughput
                        loss_rate = len(loss_rate_timestamps) / window_size; delay = rtt - min_rtt
                        reward = (throughput_factor * throughput) - (latency_factor * latency) - (loss_rate_factor * loss_rate)

                        throughput_utility = 0; delay_utility = 0; loss_rate_utility = 0
                        if max_throughput != 0: np.log(throughput / max_throughput)
                        if delay != 0: delay_factor * np.log(delay)
                        if loss_rate != 0: loss_rate_factor * np.log(loss_rate)
                        utility = throughput_utility - delay_utility + loss_rate_utility
                        prev_utility = curr_utility; curr_utility = utility; diff_utility = curr_utility - prev_utility

                        if diff_utility >= 1: reward = 10
                        elif diff_utility < 1 and diff_utility >= 0: reward = 2
                        elif diff_utility < 0 and diff_utility >= -1: reward = -2
                        elif diff_utility < -1: reward = -10

                        row = [num, idx, cwnd, pending_acks[num]['cwnd_order'], ssthresh, send_time]
                        row += [latency, rtt, loss_rate, overall_loss_rate] + [inter_send, min_inter_send, ewma_inter_send]
                        row += [ts_inter_send[i] - ts_inter_send[0] for i in range(1, TS_SIZE + 1)] 
                        row += [ratio_inter_send] + [ts_ratio_inter_send[i] - ts_ratio_inter_send[0] for i in range(1, TS_SIZE + 1)]
                        row += [inter_arr, min_inter_arr, ewma_inter_arr]
                        row += [ts_inter_arr[i] - ts_inter_arr[0] for i in range(1, TS_SIZE + 1)]
                        row += [ratio_inter_arr] + [ts_ratio_inter_arr[i] - ts_ratio_inter_arr[0] for i in range(1, TS_SIZE + 1)]
                        row += [rtt, min_rtt, ewma_rtt] + [ts_rtt[i] - ts_rtt[0] for i in range(1, TS_SIZE + 1)]
                        row += [ratio_rtt] + [ts_ratio_rtt[i] - ts_ratio_rtt[0] for i in range(1, TS_SIZE + 1)] + [throughput, reward]
                        df.loc[len(df)] = row + [0]

                        if len(df) > 5000:
                            df = df.dropna()
                            df.to_csv(CSV_NAME, mode='a', header=not os.path.exists(CSV_NAME))
                            df = df.iloc[0:0]

                        cli_socket.send(Packet(num, idx, DATA, send_time=send_time).to_bytes())
                        sent_packets += 1; lost_packets += 1
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
        df = df.dropna()
        df.to_csv(CSV_NAME, mode='a', header=not os.path.exists(CSV_NAME))
        if sent_packets > 0:
            overall_loss_rate = round((lost_packets / sent_packets) * 100, 8)
            print("Packet loss rate was: " + str(overall_loss_rate) + "%")