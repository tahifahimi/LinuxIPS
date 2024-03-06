from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import ARP
from scapy.all import rdpcap
import pandas as pd

# from scapy.layers.tls.all import TLS
# from scapy.all import load_layer, bind_layers

def preprocess(filename,
               url="https://mcfp.felk.cvut.cz/publicDatasets/CTU-Mixed-Capture-1/2015-07-28_mixed.before.infection.pcap"):
    # Define a list to store data
    data = []

    # we skip TLS layer for now
    # load_layer('tls')
    # bind_layers(TCP, TLS, dport=8443)

    # Read pcap file
    # this view of processing works for a specific size of pcaps
    packets = rdpcap(filename)


    for packet in packets:
        # extract information from inet layer
        if IP in packet:
            direction = 'inbound' if packet[IP].dst.startswith('10.0.') else 'outbound'
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            proto = packet[IP].proto

            if TCP in packet:
                sport = packet[TCP].sport
                dport = packet[TCP].dport
            elif UDP in packet:
                sport = packet[UDP].sport
                dport = packet[UDP].dport
            else:
                sport = None
                dport = None
            packet_length = len(packet)

            # Example: Extract more features as needed
            data.append([packet.time, src_ip, dst_ip, proto, sport, dport, packet_length, direction])

        elif ARP in packet:
            # Handle ARP packets separately if needed
            pass
        # extract next layer information (For now, we pass using TLS/SSL)
        # elif  

    # Convert to DataFrame for easier manipulation and analysis
    df = pd.DataFrame(data, columns=['timestamp', 'SrcIP', 'DstIP', 'Protocol', 'SrcPort', 'DstPort', 'length', 'direction'])

    # calculate packet size and Direction
    total_packet_sizes = df['length'].sum()
    individual_packet_sizes = df['length'].tolist()
    traffic_direction_counts = df['direction'].value_counts().to_dict()

    # # timing information
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    # df = df.sort_values(by='timestamp')
    # df['inter_arrival_time'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    # session_duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]


    # #statistical feature
    # packet_size_stats = df['length'].describe()
    # inter_arrival_stats = df['inter_arrival_time'].describe()

    # # flow value
    # # Already calculated as total_packet_sizes
    # total_bytes_per_flow = total_packet_sizes
    # total_packets_per_flow = len(df)


    print(df.head())
    return df, packets


def main():
    filename = "/home/tahera/Desktop/StratosphereLinuxIPS/modules/anomaly_detection/2015-07-28_mixed.before.infection.pcap"
    df, packets = preprocess(filename=filename)
    print(df.columns)
    # Group packets into flows
    flows = df.groupby(['SrcIP', 'DstIP', 'SrcPort', 'DstPort', 'Protocol'])
    # Example: Calculating statistics for each flow
    flow_stats = flows.agg(
        total_packets=pd.NamedAgg(column="length", aggfunc="count"),
        total_bytes=pd.NamedAgg(column="length", aggfunc="sum"),
        average_packet_size=pd.NamedAgg(column="length", aggfunc="mean"),
        flow_duration=pd.NamedAgg(column="timestamp", aggfunc=lambda x: x.max() - x.min()),
    )

    print(flow_stats.reset_index())

    df.sort_values(by=['timestamp'], inplace=True)  # Ensure data is sorted by timestamp

    # Calculate inter-arrival times within each flow
    df['inter_arrival_time'] = flows['timestamp'].diff().fillna(0)







if __name__ == '__main__':
    main()
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', type=str, default='path/to/dataset.csv', help='Path to the input data file.')
#     parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of the dataset to be used for testing.')
#     parser.add_argument('--max_length', type=int, default=128, help='Maximum length of input sequence.')
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training the model.')
#     parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training the model.')
#     parser.add_argument('--verbose', type=int, default=1,help='Verbosity mode. 0 = silent, 1 = printing into command line')
