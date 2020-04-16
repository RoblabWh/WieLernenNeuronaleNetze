"""
Train % plot networks in the information plane
"""
import idnns.networks.network_paramters
from idnns.networks import information_network as inet, network_paramters


def main():
    #Bulid the netowrk
    print ('Building the network')
    args = network_paramters.get_default_parser(None)
    args.num_ephocs = 10
    # args.interval_information_display = 1
    args.data_dir = 'idnns/data/'
    args.net_type = '7'
    net = inet.informationNetwork(args=args)
    net.print_information()
    print ('Start running the network')
    net.run_network()
    print ('Saving data')
    net.save_data()
    print ('Ploting figures')
    #Plot the newtork
    net.plot_network()
if __name__ == '__main__':
    main()

