########################################################################################
##                                                                                    ##
##  THIS LIBRARY IS PART OF THE SOFTWARE DEVELOPED BY THE JET PROPULSION LABORATORY   ##
##  IN THE CONTEXT OF THE GPU ACCELERATED FLEXIBLE RADIOFREQUENCY READOUT PROJECT     ##
##                                                                                    ##
########################################################################################

import numpy as np
import signal as Signal
import sys
import socket
from threading import Condition
import multiprocessing
from multiprocessing.managers import SyncManager

# version of this library
libUSRP_net_version = "2.0"

# ip address of USRP server
USRP_IP_ADDR = '127.0.0.1'

# soket used for command
USRP_server_address = (USRP_IP_ADDR, 22001)
USRP_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
USRP_socket.settimeout(1)
USRP_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# address used for data
USRP_server_address_data = (USRP_IP_ADDR, 61360)
USRP_data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
USRP_data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# queue for passing data and metadata from network
USRP_data_queue = multiprocessing.Queue()

# compression used in H5py datasets
H5PY_compression = "gzip"
HDF5_compatible_compression = 'gzip'

# Cores to use in analysis
N_CORES = 10
parallel_backend = 'multiprocessing'

# usrp output power at 0 tx gain
USRP_power = -6.15

# Colors used for plotting
COLORS = ['black', 'red', 'green', 'blue', 'violet', 'brown', 'purple']

# this enable disable warning generated when more data that expected is received in the h5 file.
dynamic_alloc_warning = True

# reproduce the RX_wrapper struct in server_settings
header_type = np.dtype([
    ('usrp_number', np.int32),
    ('front_end_code', np.dtype('|S1')),
    ('packet_number', np.int32),
    ('length', np.int32),
    ('errors', np.int32),
    ('channels', np.int32)
])

# data type expected for the buffer
data_type = np.complex64

# threading condition variables for controlling Async RX and TX thread activity
Async_condition = Condition()

# variable used to notify the end of a measure, See Decode_async_payload()
END_OF_MEASURE = False
# mutex guarding the variable above
EOM_cond = Condition()

# becomes true when a communication error occured
ERROR_STATUS = False

# in case the server has to communicate the current filename
REMOTE_FILENAME = ""

# variable ment to accumulate data coming from the data stream
DATA_ACCUMULATOR = []

# initially the status is off. It's set to True in the Sync_RX function/thread
Async_status = False

# Thread synchronizaton manager
manager = SyncManager()


def mgr_init():
    '''
    Initialization for the thread synchronization manager. This is basically propagaring the Ctrl+C command to kill the
    threads (processes). This is ment to be used in:
    >>> manager.start(mgr_init)

    :return: None
    '''
    Signal.signal(Signal.SIGINT, Signal.SIG_IGN)


# explicitly starting the manager, and telling it to ignore the interrupt signal and propagate it.
manager.start(mgr_init)

CLIENT_STATUS = manager.dict()
CLIENT_STATUS["Sync_RX_status"] = False
CLIENT_STATUS["keyboard_disconnect"] = False
CLIENT_STATUS["keyboard_disconnect_attemp"] = 0
CLIENT_STATUS["measure_running_now"] = False

# threading condition variables for controlling Sync RX thread activity
Sync_RX_condition = manager.Condition()


def to_list_of_str(user_input):
    '''
    Determines if the input is a string or a list of string. In case is not a list of string, returns a single element list; returns the list otherwise.

    Arguments:
        - string or list of strings.

    Returns:
        - list of strings.

    Note:
        - I'm assuming the strings contain filenames that can't be long 1.
    '''
    try:
        l = len(user_input[0])
    except:
        print_error(
            "Something went wrong in the string to list of string conversion. Check function input: " + str(user_input))

    if l == 1:
        return [user_input]
    elif l == 0:
        print_warning("an input name has length 0.")
        return user_input
    else:
        return user_input

def print_warning(message):
    '''
    Print a yellow warning label before message.
    :param message: the warning message.
    :return: None
    '''
    print "\033[40;33mWARNING\033[0m: " + str(message) + "."


def print_error(message):
    '''
    Print a red error label before message.
    :param message: the error message.
    :return: None
    '''
    print "\033[1;31mERROR\033[0m: " + str(message) + "."


def print_debug(message):
    '''
    Print the message in italic grey.
    :param message: the debug message.
    :return: None
    '''
    print "\033[3;2;37m" + str(message) + "\033[0m"


def print_line(msg):
    '''
    Print without new line.
    :param message: string to print.
    :return: None
    '''
    sys.stdout.write(msg)
    sys.stdout.flush()


def get_timestamp():
    '''
    Returns the timestamp formatted in a stirng.

    Returns:
        string containing the timestamp.
    '''
    return str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))