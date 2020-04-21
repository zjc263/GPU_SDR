########################################################################################
##                                                                                    ##
##  THIS LIBRARY IS PART OF THE SOFTWARE DEVELOPED BY THE JET PROPULSION LABORATORY   ##
##  IN THE CONTEXT OF THE GPU ACCELERATED FLEXIBLE RADIOFREQUENCY READOUT PROJECT     ##
##                                                                                    ##
########################################################################################
from .USRP_low_level import *
import numpy as np
from scipy import signal
from .USRP_fitting import get_fit_param
from .USRP_noise import calculate_frequency_timestream
import h5py
import time
class trigger_template(object):
    '''
    Example class for developing a trigger.
    The triggering method has to be passed as an argument of the Packets_to_file function and has to respect the directions given in the Trigger section of this documentation.
    The user is responible for the initialization of the object.
    The internal variable trigger_coltrol determines if the trigger dataset bookeep whenever the trigger method returns metadata['length']>0 or if it's controlled by the user.
    In case the trigger_control is not set on \'AUTO\' the user must take care of expanding the dataset before writing.
    '''

    def __init__(self, rate):
        self.trigger_control = "MANUAL" # OR MANUAL
        if (self.trigger_control != "AUTO") and (self.trigger_control != "MANUAL"):
            err_msg = "Trigger_control in the trigger class can only have MANUAL or AUTO value, not \'%s\'"%str(self.trigger_control)
            print_error(err_msg)
            raise ValueError(err_msg)

    def dataset_init(self, antenna_group):
        '''
        This function is called on file creation an is used to create additional datasets inside the hdf5 file.
        In order to access the datasets created here in the trigger function make them member of the class:

        >>> self.custom_dataset = antenna_group.create_dataset("amazing_dataset", shape = (0,), dtype=np.dtype(np.int64), maxshape=(None,), chunks=True)

        Note: There is no need to bookeep when (at what index) the trigger is called as this is already taken care of in the trigger dataset.
        :param antenna_group is the antenna group containing the 'error','data' and triggering datasets.
        '''

        self.trigger_group = antenna_group['trigger']

        return
    def write_trigger(self, data):

        current_len_trigger = len(self.trigger)
        self.trigger.resize(current_len_trigger+1,0)
        self.trigger[current_len_trigger] = data

    def trigger(self, data, metadata):
        '''
        Triggering function.
        Make modification to the data and metadata accordingly and return them.

        :param data: the data packet from the GPU server
        :param metadata: the metadata packet from the GPU server

        :return same as argument but with modified content.

        Note: the order of data at this stage follows the example ch0_t0, ch1_t0, ch0_t1, ch1_t1, ch0_t2, ch1_t2...
        '''

        return data, metadata


class trigger_example(object):
    '''
    Example class for developing a trigger.
    The triggering method has to be passed as an argument of the Packets_to_file function and has to respect the directions given in the Trigger section of this documentation.
    The user is responible for the initialization of the object.
    The internal variable trigger_coltrol determines if the trigger dataset bookeep whenever the trigger method returns metadata['length']>0 or if it's controlled by the user.
    In case the trigger_control is not set on \'AUTO\' the user must take care of expanding the dataset before writing.
    '''

    def __init__(self, rate = 1):
        self.trigger_control = "MANUAL" # OR MANUAL
        if (self.trigger_control != "AUTO") and (self.trigger_control != "MANUAL"):
            err_msg = "Trigger_control in the trigger class can only have MANUAL or AUTO value, not \'%s\'"%str(self.trigger_control)
            print_error(err_msg)
            raise ValueError(err_msg)

        self.signal_accumulator = 0
        self.std_threshold_multiplier = 1.5
        self.trigget_count = 0

    def dataset_init(self, antenna_group):
        '''

        TRIGGER IMPLEMENTATION EXAMPLE

        This function is called on file creation an is used to create additional datasets inside the hdf5 file.
        In order to access the datasets created here in the trigger function make them member of the class:

        >>> self.custom_dataset = antenna_group.create_dataset("amazing_dataset", shape = (0,), dtype=np.dtype(np.int64), maxshape=(None,), chunks=True)

        Note: There is no need to bookeep when (at what index) the trigger is called as this is already taken care of in the trigger dataset.
        :param antenna_group is the antenna group containing the 'error','data' and triggering datasets.
        '''

        # Create a dataset for storing some timing info. This dataset will be updated when the trigger is fired (see write trigger method)
        self.trigger_timing = antenna_group.create_dataset("timing", shape = (0,), dtype=np.dtype(np.float64), maxshape=(None,))

        # Because this is a fancy example we'll store also the triggering threshold and the length of the packet in samples (yes it's dynamically accomodated in the comm protocol)
        self.thresholds = antenna_group.create_dataset("thresholds", shape = (0,), dtype=np.dtype(np.float64), maxshape=(None,))
        self.slices = antenna_group.create_dataset("slices", shape = (0,), dtype=np.dtype(np.int32), maxshape=(None,))

    def write_trigger(self, metadata):

        self.trigget_count +=1

        current_len_trigger = len(self.trigger_timing) # check current length of the dset
        (self.trigger_timing).resize((self.trigget_count,)) # update the length (expensive disk operation ~500 cycles on x86 intel, separate client server if rate>100Msps)
        self.trigger_timing[self.trigget_count-1] = time.time() # write the data of interest

        current_len_trigger = len(self.thresholds)
        (self.thresholds).resize((self.trigget_count,))
        self.thresholds[self.trigget_count-1] = self.signal_accumulator

        current_len_trigger = len(self.slices)
        (self.slices).resize((self.trigget_count,))
        self.slices[self.trigget_count-1] = metadata['length']



    def trigger(self, data, metadata):
        '''
        Triggering function.
        Make modification to the data and metadata accordingly and return them.

        :param data: the data packet from the GPU server
        :param metadata: the metadata packet from the GPU server

        :return same as argument but with modified content.

        Note: the order of data at this stage follows the example ch0_t0, ch1_t0, ch0_t1, ch1_t1, ch0_t2, ch1_t2...

        ***IMPLEMENTATION SPECIFIC***

        This trigger check on the standard deviation of all cthe channels together and write data if the std is bigger than the previous packet.
        This is not ment as a usable object but only as a guideline for designing more functional stuff.

        '''
        if metadata['packet_number'] < 2:
            self.signal_accumulator = np.std(data)
            metadata['length'] = 0
            return [], metadata
        elif self.signal_accumulator*self.std_threshold_multiplier < np.std(data):
            print("Triggered!")
            self.write_trigger(metadata)
            return data, metadata
        else:
            self.signal_accumulator = np.std(data)
            metadata['length'] = 0
            return [], metadata
