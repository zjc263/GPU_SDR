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


class deriv_test(trigger_template):
    '''
    Just a test I wrote for the triggers.
    There is a bug somewhere an it's way too slow for a long acquisition to be sustainable.
    '''
    def __init__(self):
        trigger_template.__init__(self)
        self.stored_data = np.array([])
        self.threshold = 1.1

    def trigger(self, data, metadata):
        n_chan = metadata['channels']

        # Accumulate data
        self.stored_data = np.concatenate((self.stored_data,data))

        # Reach a condition
        if len(self.stored_data) >= 3 * metadata['length']:

            # do some analysis
            samples_per_channel = 3*metadata['length']/n_chan
            formatted_data = np.gradient( np.reshape(self.stored_data, (samples_per_channel, n_chan)).T, axis = 1)
            per_channel_average = np.abs(np.mean(formatted_data,1))
            x = sum([sum(np.abs(formatted_data[i])>(self.threshold*per_channel_average[i])) for i in range(len(formatted_data))])
            if x > 1:

                ret = self.stored_data
                metadata['length'] = len(self.stored_data)
                self.stored_data = np.array([])

                return ret, metadata
            else:
                self.stored_data = np.array([])
                metadata['length'] = 0
                return [[],],metadata
        else:
            metadata['length'] = 0
            return [[],],metadata



class amplitude_trigger(object):
    '''
    Triggers by an amplitude-based threshold. The incoming data is first accumulated
    into a 10 second packet. The data then converted to frequency and Qr data based
    on the fit parameters from the VNA. Frequency is stored in the real component and Qr
    is stored in the imaginary component. Thresholds are set via the median of the frequency timestream data
    +/- a certain number of standard deviations of the frequency timestream data. If any point of the
    frequency data on the list of triggering channels lies above these thresholds, all channels are recorded
    in the 8 ms around that glitch. 2 random 8 ms noise samples are also recorded at the beginning of
    the packet.
    '''

    def __init__(self, sample_rate, freq, tones, vna, threshold, channels):
        self.trigger_control = "AUTO" # OR MANUAL
        if (self.trigger_control != "AUTO") and (self.trigger_control != "MANUAL"):
            err_msg = "Trigger_control in the trigger class can only have MANUAL or AUTO value, not \'%s\'"%str(self.trigger_control)
            print_error(err_msg)
            raise ValueError(err_msg)


        self.stored_data = []
        self.time_index = 0
        self.rate = sample_rate
        self.freq = freq
        self.tones = tones
        self.vna = vna
        self.threshold = threshold #number of standard deviations above the mean
        self.bounds = []
        ##every 10 seconds, a new pair of high and low thresholds for each channel is added.
        self.nglitch = []
        self.glitch_indices = [] ##glitch times (by packet.)
        self.samples_per_packet = []

        vna_file = h5py.File(self.vna, 'r')
        calibration = vna_file['VNA_0'].attrs['calibration']
        chs = len(vna_file['Resonators'].attrs['tones_init'])

        self.cal = calibration
        self.index = 0

        if channels is not None:
            self.channels = channels
            print("The selected channels for triggering are:", channels)
        else:
            print("All", chs, "channels are being used for triggering.")
            self.channels = np.arange(chs)

    def dataset_init(self, antenna_group):
        self.trigger_group = antenna_group['trigger']
        return

    def write_trigger(self, data):

        current_len_trigger = len(self.trigger)
        self.trigger.resize(current_len_trigger+1,0)
        self.trigger[current_len_trigger] = data


    def trigger(self, data, metadata):
        n_chan = metadata['channels']
        self.time_index += metadata['length']/n_chan
        self.stored_data.extend(data)
        if self.time_index >= 10*self.rate: ##data accumulated into 10 second packet
            self.stored_data = np.array(self.stored_data)
            t0 = time.time()
            n_samples = len(self.stored_data)/n_chan ##The number of samples per channel in the packet
            self.samples_per_packet.append(n_samples)
            reshaped_data = np.reshape(self.stored_data, (n_samples, n_chan)).T
            srate = self.rate

            ###frequency conversion:
            ti = time.time()
            fit_params = get_fit_param(self.vna)
            n_reso = len(fit_params)
            frequencies = self.freq + self.tones

            for n in range(0, n_reso):
                noise_data = reshaped_data[n]
                p = fit_params[n]
                ##following the same order specified in calculate_frequency_timestreams
                params = (p['f0'], p['A'], p['phi'], p['D'], p['Qi'], p['Qr'], p['Qe'].real, p['Qe'].imag, p['a'])
                current_freq = frequencies[n]
                x, Qr = calculate_frequency_timestream(current_freq, noise_data*self.cal, params)
                reshaped_data[n] = x + 1j*Qr
                ##frequency is in real, Qr is in imaginary.
            tf = time.time()
            print("Time to frequency convert is", tf-ti)
            ##finding the indices of the glitches:
            hits = np.zeros(n_samples, dtype=bool) ##initially all false.
            bounds = []
            for x in range(0, len(self.channels)):
                ch = self.channels[x]
                current = reshaped_data[ch].real
                med = np.median(current)
                stddev = np.std(current)
                lo = med - self.threshold*stddev
                hi = med + self.threshold*stddev
                bounds.append([lo, hi])
                mask = np.logical_or(current<lo, current>hi)
                hits = np.logical_or(hits, mask)
            ##hits now contains the indices of all the glitches across the triggering channels
            self.bounds.append(bounds)
            hit_indices = np.nonzero(hits)[0]
            indices_diffs = np.ediff1d(hit_indices)
            count = 0
            for y in range(0, len(indices_diffs)):
                if indices_diffs[y] < (0.001*srate): ##if points are less than .001 sec apart, it is the same glitch.
                    hit_indices = np.delete(hit_indices, count+1)
                else:
                    count += 1
            ##now hit_indices only contains one marker per glitch.
            n_glitch = len(hit_indices)
            print(n_glitch, "glitches detected.")
            ##adding random noise info:
            num = int(srate*0.002) ##2 ms worth of points.
            rand1 = np.random.randint(num, high=n_samples-3*num)
            rand2 = np.random.randint(num, high=n_samples-3*num)
            hit_indices = np.concatenate((np.array([rand1, rand2]), hit_indices))
            res = np.empty([n_chan, 0])
            glitch_index = []
            for z in range(0, len(hit_indices)): ##find data around glitches
                i = hit_indices[z]
                if i>=num and i<n_samples-3*num:
                    chopped = reshaped_data[0:n_chan, (i-num):(i+3*num)]
                    res = np.concatenate((res.T, chopped.T)).T
                    glitch_index.append(i+self.index)
                else:
                    print("Glitch index", i, "not in range.")
                    n_glitch = n_glitch - 1
            self.nglitch.append(n_glitch)
            self.glitch_indices.extend(glitch_index)
            res = np.reshape(res.T, (res.size,))
            metadata['length'] = len(res)
            self.stored_data = []
            self.time_index = 0
            self.index += n_samples
            t1 = time.time()
            print("Time to run is", t1-t0)
            return res, metadata
        else: ##if data is not long enough.
            metadata['length'] = 0
            return np.array([]), metadata
