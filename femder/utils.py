# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 11:58:35 2020

@author: gutoa
"""
from __future__ import division, print_function
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
import scipy.signal.windows as win
def len_unique_items_dict(dict_obj):
    """
    Computes biggest unique for dict
    Parameters
    ----------
    dict_obj:dict

    Returns
    -------
    Max unique for dict
    """
    bool_obj = np.asarray(list(dict_obj.items()), dtype=object)[:, 1]
    unique_len = max([len(np.unique(bool_obj[i])) for i in range(len(bool_obj))])
    return unique_len


def bigger_than_n_unique_dict(dict_obj, n):
    """
    Computes indexes of keys with more than n unique items
    Parameters
    ----------
    dict_obj: dict
        Dictionary to operatoe
    n: int

    Returns
    -------
    Array of indexes
    """
    bool_obj = np.asarray(list(dict_obj.items()), dtype='object')[:, 1]
    unique_len = np.argwhere(np.asarray([len(np.unique(bool_obj[i])) for i in range(len(bool_obj))]) > n).ravel()
    return unique_len


def timer_func(func):
    """
    Time function with decorator
    Parameters
    ----------
    func: function

    Returns
    -------

    """

    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        if not rasta.HIDE_PBAR:
            print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


def nearest_multiple(x, multiple=0.05):
    """Round a given number to the selected nearest multiple."""
    return multiple * round(x / multiple)


def find_nearest(array, value):
    """
    Function to find closest frequency in frequency array.

    Parameters
    ----------
    array : array
        1D array in which to search for the closest value.
    value : float or int
        Value to be searched.

    Returns
    -------
    Closest value found and its position index.
    """
    import numpy as np

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def closest_node(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    from detect_peaks import detect_peaks
    x = np.random.randn(100)
    x[60:81] = np.nan
    # detect all peaks and plot data
    ind = detect_peaks(x, show=True)
    print(ind)

    x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    # set minimum peak height = 0 and minimum peak distance = 20
    detect_peaks(x, mph=0, mpd=20, show=True)

    x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    # set minimum peak distance = 2
    detect_peaks(x, mpd=2, show=True)

    x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    # detection of valleys instead of peaks
    detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)

    x = [0, 1, 1, 0, 1, 1, 0]
    # detect both edges
    detect_peaks(x, edge='both', show=True)

    x = [-2, 1, -2, 2, 1, 1, 3, 0]
    # set threshold = 2
    detect_peaks(x, threshold = 2, show=True)

    Version history
    ---------------
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()
        
def SBIR(IR, t_IR, fmin, fmax, winCheck=False, spectraCheck=False, ms=32, method='constant', beta=1, cosWin=False,
         ABEC=False, delta_ABEC=52):
    """

    Function to calculate Speaker Boundary Interference Response

    Parameters
    ----------
    IR, t_IR: 1D arrays, contain bothe Impulse Response magnitude and time step values.
             freq, frf, t, IR = bem(args)

    fmin, fmax: int, minimun and maximum frequency of interest.
            fmin, fmax = 20, 100

    winCheck: bool, option to view windowing in time domain.
            winCheck = True or False

    spectraCheck: bool, option to view frequency response and SBIR in frequency domain.
            spectraCheck = True or False

    modalCheck: bool, option to view room modes prediction of BEM simulation and cuboid approximation.
            modalCheck = True or False
    """

    if len(IR) < 20:
        print('IR resolution not high enough to calculate SBIR')

    if method == 'constant':
        peak = 0  # Window from the start of the IR
        dt = (max(t_IR) / len(t_IR))  # Time axis resolution
        tt_ms = round((ms / 1000) / dt)  # Number of samples equivalent to 64 ms

        # Windows
        post_peak = np.zeros((len(IR[:])))
        pre_peak = np.zeros((len(IR[:])))

        if cosWin is True:
            win_cos = win.cosine(int(2 * tt_ms)) ** 2  # Cosine squared window
        else:
            win_cos = win.tukey(int(2 * tt_ms), beta)  # Cosine window

        window = np.zeros((len(IR[:])))  # Final window
        ##
        win_cos[0:int(tt_ms)] = 1
        window[0:int(2 * tt_ms)] = win_cos
        ##

    elif method == 'peak':
        # Sample of the initial peak
        peak = detect_peaks(IR, mph=(max(IR) * 0.9), threshold=0, edge='rising', show=False)
        if len(peak) > 1:
            peak = peak[0]
            # print('More than one peak at the IR')
        #        ind[x] = 0; # Window max from the beginning
        # peak = 0  # Window from the start of the IR
        dt = (max(t_IR) / len(t_IR))  # Time axis resolution
        tt_ms = round((ms / 1000) / dt)  # Number of samples equivalent to 64 ms

        # Windows
        post_peak = np.zeros((len(IR[:])))
        pre_peak = np.zeros((len(IR[:])))
        win_cos = win.tukey(int(2 * tt_ms), beta)  # Cosine window
        window = np.zeros((len(IR[:])))  # Final window

        ms = 64
        # Sample of the initial peak
        peak = detect_peaks(IR, mph=(max(IR) * 0.9), threshold=0, edge='rising', show=False)
        if len(peak) > 1:
            peak = peak[0]
            # print('More than one peak at the IR')
        #        ind[x] = 0; # Window max from the beginning
        dt = (max(t_IR) / len(t_IR))  # Time axis resolution
        tt_ms = round((ms / 1000) / dt)  # Number of samples equivalent to 64 ms

        # Windows
        post_peak = np.zeros((len(IR[:])))
        pre_peak = np.zeros((len(IR[:])))
        win_cos = win.cosine(int(2 * tt_ms))  # Cosine window
        window = np.zeros((len(IR[:])))  # Final window
        ##
        # Cosine window pre peak
        win_cos_b = win.cosine(2 * peak + 1)
        pre_peak[0:int(peak)] = win_cos_b[0:int(peak)]
        pre_peak[int(peak)::] = 1

        # Cosine window post peak
        post_peak[int(peak):int(peak + tt_ms)] = win_cos[int(tt_ms):int(2 * tt_ms)] / max(
            win_cos[int(1 * tt_ms):int(2 * tt_ms)])  # Creating Hanning window array

        # Creating final window
        window[0:int(peak)] = pre_peak[0:int(peak)]
        #         window[0:int(peak)] = 1  # 1 from the beggining
        window[int(peak)::] = post_peak[int(peak)::]

    # Applying window
    IR_array = np.zeros((len(IR), 2))  # Creating matrix
    IR_array[:, 0] = IR[:]
    IR_array[:, 1] = IR[:] * window[:]  # FR and SBIR

    # Calculating FFT
    FFT_array = np.zeros((len(IR_array[:, 0]), len(IR_array[0, :])), dtype='complex')  # Creating matrix

    if ABEC is True:
        FFT_array_dB = np.zeros((round(len(IR_array[:, 0]) / 2) - delta_ABEC, len(IR_array[0, :])), dtype='complex')
        FFT_array_Pa = np.zeros((round(len(IR_array[:, 0]) / 2) - delta_ABEC, len(IR_array[0, :])), dtype='complex')
    else:
        FFT_array_dB = np.zeros((round(len(IR_array[:, 0]) / 2), len(IR_array[0, :])), dtype='complex')
        FFT_array_Pa = np.zeros((round(len(IR_array[:, 0]) / 2), len(IR_array[0, :])), dtype='complex')

    for i in range(0, len(IR_array[0, :])):
        iIR = IR_array[:, i]
        FFT_array[:, i] = 2 / len(iIR) * np.fft.fft(iIR)
        if ABEC is True:
            FFT_array_Pa[:, i] = FFT_array[delta_ABEC:round(len(iIR) / 2), i]
        else:
            FFT_array_Pa[:, i] = FFT_array[0:round(len(iIR) / 2), i]
    for i in range(0, len(IR_array[0, :])):
        if ABEC is True:
            FFT_array_dB[:, i] = 20 * np.log10(np.abs(FFT_array[delta_ABEC:int(len(FFT_array[:, i]) / 2),
                                                      i]) / 2e-5)  # applying log and removing aliasing and first 20 Hz
        else:
            FFT_array_dB[:, i] = 20 * np.log10(
                np.abs(FFT_array_Pa[:, i]) / 2e-5)  # applying log and removing aliasing and first 20 H

    if ABEC is False:
        freq_FFT = np.linspace(0, len(IR) / 2, num=int(len(IR) / 2))  # Frequency vector for the FFT
    else:
        freq_FFT = np.linspace(fmin, fmax, num=len(FFT_array_dB[:, 0]))  # Frequency vector for the FFT

    # View windowed Impulse Response in time domain:
    if winCheck is True:
        figWin = plt.figure(figsize=(12, 5), dpi=80, facecolor='w', edgecolor='k')
        win_index = 0
        plt.plot(t_IR, IR_array[:, win_index], linewidth=3)
        plt.plot(t_IR, IR_array[:, win_index + 1], '--', linewidth=5)
        plt.plot(t_IR, window[:] * (max(IR_array[:, 0])), '-.', linewidth=5)
        plt.title('Impulse Response Windowing', fontsize=20)
        plt.xlabel('Time [s]', fontsize=20)
        plt.xlim([t_IR[0], 0.12])#t_IR[int(len(t_IR) / 8)]])
        # plt.xticks(np.arange(t_IR[int(peak[0])], t_IR[int(len(t_IR))], 0.032), fontsize=15)
        plt.ylabel('Amplitude [-]', fontsize=20)
        plt.legend(['Modal IR', 'SBIR IR', 'Window'], loc='best', fontsize=20)
        plt.grid(True, 'both')
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.show()

    # Frequency Response and SBIR
    if spectraCheck is True:
        figSpectra = plt.figure(figsize=(12, 5), dpi=80, facecolor='w', edgecolor='k')

        if ABEC is False:
            plt.semilogx(freq_FFT[fmin:fmax + 1], FFT_array_dB[fmin:fmax + 1, 0], linewidth=3, label='Full spectrum')
            plt.semilogx(freq_FFT[fmin:fmax + 1], FFT_array_dB[fmin:fmax + 1, 1], '-.', linewidth=3, label='SBIR')
        elif ABEC is True:
            plt.semilogx(freq_FFT, FFT_array_dB[:, 0], linewidth=3, label='Full spectrum')
            plt.semilogx(freq_FFT, FFT_array_dB[:, 1], '-.', linewidth=3, label='SBIR')

        # plt.semilogx(freq_FFT[fmin:fmax + 1], 20 * np.log10(np.abs(FFT_array_Pa[fmin:fmax + 1, 1])/2e-5), ':',
        #              linewidth=3, label='SBIR 2')
        plt.legend(fontsize=15, loc='best')  # , bbox_to_anchor=(0.003, 0.13))
        # plt.title('Processed IR vs ABEC', fontsize=20)
        plt.xlabel('Frequency [Hz]', fontsize=20)
        plt.ylabel('SPL [dB ref. 20 $\mu$Pa]', fontsize=20)
        plt.gca().get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
        plt.gca().get_xaxis().set_minor_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
        plt.gca().tick_params(which='minor', length=5)  # Set major and minor ticks to same length
        plt.xticks(fontsize=15)
        plt.grid(True, 'both')
        plt.xlim([fmin, fmax])
        plt.ylim([55, 105])
        plt.yticks(fontsize=15)
        plt.tight_layout(pad=3)
        plt.show()
    return freq_FFT, FFT_array_Pa, window
    # return freq_FFT[closest_node(freq_FFT,fmin):closest_node(freq_FFT,fmax)+1], FFT_array_Pa[closest_node(freq_FFT,fmin):closest_node(freq_FFT,fmax)+1,1], window



class IR(object):
    """Perform a room impulse response computation."""

    def __init__(self, sampling_rate, duration,
            minimum_frequency, maximum_frequency):
        """
        Setup the room impulse computation.

        Parameters
        ----------
        sampling_rate : integer
            Sampling rate in Hz for the time signal.
        duration: float
            Time in seconds until which to sample the room impulse response.
        minimum_frequency: float
            Minimum sampling frequency
        maximum_frequency: float
            Maximum sampling frequency

        """
        self._number_of_frequencies = int(round(sampling_rate * duration))
        self._sampling_rate = sampling_rate
        self._duration = duration
        self._frequencies = (sampling_rate * np.arange(self._number_of_frequencies) 
                / self._number_of_frequencies)
        self._timesteps = np.arange(self._number_of_frequencies) / sampling_rate

        self._maximum_frequency = maximum_frequency
        self._minimum_frequency = minimum_frequency

        self._frequency_filter_indices = np.flatnonzero(
                (self._frequencies <= self._maximum_frequency) & 
                (self._frequencies >= self._minimum_frequency))

        self._high_pass_frequency = 2 * minimum_frequency
        self._low_pass_frequency = 2 * maximum_frequency

        self._high_pass_order = 4
        self._low_pass_order = 4

        self._alpha = 0.18  # Tukey window alpha

        
    @property
    def number_of_frequencies(self):
        """Return number of frequencies."""
        return self._number_of_frequencies

    @property
    def sampling_rate(self):
        """Return sampling rate."""
        return self._sampling_rate

    @property
    def duration(self):
        """Return duration."""
        return self._duration

    @property
    def timesteps(self):
        """Return time steps."""
        return self._timesteps

    @property
    def frequencies(self):
        """Return frequencies."""
        return self._frequencies

    @property
    def filtered_frequencies(self):
        """Return the filtered frequencies."""
        return self.frequencies[
                self._frequency_filter_indices
                ]

    @property
    def maximum_frequency(self):
        """Return maximum frequency."""
        return self._maximum_frequency

    @property
    def minimum_frequency(self):
        """Return minimum frequency."""
        return self._minimum_frequency

    @property
    def high_pass_frequency(self):
        """Return high pass frequency."""
        return self._high_pass_frequency

    @high_pass_frequency.setter
    def high_pass_frequency(self, freq):
        """Set high pass frequency."""
        self._high_pass_frequency = freq

    @property
    def low_pass_frequency(self):
        """Return low pass frequency."""
        return self._low_pass_frequency

    @low_pass_frequency.setter
    def low_pass_frequency(self, freq):
        """Set low pass frequency."""
        self._low_pass_frequency = freq

    @property
    def high_pass_filter_order(self):
        """Return high pass filter order."""
        return self._high_pass_order

    @high_pass_filter_order.setter
    def high_pass_filter_order(self, order):
        """Set high pass filter order."""
        self._high_pass_order = order

    @property
    def low_pass_filter_order(self):
        """Return low pass filter order."""
        return self._low_pass_order

    @low_pass_filter_order.setter
    def low_pass_filter_order(self, order):
        """Set low pass filter order."""
        self._low_pass_order = order


    def compute_room_impulse_response(
            self, values_at_filtered_frequencies):
        """
        Compute the room impulse response.

        Parameters
        ----------
        values_at_filtered_frequencies : array
            The frequency domain values to be transformed taken
            at the filtered frequencies.

        Output
        ------
        An array of approximate time values at the given time steps.
        
        """
        from scipy.signal import butter, freqz, tukey
        from scipy.fftpack import ifft
        
        b_high, a_high = butter(
                self.high_pass_filter_order,
                self.high_pass_frequency * 2 / self.sampling_rate, 
                'high')

        b_low, a_low = butter(
                self.low_pass_filter_order,
                self.low_pass_frequency * 2 / self.sampling_rate, 
                'low')

        high_pass_values = freqz(
                b_high, a_high, self.filtered_frequencies,
                fs=self.sampling_rate)[1]

        low_pass_values = freqz(
                b_low, a_low, self.filtered_frequencies,
                fs=self.sampling_rate)[1]

        butter_filtered_values = (values_at_filtered_frequencies * 
                np.conj(low_pass_values) * np.conj(high_pass_values))

        # windowed_values = butter_filtered_values * tukey(len(self.filtered_frequencies),
        #         min([self.maximum_frequency - self.low_pass_frequency,
        #              self.high_pass_frequency - self.minimum_frequency]) /
        #         (self.maximum_frequency - self.minimum_frequency))

        windowed_values = butter_filtered_values * tukey(len(self.filtered_frequencies), alpha=self._alpha)

        full_frequency_values = np.zeros(self.number_of_frequencies, dtype='complex128')
        full_frequency_values[self._frequency_filter_indices] = windowed_values
        full_frequency_values[-self._frequency_filter_indices] = np.conj(windowed_values)

        return ifft((full_frequency_values)) * self.number_of_frequencies
    


