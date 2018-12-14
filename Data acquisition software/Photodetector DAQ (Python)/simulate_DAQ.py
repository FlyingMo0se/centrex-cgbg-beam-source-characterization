# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 13:32:33 2018

@author: Oskari

This is a test program which is supposed to simulate a DAQ, fetch the simulated
data and plot it using the niscope module.
"""

import numpy as np
import niscope
import matplotlib.pyplot as plt

def simulate_DAQ(resource_name, options, n_samples, sample_rate, vertical_range,coupling):
    # 1. Initialize DAQ
    with niscope.Session(resource_name=resource_name,options=options) as session:
        
        # List channels to be used:
        channel_list = [0]
        
        # 2. Configure settings for data acquisition:
        #Configure timing settings
        session.configure_horizontal_timing(min_sample_rate=sample_rate, 
                                            min_num_pts = n_samples,
                                            ref_position = 0.0,
                                            num_records = 1,
                                            enforce_realtime = True)
        
        # Configure vertical (data) settings:
        session.channels[channel_list].configure_vertical(vertical_range, coupling)
        
        # 3. Allocate numpy arrays for data
        waveforms = [np.ndarray(n_samples, dtype=np.float64) for c in channel_list]
        
        # 4. Configure trigger
        session.configure_trigger_immediate()
        
        # 5. Collect data using fetch_into()
        #Initiate data collection
        with session.initiate():
            for channel, waveform in zip(channel_list, waveforms):
                session.channels[channel].fetch_into(waveform)
                
    return waveforms

def plot(waveforms):
    for waveform in waveforms:
        x = np.arange(0,len(waveform),1)
        plt.plot(x,waveform)
    
def main():
    resource_name = 'PXI1Slot2'
    options = {'simulate': True, 'driver_setup': {'Model': '5164', 'BoardType': 'PXIe', }, }
    n_samples = 1000
    sample_rate = 10000.0
    vertical_range = 2.0
    coupling = niscope.VerticalCoupling.DC
    waveforms = simulate_DAQ(resource_name, options, n_samples, sample_rate, vertical_range,coupling)
    
    plot(waveforms)
    