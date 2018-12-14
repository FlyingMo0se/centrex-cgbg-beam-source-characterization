# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 13:32:33 2018

@author: Oskari

This is a test program which is supposed to simulate a DAQ, fetch the simulated
data and plot it using the niscope module. A graphical user interface is
implemented using tkinter.
"""

import numpy as np
import niscope
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk


class Device:
    def __init__(self):
        self.resource_name = 'PXI1Slot2'
        self.options = {'simulate': True, 'driver_setup': {'Model': '5164', 'BoardType': 'PXIe', }, }
        self.n_samples = 1000
        self.sample_rate = 10000.0
        self.vertical_range = 2.0
        self.coupling = niscope.VerticalCoupling.DC
        self.channel_list = [0]
        self.waveforms = [np.ndarray(self.n_samples, dtype=np.float64) for c in self.channel_list]
        

class PhotodetectorDAQ_GUI(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.winfo_toplevel().title("CENTREX Photodtector DAQ")
        self.parent = parent
        self.pack()
        
        test_GUI(self, *args, *kwargs)
        

class test_GUI(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self,parent)
        self.pack()
        self.create_widgets()
        #self.create_figure()
        self.DAQ = Device()
        print(self.DAQ.channel_list)
        
        
        
        
    def create_widgets(self):
        self.acquire_data = tk.Button(self)
        self.acquire_data["text"] = "Press here to acquire simulated data"
        self.acquire_data["command"] = self.acquire_data
        self.acquire_data.pack(side="top")
        
        self.plot = ttk.Button(self)
        self.plot["text"] = "Plot data"
        self.plot["command"] = self.plot_waveforms
        self.plot.pack(side = "top")
        
    def create_figure(self):
        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side="bottom", fill="both", expand=True)
        

    def acquire_data(self):
        DAQ = self.DAQ
        # 1. Initialize DAQ
        with niscope.Session(resource_name=DAQ.resource_name,options=DAQ.options) as session:
                        
            # 2. Configure settings for data acquisition:
            #Configure timing settings
            session.configure_horizontal_timing(min_sample_rate=DAQ.sample_rate, 
                                                min_num_pts = DAQ.n_samples,
                                                ref_position = 0.0,
                                                num_records = 1,
                                                enforce_realtime = True)
            
            # Configure vertical (data) settings:
            session.channels[DAQ.channel_list].configure_vertical(DAQ.vertical_range, DAQ.coupling)
            
            # 3. Allocate numpy arrays for data
            DAQ.waveforms = [np.ndarray(DAQ.n_samples, dtype=np.float64) for c in DAQ.channel_list]
            
            # 4. Configure trigger
            session.configure_trigger_immediate()
            
            # 5. Collect data using fetch_into()
            #Initiate data collection
            with session.initiate():
                for channel, waveform in zip(DAQ.channel_list, DAQ.waveforms):
                    session.channels[channel].fetch_into(waveform)
                    DAQ.waveforms[channel] = waveform
    
    def plot_waveforms(self):      
        DAQ = self.DAQ
        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side="bottom", fill="both", expand=True)
        
        #Finally, plot the waveforms
        for waveform in DAQ.waveforms:
            x = np.arange(0,len(waveform),1)
            a.plot(x,waveform)

root = tk.Tk()
GUI = PhotodetectorDAQ_GUI(root)
GUI.mainloop()
    