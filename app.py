from io import BytesIO
from pickle import FALSE
from re import X
from tkinter import Button
from tkinter.simpledialog import SimpleDialog
from blinker import Signal
from matplotlib.axis import XAxis
import numpy as np
import scipy
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.express import colors
import matplotlib.pyplot as plt
import streamlit.components.v1 as com
import time
import xlsxwriter
from collections import deque
import hydralit_components as hc
from streamlit_option_menu import option_menu
from streamlit import button
import scipy as sc
from scipy.interpolate import interp1d, interp2d,splev
from math import ceil,floor


st.set_page_config(page_title="sampling studio", page_icon=":bar_chart:",layout="wide")


#hiding copyright things
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
with open("style.css") as source_des:
    st.markdown(f"""<style>{source_des.read()}</style>""", unsafe_allow_html=True)


ColorMinMax = st.markdown(''' <style> div.stSlider > div[data-baseweb = "slider"] > div[data-testid="stTickBar"] > div {
    background: rgb(1 1 1 / 0%); } </style>''', unsafe_allow_html = True)


Slider_Cursor = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
    background-color: rgb(14, 38, 74); box-shadow: rgb(14 38 74 / 20%) 0px 0px 0px 0.2rem;} </style>''', unsafe_allow_html = True)

    
Slider_Number = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div > div
                                { color: rgb(14, 38, 74); } </style>''', unsafe_allow_html = True)
    

col = f''' <style> div.stSlider > div[data-baseweb = "slider"] > div > div {{
    background: linear-gradient(to right,rgba(151, 166, 195, 0.25)  0%, 
                                 rgba(151, 166, 195, 0.25) , 
                                rgb(1, 183, 158), 
                                rgb(1, 183, 158) 100%); }} </style>'''

ColorSlider = st.markdown(col, unsafe_allow_html = True)

Fs = 10000    #Sampling Frequency    
t = np.arange(0, 1 + 1 / Fs, 1 / Fs)    # Time

def find_amplitude(signal):
    np_fft = np.fft.fft(signal)
    amplitudes = 2 / 10002* np.abs(np_fft) 
    return ceil(max(amplitudes))


def max_frequency(magnitude=[],time=[]):
    sample_period = time[1]-time[0]
    n_samples = len(time)
    fft_magnitudes=np.abs(np.fft.fft(magnitude))
    fft_frequencies = np.fft.fftfreq(n_samples, sample_period)
    fft_clean_frequencies_array = []
    for i in range(len(fft_frequencies)):
        if fft_magnitudes[i] > 22:
            fft_clean_frequencies_array.append(fft_frequencies[i])
    if len(fft_clean_frequencies_array)==0:
        return 0
    else:    
        max_freq = max(fft_clean_frequencies_array)
        if max_freq >42:
            return floor(max_freq)
        else: return ceil(max_freq) 



# Initialization of session state
if 't' not in st.session_state:
    st.session_state['t'] = t
if 'sum' not in st.session_state:
    st.session_state['sum'] = 0    

if 'signals table' not in st.session_state:
    st.session_state['signals table'] = []
if 'signal' not in st.session_state:
        #st.session_state['signal']=1*np.sin(2*np.pi*1*st.session_state['t'])
        st.session_state['signal'] = np.zeros(len(st.session_state['t']))
        for i in range(len(st.session_state['t'])):
            signal = 1*np.sin(2*np.pi*1*st.session_state['t'][i])
            st.session_state['signal'][i] += signal
        st.session_state['signals table'].append([1,1])


if 'noise' not in st.session_state:
    st.session_state['noise'] = 1

if 'up_noise' not in st.session_state:
    st.session_state['up_noise'] = 1

if 'sample_by_frequency' not in st.session_state:
    st.session_state['sample_by_frequency'] = 1

if 'sample_rate' not in st.session_state:
    st.session_state['sample_rate'] = 2 

def update_signal(magnitude, frequency):
    for i in range(len(st.session_state['t'])):
        st.session_state['signal'][i] += magnitude * \
            np.sin(2*np.pi*frequency*st.session_state['t'][i])        
 
#Adding noise to signal
def Noise(Data, number,n):
    snr = 10.0**(number/10.0)
    power_signal = Data.var()   #power signal
    Noise = power_signal/snr
    noise_signal = sc.sqrt(Noise)*sc.randn(n)    #Noise Signal
        
    return noise_signal

                
#download a file as excel
def download(time , magnitude):
    output = BytesIO()
    # Write files to in-memory strings using BytesIO
    # See: https://xlsxwriter.readthedocs.io/workbook.html?highlight=BytesIO#constructor
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet()
    worksheet.write(0,0,0)
    worksheet.write(0,1,0)
    worksheet.write_column(1,0,time)
    worksheet.write_column(1,1,magnitude)
    add_number_x=[-2.93915E-15]
    add_number_y=[1]
    worksheet.write_column(10001,1,add_number_x)
    worksheet.write_column(10001,0,add_number_y)

    workbook.close()
    #Button of downloading
    st.download_button(
        label="Download",
        data=output.getvalue(),
        file_name="signal.xlsx",
        mime="application/vnd.ms-excel"
    )

upload_file= st.sidebar.file_uploader("")

if upload_file:
    signal_upload=pd.read_excel(upload_file)
    st.session_state['signal']= signal_upload[signal_upload.columns[1]]
    x_signal= signal_upload[signal_upload.columns[0]]
    amplitude=find_amplitude(st.session_state['signal'])
    frequency= max_frequency(st.session_state['signal']/amplitude,x_signal)
    if frequency==0:
        signal_figure= px.line(x_signal,st.session_state['signal'], title="Reconstructed signal that is almost a line")
       
    else:  
        frequency_added=st.sidebar.slider("Frequency of the added signal", min_value=1)
        amp_added=st.sidebar.slider("amplitude of the added signal", min_value=1)
        if st.sidebar.button('Add signal'): 
            update_signal(amp_added, frequency_added)
            st.session_state['signals table'].append([amp_added, frequency_added])
            st.session_state['signal']+=st.session_state['signal']
            signal_figure = px.line(st.session_state['signal'], x=t, y=st.session_state['signal']).update_layout(xaxis_title="Time (Sec)", yaxis_title="Amplitude")
        undo_signals = st.sidebar.multiselect("Remove signals", options=st.session_state['signals table'])
        for item in undo_signals:    # #remove signals
            update_signal(-1.0*item[0], item[1])
            for item2 in st.session_state['signals table']:
                if item == item2:
                    st.session_state['signals table'].remove(item2)
        signal_figure = px.line(x=t, y=st.session_state['signal']).update_layout(xaxis_title="Time (Sec)", yaxis_title="Amplitude")
        sampleByFreqUp_ck=st.sidebar.checkbox('Sample by frequency')
        if sampleByFreqUp_ck:
            sampleByFreq_sl = st.sidebar.slider("Frequency", min_value=1,value=st.session_state['sample_by_frequency'])
            frequency_sample=sampleByFreq_sl
            st.session_state['sample_by_frequency']= sampleByFreq_sl
        else:
            sampleRate = st.sidebar.slider("sample rate", min_value=0,max_value=10,value=st.session_state['sample_rate'])
            frequency_sample= sampleRate*frequency
            st.session_state['sample_rate']= sampleRate
        
        noise_ck = st.sidebar.checkbox('Add Noise') 
        if noise_ck:
            number = st.sidebar.slider('Insert SNR',min_value=1, value=st.session_state['up_noise'])
            new_signal = Noise(st.session_state['signal'], number,10001)
            st.session_state['up_noise'] = number
            st.session_state['signal'] = amplitude * np.sin(2 * np.pi * frequency * t) + new_signal
        
        signal_figure = px.line(x=t, y=st.session_state['signal']).update_layout(xaxis_title="Time (Sec)", yaxis_title="Amplitude")
        
        #sampling func
        if frequency_sample!=0:
            T=1/frequency_sample
            n_Sample=np.arange(0,1/T)
            t_sample = n_Sample * T
            
            amplitude=find_amplitude(st.session_state['signal'])
            frequency=max_frequency(st.session_state['signal']/amplitude,x_signal)
            signal_sample = amplitude * np.sin(2 * np.pi * frequency * t_sample)
            signal_figure.add_scatter(x=t_sample, y=signal_sample,mode="markers",name="samples points", marker={"color":"black"})
            
            Inter=st.checkbox("interpolation")
            if Inter:
                sum=0
                for i in n_Sample:
                    s_sample = amplitude * np.sin(2 * np.pi * frequency *i* T)
                    sum+= np.dot(s_sample,np.sinc((t-i*T)/T))
                signal_figure.add_scatter(x=t, y=sum, mode="lines",name="Reconstructed signal", line={"color":"red"})
    
        st.plotly_chart(signal_figure, use_container_width=True)    
    download(x_signal,st.session_state['signal'])
else:
    #drawing normal sine
    frequency = st.sidebar.slider("Frequency", min_value=0,value=1)
    st.session_state['frequency']=frequency
    amplitude = st.sidebar.slider("Amplitude", min_value=0,value=1)
    st.session_state['amp']=amplitude
    st.session_state['signal'] = amplitude * np.sin(2 * np.pi * frequency * t)
    
    
    #adding signals
    if st.sidebar.button("Add Signal"):
        update_signal(amplitude, frequency)
        st.session_state['signals table'].append([amplitude, frequency])
        fig = px.line(x=t, y=st.session_state['signal']).update_layout(xaxis_title="Time (Sec)", yaxis_title="Amplitude")

    undo_signals = st.sidebar.multiselect("Remove signals", options=st.session_state['signals table'])
    for item in undo_signals:    #remove signals
        update_signal(-1.0*item[0], item[1])
        for item2 in st.session_state['signals table']:
            if item == item2:
                st.session_state['signals table'].remove(item2)
    fig = px.line(x=t, y=st.session_state['signal']).update_layout(xaxis_title="Time (Sec)", yaxis_title="Amplitude")

                
    sampleByfreq_ck=st.sidebar.checkbox('Sample By Frequency')
    
    if sampleByfreq_ck:
        sampleByFreq_sl = st.sidebar.slider("Frequency", min_value=1,value=st.session_state['sample_by_frequency'])
        frequency_sample=sampleByFreq_sl
        st.session_state['sample_by_frequency']= sampleByFreq_sl
    else:
        sampleRateUp = st.sidebar.slider("Sample rate", min_value=0,max_value=10,value=st.session_state['sample_rate'])
        frequency_sample= sampleRateUp*frequency
        st.session_state['sample_rate']= sampleRateUp
    
    noise = st.sidebar.checkbox('Add noise')
    if noise:
        
        number = st.sidebar.slider('Insert SNR',min_value=1,value=st.session_state['noise'])
        new_signal = Noise(st.session_state['signal'], number,10001)
        st.session_state['noise'] =number
        st.session_state['signal'] = amplitude * np.sin(2 * np.pi * frequency * t) + new_signal
        
    
    fig = px.line(x=t, y=st.session_state['signal']).update_layout(xaxis_title="Time (Sec)", yaxis_title="Amplitude")
    fig.update_yaxes(title_font=dict(size=18,family="Arial"))
    fig.update_xaxes(title_font=dict(size=18,family="Arial"))
    
    
    #sampling func
    if frequency_sample!=0:
        T=1/frequency_sample
        n_Sample=np.arange(0,1/T)
        t_sample = n_Sample * T
        
        amplitude=find_amplitude(st.session_state['signal'])
        frequency=max_frequency(st.session_state['signal']/amplitude,t)
        signal_sample = amplitude * np.sin(2 * np.pi *frequency* t_sample)

            
        fig.add_scatter(x=t_sample, y=signal_sample, mode="markers",name="samples points", marker={"color":"black"})
        
        
        Inter=st.checkbox("interpolation")
        if Inter:
            sum=0
            for i in n_Sample:
                s_sample = amplitude* np.sin(2 * np.pi * frequency *i* T)
                sum+= np.dot(s_sample,np.sinc((t-i*T)/T))
            st.session_state['signal']=sum
            fig.add_scatter(x=t, y=st.session_state['signal'], mode="lines",name="Reconstructed signal", line={"color":"red"})


        #fig.add_scatter(x=t_sample, y=sample_s, mode="markers",name="samples points", marker={"color":"black"})
    st.plotly_chart(fig, use_container_width=True)
    download(t,st.session_state['signal'])
