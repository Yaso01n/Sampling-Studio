
from io import BytesIO
from re import X
from tkinter import Button
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

Fs = 1000    #Sampling Freqyency    
t = np.arange(0, 1 + 1 / Fs, 1 / Fs)    # Time

def find_amplitude(signal):
    np_fft = np.fft.fft(signal)
    amplitudes = 2 / 1002* np.abs(np_fft) 
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
    max_freq = max(fft_clean_frequencies_array)
    if max_freq >42:
        return floor(max_freq)
    else: return ceil(max_freq) 


def demo():
    y_demo=np.sin(2 * np.pi * t)
    return y_demo


 
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
    worksheet.write_column(0,0,time)
    worksheet.write_column(0,1,magnitude)
    add_number_x=[-2.93915E-15]
    add_number_y=[1]
    worksheet.write_column(1001,1,add_number_x)
    worksheet.write_column(1001,0,add_number_y)

    workbook.close()
    #Button of downloading
    st.download_button(
        label="Download",
        data=output.getvalue(),
        file_name="signal.xlsx",
        mime="application/vnd.ms-excel"
    )


def sum_signal(data, new1):
    newSignal = data + new1
    return newSignal

def add_signal():
    Add_F= st.sidebar.slider("F(max)")
    Add_Am = st.sidebar.slider("Amp.")        
    Include_signal= Add_Am * np.sin( 2 * np.pi * Add_F* t)
    return Include_signal, Add_Am, Add_F;




# horizontal menu
selected2 = option_menu(None, ["Generate"], 
    icons=['play'], 
    menu_icon="cast", default_index=0, orientation="horizontal" ,
    styles={
        "container": {"padding": "0 px"},
        "icon": {"color": "black", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "grey"},
    }

    )

# def callback():
#     st.session_state.button_clicked=True
# st.button(
#     label='upload',
#     on_click=callback()
# )
upload_ck= st.sidebar.checkbox("Upload")
if upload_ck:
    upload_file= st.file_uploader("Browse")
    if upload_file:
        signal_upload=pd.read_excel(upload_file)
        y_signal= signal_upload[signal_upload.columns[1]]
        x_signal= signal_upload[signal_upload.columns[0]]
        amplitude=find_amplitude(y_signal)
        frequency= max_frequency(y_signal/amplitude,x_signal)


    
        sampleRate = st.sidebar.slider("sample rate", min_value=0,max_value=10)
        
        noise_ck = st.sidebar.checkbox('Add Noise') 

        if noise_ck:
            number = st.sidebar.slider('Insert SNR')
            new_signal = Noise(y_signal, number,1001)
            y_signal = amplitude * np.sin(2 * np.pi * frequency * t) + new_signal
        
        signal_figure= px.line(signal_upload, x=x_signal, y=y_signal, title="The normal signal")
        addSignal = st.sidebar.checkbox('Add Signal')
        
        if addSignal:
            added, addedAmp, addedFreq= add_signal()
            sumSignal = st.sidebar.button('Sum Signals')
            signal_figure.add_scatter(x=t, y=added, mode="lines",name="Added signal",line={"color":"#e0b0ff"})
            if sumSignal:
                    y_signal= sum_signal(y_signal,added)
                    frequency=frequency+addedFreq
                    amplitude=amplitude+addedAmp
                    signal_figure = px.line(y_signal, x=t, y=y_signal)
                    
       
    
        #sampling func
        frequency_sample=frequency*sampleRate
        if frequency_sample!=0:
            T=1/frequency_sample
            n_Sample=np.arange(0,1/T)
            t_sample = n_Sample * T
            
            amplitude=find_amplitude(y_signal)
            frequency=max_frequency(y_signal/amplitude,x_signal)


            signal_sample = amplitude * np.sin(2 * np.pi * frequency * t_sample)

            signal_figure.add_scatter(x=t_sample, y=signal_sample,mode="markers",name="samples points", marker={"color":"black"})
            
            Inter=st.checkbox("interpolation")
            if Inter:
                sum=0

                amplitude=find_amplitude(y_signal)
                frequency=max_frequency(y_signal/amplitude,x_signal)
                
                for i in n_Sample:
                    s_sample = amplitude * np.sin(2 * np.pi * frequency *i* T)
                    sum+= np.dot(s_sample,np.sinc((t-i*T)/T))

                signal_figure.add_scatter(x=t, y=sum, mode="lines",name="Reconstructed signal", line={"color":"red"})
            
        st.plotly_chart(signal_figure, use_container_width=True)
        

        download(x_signal,y_signal)

elif selected2=="Generate":

    #drawing normal sine
    frequency = st.sidebar.slider("Frequency", min_value=1)
    amplitude = st.sidebar.slider("Amplitude", min_value=1)
    sampleRate = st.sidebar.slider("sample rate", min_value=0,max_value=10)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    frequency_sample= sampleRate*frequency
    noise = st.sidebar.checkbox('Add noise')
    

    if noise:
        
        number = st.sidebar.slider('Insert SNR')
        new_signal = Noise(signal, number,1001)
        signal = amplitude * np.sin(2 * np.pi * frequency * t) + new_signal
    
    addSignal = st.sidebar.checkbox('Add Signal')
    

    fig = px.line(signal, x=t, y=signal).update_layout(xaxis_title="Time", yaxis_title="Amplitude")

    
    if addSignal:
        added, addedAmp, addedFreq= add_signal()
        sumSignal = st.sidebar.button('Sum Signals')
        fig.add_scatter(x=t, y=added, mode="lines",name="added signal",line={"color":"#e0b0ff"})
        if sumSignal:
                signal= sum_signal(signal,added)  
                frequency=frequency+addedFreq
                amplitude=amplitude+addedAmp
                fig = px.line(signal, x=t, y=signal)

                
    
    #sampling func
    if frequency_sample!=0:
        T=1/frequency_sample
        n_Sample=np.arange(0,1/T)
        t_sample = n_Sample * T
        
        amplitude=find_amplitude(signal)
        frequency=max_frequency(signal/amplitude,t)
        signal_sample = amplitude * np.sin(2 * np.pi *frequency* t_sample)
    
            
        fig.add_scatter(x=t_sample, y=signal_sample, mode="markers",name="samples points", marker={"color":"black"})
        
        
        Inter=st.checkbox("interpolation")
        if Inter:
            sum=0
            amplitude=find_amplitude(signal)
            frequency=max_frequency(signal/amplitude,t)
            for i in n_Sample:
                s_sample = amplitude* np.sin(2 * np.pi * frequency *i* T)
                sum+= np.dot(s_sample,np.sinc((t-i*T)/T))
        
            fig.add_scatter(x=t, y=sum, mode="lines",name="Reconstructed signal", line={"color":"red"})

            
    st.plotly_chart(fig,  use_container_width=True)

    download(t,signal)
    













