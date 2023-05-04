import numpy as np
import pandas as pd
import os, shutil
import math
import pyodbc
from IPython.display import clear_output

from tkinter import *
from tkinter.messagebox import askokcancel

import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import n_colors

import Sivarita


def loadData(datapath):

    #
    user_list = []
    sessions_list = []
    df_activity = []
    df_list = []
    trials_list = []
    activity_list = []
    activity_type_list = []
    upper_size = []
    down_size = []
    num_rep = []
    arm = []


    activity_folders = next(os.walk(datapath))[1]

    for activity in activity_folders:
        session_folders = next(os.walk(datapath + activity))[1]
        for ses in session_folders:
            path = datapath + activity + "/" + ses
            user_list.append('Alberto_Villegas')
            sessions_list.append(ses)
            activity_list.append(activity)

            df = Sivarita.loadData(path)
            df_list.append(df)

            df_2, df_trials = Sivarita.loadDataActivity(path)
            activity_type_list.append(df_2.modo[0])
            upper_size.append(df_2.upper_size[0])
            down_size.append(df_2.fore_size[0])
            num_rep.append(df_2.trials[0])
            df_activity.append(df_2)
            trials_list.append(df_trials)


    data = {
        'usuario': user_list,
        'sesion': sessions_list,
        'actividad': activity_list,
        'tipo_actividad': activity_type_list,
        'upper_size': upper_size,
        'fore_size': down_size,
        'num_rep': num_rep,
        'dataFrame': df_list
    }

    clear_output(wait=True)

    df = pd.DataFrame(data)

    return df

def downloadDataFromTeams(path_folder): #\2022_rubidium_longitudinal\_data_
# Descargar datos de la carpeta "path" de teams.
# [IMPORTANTE] Es necesario tener sincronizado el canal Experimentaciones de Teams con el ordenador.

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    answer = askokcancel("Descargar datos de Teams", "Quieres descargar y actualizar los datos de la carpeta data de Teams?")

    root.destroy()

    if answer == True:
        
        path = os.path.join(os.path.join(os.environ['ONEDRIVE']), '') 
        path = path.split('\\')
        src = ''
        for p in path[0:-2]:
            src = src + p + '/'
        teams = "/UMH/RAI - Experimentaciones"
        src = src + teams + path_folder
        # src = r"C:\Users\%s\UMH\RAI - Experimentaciones\2023_evaluacion_eeg-emg\EMG\_data_" % os.getlogin()
        
        dst = "_data_"

        #Remove "_data_" folder
        try:
            shutil.rmtree(dst, ignore_errors=False, onerror=None)
        except:
            print("Carpeta src no se puede eliminar porque no existe")
            
        # Download "_data_" folder
        destination = shutil.copytree(src, dst)
            
    return

def process_all_params(df_data):

    session_list = []
    activity_list = []
    activity_type_list = []

    q1_trajectory = []
    q2_trajectory = []
    q3_trajectory = []
    q4_trajectory = []
    q5_trajectory = []
    q6_trajectory = []
    q7_trajectory = []

    max_q1 = []
    max_q2 = []
    max_q3 = []
    max_q4 = []
    max_q5 = []
    max_q6 = []
    max_q7 = []

    min_q1 = []
    min_q2 = []
    min_q3 = []
    min_q4 = []
    min_q5 = []
    min_q6 = []
    min_q7 = []

    

    for i_row in range(df_data.shape[0]):

        session_list.append(df_data.sesion[i_row])
        activity_list.append(df_data.actividad[i_row])
        activity_type_list.append(df_data.tipo_actividad[i_row])

        clear_output(wait=True)
        df = df_data.dataFrame[i_row]

        #MAX ANGLES
        max_q1.append(Sivarita.maxAngle(df,'q1'))
        max_q2.append(Sivarita.maxAngle(df,'q2'))
        max_q3.append(Sivarita.maxAngle(df,'q3'))
        max_q4.append(Sivarita.maxAngle(df,'q4'))
        max_q5.append(Sivarita.maxAngle(df,'q5'))
        max_q6.append(Sivarita.maxAngle(df,'q6'))
        max_q7.append(Sivarita.maxAngle(df,'q7'))

        #MIN ANGLES
        min_q1.append(Sivarita.minAngle(df,'q1'))
        min_q2.append(Sivarita.minAngle(df,'q2'))
        min_q3.append(Sivarita.minAngle(df,'q3'))
        min_q4.append(Sivarita.minAngle(df,'q4'))
        min_q5.append(Sivarita.minAngle(df,'q5'))
        min_q6.append(Sivarita.minAngle(df,'q6'))
        min_q7.append(Sivarita.minAngle(df,'q7'))
        #q1_trajectory.append()

    data = {

        'session': session_list, 'activity': activity_list, 'tipo_actividad': activity_type_list,
        'MAX_Q1': max_q1, 'MAX_Q2': max_q2, 'MAX_Q3': max_q3, 'MAX_Q4': max_q4, 'MAX_Q5': max_q5, 'MAX_Q6': max_q6, 'MAX_Q7': max_q7,
        'MIN_Q1': min_q1, 'MIN_Q2': min_q2, 'MIN_Q3': min_q3, 'MIN_Q4': min_q4, 'MIN_Q5': min_q5, 'MIN_Q6': min_q6, 'MIN_Q7': min_q7

    }

    df = pd.DataFrame(data)

    return df

