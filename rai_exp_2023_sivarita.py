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
    user_id_list = []
    sessions_list = []
    df_activity = []
    df_list = []
    df_IA_list = []
    trials_list = []
    activity_list = []
    activity_type_list = []
    upper_size = []
    down_size = []
    num_rep = []
    arm = []

    users_folders = next(os.walk(datapath))[1]
    for user in users_folders:
        activity_folders = next(os.walk(datapath + user + "/"))[1]
        for activity in activity_folders:
            session_folders = next(os.walk(datapath + user + "/" + activity))[1]
            for ses in session_folders:

                #Añadir el nuevo path  de la sesion
                path = datapath + user + "/" + activity + "/" + ses

                #Añadir los datos extras
                user_list.append(user)
                #user_id_list.append(df_excel.loc[df_excel['nombre'] == user, "id"].values[0])
                sessions_list.append(ses)
                activity_list.append(activity)

                # Cargar los DataJoints.bin
                df = Sivarita.loadData(path)
                df_list.append(df)

                # Cargar los Data_Activity.bin
                df_2, df_trials = Sivarita.loadDataActivity(path)
                activity_type_list.append(df_2.modo[0])
                upper_size.append(df_2.upper_size[0])
                down_size.append(df_2.fore_size[0])
                arm.append(chr(round(df_2.brazo[0])))
                num_rep.append(df_2.trials[0])
                df_activity.append(df_2)
                trials_list.append(df_trials)

                # Cargar los Data_IA.bin
                df_IA = Sivarita.loadDataIA(path)
                df_IA_list.append(df_IA)



    data = {
        'usuario': user_list,
        'sesion': sessions_list,
        'actividad': activity_list,
        'tipo_actividad': activity_type_list,
        'upper_size': upper_size,
        'fore_size': down_size,
        'brazo': arm,
        'num_rep': num_rep,
        'dataFrame': df_list,
        'dataFrameIA': df_IA_list
    }

    clear_output(wait=True)

    df = pd.DataFrame(data)

    return df

def downloadDataFromTeams(path_folder): #\Resultados\_data_
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
        teams = "/UMH/RAI - SIVARITA"
        src = src + teams + path_folder
        # src = r"C:\Users\%s\UMH\RAI - Experimentaciones\2023_sivarita\_data_" % os.getlogin()
        
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

    user_list = []
    session_list = []
    activity_list = []
    activity_type_list = []
    arm_list = []

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

    dtw_err_list = []
    dtw_org_err_list = []
    dtw_porc_err_list = []



    for i_row in range(df_data.shape[0]):

        user_list.append(df_data.usuario[i_row])
        session_list.append(df_data.sesion[i_row])
        activity_list.append(df_data.actividad[i_row])
        activity_type_list.append(df_data.tipo_actividad[i_row])
        arm_list.append(df_data.brazo[i_row])

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


        #DTW
        df_dtw = df_data.dataFrameIA[i_row]
        tarea = df_data.tipo_actividad[i_row]
        dtw_org_error,dtw_error, porc_error = Sivarita.computeDTW(df_dtw, tarea)
        dtw_org_err_list.append(dtw_org_error)
        dtw_err_list.append(dtw_error)
        dtw_porc_err_list.append(porc_error)


    data = {

        'usuario': user_list, 'session': session_list, 'activity': activity_list, 'tipo_actividad': activity_type_list, 'brazo': arm_list,
        'DTW_org_error': dtw_org_err_list,'DTW_error': dtw_err_list, 'Porcentaje_pred': dtw_porc_err_list,
        'MAX_Q1': max_q1, 'MAX_Q2': max_q2, 'MAX_Q3': max_q3, 'MAX_Q4': max_q4, 'MAX_Q5': max_q5, 'MAX_Q6': max_q6, 'MAX_Q7': max_q7,
        'MIN_Q1': min_q1, 'MIN_Q2': min_q2, 'MIN_Q3': min_q3, 'MIN_Q4': min_q4, 'MIN_Q5': min_q5, 'MIN_Q6': min_q6, 'MIN_Q7': min_q7

    }

    df = pd.DataFrame(data)

    return df

def filter_byActivities():
    return

def process_byGroups(df_main, df_params, group):
    #hawer = df_main.nombre.unique()
    nombres = df_main[df_main.grupo == group].nombre.to_list()

    user_list = []
    mean_MAXQ1 = []
    mean_MAXQ2 = []
    mean_MAXQ3 = []
    mean_MAXQ4 = []
    mean_MAXQ5 = []
    mean_MAXQ6 = []
    mean_MAXQ7 = []

    mean_MINQ1 = []
    mean_MINQ2 = []
    mean_MINQ3 = []
    mean_MINQ4 = []
    mean_MINQ5 = []
    mean_MINQ6 = []
    mean_MINQ7 = []

    mean_dtw_error = []
    mean_dtw_org_error = []
    mean_porc_pred = []


    for user in nombres:
        # Filtrar brazo
        #brazo = df_main[df_main.nombre == user].brazo.to_list()[0]
        if user in df_params['usuario'].values:
            # User
            user_list.append(user)

            # Mean MAX Q
            mean_MAXQ1.append(np.nanmean(df_params[(df_params.usuario==user)].MAX_Q1.to_numpy()))
            mean_MAXQ2.append(np.nanmean(df_params[(df_params.usuario==user)].MAX_Q2.to_numpy()))
            mean_MAXQ3.append(np.nanmean(df_params[(df_params.usuario==user)].MAX_Q3.to_numpy()))
            mean_MAXQ4.append(np.nanmean(df_params[(df_params.usuario==user)].MAX_Q4.to_numpy()))
            mean_MAXQ5.append(np.nanmean(df_params[(df_params.usuario==user)].MAX_Q5.to_numpy()))
            mean_MAXQ6.append(np.nanmean(df_params[(df_params.usuario==user)].MAX_Q6.to_numpy()))
            mean_MAXQ7.append(np.nanmean(df_params[(df_params.usuario==user)].MAX_Q7.to_numpy()))

            # Mean MIN Q
            mean_MINQ1.append(np.nanmean(df_params[(df_params.usuario==user)].MIN_Q1.to_numpy()))
            mean_MINQ2.append(np.nanmean(df_params[(df_params.usuario==user)].MIN_Q2.to_numpy()))
            mean_MINQ3.append(np.nanmean(df_params[(df_params.usuario==user)].MIN_Q3.to_numpy()))
            mean_MINQ4.append(np.nanmean(df_params[(df_params.usuario==user)].MIN_Q4.to_numpy()))
            mean_MINQ5.append(np.nanmean(df_params[(df_params.usuario==user)].MIN_Q5.to_numpy()))
            mean_MINQ6.append(np.nanmean(df_params[(df_params.usuario==user)].MIN_Q6.to_numpy()))
            mean_MINQ7.append(np.nanmean(df_params[(df_params.usuario==user)].MIN_Q7.to_numpy()))

            # Mean DTW error
            mean_dtw_error.append(np.nanmean(df_params[(df_params.usuario==user)].DTW_error.to_numpy()))
            mean_dtw_org_error.append(np.nanmean(df_params[(df_params.usuario==user)].DTW_org_error.to_numpy()))

            # Porc. pred
            mean_porc_pred.append(np.nanmean(df_params[(df_params.usuario==user)].Porcentaje_pred.to_numpy()))
            

    data = {
        'user': user_list,
        'mean_MAXQ1': mean_MAXQ1, 'mean_MAXQ2': mean_MAXQ2, 'mean_MAXQ3': mean_MAXQ3, 'mean_MAXQ4': mean_MAXQ4, 'mean_MAXQ5': mean_MAXQ5,
        'mean_MAXQ6': mean_MAXQ6, 'mean_MAXQ7': mean_MAXQ7,

        'mean_MINQ1': mean_MINQ1, 'mean_MINQ2': mean_MINQ2, 'mean_MINQ3': mean_MINQ3, 'mean_MINQ4': mean_MINQ4, 'mean_MINQ5': mean_MINQ5,
        'mean_MINQ6': mean_MINQ6, 'mean_MINQ7': mean_MINQ7,

        'mean_dtw_org_error':mean_dtw_org_error, 'mean_dtw_error': mean_dtw_error, 'mean_porc_pred': mean_porc_pred
    }

    df_mean = pd.DataFrame(data)

    return df_mean

def process_byGroup_ROM(df_main, df_params, group):

    nombres = df_main[df_main.grupo == group].nombre.to_list()

    user_list = []

    rom_q1 = []
    rom_q2 = []
    rom_q3 = []
    rom_q4 = []
    rom_q5 = []
    rom_q6 = []
    rom_q7 = []

    mean_dtw_error = []
    mean_dtw_org_error = []
    mean_porc_pred = []


    for user in nombres:
        if user in df_params['usuario'].values:
            # User
            user_list.append(user)

            #ROM
            rom_q1.append(df_params[(df_params.usuario==user)].MAX_Q1.max() - df_params[(df_params.usuario==user)].MIN_Q1.min())
            rom_q2.append(df_params[(df_params.usuario==user)].MAX_Q2.max() - df_params[(df_params.usuario==user)].MIN_Q2.min())
            rom_q3.append(df_params[(df_params.usuario==user)].MAX_Q3.max() - df_params[(df_params.usuario==user)].MIN_Q3.min())
            rom_q4.append(df_params[(df_params.usuario==user)].MAX_Q4.max() - df_params[(df_params.usuario==user)].MIN_Q4.min())
            rom_q5.append(df_params[(df_params.usuario==user)].MAX_Q5.max() - df_params[(df_params.usuario==user)].MIN_Q5.min())
            rom_q6.append(df_params[(df_params.usuario==user)].MAX_Q6.max() - df_params[(df_params.usuario==user)].MIN_Q6.min())
            rom_q7.append(df_params[(df_params.usuario==user)].MAX_Q7.max() - df_params[(df_params.usuario==user)].MIN_Q7.min())

            # Mean DTW error
            mean_dtw_error.append(np.nanmean(df_params[(df_params.usuario==user)].DTW_error.to_numpy()))
            mean_dtw_org_error.append(np.nanmean(df_params[(df_params.usuario==user)].DTW_org_error.to_numpy()))

            # Porc. pred
            mean_porc_pred.append(np.nanmean(df_params[(df_params.usuario==user)].Porcentaje_pred.to_numpy()))
            

    data = {
        'user': user_list,
        'ROM_Q1': rom_q1, 'ROM_Q2': rom_q2, 'ROM_Q3': rom_q3, 'ROM_Q4': rom_q4, 'ROM_Q5': rom_q5,
        'ROM_Q6': rom_q6, 'ROM_Q7': rom_q7,

        'mean_dtw_org_error':mean_dtw_org_error, 'mean_dtw_error': mean_dtw_error, 'mean_porc_pred': mean_porc_pred
    }

    df_mean = pd.DataFrame(data)

    return df_mean 

def filter_per_Activities(df):

    activity_list = ['MoveGlass', 'Move_Cube', 'PaintForms', 'TouchGame']
    nombres = df.usuario.unique()
    death_note = []
    
    for act in activity_list:
        # Filtrar el DataFrame para la actividad actual
        df_filtrado = df[df['activity'] == act]

        # Contar las ocurrencias de la actividad actual para cada usuario
        conteo_actividad = df_filtrado.groupby('usuario').size().reset_index(name='Conteo')

        for user in nombres:

            esta_presente = (conteo_actividad['usuario'] == user).any()
            #Comprobar si está el usuario
            if not esta_presente:
                # Añadir a la Death Note
                death_note.append(user)
    
    #Eliminar duplicados
    death_note = list(set(death_note))
    for name in death_note:
        # Seleccionar las filas que cumplen con la condición
        filas_a_eliminar = df.loc[df['usuario'] == name]
        # Eliminar las filas seleccionadas
        df = df.drop(filas_a_eliminar.index)
    
    return df



