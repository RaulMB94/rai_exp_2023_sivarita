import struct
import pandas as pd
import math 
import os
import numpy as np
np.seterr(invalid='ignore')
from numpy.linalg import norm

import plotly.graph_objects as go
import plotly.express as px

import scipy
import sqlite3

#############################################
#
#   Read Data
# 
#############################################

def loadData(folder_selected):

    file = open(folder_selected + "/Data_Joints.bin","rb")

    pkg_size = (9) * 8 # doubles 8 bytes

    timeStamp = []
    repetition = []
    q1 = []
    q2 = []
    q3 = []
    q4 = []
    q5 = []
    q6 = []
    q7 = []

    block = file.read(pkg_size)

    while len(block) > 0:
        (d1,d2,d3,d4,d5,d6,d7,rep,t) = struct.unpack("ddddddddd", block)

        q1.append(math.degrees(d1))
        q2.append(math.degrees(d2))
        q3.append(math.degrees(d3))
        q4.append(math.degrees(d4))
        q5.append(math.degrees(d5))
        q6.append(math.degrees(d6))
        q7.append(math.degrees(d7))

        timeStamp.append(t)
        repetition.append(rep)


        block = file.read(pkg_size)
    
    data = { 
            'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4, 'q5': q5, 'q6': q6, 'q7': q7, 'repetition': repetition, 'timeStamp': timeStamp
            }
    
    df = pd.DataFrame(data)

    return df

def loadDataActivity(folder):
    file = open(folder + "/Data_Activity.bin","rb")
    file_size = os.path.getsize(folder + "/Data_Activity.bin")

    actividad = []
    modo = []
    frec = []
    k = []
    brazo = []
    up_size = []
    down_size = []
    n_rep = []
    current_rep = []
    time_init = []
    time_end = []

    pkg_size = (8) * 8 # doubles 8 bytes

    block = file.read(pkg_size)
    (d1,d2,d3,d4,d5,d6,d7,d8) = struct.unpack("dddddddd", block)

    actividad.append(d1)
    modo.append(getActivityType(d1, d2))
    frec.append(d3)
    k.append(d4)
    brazo.append(d5)
    up_size.append(d6)
    down_size.append(d7)
    n_rep.append(d8)
    
    if(file_size != 72):
        block = file.read(3*8)
        for i in range(int(d8)):
            (repeticion,t_inicial,t_final) = struct.unpack('ddd',block)
            current_rep.append(repeticion)
            time_init.append(t_inicial)
            time_end.append(t_final)

            if i <= d8:
                block = file.read(3*8)
    else:
        block = file.read(8)
        (repeticion) = struct.unpack('d',block)
        current_rep.append(-1)
        time_init.append(-1)
        time_end.append(-1)

    file.close()

    data = {
        'actividad': actividad, 'modo': modo, 'frecuencia': frec, 'k': k, 'brazo': brazo,
        'upper_size': up_size, 'fore_size': down_size, 'trials': n_rep
    }

    data_trials = {
        'repeticion': current_rep, 'time_init': time_init, 'time_end': time_end
    }

    df_trials = pd.DataFrame(data_trials)
    df = pd.DataFrame(data)
    # nueva_serie = pd.Series(current_rep)
    # print(nueva_serie)
    # df['currentRep'] = nueva_serie

    return (df, df_trials)

def loadDataIA(folder_selected):
    file = open(folder_selected + "/Data_IA.bin", "rb")

    pkg_size = (15) * 8

    data_org_1 = []
    data_pred_1 = []

    data_org_2 = []
    data_pred_2 = []

    data_org_3 = []
    data_pred_3 = []

    data_org_4 = []
    data_pred_4 = []

    data_org_5 = []
    data_pred_5 = []

    data_org_6 = []
    data_pred_6 = []

    data_org_7 = []
    data_pred_7 = []

    tarea = []

    block = file.read(pkg_size)

    while len(block) > 0:
        (do1,dp1,do2,dp2,do3,dp3,do4,dp4,do5,dp5,do6,dp6,do7,dp7,t) = struct.unpack("ddddddddddddddd", block)

        data_org_1.append(do1)
        data_pred_1.append(dp1)

        data_org_2.append(do2)
        data_pred_2.append(dp2)

        data_org_3.append(do3)
        data_pred_3.append(dp3)

        data_org_4.append(do4)
        data_pred_4.append(dp4)

        data_org_5.append(do5)
        data_pred_5.append(dp5)

        data_org_6.append(do6)
        data_pred_6.append(dp6)

        data_org_7.append(do7)
        data_pred_7.append(dp7)

        tarea.append(t)

        block = file.read(pkg_size)
    
    data = {'tarea': tarea, 
            'data_org_1': data_org_1, 'data_pred_1': data_pred_1, 'data_org_2': data_org_2, 'data_pred_2': data_pred_2, 'data_org_3': data_pred_3, 'data_org_4': data_org_4, 
            'data_pred_4': data_pred_4, 'data_org_5': data_org_5, 'data_pred_5':data_pred_5, 'data_org_6': data_org_6, 'data_pred_6': data_pred_6, 'data_org_7': data_org_7,
            'data_pred_7':data_pred_7
            }
    
    df = pd.DataFrame(data)

    return df 
#############################################
#
#   Trials
# 
#############################################

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]

def getTrials(df):


    return 

#############################################
#
#   
# 
#############################################
def getActivityType(id, modo):
    if (id == 0): # MOVEGLASS
        if (modo == 1):
            actividad = 'BEBER'   # Beber
        elif (modo == 2): 
            actividad = 'LLENADO'   # Llenado
        elif (modo == 3):
            actividad = 'BIMANUAL'   # Bimanial
    elif (id == 1): # MOVECUBE
        if (modo == 1):
            actividad = 'DEPOSITAR'   # Depositar
        elif (modo == 2):
            actividad = 'SORTEAR'   # Sortear
        elif (modo == 3):
            actividad = 'DESPLAZAR'   # Desplazamiento
    elif (id == 2): # Paint
        if (modo == 1):
            actividad = 'TRIÁNGULO'   # Triangulo
        elif (modo == 2):
            actividad = 'CUADRADO'   # Cuadrado
        elif (modo == 3):
            actividad = 'CÍRCULO'   # Círculo
    elif (id == 3): # Tocar
        if (modo == 1):
            actividad ='HOMBRO'   # Hombro
        elif (modo == 2):
            actividad = 'CABEZA'   # Cabeza
        elif (modo == 3):
            actividad = 'RODILLA'   # Rodilla
        elif (modo == 4):
            actividad =  'NARIZ'  # Nariz
    return actividad

#############################################
#
#   Parameters
# 
#############################################

def maxAngle(df, column):

    filas_max_angle = df.loc[df['repetition'] == 1]
    valor_maximo = filas_max_angle[column].max()

    return valor_maximo

def minAngle(df, column):

    filas_min_angle = df.loc[df['repetition'] == 1]
    valor_minimo = filas_min_angle[column].min()

    return valor_minimo

def computeVelocity(df):


    # t = np.array(df.timeStamp)
    # x = np.array(df.x)
    # y = np.array(df.y)
    
    # # Compute distance
    # a = np.array([ x[0:-1], y[0:-1] ])
    # b = np.array([ x[1:], y[1:] ])
    # distance = np.sqrt(np.sum(np.power((b - a),2),axis=0))

    # # Compute elapsed time
    # t1 = np.array(t[1:])
    # t2 = np.array(t[0:-1])
    # elapsed_time =  t2 - t1

    # # Compute velocity
    # velocity = np.abs(distance/elapsed_time)
    # velocity = scipy.signal.medfilt(velocity, kernel_size=9)

    # # Compute params to normalize velocity values
    # tmp = []
    # velocity_trials = []
    # velocity_min = 0
    # velocity_max = 0
    # for i_trial in range(df_trials.shape[0]):

    #     i_ini = df_trials.index_start[i_trial]
    #     i_end = df_trials.index_end[i_trial]

    #     vtrial = velocity[i_ini:i_end+1]
        
    #     velocity_trials.append(vtrial)

    #     vtrial = vtrial[~np.isnan(vtrial)]

    #     tmp_list = vtrial.tolist()
    #     tmp_list.append(velocity_min)
    #     tmp_list.append(velocity_max)

    #     velocity_min = np.min(tmp_list)
    #     velocity_max = np.max(tmp_list)

    # velocity_mean_trials = []
    # velocity_max_trials = []
    # velocity_rms_trials = []

    # for i_trial in range(len(velocity_trials)):
    #     velocity_norm = velocity_trials[i_trial]/velocity_max
        
    #     velocity_mean_trials.append(np.mean(velocity_norm))
    #     velocity_max_trials.append(np.max(velocity_norm))
    #     velocity_rms_trials.append(np.around(np.sqrt(np.square(velocity_norm).mean()),3))

    # return velocity_mean_trials, velocity_rms_trials, velocity_max_trials
    return 




#############################################
#
#   PLOT Functions
# 
#############################################


