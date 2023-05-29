import struct
import pandas as pd
import math 
import os
import numpy as np
np.seterr(invalid='ignore')
from numpy.linalg import norm

import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt   # plotting

import scipy
import sqlite3

#############################################
#
#   Read Data
# 
#############################################

def loadData(folder_selected):

    timeStamp = []
    repetition = []
    q1 = []
    q2 = []
    q3 = []
    q4 = []
    q5 = []
    q6 = []
    q7 = []

    if os.path.exists(folder_selected + "/Data_Joints.bin") and os.path.getsize(folder_selected + "/Data_Joints.bin") != 0:

        file = open(folder_selected + "/Data_Joints.bin","rb")

        pkg_size = (9) * 8 # doubles 8 bytes

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

    if os.path.exists(folder_selected + "/Data_IA.bin") and os.path.getsize(folder_selected + "/Data_IA.bin") != 0:

        file = open(folder_selected + "/Data_IA.bin", "rb")

        pkg_size = (15) * 8



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

            tarea.append(int(t))

            block = file.read(pkg_size)
    
    data = {'tarea': tarea, 
            'data_org_1': data_org_1, 'data_pred_1': data_pred_1, 'data_org_2': data_org_2, 'data_pred_2': data_pred_2, 'data_org_3': data_org_3, 'data_pred_3': data_pred_3,
            'data_org_4': data_org_4, 'data_pred_4': data_pred_4, 'data_org_5': data_org_5, 'data_pred_5': data_pred_5, 'data_org_6': data_org_6, 'data_pred_6': data_pred_6,
            'data_org_7': data_org_7, 'data_pred_7':data_pred_7
            }
    
    df = pd.DataFrame(data)

    return df

def loadCSVdata():
    df = pd.read_csv('datos_expertos.csv')
    # print(df.info())
    # # Head of de data
    # print("\nHEAD:\n",df.head())
    # # Basic statistics of the data:
    # print("\nDESCRIBE:\n",df.describe())

    # print("\nSAPE:\n",df.shape)
    # ##check for any null/empty values
    # print("\nEMPTY VALUES:\n",df.isnull().any().sum()) 

    # # Plot the time series
    # plt.style.use('fivethirtyeight')
    # df.plot(subplots=True,
    #         layout=(4, 3),
    #         figsize=(22,22),
    #         fontsize=10, 
    #         linewidth=2,
    #         sharex=False,
    #         title='Visualization of the original Time Series')
    # plt.show()

    return df
#############################################
#
#   Trials
# 
#############################################

def find_indices(df):
    return 

def getTrials(df):
# Encontrar los índices donde cambia el valor de actividad
    indices_cambio = df.index[df['repetition'].diff() != 0]

    # Calcular los intervalos de repeticiones
    intervalos = []
    for i in range(1, len(indices_cambio), 2):
        if i < len(indices_cambio) - 1:
            inicio_intervalo = indices_cambio[i]
            fin_intervalo = indices_cambio[i + 1] - 1
            intervalo = (inicio_intervalo, fin_intervalo)
            intervalos.append(intervalo)

    return intervalos

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

def getActivityType2(id):
    if(id == 1):
        actividad = 'BEBER'
    elif(id == 2):
        actividad = 'LLENADO'
    elif(id == 3):
        actividad = 'BIMANUAL'
    elif(id == 4):
        actividad = 'DEPOSITAR'
    elif(id == 5):
        actividad = 'SORTEAR'
    elif(id == 6):
        actividad = 'DESPLAZAR'
    elif(id == 7):
        actividad = 'TRIÁNGULO'
    elif(id == 8):
        actividad = 'CUADRADO'
    elif(id == 9):
        actividad = 'CÍRCULO'
    elif(id == 10):
        actividad = 'HOMBRO'
    elif(id == 11):
        actividad = 'CABEZA'
    elif(id == 12):
        actividad = 'RODILLA'
    elif(id == 13):
        actividad = 'NARIZ'
    
    return actividad
#############################################
#
#   Parameters
# 
#############################################

def maxAngle(df, column):

    # Encontrar los índices donde cambia el valor de actividad
    intervalos = getTrials(df)

    # Obtener los máximos de cada intervalo de repetición
    max_anglesList = []
    for inicio, fin in intervalos:        
        max_anglesList.append(df[column][inicio:fin].max())
    
    #Obetener la media
    valor_maximo = np.mean(max_anglesList)

    return valor_maximo

def minAngle(df, column):

    # Encontrar los índices donde cambia el valor de actividad
    intervalos = getTrials(df)

    #Obtener los mínimos de cada intervalo de repetición
    min_anglesList = []
    for inicio, fin in intervalos:        
        min_anglesList.append(df[column][inicio:fin].min())

    #Obetener la media
    valor_minimo = np.mean(min_anglesList)

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

def computeDTW(df, tarea):

    correctly_clasf = []
    miss_clasf = []
    porcentaje_pred_list = []

    for row in range(df.shape[0]):
        # Calcular las medias de todas las joints
        sum_org = df['data_org_1'][row] + df['data_org_2'][row] + df['data_org_3'][row] + df['data_org_4'][row] + df['data_org_5'][row] + df['data_org_6'][row] + df['data_org_7'][row]
        sum_pred = df['data_pred_1'][row] + df['data_pred_2'][row] + df['data_pred_3'][row] + df['data_pred_4'][row] + df['data_pred_5'][row] + df['data_pred_6'][row] + df['data_pred_7'][row]
        mean_org = sum_org / 7
        mean_pred = sum_pred / 7
        
        # Representar los datos como puntos utilizando Matplotlib      
        if(getActivityType2(df['tarea'][row]) == tarea):
            correctly_clasf.append([mean_org, mean_pred])# good clasificated
        else:
            miss_clasf.append([mean_org, mean_pred])

        porcentaje_pred = len(correctly_clasf) / (len(correctly_clasf) + len(miss_clasf)) * 100
        porcentaje_pred_list.append(porcentaje_pred)

    porcentaje_pred_mean = np.mean(porcentaje_pred_list)

    return (mean_pred, porcentaje_pred_mean)

##functions for extracting sEMG features
def rms(data): ##root mean square
    return  np.sqrt(np.mean(data**2,axis=0))  

def SSI(data): ##Simple Square Integral
    return np.sum(data**2,axis=0)

def abs_diffs_signal(data): ##absolute differential signal
    return np.sum(np.abs(np.diff(data,axis=0)),axis=0)



#############################################
#
#   PLOT Functions
# 
#############################################
def plotDTWparam(df, tarea):

    correctly_clasf = []
    miss_clasf = []

    max_list = []
    
    if len(df) == 0:
        return
    else:
        for row in range(df.shape[0]):
            # Calcular las medias de todas las joints
            sum_org = df['data_org_1'][row] + df['data_org_2'][row] + df['data_org_3'][row] + df['data_org_4'][row] + df['data_org_5'][row] + df['data_org_6'][row] + df['data_org_7'][row]
            sum_pred = df['data_pred_1'][row] + df['data_pred_2'][row] + df['data_pred_3'][row] + df['data_pred_4'][row] + df['data_pred_5'][row] + df['data_pred_6'][row] + df['data_pred_7'][row]
            mean_org = sum_org / 7
            mean_pred = sum_pred / 7
            
            max_list.append(mean_org)
            max_list.append(mean_pred)

            # Representar los datos como puntos utilizando Matplotlib      
            if(getActivityType2(df['tarea'][row]) == tarea):
                correctly_clasf.append([mean_org, mean_pred])# good clasificated
            else:
                miss_clasf.append([mean_org, mean_pred])                # missclasificated

        # Trasponer de lista a array y obtener los valores x e y para la gráfica
        # Plot de los datos
        if len(correctly_clasf) != 0:
            x_correct, y_correct = np.array(correctly_clasf).T
            plt.scatter(x_correct, y_correct, label= 'Gesture Correctly Classified', marker='o')
            
        if len(miss_clasf) != 0:
            x_miss, y_miss = np.array(miss_clasf).T
            plt.scatter(x_miss, y_miss, label='Missclassified Gesture', marker='x')

        # Ajustes para la gráfica
        max_size = np.max(max_list) + 2
        plt.plot([0, max_size], [0, max_size], ls="--", c=".3")
        plt.xlim([0,max_size])
        plt.ylim([0,max_size])
        plt.xlabel('DTW distance to True Gesture')
        plt.ylabel('DTW distance to Predicted Gesture')
        plt.title('Gráfico de DTW para {0}'.format(tarea))
        plt.legend()
        plt.show()


