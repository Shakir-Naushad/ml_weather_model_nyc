import tflite_runtime.interpreter as tflite
from coral.enviro.board import EnviroBoard
from luma.core.render import canvas
import numpy as np
from time import sleep

enviro = EnviroBoard()

ALPHA_COEFF = 17.625
BETA_COEFF = 243.04

temp_data = []
humid_data = []
press_data = []
dew_data = []

labels = ['Normal', 'Rain', 'Thunder', 'Fog', 'Snow']

model_path = 'model_v2.tflite'

def tempInF(temp):
    return (temp * (9/5)) + 32

def presInHG(press):
    return np.divide(press, 3.386)

#We can calculate the Dew Point from the temperture and humditiy

def alpha_func(temp,humid):
    value = np.log(humid/100) + ALPHA_COEFF*temp/(BETA_COEFF+temp)
    return value

def dew_calc(temp, humid):
    alpha = alpha_func(temp,humid)
    num = BETA_COEFF * alpha
    denom = ALPHA_COEFF - alpha
    dew = num / denom
    return dew

#Initialize the board to read and use the machine model

def init(path):
    interpreter = tflite.Interpreter(path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

#Draws the message to the LED screen on the board

def update_display(display, msg, case):
    with canvas(display) as draw:
        if case == 1:
            draw.multiline_text((0,0), msg, fill='white')
        if case == 2:
            draw.multiline_text((0,-16), msg, fill='white')

def nump(array):
    return np.array(array)

#Save the raw data measurements

def get_data():
    temp = enviro.temperature
    humid = enviro.humidity
    dew = dew_calc(temp,humid)
    pres = enviro.pressure

    temp_data.append(temp)
    humid_data.append(humid)
    dew_data.append(dew)
    press_data.append(pres)

#Converts the raw data into usable data for the model

def inputData():
    npTemp = nump(temp_data)
    npHumid = nump(humid_data)
    npDew = nump(dew_data)
    npPress = nump(press_data)
    npTemp = tempInF(npTemp)
    npPress = presInHG(npPress)
    npDew = tempInF(npDew)
    output = np.array([npTemp.max(), npTemp.mean(), npTemp.min(), npDew.max(), npDew.mean(), npDew.min(), npHumid.max(), npHumid.mean(), npHumid.min(), npPress.max(), npPress.mean(), npPress.min()])
    return output

def translate(array):
    msg = 'Current Weather:\n'
    for i in range(5):
        if array[i] >= 50: #Because the model can predicted multiple events occuring at the same time, I arbitarily chose 50 as the "activation" point
           # msg += f"{array[i]:.0f}% {labels[i]} \t"
           msg += f"{labels[i]} \t"
    return msg


interpreter, inputDet, outputDet = init(model_path)

while(True):
    get_data()
    inputs = inputData()

    input_data = np.array(inputs, dtype = np.float32, ndmin=2)
    interpreter.set_tensor(inputDet[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(outputDet[0]['index'])

    output_data = output_data[0] * 100

    msg = translate(output_data)

    print(msg)
    print("------")
    update_display(enviro.display,msg,1)
    sleep(1.5)

