import os
from tkinter import *
from tkinter import messagebox

import pickle as cPickle
from random import shuffle
import numpy as np

import multilayer_perceptron as nn

__autor__ = 'hwpoison'
__date__ = '25/07/2022'

"""
Controles:
-Con el click del mouse se selecciona una celda.
-Presionando una vez el click derecho, se activa el modo selección,
por donde pase el puntero se seleccionara una celda o se desactivará si esta
ya estaba previamente activada.
-Presionandolo otra vez este se desactiva.
"""

##MLP  Parameters##
# Input matrix
GRID_WIDTH = 7
GRID_HEIGHT = 8
# Data
CHARACTERS_SET = "_ABCDEFGHIJKLMNÑOPQRSTUVWXYZ0123456789!"
MODEL_FILE_NAME = "brain.saved"
# Layer
HIDDEN_LAYER = 20
# Train settings
LEARN_RATE = 0.001
DEFAULT_ACTIVATION = 'tanh'
TRAIN_EPOCHS = 1200
###################
##GUI Options##
CELL_SIZE = 4
COLOR_ENABLED = '#1EFF1A'
COLOR_DISABLED = '#35A4E7'

TEXT_TITLE = "MLP v2 by " + __autor__
TEXT_TRAIN = "Entrenar"
TEXT_PREDICT = "Predecir"

TEXT_FORGET = "Olvidar"
TEXT_FORGET_DESC = "Reinicia la red"

TEXT_CLEAR = "Limpiar"
TEXT_CLEAR_DESC = "Resetea los valores de la grilla"
TEXT_EXPECTED_OUTPUT = "Por favor, introducir una salida esperada."
TEXT_REFORCE = "Refuerzo"
TEXT_REFORCE_DESC = "Por cada entrenamiento, refuerza el \
aprendizaje repasando nuevamente las muestras previamente enseñadas.\n"


class InputGrid(Frame):

    def __init__(self, width=5, height=4):
        super().__init__()
        self.width = width
        self.height = height
        self.button_grid = {}
        self.selection_mode = False
        self.predicting_control = False
        self.init_grid()
        self.grid(row=1, column=0)

    def create_button(self, text):
        # Genera un boton y asigna evento de estado
        boton = Button(self,
                       width=CELL_SIZE,
                       height=int(CELL_SIZE/2),
                       bg=COLOR_DISABLED,
                       bd=1,
                       # text=str(text)
                       )
        boton.bind('<Enter>', lambda x: self.update_input(boton, mode='enter'))
        boton.bind('<ButtonPress-3>',
                   lambda x: self.update_input(boton, mode='m_selection'))
        boton.bind('<ButtonPress-1>',
                   lambda x: self.update_input(boton, mode='normal'))

        return boton

    def draw(self, inputs):
        # Dibuja patrón proveniente de la red
        for button, input_ in zip(self.button_grid, inputs):
            self.update_input(button, value=1 if input_ >= 0.9 else 0)
        self.predicting_control = True

    def update_input(self, boton, value=None, mode=None):
        # Estado de botón al hacerle click, 0 o 1
        if(mode == 'enter' and self.selection_mode is not True):
            return
        if(mode == 'm_selection'):
            if not self.selection_mode:
                self.selection_mode = True
            else:
                self.selection_mode = False
                return
        if(mode == 'normal'):
            self.selection_mode = False
        if(self.predicting_control is True):
            # resetea la grilla luego de predecir para
            # no tener que andar olvidando cada vez
            self.predicting_control = False
            self.reset()

        if(value is not None):
            boton['bg'] = COLOR_ENABLED if value == 1 else COLOR_DISABLED
            self.button_grid[boton] = value

        elif(self.button_grid[boton] == 1):
            boton['bg'] = COLOR_DISABLED
            self.button_grid[boton] = 0
        else:
            self.button_grid[boton] = 1
            boton['bg'] = COLOR_ENABLED
        boton.update()

    def init_grid(self):
        # Inicializa la grilla
        control = 0
        for x in range(self.width):
            for y in range(self.height):
                button = self.create_button(control)
                button.grid(row=y, column=x)
                self.button_grid[button] = 0
                control += 1

    def reset(self):
        # Resetea todos los botones de la grilla
        for button in self.button_grid:
            self.button_grid[button] = 0
            button['bg'] = COLOR_DISABLED

    def get_inputs(self):
        # Obtiene la matriz de la grilla
        return [w for w in self.button_grid.values()]


class CreateToolTip(object):
    '''
    create a tooltip for a given widget
    '''

    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.close)

    def enter(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(self.tw, text=self.text, justify='left',
                      background='white', relief='solid', borderwidth=1,
                      font=("times", "10", "normal"))
        label.pack(ipadx=1)

    def close(self, event=None):
        if self.tw:
            self.tw.destroy()


class App(object):
    def __init__(self):
        self.window = Tk()
        self.NeuralNetwork = None
        self.reinforcement_data_check = IntVar()
        self.net_prediction_label = StringVar()
        self.batch =  []
        self.characters = CHARACTERS_SET
        self.expected_output = None
        self.init_nnetwork()
        self.init_panel()
        self.init_window()

    def init_window(self):
        screen_width = self.window.winfo_screenwidth()//2
        screen_height = self.window.winfo_screenheight()//2
        x = int((screen_width - self.window.winfo_reqwidth()) / 2)
        y = int((screen_height - self.window.winfo_reqheight()) / 2)
        self.window.withdraw()
        self.window.update_idletasks()
        self.window.geometry(f"+{x}+{y}")
        self.window.title(TEXT_TITLE)
        self.window.resizable(False, False)
        self.window.deiconify()
        self.window.mainloop()

    def predict(self):
        # passing the grid matrix to the network
        result = self.NeuralNetwork.input(self.panel_inputs.get_inputs())
        raw_output = np.around(result, decimals=GRID_WIDTH*GRID_HEIGHT)
        self.panel_inputs.predicting_control = True

        print("Salida actual de la red:")
        print(raw_output)

        # print top 5 characters
        dtype = [("character",object),("activation",float) ]
        probabilities = np.array(list(zip(self.characters, raw_output)), dtype=dtype)
        top_5 = np.sort(probabilities, order='activation')[::-1][:5]
        print("*****Top 5*****")
        for character, activation in top_5:
            print(f"{character} => {activation}")
        print("***************")
        predicted_letter = self.characters[self.characters.index(top_5[0][0])]
        print("Predicted:" , predicted_letter)
        self.net_prediction_label.set(" - ".join([f" {charc}({fire:.2f})" for charc, fire in top_5[:3]]) )
        
    def train(self):
        print("Aprendiendo...", end="")
        panel_inputs = self.panel_inputs.get_inputs()
        net_input = np.array(panel_inputs)
        net_expect = np.array(panel_inputs)
        
        expected_output = self.expected_output.get().upper()
        if not expected_output or not expected_output in self.characters:
            messagebox.showerror("Error",f"{TEXT_EXPECTED_OUTPUT} ({self.characters})")
            return

        index = self.characters.index(expected_output)

        net_expect = np.zeros(len(self.characters))
        net_expect[index] = 1
        
        self.batch.append([net_input, net_expect])
        shuffle(self.batch)

        if not self.reinforcement_data_check.get():
            # online training
            self.NeuralNetwork.train(np.asarray(net_input), np.asarray(net_expect))
        else:
            # batch training
            self.NeuralNetwork.train(
                np.asarray([x[0] for x in self.batch]), 
                np.asarray([x[1] for x in self.batch]), TRAIN_EPOCHS)

        print(f"Minimo local conseguido:", self.NeuralNetwork.results['MSE'][-1])
        print("Listo!")
        
        self.save_profile()
        self.panel_inputs.reset()

    def save_profile(self):
        # save model status with his weights and grid preferences.
        actual_profile = {
            "model":self.NeuralNetwork,
            "gui_grid_width":GRID_WIDTH,
            "gui_grid_height":GRID_HEIGHT,
            "hidden_layer_size":HIDDEN_LAYER,
            "learn_rate":LEARN_RATE,
            "train_epochs":TRAIN_EPOCHS,
            "characters_set": CHARACTERS_SET
        }

        with open(MODEL_FILE_NAME, "wb") as save_profile_file:
            cPickle.dump(actual_profile, save_profile_file)
            print("Cambios guardados..")

    def detect_saved_profile(self):
        if os.path.exists(MODEL_FILE_NAME):
            return True
        return False 

    def load_profile(self):
        global GRID_WIDTH, GRID_HEIGHT, TRAIN_EPOCHS, CHARACTERS_SET
        # load profile from file
        print("Cargando perfil guardado...")
        with open(MODEL_FILE_NAME, "rb") as load_profile:
            loaded_profile = cPickle.load(load_profile)

        # set gui preferences
        GRID_WIDTH = loaded_profile['gui_grid_width']
        GRID_HEIGHT = loaded_profile['gui_grid_height']
        TRAIN_EPOCHS = loaded_profile['train_epochs']
        CHARACTERS_SET = loaded_profile['characters_set']
        # load nn status
        self.NeuralNetwork = loaded_profile['model']

        print("Modelo cargado desde archivo.")

    def init_nnetwork(self):
        # Red neuronal de HIDDEN_LAYER capas
        # 1: capa oculta de HIDDEN_LAYER neuronas,
        #   con (GRID_WEIGHT*GRID_HEIGHT) entradas cada una
        # 2: salida de (GRID_WEIGHT*GRID_HEIGHT) neuronas

        if self.detect_saved_profile():
            self.load_profile()
            return True

        input_len = GRID_HEIGHT*GRID_WIDTH
        output_len = len(self.characters)

        model = [
            nn.NeuralLayer(input_len, HIDDEN_LAYER), # input layer
            nn.NeuralLayer(HIDDEN_LAYER, output_len), # output layer
        ]
        self.NeuralNetwork = nn.NeuralNetwork(model)
        self.NeuralNetwork.learn_rate = LEARN_RATE
        self.NeuralNetwork.default_activation = DEFAULT_ACTIVATION

        print(f"Red Neuronal inicializada.")

        return True 

    def forget(self):
        print("Olvidando...")
        os.remove("saved_model.pickle")
        self.init_nnetwork()

    def clear(self):
        self.panel_inputs.reset()

    def init_panel(self):
        self.panel_inputs = InputGrid(GRID_WIDTH, GRID_HEIGHT)
        control = Frame(self.window)
        Button(control, text=TEXT_TRAIN,
               command=lambda: self.train()).pack(side=LEFT)
        Button(control, text=TEXT_PREDICT,
               command=lambda: self.predict()).pack(side=LEFT)

        clear_btn = Button(control, text=TEXT_CLEAR,
                           command=lambda: self.clear())
        CreateToolTip(clear_btn, TEXT_CLEAR_DESC)
        clear_btn.pack(side=LEFT)

        forget_btn = Button(control, text=TEXT_FORGET,
                            command=lambda: self.forget())
        CreateToolTip(forget_btn, TEXT_FORGET_DESC)
        forget_btn.pack(side=LEFT)

        reinforcement_chk = Checkbutton(
            control, text=TEXT_REFORCE, variable=self.reinforcement_data_check)
        reinforcement_chk.select()
        CreateToolTip(reinforcement_chk, TEXT_REFORCE_DESC)
        reinforcement_chk.pack(side=LEFT)
        Label(control, text="|").pack(side=LEFT)
        Label(control, text="Salida Esperada:").pack(side=LEFT)
        self.expected_output = Entry(control)
        self.expected_output.pack(side=RIGHT)
        control.grid(column=0, row=0, pady=10, padx=10)

        # bottom labels
        Label(self.window, text="Predición:").grid(column=0, row=4)
        Label(self.window, textvar=self.net_prediction_label, font=("Arial", 15)).grid(column=0, row=5)
        self.net_prediction_label.set("...")


if __name__ == '__main__':
    app = App()
