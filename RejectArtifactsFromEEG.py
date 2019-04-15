import os
import sys
import numpy as np
from scipy.io import loadmat 
from scipy import signal
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import LinearFIR as filtro

class loadFile(object):
    ''' 
    Clase encargada de cargar los datos de la señal almacenados en un archivo 
    con formato txt.
    '''
    def __init__(self):
        '''
        Método inicializador. 
        '''
        self.__path = ''
    
    def setPath(self, path):
        '''
        Asiganamos la ruta del sistema en donde se encuentra el archivo con la 
        señal.
        '''
        self.__path = path
    
    def getPath(self):
        '''
        Método que recupera la ruta del archivo en donde se encuentra la señal.
        '''
        return self.__path
    
    def getFormat(self):
        '''
        Método encargado de obtener el formato del archivo cargado. 
        '''
        (name, extension) = os.path.splitext(self.getPath())
        return extension
    
    def validationfile(self, extension):
        '''
        Método encargado de validar que el archivo que se está cargando es de 
        formato txt 
        '''
        if extension == ".txt":
            try:
                data = open(self.getPath(),'r')
                data.close()
                return True
            except:
                print("The file: %s was not found" %(self.getPath()))
                return False
        else:
            print("The format file (%s) not available." %(extension))

    def openFileTxt(self, delimiter, skiprows, usecols):
        '''
        Método encargado de abrir archivos txt con numpy. 
        '''
        data = np.loadtxt(self.getPath(), delimiter = delimiter, skiprows = skiprows, usecols = usecols)
        return data
    
class PreprocessingSignal(object):
    '''
    Clase encargada de preprocesar la señal con los datos.
    Su funcinamiento se basa en filtrar la señal y visualizar tanto la señal
    original como la filtrada. 
    '''
    def __init__(self):
        '''
        Método inicializado. 
        En ella creamos las variables globales de nustra clase.
        '''
        self.__signal = 0
        self.__fs = 0    
        self.__channels = 0
    
    def setSignal(self, signal):
        '''
        Asignamos la señal cargada por el usuario.
        '''
        self.__signal = signal
    
    def getSignal(self):
        '''
        Recuperamos la señal.
        '''
        return self.__signal
    
    def setSampleFrecuency(self, fs):
        '''
        Asignamos la frecuencia de muestreo de la señal.
        '''
        self.__fs = fs
    
    def getsampleFrecuency(self):
        '''
        Método que recupera la frecuencia de muestreo de la señal.
        '''
        return self.__fs
        
    def getIntervalTime(self):
        '''
        Método encarado de obtener el vector de tiempo. Este vector es el usado
        para graficar los datos. 
        '''
        x = self.__signal.shape
        time = np.arange(0, x[0]/self.__fs, 1/self.__fs)
        return time
    
    def filter(self):
        '''
        Método encargado de filtrar los datos de la señal cargada por el usuario.
        Hace uso del archivo LinearFIR.py.
        '''
        self.signalFiltered = filtro.eegfiltnew(self.__signal, self.__fs, 1, 50, 0, 0)
        return self.signalFiltered
    
    def plotingAll(self, width = 5, height = 4):
        '''Método encargado de graficar los datos de la señal cargada y la 
        señal filtrada. Debido a que al filtrar la señal el offset es eliminado
        por completo, se le agrega un nivel DC con el objetivo de evitar la 
        superposición de los canales de la señal.
        '''
        size = self.__signal.shape
        DCLevel = np.zeros((1, size[1]), dtype = np.int)
        inicio = 0
        for i in range(0, size[1]):
            DCLevel[0,i]= inicio
            inicio+=100
        self.__time = self.getIntervalTime()
        plt.figure(figsize = (width, height))
        plt.plot(self.__time, self.signalFiltered+DCLevel)
        plt.title("EEG CHANNELS [Filtered Signal 50 Hz]")
        plt.xlabel("Time [s]")
        plt.grid()
        
        plt.figure(figsize = (width, height))
        plt.plot(self.__time, self.__signal)
        plt.title("EEG CHANNELS [Original Signal]")
        plt.xlabel("Time [s]")
        plt.grid()
        
        plt.show()
    
class ProcessingSignal(object):
    '''
    Clase encargada de encontrar las épocas atípicas presentes en la señal EEG. 
    Una vez encontradas es capaz de eliminarlas y retornar la señal sin estos 
    valores atípicos. 
    '''
    def segmentationEpochs(self, data, time, fs, epocs):
        '''
        Método que segmenta la señal en las épocas que el usuario desee.
        Devuelve un arrego de ceros y unos el cual es un vector de rechazo. Los
        valores en cero corresponden a los segmentos que deben ser rechazados.
        '''
        #Se obtiene la dimension de los datos de la señal
        x = data.shape
        #Se obtiene el modulo de la señal divido por el numero de muestras en que
        #el usuario quiere dividir las epocas.
        residue = x[0]%(fs*epocs) 
        #Se obtiene el tiempo de adquision de la señal dividiendo la longitud de la señal
        #por la frecuencia de muestreo.
        totalTime = x[0]/fs
        #Se segmenta la señal en el numero de epocas que el usuario desee
        #primero se obtiene la lista de la señal desde 0 hasta su longitud menos el residuo
        #en todos los canales lo que da resultado un arreglo de 3 dimensiones y 
        #se utiliza como secciones el tiempo divido en el numero de epocas (division entera) 
        segmentedData = np.split(data[0:x[0]-residue, :], int(totalTime//epocs))
        #Se segmenta el tiempo de la misma manera que se segmentaron los datos
        t = np.split(time[0:x[0]-residue], int(totalTime//epocs))
        #La funcion retorna dos arreglos, con la segmentacion y el tiempo
        return np.array(segmentedData), np.array(t)

    def extremeValues(self, data, umbral):
        '''
        Método que compara cada uno de los valores máximos y mínimos presentes 
        en cada uno de los segmentos de la señales con los valores ingresados en 
        el umbral. Si el valor máximo o mínimo excede los umbrales, este 
        segmento es rechazado. 
        Devuelve un arrego de ceros y unos el cual es un vector de rechazo. Los
        valores en cero corresponden a los segmentos que deben ser rechazados.
        '''
        #Se obtiene la longitud de los datos
        size = data.shape
        #Se obtienen los valores maximos y minimos del umbral
        maximun = data.max(axis=1)
        minimun = data.min(axis=1)
        #Se obtiene un arreglo de 1 con el tamaño de la señal que se utilizara como
        #vector de rechazo
        toRefuse = np.ones((size[0], 1), dtype = np.int)
        #Se crea un for que recorre desde cero hasta 
        for i in range(0,size[2]):
            index = np.where(maximun[:,i]>umbral[0])
            toRefuse[index]=0
            index = np.where(minimun[:,i]<umbral[1])
            toRefuse[index]=0
        return toRefuse.transpose()[0]
        
    def linearTrends(self, data, time, umbral):
        '''
        Método que calcula la recta que mejor se ajusta a cada uno de los 
        segmentos de la señal. Una vez calculados compara la pendiente de cada
        uno de los segmentos con las pendientes umbrales ingresados por el 
        usuario, si estos están por fuera de rango, estos segmentos son 
        rechazados.
        Devuelve un arrego de ceros y unos el cual es un vector de rechazo. Los
        valores en cero corresponden a los segmentos que deben ser rechazados.
        '''
        ajuste =np.array([np.polynomial.polynomial.polyfit(time[i,:], data[i,:,:], 1) for i in range(0,time.shape[0])])
        size = data.shape
        toRefuse = np.ones((size[0], 1), dtype = np.int)
        for i in range(0, size[2]):
            index = np.where(ajuste[:,1,i]>umbral[0])
            toRefuse[index]=0
            index = np.where(ajuste[:,1,i]<umbral[1])
            toRefuse[index]=0
        return toRefuse.transpose()[0]
    
    def improbability(self, data, umbral):
        '''
        Método que calcula la curtosis de cada segmento de la señal y los 
        compara con los valores umbrales ingresados. Si estos valores están por 
        fuera del rango ingreado, estos valores son rechazados.
        '''
        distribution = kurtosis(data, axis=1)
        size = data.shape
        toRefuse = np.ones((size[0], 1), dtype = np.int)
        for i in range(0, size[2]):
            index = np.where(distribution[:,i]>umbral[0])
            toRefuse[index]=0
            index = np.where(distribution[:,i]<umbral[1])
            toRefuse[index]=0
        return toRefuse.transpose()[0]
    
    def spectralPattern(self, data, fs, umbral): 
        '''
        Método encargado de echazar los valores haciendo uso de las potencias 
        arrojadas por el periodograma de Welch calculado ca cada uno de los
        segmentos de la señal. Una vez calculado el periodograma, se saca el 
        valor medio de esta y se le resta al mismo periodograma. El resultado se
        compara con los umbrales ingresadosy son rechazados si están por fuera 
        del rango.
        Devuelve un arrego de ceros y unos el cual es un vector de rechazo. Los
        valores en cero corresponden a los segmentos que deben ser rechazados.
        '''                  
        size = data.shape
        potenciasMax = np.zeros((size[0], size[2]), dtype=np.float32)
        potenciasMin = np.zeros((size[0], size[2]), dtype=np.float32)
        toRefuse = np.ones((size[0],1), dtype=np.int)
        for j in range(0,size[2]):
            for i in range(0,size[0]):
                frec, Pxx = signal.welch(data[i,:,j], fs, "hanning", size[1])
                mean = np.mean(Pxx)
                x = Pxx-mean
                potenciasMax[i,j]=x.max()
                potenciasMin[i,j]=x.min()
        for i in range(0,size[2]):
            index = np.where(potenciasMax[:,i]>umbral[0])
            toRefuse[index]=0
            index = np.where(potenciasMin[:,i]<umbral[1])
            toRefuse[index]=0
        return toRefuse.transpose()[0]

    def reject(self, data, time, toRefuse, plot = True, title = "Title"):
        '''
        Método encargado de eliminar de la señal los segmentos rechazados por 
        los métodos anteriores. A partir de el vector toRefuse, se evalua la 
        señal, indices con valores 1 permanecen en la señal mientras que los 
        indices con valores en 0 son rechazados.
        '''
        size = data.shape
        DCLevel = np.zeros((1, size[2]), dtype = np.int)
        inicio = 0
        for i in range(0, size[2]):
            DCLevel[0,i]= inicio
            inicio+=100
        extreme = data[toRefuse==1, :, :]
        size = extreme.shape
        signal = np.zeros((size[0]*size[1], size[2]), dtype = np.int)
        for i in range(0, size[2]):
            signal[:,i]=extreme[:,:,i].ravel()
        if plot == True:
            plt.plot(time[0:signal.shape[0]],signal+DCLevel)
            plt.title(title)
            plt.xlabel('Time[s]')
            plt.grid()
            plt.show()
        return signal, time[0:signal.shape[0]], DCLevel
                
class PlotView(object):
    '''
    Clase que permite graficar y moverse através de de las épocas de la señal.
    Esto permite una mejor visualización y análisis de los datos.
    '''
    def __init__(self, data, fs, time, epoch, title):
        '''
        Método inicializador de la clase.
        En el se inicializan las variables globales de la clase.
        '''
        self.__data = data
        self.__fs = fs
        self.__time = time
        self.__epoch = epoch
        self.__title = title
    
    def getIntervalTime(self, signal, time, fs, epoch):
        '''
        Método encargado de obtener el vector de tiempo.
        '''
        x = signal.shape
        sobrante = x[0]%(fs*epoch)
        data = signal[0:x[0]-sobrante, :]
        time = time[0:x[0]-sobrante]
        return data, time
    
    def multiView(self):
        '''
        Método que obtinen las variables iniciales e inicializa la gráfica.
        '''
        self.remove_keymap_conflicts({"left", "right"})
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(13,6))
        ax1.set_title(self.__title)
        ax1.set_xlabel("Time [s]")
        ax2.set_title(self.__title+" for periods of %i seconds"%self.__epoch)
        ax2.set_xlabel("Time [s]")
        ax1.data, ax1.time = self.getIntervalTime(self.__data,self.__time, self.__fs, self.__epoch)
        ax1.fs = self.__fs
        ax1.inicio = 0
        ax1.final = self.__fs*self.__epoch
        ax1.epoch = self.__epoch
        ax1.plot(ax1.time, ax1.data)
        ax1.axvspan(ax1.time[ax1.inicio],ax1.time[ax1.final], alpha = 0.5) 
        ax2.plot(ax1.time[ax1.inicio:ax1.final], ax1.data[ax1.inicio:ax1.final])
        fig.canvas.mpl_connect('key_press_event', self.press_key)
        plt.show()
    
    def remove_keymap_conflicts(self, new_keys_set):
        '''
        Método encargado de remover los botones que se usaran en el slicer, 
        esto con el objetivo de que no se presenten conflictos con otro tipo
        de funciones.
        '''
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)
    
    def press_key(self, event):
        '''
        Método que permive ejecuar determinada función dependiendo de qué tecla
        es precionada. 
        '''
        fig = event.canvas.figure
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        if event.key == 'left':
            self.Back(ax1, ax2)
            
        elif event.key == 'right':
            self.Next(ax1, ax2)
        ax1.set_title(self.__title)
        ax1.set_xlabel("Time [s]")
        ax2.set_title(self.__title+" for periods of %i seconds"%self.__epoch)
        ax2.set_xlabel("Time [s]")   
        fig.canvas.draw()
    
    def Back(self, ax1, ax2):
        '''
        Método encargado de mostrar la imagen anterior
        '''
        data = ax1.data
        time = ax1.time
        ax1.clear()
        ax2.clear()
        if ax1.inicio == 0:
            ax1.final = len(data)-1
            ax1.inicio = ax1.final-(ax1.fs*ax1.epoch)
        elif ax1.inicio == (ax1.fs*ax1.epoch)-1:
            ax1.inicio=0
            ax1.final=(ax1.fs*ax1.epoch)  
        else:
            ax1.inicio = ax1.inicio-(ax1.fs*ax1.epoch)
            ax1.final = ax1.final-(ax1.fs*ax1.epoch)
        ax1.plot(time, data)
        ax1.axvspan(ax1.time[ax1.inicio],ax1.time[ax1.final], alpha = 0.5) 
        ax2.plot(time[ax1.inicio:ax1.final], data[ax1.inicio:ax1.final])
        
    def Next(self, ax1, ax2):
        '''
        Método encargado de mostrar la imagen siguiente.
        '''
        data = ax1.data
        time = ax1.time
        ax1.clear()
        ax2.clear()
        if ax1.final == len(data)-1:
            ax1.final = ax1.fs*ax1.epoch
            ax1.inicio = 0
        elif ax1.final == len(data)-ax1.fs*ax1.epoch:
            ax1.final = len(data)-1
            ax1.inicio = len(data)-ax1.fs*ax1.epoch
        else:
            ax1.inicio = ax1.inicio+(ax1.fs*ax1.epoch)
            ax1.final = ax1.final+(ax1.fs*ax1.epoch)
        ax1.plot(time, data)
        ax1.axvspan(ax1.time[ax1.inicio],ax1.time[ax1.final], alpha = 0.5)  
        ax2.plot(time[ax1.inicio:ax1.final], data[ax1.inicio:ax1.final])

            

def loadSignal(path, fs, delimiter, skiprows, usecols, plot = True):
    '''
    Función encargada de la gestión de carga de la señal. Esta función devuelve
    los datos procedentes de la carga de la señal, la señal filtrada, y el tiempo.
    Permite al usuario graficar o no los datos retornados.
    '''
    file = loadFile()
    fig = PreprocessingSignal()
    file.setPath(path)
    ext = file.getFormat()
    bandera = file.validationfile(ext)
    if bandera == True:
        if ext == ".txt":
            try:
                data = file.openFileTxt(delimiter = delimiter, skiprows = skiprows, usecols = usecols)
            
            except:
                print("Error loading the file. Some of the data may not be of type digit.")
                sys.exit(1)
    else:
        sys.exit(1)           
    fig.setSignal(data)
    fig.setSampleFrecuency(fs)
    filteredData = fig.filter()
    time = fig.getIntervalTime()
    if plot == True:
        fig.plotingAll(8,6)
    return data, filteredData, time

def rejectionArtifacts(data, time, fs, epocs, type, umbral, plot = True, title="Title"):
    '''
    Función encargada de el control del rechazo de los segmentos atípicos dentro
    de la señal.
    data = señal filtrda.
    time = vectro de tiempo.
    fs = frecuencia de muestreo de la señal.
    epocs = cada cuantos segundos quiere dividir la señal.
    type = el tipo de rechazo a aplicar.
        extreme = Rechazo por valores extremos.
        linear = Rechazo por tendencias lineales.
        improbability = Rechazo por datos improbable.
        espectral = rechazo por patrones espectrales.
    umbral = tupla o lista con los umbrales.
        (max, min)
    plot = Valor booleano que permite mostrar las gráficas de los procesos 
        ejecutados.
    title = Título del gráfico.
    '''
    processing = ProcessingSignal()
    segmentedData, times = processing.segmentationEpochs(data, time, fs, epocs)
    if type == 'extreme':
        reject = processing.extremeValues(segmentedData, umbral)
        
    elif type == 'linear':
        reject = processing.linearTrends(segmentedData, times, umbral)
    
    elif type == "improbability":
        reject = processing.improbability(segmentedData, umbral)
        
    elif type == "espectral":
        reject = processing.spectralPattern(segmentedData, fs, umbral)
    else: 
        print("Type of process not supported (%s)"%type)
        reject = ''
        sys.exit(1)
        
    signal, t, DCLevel = processing.reject(segmentedData, time, reject, plot, title)
    
    return signal, t, DCLevel

def EEGSpectral(data, channel, fs, nperseg):
    '''
    Función encargada de mostrar el periodograma de Welch de una señal con 
    múltiples canales.
    Brinda información sobre los rangos de frecuencias entre los que se
    encuentran los picos Delta, Theta, Alpha, Beta y Gamma.
    '''
    Frec, Pxx = signal.welch(data[:, channel], fs, "hanning", nperseg)
    plt.figure(figsize=(10,6))
    plt.semilogy(Frec, Pxx)
    plt.title('Spectrum (EEG)')
    plt.xlabel('Frecuency [Hz]')
    plt.ylabel('Power Density [μV^2/Hz]')
    #Delta peak
    plt.axvspan(0, 4, facecolor='g', alpha=0.5)
    plt.text(1,0.01,'Delta', rotation = 90)
    #Theta peak
    plt.axvspan(4, 8, facecolor='r', alpha=0.5)
    plt.text(5,0.01,'Theta', rotation = 90)
    #Alpha peak
    plt.axvspan(8, 12, facecolor='y', alpha=0.5)
    plt.text(9,0.01,'Alpha', rotation = 90)
    #Beta peak
    plt.axvspan(12, 30, facecolor='b', alpha=0.5)
    plt.text(21,0.01,'Beta', rotation = 90)
    #Gamma peak 
    plt.axvspan(30,np.max(Frec), facecolor='m', alpha=0.5)
    plt.text(45,0.01,'Gamma', rotation = 90)
    plt.grid()
    plt.show()

##
###IMPLEMENTACION
###Cargar la señal y sus canales
##
##
##data, filtered, time = loadSignal("Electrobisturi_3min.txt", 250, ',', 6, (1,2,3,4,5,6,7,8), plot =True)
##
###Realizar cada uno de los metodos de rechazo para cada una de las epocas
##signal1, t1, DCLevel1 = rejectionArtifacts(filtered, time, 250, 2, "extreme", (75, -75),True,'Rejection to Extrem Values')
##signal2, t2, DCLevel2 = rejectionArtifacts(filtered, time, 250, 2, "linear", (5, -5), True,'Rejection to Linear Trends')
##signal3, t3, DCLevel3 = rejectionArtifacts(filtered, time, 250, 2, "improbability", (1, -1), True,'Rejection to Kurtosis Method')
##signal4, t4, DCLevel4 = rejectionArtifacts(filtered, time, 250, 2, "espectral", (300, -3),True,'Rejection to Spectral pattern')
##
###Se realizar la visualizacion de todas las epocas de un mismo canal
##x = PlotView(signal1+DCLevel1, 250, t1, 2, "Rejection to Extrem Values")
### x = PlotView(signal2+DCLevel2, 250, t2, 2, "Rejection to Linear Trends")
### x = PlotView(signal3+DCLevel3, 250, t3, 2, "Rejection to Kurtosis Method")
### x = PlotView(signal4+DCLevel4, 250, t4, 2, "Rejection to Spectral pattern")
##x.multiView()
##
###METODOS DE FILTRADO
###Primer metodo de filtrado 
##signal1, t1, DCLevel1 = rejectionArtifacts(filtered, time, 250, 2, "extreme", (75,-75),False,None)
##signal2, t2, DCLevel2 = rejectionArtifacts(signal1, t1, 250, 2, "linear", (5, -5),False,None)
##signal3, t3, DCLevel3 = rejectionArtifacts(signal2, t2, 250, 2, "improbability", (1, -1),False,None)
##signal4, t4, DCLevel4 = rejectionArtifacts(signal3, t3, 250, 2, "espectral",(200, -2),False,None)
##
###DENSIDAD ESPECTRAL 
###Densidad Espectral Primer metodo
##EEGSpectral(signal4, 0, 250, len(signal4[:,0])/8)
##
###Segundo metodo de filtrado 
### signal1, t1, DCLevel1 = rejectionArtifacts(filtered, time, 250, 2, "extreme", (100,-100),False,None)
### signal4, t4, DCLevel4 = rejectionArtifacts(signal1, t1, 250, 2, "espectral",(200, -2),False,None)
##
###Densidad Espectral Segundo metodo
### EEGSpectral(signal4, 0, 250, len(signal4[:,0])/8)
##
###Tercer metodo de filtrado 
##signal3, t3, DCLevel3 = rejectionArtifacts(filtered, time, 250, 2, "improbability", (1, -1),False,None)
##signal2, t2, DCLevel2 = rejectionArtifacts(signal3, t3, 250, 2, "linear", (4, -4),False,None)
##
###Densidad Espectral Tercer metodo
##EEGSpectral(signal2, 0, 250, len(signal2[:,0])/8)






