import cv2
import numpy as np
import os

#Constantes opciones del menu principal 
CANTIDAD_OPCIONES_MENU:int = 8
OPCION_MENU_ABM_PEDIDOS:int = 1
OPCION_MENU_RECORRIDO_POR_ZONA:int = 2
OPCION_MENU_PROCESAR_PEDIDOS_TRANSPORTE:int = 3
OPCION_LISTAR_PEDIDOS_PROCESADOS:int = 4
OPCION_VALORIZAR_PEDIDOS_ROSARIO:int = 5
OPCION_ARTICULO_MAS_PEDIDO:int = 6
OPCION_INCIALIZAR_CINTA_TRANSPORTADORA:int = 7



def load_yolo():
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	classes = []
	with open("coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()] 
	
	output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers


def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels


def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs


def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids

def get_color(path):
	img = cv2.imread(path)
	color:str = ""
	b = img[:, :, :1]
	g = img[:, :, 1:2]
	r = img[:, :, 2:]

	# computing the mean
	b_mean = int(np.mean(b))
	g_mean = int(np.mean(g))
	r_mean = int(np.mean(r))
	print(b_mean)
	print(g_mean)
	print(r_mean)

	# displaying the most prominent color
	if (b_mean > g_mean and b_mean > r_mean):
		color = "Blue"
	elif (g_mean > r_mean and g_mean > b_mean):
		color = "Green"
	elif (r_mean > b_mean and r_mean > g_mean):
		color = "Red"
	elif (g_mean == r_mean and (b_mean != r_mean or b_mean != g_mean)):
		color = "Yellow"
	elif (r_mean == b_mean and r_mean == g_mean):
		color = "Black"
	
	return color

def count (label, cup, bottle, colores):
	if label == "cup":
		if len(cup) == 1:
			cup.append(1)
			if colores not in cup[0]:
				cup[0][colores] = 1
		else:
			cup[1] = cup[1] + 1
			if colores not in cup[0]:
				cup[0][colores] = 1
			else: cup[0][colores] += 1
	elif label == "bottle":
		if len(bottle) == 1:
			bottle.append(1)
			if colores not in bottle[0]:
				bottle[0][colores] = 1
		else:
			bottle[1] = bottle[1] + 1
			if colores not in bottle[0]:
				bottle[0][colores] = 1
			else: 
				bottle[0][colores] += 1
	else:
		print("PROCESO DETENIDO, se reanuda en 1 minuto")

def draw_labels(path, boxes, confs, colors, class_ids, classes, img, cup, bottle):
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	colores = get_color(path)
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y + 20), font, 1, color, 1)
			print(label + "  " + colores)
			count(label, cup, bottle, colores)
			

def image_detect(img_path, cup, bottle): 
	model, classes, colors, output_layers = load_yolo()
	image, height, width, channels = load_image(img_path)
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
	draw_labels(img_path, boxes, confs, colors, class_ids, classes, image, cup, bottle)
	cv2.waitKey(0)


def imprimir_opciones_logistik() -> None:
    """
        Imprime las posibles opciones de menu por pantalla para que el 
        usuario sepa que puede hacer con el programa principal
    """
    print("(1) - Alta, Baja o Modificación de Pedidos")
    print("(2) - Determinar recorrido por zona geografica")
    print("(3) - Procesar pedidos optimizando carga de transporte")
    print("(4) - Listar pedidos procesados")
    print("(5) - Valorizar pedidos de la ciudad de 'Rosario'")
    print("(6) - Artículo más pedido")
    print("(7) - Inicializar cinta transportadora")
    print("(8) - SALIR ")
    
def opcion_valida(opcion:str, cantidad_opciones:int) -> bool:
    """ 
    Valida si la opcion ingresada es válida de acuerdo al menu del programa 

    Parametros
    ----------
    opcion: str 
        La opcion ingresada por el usuario a través del teclado 
    cantidad_opciones: int 
        La cantidad de opciones que puede tener el menu del programa

    Retorno
    -------
        bool 
            Retorna un booleano que indica la validación de la opción
            Evaluando que el ingreso no esa una opciń en blanco, vacía o fuera
            del rango establecido por cantidad_opciones    
    """

    return not (opcion.isspace() or len(opcion) == 0) and (opcion.isnumeric() and int(opcion) in range(1, cantidad_opciones + 1))

def menu() -> int:
    """
        Hace la impresion del menu y valida que la opcion ingresada 
        por el usuario corresponda con alguna de las posibles opciones
    """
    print("")
    print("Bienvenido Logistik, por favor escoge una opción: ")
    imprimir_opciones_logistik()
    opcion_user:str = input("")

    while not opcion_valida(opcion_user, CANTIDAD_OPCIONES_MENU):
        print("Por favor ingrese una opción válida: ")
        imprimir_opciones_logistik()
        opcion_user = input("")
    
    return int(opcion_user)

def funcion_opcion_1():
    pass 

def funcion_opcion_2():
    pass 

def funcion_opcion_3():
    pass 

def funcion_opcion_4():
    pass 

def funcion_opcion_5():
    pass 

def funcion_opcion_6():
    pass 

def lector_imagenes():
    bottle = [{}]
    cup = [{}]
    # image_path = os.getcwd() + "/Lote0001/75.jpg"
    # image_detect(image_path, cup, bottle)

    input_imagen_path = os.getcwd() + "/Lote0001"
    nombre_archivos = os.listdir(input_imagen_path)

    for archivo in nombre_archivos:
        imagen_path = input_imagen_path + "/" + archivo
        image_detect(imagen_path, cup, bottle)

    print(cup)
    print(bottle)
    cv2.destroyAllWindows()


def main():
    opcion_menu:int = 0
  
    while (not opcion_menu == CANTIDAD_OPCIONES_MENU):
        opcion_menu = menu()
       
        if (opcion_menu == OPCION_MENU_ABM_PEDIDOS):
            funcion_opcion_1()

        elif (opcion_menu == OPCION_MENU_RECORRIDO_POR_ZONA):
            funcion_opcion_2()

        elif (opcion_menu == OPCION_MENU_PROCESAR_PEDIDOS_TRANSPORTE):
            funcion_opcion_3()
        
        elif (opcion_menu == OPCION_LISTAR_PEDIDOS_PROCESADOS):
            funcion_opcion_4()

        elif (opcion_menu == OPCION_VALORIZAR_PEDIDOS_ROSARIO):
            funcion_opcion_5()

        elif (opcion_menu == OPCION_ARTICULO_MAS_PEDIDO):
            funcion_opcion_6()

        elif (opcion_menu == OPCION_INCIALIZAR_CINTA_TRANSPORTADORA):
            lector_imagenes()

    print("--- ¡ Nos vemos en la próxima LOGISTIK ! ----")

main()