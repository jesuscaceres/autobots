import ssl
import certifi as certifi
import cv2
import geopy
import numpy as np
import os
import json
import csv
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy import distance

# Constantes opciones del menu principal
CANTIDAD_OPCIONES_MENU: int = 8
OPCION_MENU_ABM_PEDIDOS: int = 1
OPCION_MENU_RECORRIDO_POR_ZONA: int = 2
OPCION_MENU_PROCESAR_PEDIDOS_TRANSPORTE: int = 3
OPCION_LISTAR_PEDIDOS_PROCESADOS: int = 4
OPCION_VALORIZAR_PEDIDOS_ROSARIO: int = 5
OPCION_ARTICULO_MAS_PEDIDO: int = 6
OPCION_INCIALIZAR_CINTA_TRANSPORTADORA: int = 7

# Constantes geolocalizacion
LATITUD_35_GRADOS: float = 35
LATITUD_40_GRADOS: float = 40
PROVINCIA_PUNTO_PARTIDA: str = "Buenos Aires"
CIUDAD_PUNTO_PARTIDA: str = "CABA"
PAIS: str = "Argentina"

# Constantes Zonas Geograficas (Claves)
ZONA_CABA: str = "CABA"
ZONA_CENTRO: str = "CENTRO"
ZONA_SUR: str = "SUR"
ZONA_NORTE: str = "NORTE"

# Precio en dólares
PRECIO_BOTELLA: float = 15
PRECIO_VASO: float = 8
# Peso en kilogramos
PESO_BOTELLA: float = 0.450
PESO_VASO: float = 0.350

# Traductor colores
COLORES: dict = {
    "Black": "Negro",
    "Blue": "Azul",
    "Yellow": "Amarillo",
    "Green": "Verde",
    "Red": "Rojo"
}


def cargar_yolo() -> tuple:
    """
    Carga los archivos de YOLO
    """
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        clases = [line.strip() for line in f.readlines()]

    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colores_aleatorios = np.random.uniform(0, 255, size=(len(clases), 3))
    return net, clases, colores_aleatorios, output_layers


def leer_imagen(dir_imagen: str) -> tuple:
    """
    Parametros
    ----------
    Recibe como parámetro la dirección en donde está ubicada la imagen en nuestra PC

    Retorno
    -------
        float 
            Retorna la imagen, su altura, ancho y canal    
    """
    img = cv2.imread(dir_imagen)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    altura, ancho, canal = img.shape
    return img, altura, ancho, canal


def detectar_objetos(img, net, output_layers: list):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return blob, outputs


def obtener_dimension_box(outputs, altura, ancho) -> tuple:
    """ 
    Parametros
    ----------
    Recibe como parámetro el output del obeto detectado, su altura y ancho

    Retorno
    -------
        Retorna las medidas y la posicion de donde se ubica el cuadro y el texto en la imagen  
    """

    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                centro_x = int(detect[0] * ancho)
                centro_y = int(detect[1] * altura)
                w = int(detect[2] * ancho)
                h = int(detect[3] * altura)
                x = int(centro_x - w / 2)
                y = int(centro_y - h / 2)

                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def obtener_color(path) -> str:
    """ 
    Obtiene el color de la imagen

    Parametros
    ----------
    Recibe como parámetro la dirección en donde está ubicada la imagen

    Retorno
    -------
        str 
            Retorna el color 
    """
    img = cv2.imread(path)
    color: str = ""

    b = img[:, :, :1]
    g = img[:, :, 1:2]
    r = img[:, :, 2:]

    b_mean = int(np.mean(b))
    g_mean = int(np.mean(g))
    r_mean = int(np.mean(r))

    # estableciendo el color dominante

    if b_mean > g_mean and b_mean > r_mean:
        color = "Blue"
    elif g_mean > r_mean and g_mean > b_mean:
        color = "Green"
    elif r_mean > b_mean and r_mean > g_mean:
        color = "Red"
    elif g_mean == r_mean and (b_mean != r_mean or b_mean != g_mean):
        color = "Yellow"
    elif r_mean == b_mean and r_mean == g_mean:
        color = "Black"

    return color


def contador_producto_color(etiqueta_nombre: str, copa: list, botella: list, colores: str) -> None:
    """ 
    Imprime en pantalla si el proceso se detiene por algún producto distinto a los del catálogo.

    Parametros
    ----------
    Recibe como parámetro el nombre del objeto detectado, la lista de la copa y botella en donde se lleva el registro
        de la cantidad de cada uno y el color.
    """
    if etiqueta_nombre == "cup":
        if len(copa) == 1:
            copa.append(1)
            if colores not in copa[0]:
                copa[0][colores] = 1
        else:
            copa[1] = copa[1] + 1
            if colores not in copa[0]:
                copa[0][colores] = 1
            else:
                copa[0][colores] += 1
    elif etiqueta_nombre == "bottle":
        if len(botella) == 1:
            botella.append(1)
            if colores not in botella[0]:
                botella[0][colores] = 1
        else:
            botella[1] = botella[1] + 1
            if colores not in botella[0]:
                botella[0][colores] = 1
            else:
                botella[0][colores] += 1
    else:
        print("\tPROCESO DETENIDO, se reanuda en 1 minuto")


def dibujar_cuadro_nombre(path, boxes, confs, colors, class_ids, classes, img, copa, botella) -> None:
    """ 
    Dibuja en mi imagen el cuadrado con el nombre del objeto que se detecta. Esta función es utilizada si se quiere imprimir 
        la imagen.

    Parametros
    ----------
    Recibe como parámetro toda la información necesario previamente calculada.
 
    """
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colores = obtener_color(path)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 20), font, 1, color, 1)
            contador_producto_color(label, copa, botella, colores)


def detectar_imagen(dir_imagen, copa: list, botella: list) -> None:
    modelo, clases, colores, output_layers = cargar_yolo()
    image, altura, ancho, channels = leer_imagen(dir_imagen)
    blob, outputs = detectar_objetos(image, modelo, output_layers)
    boxes, confs, class_ids = obtener_dimension_box(outputs, altura, ancho)
    dibujar_cuadro_nombre(dir_imagen, boxes, confs, colores, class_ids, clases, image, copa, botella)
    cv2.waitKey(0)


def imprimir_opciones_logistik() -> None:
    """
        Imprime las posibles opciones de menu por pantalla para que el
        usuario sepa que puede hacer con el programa principal
    """
    print('')
    print("\t(1) - Alta, Baja o Modificación de Pedidos")
    print("\t(2) - Determinar recorrido por zona geografica")
    print("\t(3) - Procesar pedidos optimizando carga de transporte")
    print("\t(4) - Listar pedidos procesados")
    print("\t(5) - Valorizar pedidos de la ciudad de 'Rosario'")
    print("\t(6) - Artículo más pedido")
    print("\t(7) - Inicializar cinta transportadora")
    print("\t(8) - Salir ")


def opcion_valida(opcion: str, cantidad_opciones: int) -> bool:
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

    return not (opcion.isspace() or len(opcion) == 0) and (
            opcion.isnumeric() and int(opcion) in range(1, cantidad_opciones + 1))


def menu() -> int:
    """
        Hace la impresion del menu y valida que la opcion ingresada
        por el usuario corresponda con alguna de las posibles opciones
    """
    print("")
    print("\tBienvenido Logistik, por favor escoge una opción: ")
    imprimir_opciones_logistik()
    opcion_user: str = input('\n\t\tIngrese su opción: ')

    while not opcion_valida(opcion_user, CANTIDAD_OPCIONES_MENU):
        print("\n\tPor favor ingrese una opción válida: ")
        imprimir_opciones_logistik()
        opcion_user = input('\n\t\tIngrese su opción: ')

    return int(opcion_user)


def obtener_zonas_geograficas(_pedidos: dict) -> dict:
    """ 
    Obtiene el listado de zonas geográficas de Argentina 
    agrupadas con las ciudades que están actualmente en pedidos 
    clasificadas segun latitud, de acuerdo a lo siguiente: 
        
        Zona Norte: Todas las ciudades cuya latitud sea menor a 35°
        Zona centro: Todas las ciudades entre la latitud 35° y 40°
        Zona Sur: Todas las ciudades cuya latitud sea mayor a 40°.
        CABA: Todos los pedidos que sean de CABA.

    Parametros
    ----------
    _pedidos: dict 
        La estructura que almacena todos los pedidos y la información 
        de cada uno 

    Retorno
    -------
    zonas_geograficas: dict 
        El diccionario que contiene como clave:zona y valor otro diccionario 
        que contiene como clave: Ciudad y Valor: Coordenadas (el punto geografico)
        de la ciudad, obtenido haciendo uso de la librería Geopy. 

    """
    geolocalizador = Nominatim(user_agent="autobots")
    ciudades: dict = {}
    zonas_geograficas: dict = {
        ZONA_CABA: {},
        ZONA_NORTE: {},
        ZONA_SUR: {},
        ZONA_CENTRO: {}
    }

    for nro_pedido, datos_pedido in _pedidos.items():
        pedido_ciudad: str = datos_pedido.get("ciudad", "CABA")
        zona: str = ''
        ubicacion_ciudad: tuple = ()

        if pedido_ciudad not in ciudades.keys():
            # Solo proceso ciudades que no he procesado
            pedido_provincia: str = datos_pedido.get("provincia", "Buenos Aires")
            geo_ubicacion = geolocalizador.geocode(f"{pedido_ciudad}, {pedido_provincia}, {PAIS}")
            ubicacion_ciudad: tuple = (float(geo_ubicacion.latitude), float(geo_ubicacion.longitude))
            ciudades[pedido_ciudad] = ubicacion_ciudad
            zona = ZONA_CABA

        if pedido_ciudad == ZONA_CABA:
            zona = zona
        elif abs(ubicacion_ciudad[0]) < LATITUD_35_GRADOS:
            zona = ZONA_NORTE
        elif abs(ubicacion_ciudad[0]) < LATITUD_40_GRADOS:
            zona = ZONA_CENTRO
        else:
            zona = ZONA_SUR

        zonas_geograficas[zona][pedido_ciudad] = ubicacion_ciudad

    return zonas_geograficas


def obtener_punto_partida() -> tuple:
    """ 
    Obtiene las coordenadas para el punto de partida.
    En este caso, el punto de partida desde donde se van 
    a enviar los pedidos, siempre se va a considerar como CABA, Buenos Aires. 

    Parametros
    ----------
    None 

    Retorno
    -------
    punto_partida: tuple 
        Las coordenadas que definen al punto de partida
        (Latitud, Longitud) obtenidas con la librería geopy

    """
    geolocalizador = Nominatim(user_agent="autobots")
    geo_ubicacion_punto_partida = geolocalizador.geocode(
        f"{CIUDAD_PUNTO_PARTIDA}, {PROVINCIA_PUNTO_PARTIDA}, {PAIS}")
    punto_partida: tuple = (float(geo_ubicacion_punto_partida.latitude), float(geo_ubicacion_punto_partida.longitude))

    return punto_partida


def calcular_recorrido_por_zona(zonas_geograficas: dict, zona: str, punto_partida: tuple) -> list:
    """ 
    Calcula el recorrido óptimo de ciudades a 
    recorrer por zona geográfica considerando al punto de partida 

    Parametros
    ----------
    zonas_geograficas: dict 
        La estructura que contiene cada zona geográfica con la información
        de sus ciudaes (y las coordenadas de la misma) de acuerdo a las ciudades
        que hayan entre los pedidos actuales. 
    zona: str 
        Es la zona a la cuál queremos determinar el recorrido óptimo. 
        La zona representa una clave dentro del diccionario de zonas_geograficas
    punto_partida: tuple 
        Representa la tupla con las coordenadas del punto de partida 

    Retorno
    -------
        recorrido: list
        La lista de ciudades ordenadas de acuerdo al recorrido óptimo de la zona
        indicada. 
    """

    ciudades: dict = zonas_geograficas.get(zona, {})
    tamanio_recorrido: int = len(ciudades)
    punto_comparacion: tuple = punto_partida
    recorrido: list = []

    for i in range(tamanio_recorrido):
        ciudad_mas_cerca: str = \
            sorted(ciudades.items(), key=lambda x: distance.distance(x[1], punto_comparacion).km)[0][0]
        punto_comparacion = ciudades.get(ciudad_mas_cerca)
        recorrido.append(ciudad_mas_cerca)
        del ciudades[ciudad_mas_cerca]

    return recorrido


def imprimir_opciones_zonas_geograficas() -> None:
    """
        Imprime las opciones para las zonas geograficas
    """
    print("")
    print("\tEscoge una zona geográfica para determinar el recorrido óptimo a realizar: ")
    print("\t(1) - CABA")
    print("\t(2) - ZONA NORTE ")
    print("\t(3) - ZONA CENTRO ")
    print("\t(4) - ZONA SUR ")


def recorrido_por_zona(_pedidos: dict) -> None:
    """ 
    Imprime por consola el recorrido óptimo de ciudades según 
    una zona geografica (solicitada previamente al usuario). 

    Parametros
    ----------
    _pedidos: dict
        La estructura que almacena todos los pedidos y la información 
        de cada uno
    
    Retorno
    -------
        None: Imprime directamente el recorrido por pantalla
    """

    # Solo imprime por pantalla el recorrido optimo para la zona ingresada por el usuario
    listado_zonas: list[str] = [ZONA_CABA, ZONA_NORTE, ZONA_CENTRO, ZONA_SUR]
    imprimir_opciones_zonas_geograficas()
    opcion_user: str = input('\n\t\tIngrese su opción: ')

    while not opcion_valida(opcion_user, len(listado_zonas)):
        print("\n\tPor favor ingrese una opción válida: ")
        imprimir_opciones_zonas_geograficas()
        opcion_user = input('\n\t\tIngrese su opción: ')

    print("\n\tCalculando recorrido")
    zonas_geograficas: dict = obtener_zonas_geograficas(_pedidos)
    print("\n\tEl recorrido más óptimo para la zona ingresada es: ", end='')
    recorrido: list = calcular_recorrido_por_zona(zonas_geograficas, listado_zonas[int(opcion_user) - 1],
                                                  obtener_punto_partida())
    print(", ".join(ciudad for ciudad in recorrido))


def armar_archivo(lista_de_datos:list)->None:
    #Arma el archivo salida.txt con los datos pertinentes.
    contador:int = 0
    with open("salida.txt","a+") as salida:
        for elemento in (lista_de_datos[0:3]):
            salida.write(elemento)
            salida.write("\n")
        lista = ", ".join(lista_de_datos[3])
        salida.write(lista)
        salida.write("\n")


def sumar_peso(productos:dict,peso:float)->float:
    #Pre: ingresa los productos y el peso del pedido hasta el momento.
    #Devuelve la suma del peso del pedido agregado al anterior.

    cantidad_botellas: int = 0
    cantidad_vasos: int = 0
    peso_b: float = 0
    peso_v: float = 0
    for codigo,colores in productos.items():
        if (codigo == "1334"):
            cantidad = colores
            for color,cantidad in cantidad.items():
                cantidad_botellas +=cantidad["cantidad"]
    for codigo,colores in productos.items():
        if (codigo == "568"):
            cantidad = colores
            for color,cantidad in cantidad.items():
                cantidad_vasos +=cantidad["cantidad"]

    peso = cantidad_botellas * PESO_BOTELLA + cantidad_vasos * PESO_VASO

    return peso


def chequeo_de_peso(peso:float,peso_anterior:float,dict_utilitarios_peso:dict):
    #Verifica que el peso se mantenga dentro del rango dado por los utilitarios restantes.

    if ("Utilitario 003" in dict_utilitarios_peso and peso <= 500):
        peso = peso
    elif("Utilitario 001" in dict_utilitarios_peso and peso <= 600):
        peso = peso
    elif("Utilitario 002" in dict_utilitarios_peso and peso <= 1000):
        peso = peso
    elif("Utilitario 004" in dict_utilitarios_peso and peso <= 2000):
        peso = peso
    else:
        peso = peso_anterior
    
    return peso


def seleccionar_utilitario(dict_utilitarios_peso:dict,peso:float)->str:
    #Selecciona un utilitario del diccionario de acuerdo al peso.

    utilitario:str = ""

    for utilitarios,carga in dict_utilitarios_peso.items():
        if (peso <= carga):
            utilitario = utilitarios

            return utilitario 


def modifica_diccionario_utilitarios(dict_utilitarios_peso:dict,utilitario:str)->dict:
    #Quita el utilitario seleccionado para el pedido.
    del dict_utilitarios_peso[utilitario]

    
    return dict_utilitarios_peso


def cambiar_a_enviado_en_pedidos(lista_de_envios_a_modificar:list,pedidos:dict)->dict:
    #Cambia a enviados los pedidos que entraron en los utilitarios.
    pedidos_enviados:dict = dict()
    for ciudad in lista_de_envios_a_modificar:
        for numero,datos in pedidos.items():
            if (datos["ciudad"] == ciudad):
                envio = datos
                datos["enviado"] = True

    return pedidos


def armar_salida_texto(recorrido:list,zone:str,dict_utilitarios_peso:dict,pedidos:dict)->dict:

    #Crea una lista de datos para agregar en salida.txt
    #Pre: recibe el recorrido y la zona de dicho recorrido
    #Pos: envia el listado de datos para crear ó agregar en salida.txt

    peso_botella: float = 0
    peso_vaso: float = 0
    peso: float = 0
    peso_anterior: float = 0
    lista_de_datos:list = list()
    lista_ciudades_enviados:list = list()
    lista_de_envios_a_modificar: list = list()
    lista_de_datos.append(zone)

    for ciudad in recorrido:
        for numero,datos in pedidos.items():
            if (datos["ciudad"] == ciudad):
                productos = datos["productos"]
                peso_anterior += peso
                peso += sumar_peso(productos,peso)
                control = chequeo_de_peso(peso,peso_anterior,dict_utilitarios_peso)
                if (control == peso):
                    lista_ciudades_enviados.append(ciudad)
                    lista_de_envios_a_modificar.append(ciudad)
                else:
                    if(len(lista_ciudades_enviados)==1):
                        peso = chequeo_de_peso(peso,peso_anterior,dict_utilitarios_peso)
                    else:
                        peso = chequeo_de_peso(peso,peso_anterior,dict_utilitarios_peso)
    
    utilitario: str = seleccionar_utilitario(dict_utilitarios_peso,peso)
    dict_utilitarios_peso = modifica_diccionario_utilitarios(dict_utilitarios_peso,utilitario)

    if (peso == 0 or peso > 2000):
        peso = "No enviado."
    else:
        peso = int(peso)
        peso = str(peso)+"KG"

    lista_de_datos.append(utilitario)
    lista_de_datos.append(peso)
    lista_de_datos.append(lista_ciudades_enviados)

    armar_archivo(lista_de_datos)

    cambiar_a_enviado_en_pedidos(lista_de_envios_a_modificar,pedidos)
    
    return dict_utilitarios_peso


def procesar_pedido_por_utilitario(recorrido_norte:list,recorrido_centro:list,recorrido_sur:list,recorrido_caba:list,pedidos:dict)->None:
    #Inicializa el armado de pedidos de acuerdo a la zona
    #Pre: recibe todos los recorridos y el pedido.
    #Post: arma zona por zona los pedidos 

    sin_datos:list = list()

    dict_utilitarios_peso:dict = {"Utilitario 003":500,"Utilitario 001":600,"Utilitario 002":1000,"Utilitario 004":2000}

    if (recorrido_norte == []):
        sin_datos = ["Zona Norte:","Sin pedidos","",[]]
        armar_archivo(sin_datos)
    else:
        dict_utilitarios_peso = armar_salida_texto(recorrido_norte,"Zona Norte:",dict_utilitarios_peso,pedidos)
    
    if (recorrido_centro == []):
        sin_datos = ["Zona Centro:","Sin pedidos","",[]]
        armar_archivo(sin_datos)
    else:
        dict_utilitarios_peso = armar_salida_texto(recorrido_centro,"Zona Centro:",dict_utilitarios_peso,pedidos)
    
    if (recorrido_sur == []):
        sin_datos = ["Zona Sur:","Sin pedidos","",[]]
        armar_archivo(sin_datos)
    else:
        dict_utilitarios_peso = armar_salida_texto(recorrido_sur,"Zona Sur:",dict_utilitarios_peso,pedidos)
    
    if (recorrido_caba == []):
        sin_datos = ["Zona CABA:","Sin pedidos","",[]]
        armar_archivo(sin_datos)
    else:
        dict_utilitarios_peso = armar_salida_texto(recorrido_caba,"CABA",dict_utilitarios_peso,pedidos)


def armado_de_salidatxt(pedidos:dict)->None:

    zonas_geograficas = obtener_zonas_geograficas(pedidos)
    punto_partida = obtener_punto_partida()

    recorrido_norte: list = calcular_recorrido_por_zona(zonas_geograficas,"NORTE",punto_partida)
    recorrido_centro: list = calcular_recorrido_por_zona(zonas_geograficas,"CENTRO",punto_partida)
    recorrido_sur: list = calcular_recorrido_por_zona(zonas_geograficas,"SUR",punto_partida)
    recorrido_caba: list = calcular_recorrido_por_zona(zonas_geograficas,"CABA",punto_partida)

    procesar_pedido_por_utilitario(recorrido_norte,recorrido_centro,recorrido_sur,recorrido_caba,pedidos)

    print("")
    print("\t\tLos datos de pedidos que han sido enviados pueden encontrarse en el archivo salida.txt")


def ordenar_fecha(elem):
    return datetime.strptime(elem[2], '%d/%m/%Y')


def mostrar_pedidos_completos(pedidos):
    cantidad_completados: int = 0
    pedido_entregado: list = []
    for key in pedidos:
        nombre: str = pedidos[key]["cliente"]
        fecha: str = pedidos[key]["fecha"]
        for items in pedidos[key]:
            if items == "enviado":
                if pedidos[key][items]:
                    cantidad_completados += 1
                    pedido_entregado.append([nombre, key, fecha])
    pedido_entregado.sort(key=ordenar_fecha)
    return cantidad_completados, pedido_entregado


def imprimir_pedidos_ordenados(pedidos):
    cantidad, pedido_completo = mostrar_pedidos_completos(pedidos)
    print(f"\t\tSe entregaron {cantidad} pedidos:")
    print("")
    for pedido in pedido_completo:
        cliente: str = pedido[0]
        numero: int = pedido[1]
        fecha: str = pedido[2]
        print(f"\t\tEl pedido número {numero} del día {fecha} a nombre de {cliente}.")


def imprimir_total(articulos_enviados: dict, ciudad: str) -> None:
    """Imprime por pantalla el coste total de los artículos enviados a determinada ciudad.
    Args:
        articulos_enviados (dict): Diccionario con los articulos enviados detallando cantidad y costos.
        ciudad (str): Ciudad dónde fueron enviados los artículos.
    """
    if len(articulos_enviados) > 0:
        print(f"\n\t\t\tSe enviaron los siguientes artículos a la ciudad de '{ciudad}':")
        for key in articulos_enviados.keys():
            precio = PRECIO_BOTELLA if key == "1334" else PRECIO_VASO
            if key == "568":
                print(f"\n\t\t({articulos_enviados[key]['cantidad']}) vasos x ${precio} usd c/u")
            else:
                print(f"\n\t\t({articulos_enviados[key]['cantidad']}) botellas x ${precio} usd c/u")
            print(f"\t\tSubtotal --- ${articulos_enviados[key]['bruto']} usd")
            print(f"\t\tDescuento -- {articulos_enviados[key]['descuento']}%")
            print(f"\t\t{'-' * 30}")
            print(f"\t\tTOTAL ------ ${articulos_enviados[key]['neto']} usd")
    else:
        print(f"\n\t\tNo se ha envíado ningún artículo a {ciudad}.")


def obtener_valor_total_por_ciudad(_pedidos: dict) -> None:
    """Permite conocer el valor total de los articulos enviados a determinada ciudad.
    Args:
         _pedidos (dict): Diccionario que contiene la estructura base de los pedidos.
    """
    articulos_enviados: dict = {}
    for nro_pedido in _pedidos.keys():
        if _pedidos[nro_pedido]["enviado"] and (_pedidos[nro_pedido]["ciudad"].upper() == "Rosario".upper()):
            productos: dict = _pedidos[nro_pedido]["productos"]
            descuento: float = _pedidos[nro_pedido]["descuento"]
            for codigo in productos.keys():
                colores: dict = productos[codigo]
                precio: float = PRECIO_BOTELLA if codigo == "1334" else PRECIO_VASO
                for color in colores.keys():
                    cantidad: int = colores[color]["cantidad"]
                    if codigo not in articulos_enviados.keys():
                        articulos_enviados[codigo] = {
                            "cantidad": cantidad,
                            "descuento": descuento,
                            "bruto": float(precio * cantidad),
                            "neto": (precio * cantidad) * (100 - descuento) / 100
                        }
                    else:
                        item = articulos_enviados[codigo]
                        item["cantidad"] += cantidad
                        item["bruto"] += float(precio * cantidad)
                        item["neto"] += (precio * cantidad) * (100 - descuento) / 100

    imprimir_total(articulos_enviados, "Rosario")


def articulo_mas_pedido(pedidos):
    contador_vaso = ["VASO", 0]
    contador_botella = ["BOTELLA", 0]
    for numero in pedidos:
        producto: dict = pedidos[numero]["productos"]
        for codigo in producto:
            for colores in producto[codigo]:
                color = producto[codigo][colores]
                if codigo == "568":
                    contador_vaso[1] += color["cantidad"]
                else:
                    contador_botella[1] += color["cantidad"]
    if contador_vaso[1] > contador_botella[1]:
        return contador_vaso
    else:
        return contador_botella


def articulo_mas_entregado(pedidos):
    vasos_entregados = 0
    botellas_entregadas = 0
    for key in pedidos:
        productos = pedidos[key]["productos"]
        enviado = pedidos[key]["enviado"]
        if enviado:
            for producto in productos:
                color_producto = productos[producto]
                for item in color_producto:
                    cantidad_descuento = color_producto[item]
                    if producto == "568":
                        vasos_entregados += cantidad_descuento["cantidad"]
                    if producto == "1334":
                        botellas_entregadas += cantidad_descuento["cantidad"]
    return vasos_entregados, botellas_entregadas


def imprimir_articulo_mas_vendido(pedidos):
    articulo_vendido = articulo_mas_pedido(pedidos)
    vasos_entregados, botellas_entregadas = articulo_mas_entregado(pedidos)
    if articulo_vendido[0] == "VASO":
        print(f"El artículo más solicitado es el {articulo_vendido[0]} y se entregaron {vasos_entregados} de ellos.")
    else:
        print(f"El artículo más solicitado es la {articulo_vendido[0]} y se entregaron {botellas_entregadas} de ellas.")


def escribir_productos(diccionario: dict, archivo):
    for color, cantidad in diccionario.items():
        archivo.write(f"{COLORES.get(color)} {cantidad} \n")


def escribir_productos_procesados(botellas: list, copas: list) -> None:
    """ 
    Escribe el total de productos procesados en los archivos .txt 
    correspondientes, clasificados por color 

    Parametros
    ----------
    botellas: list[{}]
        Lista con los diccionarios de cada color y el total procesado
        de botellas, que se va a escribir en botellas.txt
    copas: list[{}] 
        Lista con los diccionarios de cada color y el total procesado
        de copas, que se va a escribir en el archivo copas.txt

    Retorno
    -------
        None
            La funcion directamente escribe en los archivos botellas.txt y vasos.txt
            pero no retorna nada. 
    """

    with open("botellas.txt", "w") as archivo_botellas:
        escribir_productos(botellas[0], archivo_botellas)

    with open("vasos.txt", "w") as archivo_copas:
        escribir_productos(copas[0], archivo_copas)


def inicializar_cinta_transportadora() -> None:
    """ 
    Función que ejecuta el resto de funciones para poder determinar el producto y color de la carpeta de lotes.
    """
    print()
    print(f"\tIniciando cinta transportadora..")
    print(f"\tReconociendo productos..")
    botella: list = [{}]
    copa: list = [{}]
    input_imagen_path = os.getcwd() + "/Lote0001"
    nombre_archivos = os.listdir(input_imagen_path)

    for archivo in nombre_archivos:
        imagen_path = input_imagen_path + "/" + archivo
        detectar_imagen(imagen_path, copa, botella)

    cv2.destroyAllWindows()
    # Escribo la totalizacion de productos en los archivos pedidos
    escribir_productos_procesados(botella, copa)

    print("")
    print(f"\tProceso finalizado con éxito")
    print(f"\tLas cantidades procesadas por color se pueden visualizar en los archivos botellas.txt y vasos.txt")


def cargar_pedidos() -> dict:
    """Lee un archivo con extensión .csv para cargar los pedidos en un diccionario en memoria

    Returns:
        dict: Un diccionario representando la estructura de los pedidos
    """
    with open('csv/pedidos.csv', newline='', encoding='utf-8') as archivo_csv:
        lector = csv.reader(archivo_csv, delimiter=',')
        lista_pedidos = list(lector)

    pedidos_archivo: dict = {}
    for indice in range(1, len(lista_pedidos)):
        registro_actual: list = lista_pedidos[indice]
        nro_pedido: str = registro_actual[0]
        if nro_pedido not in pedidos_archivo.keys():
            pedidos_archivo[nro_pedido] = {
                "fecha": registro_actual[1],
                "cliente": registro_actual[2],
                "ciudad": str(registro_actual[3]),
                "provincia": str(registro_actual[4]),
                "productos": {
                    registro_actual[5]: {
                        str(registro_actual[6]).lower(): {
                            "cantidad": int(registro_actual[7])
                        }
                    }
                },
                "descuento": float(registro_actual[8]),
                "enviado": False
            }
        else:
            productos: dict = pedidos_archivo[str(nro_pedido)]["productos"]
            codigo: str = registro_actual[5]
            if codigo in productos.keys():
                items: dict = productos[codigo]
                items[str(registro_actual[6]).lower()] = {
                    "cantidad": int(registro_actual[7])
                }
            else:
                productos[codigo] = {
                    str(registro_actual[6]).lower(): {
                        "cantidad": int(registro_actual[7])
                    }
                }
    return pedidos_archivo


def leer_opcion(opciones: list) -> str:
    """Muestra las posibles opciones y lee la opción ingresada por el usuario.

    Args:
        opciones (list[str]): La lista de opciones posibles.

    Returns:
        str: La opción elegida.
    """
    print('')
    for i, opcion in enumerate(opciones):
        print(f"\t{i + 1}. {opcion}")
    return input('\n\t\tIngrese su opción: ')


def obtener_valor_positivo(campo: str) -> int:
    """Le pide un valor entero positivo al usuario. Si este es incorrecto, pide ingresar nuevamente.

    Args:
         campo (str): La etiqueta del campo a ser solicitado.

    Returns:
        int: El valor ingresado.
    """
    valor: int = 0
    while not valor > 0:
        try:
            valor = int(input(f"\n\t[*] {campo}: "))
            if valor <= 0:
                raise ValueError
        except ValueError:
            print("\n\tValor incorrecto. Debe ingresar un número entero positivo.")
    return valor


def obtener_valor_en_rango(campo: str, inicio: int, fin: int) -> float:
    """Le pide un valor al usuario que esté dentro de determinado rango. Si este es incorrecto, pide ingresar nuevamente.

    Args:
         campo (str): La etiqueta del campo a ser solicitado.
         inicio (int): Valor inicial del rango
         fin (int): Valor tope del rango

    Returns:
        float: El valor ingresado.
    """
    valor: float = -1
    while not (0 <= valor <= 100):
        try:
            valor = float(input(f"\n\t[*] {campo}: "))
            if valor < inicio or valor > fin:
                raise ValueError
        except ValueError:
            print("\n\tValor incorrecto. Debe ingresar un valor entre 0 y 100 (inclusive).")
    return valor


def obtener_opciones_validas(lista: list) -> list:
    """Devuelve la lista de posibles opciones validas

    Args:
        lista (list[str]): Posibles opciones

    Returns:
        list[str]: Lista de posibles opciones. Ej: ["1", "2", "3"]
    """
    return list(map(lambda x: str(x), list(range(1, len(lista) + 1))))


def obtener_color_valido(opcion_articulo: str) -> str:
    """Le pide al usuario un color de acuerdo al tipo de articulo. Si no es válido, pide ingresar nuevamente.

    Args:
        opcion_articulo (str): Tipo de artículo elegido. 1: Botella, 2: Vasos

    Returns:
        str: El color elegido.
    """
    colores_botella: list[str] = ["Verde", "Rojo", "Azul", "Negro", "Amarillo"]
    colores_vaso: list[str] = ["Negro", "Azul"]
    opcion_color: str = ''
    color: str = ''
    if opcion_articulo == "1":
        opciones_validas: list[str] = obtener_opciones_validas(colores_botella)
        while opcion_color not in opciones_validas:
            opcion_color = leer_opcion(colores_botella)
            if opcion_color in opciones_validas:
                color = colores_botella[int(opcion_color) - 1].lower()
            else:
                print("\n\tIngrese una opción válida.")
    else:
        opciones_validas: list[str] = obtener_opciones_validas(colores_vaso)
        while opcion_color not in opciones_validas:
            opcion_color = leer_opcion(colores_vaso)
            if opcion_color in opciones_validas:
                color = colores_vaso[int(opcion_color) - 1].lower()
            else:
                print("\n\tIngrese una opción válida.")
    return color


def obtener_articulo_valido() -> tuple:
    """Le pide al usuario que ingrese el tipo de articulo que desea.
    1. Botella
    2. Vaso

    Returns:
          (str, str): El código del artículo y la opción elegida.
    """
    opcion_articulo: str = ''
    codigo: str = ''
    print("\n\t<<< Lista de artículos >>>")
    while opcion_articulo not in ["1", "2"]:
        opcion_articulo = leer_opcion(["Botella", "Vaso"])
        if opcion_articulo == '1':
            codigo = '1334'
        elif opcion_articulo == '2':
            codigo = '568'
        else:
            print('\n\tIngrese una opción válida.')
    return codigo, opcion_articulo


def obtener_fecha_valida() -> str:
    """Le pide al usuario que ingrese una fecha válida según el formato dd/mm/yyyy.
    Si no es válida, pide ingresar nuevamente la fecha.

    Returns:
        str: La fecha ingresada.
    """
    fecha_valida: bool = False
    fecha: str = ''
    while not fecha_valida:
        fecha = input("\n\t[*] Fecha: ")
        formato: str = "%d/%m/%Y"
        try:
            fecha_valida = bool(datetime.strptime(fecha, formato))
        except ValueError:
            print("\n\t\tIngrese una fecha válida. Debe respetar el formato dd/mm/yyyy")
            fecha_valida = False
    return fecha


def agregar_nuevos_articulos(productos: dict) -> None:
    """Permite agregar contínuamente artículos.

    Args:
          productos (dict): Diccionario que contiene la información de productos: código, color y cantidad.
    """
    opcion: str = ''
    while opcion != 'N':
        codigo, opcion_articulo = obtener_articulo_valido()
        color = obtener_color_valido(opcion_articulo)
        cantidad = obtener_valor_positivo("Cantidad")

        if codigo not in productos.keys():
            productos[codigo] = {
                color: {
                    "cantidad": cantidad
                }
            }
        else:
            items: dict = productos[codigo]
            if color not in items.keys():
                items[color] = {
                    "cantidad": cantidad
                }
            else:
                print(
                    f"\n\t\tPara el artículo cod-{codigo} ya fue ingresado el color {color}. Si desea, después puede modificar el pedido.")
        opcion = input("\n\tDesea agregar otro artículo? [S/N]  ").upper()


def cargar_productos() -> dict:
    """Permite ingresar continuamente artículos hasta que el usuario lo indique.

    Returns:
        dict: Un diccionario que representa la cantidad de productos ingresados por color.
    """
    productos: dict = {}
    agregar_nuevos_articulos(productos)
    return productos


def crear_pedido(_pedidos: dict) -> None:
    """Permite dar de alta un nuevo pedido.

    Args:
        _pedidos (dict): Diccionario que contiene la estructura base de los pedidos.
    """
    fecha: str = obtener_fecha_valida()
    cliente: str = input("\n\t[*] Cliente: ")
    ciudad: str = input("\n\t[*] Ciudad: ")
    provincia: str = input("\n\t[*] Provincia: ")
    productos: dict = cargar_productos()
    descuento: float = obtener_valor_en_rango("Descuento", 0, 100)

    nro_pedido = len(_pedidos) + 1
    _pedidos[str(nro_pedido)] = {
        "fecha": fecha,
        "cliente": cliente,
        "ciudad": ciudad,
        "provincia": provincia,
        "productos": productos,
        "descuento": descuento,
        "enviado": False
    }
    print("\n\t\tNuevo pedido agregado correctamente.")


def modificar_campo(_dict: dict, key: str, campo: str) -> None:
    """Permite modificar un campo específico de un pedido.

    Args:
        _dict (dict): Diccionario que contiene la estructura base de los pedidos.
        key (str): Clave de la propiedad a modificar.
        campo (str): Propiedad a modificar
    """
    print(f"\n\t\tValor anterior: {_dict[key][campo]}")
    if campo == "fecha":
        nuevo_valor = obtener_fecha_valida()
    elif campo == "cantidad":
        nuevo_valor = obtener_valor_positivo("Cantidad")
    elif campo == "descuento":
        nuevo_valor = obtener_valor_en_rango("Descuento", 0, 100)
    else:
        nuevo_valor = input("\n\t\tNuevo valor: ")
    _dict[key][campo] = nuevo_valor


def eliminar_color(colores: dict, id_articulo: str) -> None:
    """Permite eliminar un color de la lista de articulos cargados.

    Args:
        colores (dict): Diccionario de articulos por colores.
        id_articulo (str): Define de qué articulo se trata, botella o vaso.
    """
    color: str = obtener_color_valido(id_articulo)
    if color in colores.keys():
        del colores[color]
        print(f"\n\t\tSe eliminó el color {color}.")
    else:
        print("\n\t\tNo existe un artículo con ese color.")


def modificar_color(colores: dict, id_articulo: str) -> None:
    """Permite modificar un color de la lista de articulos cargados.

    Args:
        colores (dict): Diccionario de articulos por colores.
        id_articulo (str): Define de qué articulo se trata, botella o vaso.
    """
    color: str = obtener_color_valido(id_articulo)
    if color in colores.keys():
        opcion_campo = ''
        while opcion_campo != '2':
            opcion_campo = leer_opcion(["Cantidad", "Salir"])
            if opcion_campo == '1':
                modificar_campo(colores, color, "cantidad")
            elif opcion_campo == '2':
                continue
            else:
                print("\n\n\t\tIngrese una opción válida.")
    else:
        print("\n\t\tNo existe un artículo con ese color.")


def agregar_color(colores: dict, id_articulo: str) -> None:
    """Permite agregar un nuevo color a la lista de articulos cargados.

    Args:
        colores (dict): Diccionario de articulos por colores.
        id_articulo (str): Define de qué articulo se trata, botella o vaso.
    """
    color: str = obtener_color_valido(id_articulo)
    if color in colores.keys():
        print("\n\t\tYa existe este color.")
    else:
        cantidad: int = obtener_valor_positivo("Cantidad")
        colores[color] = {
            "cantidad": cantidad
        }


def modificar_propiedades_articulos(productos: dict) -> None:
    """Permite modificar las propiedades de un articulo particular: color y cantidad.

    Args:
        productos (dict): Diccionario con la información de los artículos actualmente cargados.
    """
    codigo: str = input("\n\tIngrese el código del producto a modificar: ")
    if codigo in productos.keys():
        id_articulo: str = "1" if codigo == "1334" else "2"
        opcion: str = ''
        colores: dict = productos[codigo]
        while opcion != '4':
            opcion = leer_opcion(["Nuevo color", "Modificar color", "Eliminar color", "Salir"])
            if opcion == '1':
                agregar_color(colores, id_articulo)
            elif opcion == '2':
                modificar_color(colores, id_articulo)
            elif opcion == '3':
                eliminar_color(colores, id_articulo)
            elif opcion == '4':
                continue
            else:
                print("\n\t\tIngrese una opción válida.")
    else:
        print("\n\t\tNo existe un artículo con ese código.")


def modificar_articulos(_pedidos: dict, nro_pedido: str) -> None:
    """Muestra los artículos actuales para determinado pedido con la posibilidad de modificarlos.

    Args:
        _pedidos (dict): Diccionario que contiene la estructura base de los pedidos.
        nro_pedido (str): Identificador del pedido
    """
    accion: str = ''
    while accion != '4':
        accion = leer_opcion(["Agregar artículo", "Modificar artículo", "Eliminar articulo", "Salir"])
        print("\n\t\tArtículos actuales: ")
        productos: dict = _pedidos[nro_pedido]['productos']
        print(json.dumps(productos, indent=4, ensure_ascii=False))
        if accion == '1':
            agregar_nuevos_articulos(productos)
        elif accion == '2':
            modificar_propiedades_articulos(productos)
        elif accion == '3':
            codigo, _ = obtener_articulo_valido()
            if codigo in productos.keys():
                del productos[codigo]
                print(f"\n\t\tSe eliminó el artículo {codigo}.")
            else:
                print("\n\t\tNo existe un artículo con ese código.")


def modificar_pedido(_pedidos: dict) -> None:
    """Permite modificar un pedido existente.

    Args:
        _pedidos (dict): Diccionario que contiene la estructura base de los pedidos.
    """
    if len(_pedidos) > 0:
        print("\n\t\tPedidos actuales:", end=' ')
        lista_nro_pedidos: list[str] = []
        for key in _pedidos.keys():
            if not _pedidos[key]["enviado"]:
                lista_nro_pedidos.append(key)
        print(", ".join(f"[{nro}]" for nro in lista_nro_pedidos))
        nro_pedido = input("\n\tIngrese el número de pedido a modificar: ")
        if nro_pedido in _pedidos.keys():
            opcion_modificar = ''
            while opcion_modificar != '7':
                opcion_modificar = leer_opcion(
                    ["Fecha", "Cliente", "Ciudad", "Provincia", "Productos", "Descuento", "Salir"])
                if opcion_modificar == '1':
                    modificar_campo(_pedidos, nro_pedido, "fecha")
                elif opcion_modificar == '2':
                    modificar_campo(_pedidos, nro_pedido, "cliente")
                elif opcion_modificar == '3':
                    modificar_campo(_pedidos, nro_pedido, "ciudad")
                elif opcion_modificar == '4':
                    modificar_campo(_pedidos, nro_pedido, "provincia")
                elif opcion_modificar == '5':
                    modificar_articulos(_pedidos, nro_pedido)
                elif opcion_modificar == '6':
                    modificar_campo(_pedidos, nro_pedido, "descuento")
        else:
            print("\n\t\tNo existe ningún pedido con ese número.")
    else:
        print("\n\t\tNo hay pedidos cargados actualmente.")


def eliminar_pedido(_pedidos: dict) -> None:
    """Permite eliminar un pedido ingresando el número.

    Args:
        _pedidos (dict): Diccionario que contiene la estructura base de los pedidos.
    """
    print("\n\t\tPedidos actuales:", end=' ')
    lista_nro_pedidos: list[str] = []
    for key in _pedidos.keys():
        if not _pedidos[key]["enviado"]:
            lista_nro_pedidos.append(key)
    print(", ".join(f"[{nro}]" for nro in lista_nro_pedidos))
    nro_pedido = input("\n\t\tIngrese el número de órden a eliminar: ")
    if nro_pedido in _pedidos.keys():
        del _pedidos[nro_pedido]
    else:
        print("\n\tNo existe ningún pedido con ese número.")


def listar_pedidos(_pedidos: dict) -> None:
    """Permite listar los pedidos que se encuentran cargados actualmente.

    Args:
        _pedidos (dict): Diccionario que contiene la estructura base de los pedidos.
    """
    if len(_pedidos) > 0:
        print(json.dumps(_pedidos, indent=4, ensure_ascii=False))
    else:
        print("\n\t\tNo existen pedidos cargados actualmente.")


def pedidos_abm(_pedidos: dict) -> None:
    """Muestra un menú que permite el alta, baja y modificación de pedidos.

    Args:
        _pedidos (dict): Diccionario que contiene la estructura base de los pedidos.
    """
    opcion: str = ''
    while opcion != '5':
        opcion = leer_opcion(["Crear pedido", "Modificar pedido", "Eliminar pedido", "Listar pedidos", "Salir"])
        if opcion == '1':
            crear_pedido(_pedidos)
        elif opcion == '2':
            modificar_pedido(_pedidos)
        elif opcion == '3':
            eliminar_pedido(_pedidos)
        elif opcion == '4':
            listar_pedidos(_pedidos)
        elif opcion == '5':
            print('')
        else:
            print("\n\t\t'Por favor, ingrese una opción válida.")


def inicializar_geolocalizador():
    ctx: ssl.SSLContext = ssl.create_default_context(cafile=certifi.where())
    geopy.geocoders.options.default_ssl_context = ctx
    geopy.geocoders.options.default_user_agent = 'autobots'


def main():
    opcion_menu: int = 0
    pedidos: dict = cargar_pedidos()
    inicializar_geolocalizador()

    while not opcion_menu == CANTIDAD_OPCIONES_MENU:
        opcion_menu = menu()

        if opcion_menu == OPCION_MENU_ABM_PEDIDOS:
            pedidos_abm(pedidos)
        elif opcion_menu == OPCION_MENU_RECORRIDO_POR_ZONA:
            recorrido_por_zona(pedidos)
        elif opcion_menu == OPCION_MENU_PROCESAR_PEDIDOS_TRANSPORTE:
            armado_de_salidatxt(pedidos)
        elif opcion_menu == OPCION_LISTAR_PEDIDOS_PROCESADOS:
            imprimir_pedidos_ordenados(pedidos)
        elif opcion_menu == OPCION_VALORIZAR_PEDIDOS_ROSARIO:
            obtener_valor_total_por_ciudad(pedidos)
        elif opcion_menu == OPCION_ARTICULO_MAS_PEDIDO:
            imprimir_articulo_mas_vendido(pedidos)
        elif opcion_menu == OPCION_INCIALIZAR_CINTA_TRANSPORTADORA:
            inicializar_cinta_transportadora()

    print("\n\t --- ¡ Nos vemos en la próxima LOGISTIK ! ----")
    print("")


main()
