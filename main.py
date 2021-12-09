#Constantes opciones del menu principal 
CANTIDAD_OPCIONES_MENU:int = 8
OPCION_MENU_ABM_PEDIDOS:int = 1
OPCION_MENU_RECORRIDO_POR_ZONA:int = 2
OPCION_MENU_PROCESAR_PEDIDOS_TRANSPORTE:int = 3
OPCION_LISTAR_PEDIDOS_PROCESADOS:int = 4
OPCION_VALORIZAR_PEDIDOS_ROSARIO:int = 5
OPCION_ARTICULO_MAS_PEDIDO:int = 6
OPCION_INCIALIZAR_CINTA_TRANSPORTADORA:int = 7

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

def funcion_opcion_7():
    pass 

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
            funcion_opcion_7()

    print("--- ¡ Nos vemos en la próxima LOGISTIK ! ----")

main()