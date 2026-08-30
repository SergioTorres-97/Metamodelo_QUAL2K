from typing import Dict, Any, List


class Q2KConfig:
    """
    Maneja todas las configuraciones del modelo QUAL2K.
    Permite modificar fácilmente parámetros del header, rates, etc.
    """

    def __init__(self, header_dict: Dict[str, Any]):
        """
        Inicializa la configuración con el diccionario de header.

        Args:
            header_dict: Diccionario con configuración del header
        """
        self.header_dict = header_dict
        self._inicializar_configuraciones_default()

    def _inicializar_configuraciones_default(self):
        """Inicializa todas las configuraciones con valores por defecto"""

        # Light data
        self.light_dict = {
            # --- Extincion de luz (Light extinction) ---
            "PAR":   0.47,       #        Fraccion de radiacion solar fotosinticamente disponible (PAR)
            "kep":   0.2,        #        Extincion de luz de fondo (background) (/m)
            "kela":  0.0088,     #        Extincion de luz lineal por clorofila (1/m-(ugA/L))
            "kenla": 0.054,      #        Extincion de luz no lineal por clorofila (1/m-(ugA/L)^2/3)
            "kess":  0.052,      #        Extincion de luz por solidos inorganicos en suspension ISS (1/m-(mgD/L))
            "kepom": 0.174,      #        Extincion de luz por detrito POM (1/m-(mgD/L))
            # --- Modelo de radiacion solar de onda corta (Solar shortwave radiation) ---
            "solarMethod":  "Bras",  #    Modelo de atenuacion atmosferica solar (Bras / Ryan-Stolzenbach)
            "nfacBras":     2,       #    Coef. turbidez atmosferica modelo Bras (2=claro, 5=contaminado)
            "atcRyanStolz": 0.8,     #    Coef. transmision atmosferica modelo Ryan-Stolzenbach (0.70-0.91)
            # --- Radiacion infrarroja atmosferica descendente (Longwave IR) ---
            "longatMethod": "Brunt",          #  Modelo de emisividad atmosferica (Brunt / ...)
            # --- Evaporacion y conveccion/conduccion del aire ---
            "fUwMethod": "Brady-Graves-Geyer",  # Funcion de velocidad de viento para evaporacion y conveccion
            # --- Parametros termicos del sedimento (Sediment heat) ---
            "Hsed":   15,        #        Espesor termico del sedimento (cm)
            "alphas": 0.0064,    #        Difusividad termica del sedimento (cm2/s)
            "rhos":   1.6,       #        Densidad del sedimento (g/cm3)
            "rhow":   1,         #        Densidad del agua (g/cm3)
            "Cps":    0.4,       #        Calor especifico del sedimento (cal/(g·°C))
            "Cpw":    1,         #        Calor especifico del agua (cal/(g·°C))
            # --- Diagenesis del sedimento (Sediment diagenesis) ---
            "SedComp": "Yes",    #        Calcular DBO sedimento y flujos de nutrientes (Yes/No)
        }

        # Diffuse sources
        self.diffuse_sources_dict = {"ndiff": 0, "sources": []}

        # Rates generales
        # Rates generales organizados por su posición exacta en el archivo .q2k
        self.rates_dict = {
            # --- LÍNEA 1: Sólidos inorgánicos y Estequiometría ---
            "vss": 0.1,       # vi: Velocidad de sedimentacion solidos inorganicos (m/d)
            "mgC": 40,        # Relacion carbono (gC/gC)
            "mgN": 7.2,       # Relacion nitrogeno (gN/gN)
            "mgP": 1,         # Relacion fosforo (gP/gP)
            "mgD": 100,       # Relacion peso seco (gD/gD)
            "mgA": 1,         # Relacion clorofila (gA/gA)

            # --- LÍNEA 2: Reaireación térmica y O2 estequiométrico ---
            "tka": 1.024,     # qa: Correccion de temperatura para reareacion
            "roc": 2.69,      # roc: O2 consumido por oxidacion de carbono (gO2/gC)
            "ron": 4.570,    # ron: O2 consumido por nitrificacion (gO2/gN) *Nota: ya multiplicado para el .q2k

            # --- LÍNEA 3: Inhibición O2, CBOD lento, CBOD rápido y N Orgánico ---
            "Ksocf": 0.6,     # Param. inhibicion O2 oxidacion CBOD rapido
            "Ksona": 0.6,     # Param. inhibicion O2 nitrificacion
            "Ksodn": 0.6,     # Param. mejora O2 desnitrificacion
            "Ksop":  0.6,     # Param. inhibicion O2 resp. fitoplancton
            "Ksob":  0.6,     # Param. mejora O2 resp. alga bentonica
            "khc":   0,       # Tasa hidrolisis CBOD lento
            "tkhc":  1.07,    
            "kdcs":  0,       # Tasa oxidacion CBOD lento
            "tkdcs": 1.047,   
            "kdc":   0.09,    # Tasa oxidacion CBOD rapido
            "tkdc":  1.047,   
            "khn":   0.015,   # Tasa hidrolisis N organico
            "tkhn":  1.07,    
            "von":   0.0005,  # Velocidad sedimentacion N organico

            # --- LÍNEA 4: Amonio, Nitrato, P Orgánico y P Inorgánico ---
            "kn":    0.08,    # Tasa nitrificacion
            "tkn":   1.07,    
            "ki":    0.1,     # Tasa desnitrificacion
            "tki":   1.07,    
            "vdi":   0.8,     # Coef. transferencia desnitrificacion sedimento
            "tvdi":  1.07,    
            "khp":   0.03,    # Tasa hidrolisis P organico
            "tkhp":  1.07,    
            "vop":   0.001,   # Velocidad sedimentacion P organico
            "vip":   0.8,     # Velocidad sedimentacion P inorganico
            "kspi":  1,       # Const. media saturacion O2 atenuacion P sedimento
            "Kdpi":  1000,    # Coeficiente sorcion P inorganico

            # --- LÍNEA 5: Fitoplancton (Tasas principales y luz) ---
            "kga":   3.8,     # Tasa maxima crecimiento fitoplancton
            "tkga":  1.07,    
            "krea":  0.15,    # Tasa respiracion fitoplancton
            "tkrea": 1.07,    
            "kexa":  0.3,     # Tasa excrecion fitoplancton
            "tkexa": 1.07,    
            "kdea":  0.1,     # Tasa muerte fitoplancton
            "tkdea": 1.07,    
            "ksn":   100,     # Const. media saturacion N externo fitoplancton
            "ksp":   10,      # Const. media saturacion P externo fitoplancton
            "ksc":   0.000013,# Const. media saturacion carbono inorganico fito.
            "Isat":  250,     # Constante luz fitoplancton

            # --- LÍNEA 6: Fitoplancton y Algas Bentónicas (Parte 1) ---
            "khnx":  25,           # Preferencia amonio fitoplancton
            "va":    0,            # Velocidad sedimentacion fitoplancton
            "typeF": "Zero-order", # Modelo crecimiento alga bentonica (TEXTO)
            "kgaF":  200,          # Tasa max crecimiento alga bentonica
            "tkgaF": 1.07,         
            "kreaF": 0.2,          # Tasa respiracion alga bentonica
            "tkreaF": 1.07,        
            "kexaF": 0.12,         # Tasa excrecion alga bentonica
            "tkexaF": 1.07,        
            "kdeaF": 0.1,          # Tasa muerte alga bentonica
            "abmax": 1000,         # Capacidad de carga alga bentonica (VA AL FINAL DE LA LÍNEA 6)

            # --- LÍNEA 7: Algas Bentónicas (Parte 2) y Detrito (POM) ---
            "tkdeaF": 1.07,        # Correccion temp. muerte alga bentonica
            "ksnF":   300,         # Const. media saturacion N externo alga bent.
            "kspF":   100,         # Const. media saturacion P externo alga bent.
            "kscF":   0.000013,    # Const. media saturacion C inorg. alga bent.
            "Isatf":  100,         # Constante luz alga bentonica
            "khnxF":  25,          # Preferencia amonio alga bentonica
            "kdt":    0.23,        # Tasa disolucion detrito
            "tkdt":   1.07,        
            "ffast":  1,           # Fraccion disolucion a CBOD rapido
            "vdt":    0.008,       # Velocidad sedimentacion detrito

            # --- LÍNEA 8: Fitoplancton - Consumo de lujo (Luxury uptake) ---
            "NINpmin":   0, 
            "NIPpmin":   0, 
            "NINpupmax": 0, 
            "NIPpupmax": 0, 
            "KqNp":      0, 
            "KqPp":      0, 

            # --- LÍNEA 9: Algas Bentónicas - Consumo de lujo (Luxury uptake) ---
            "NINbmin":   0.72, 
            "NIPbmin":   0.1, 
            "NINbupmax": 72, 
            "NIPbupmax": 5, 
            "KqNb":      0.9, 
            "KqPb":      0.13, 

            # --- LÍNEA 10: Patógenos ---
            "kpath":  0.8, 
            "tkpath": 1.07, 
            "vpath":  1, 
            "aPath":  1, # Factor de eficiencia de luz (0.00 en Excel, 0.001 real)

            # --- LÍNEAS 11, 12, 13: Constituyentes genéricos ---
            "consts": [
                {"kconst": 0, "tkconst": 1, "vconst": 0}, # Constituyente i
                {"kconst": 0, "tkconst": 1, "vconst": 0}, # Constituyente ii
                {"kconst": 0, "tkconst": 1, "vconst": 0}, # Constituyente iii
            ],

            # --- LÍNEA 14: Tipos de saturación (Modelos de luz y O2) ---
            "saturation_types": [
                "Exponential", "Exponential", "Exponential", 
                "Exponential", "Exponential", "Half saturation", "Half saturation"
            ],

            # --- LÍNEA 15: Métodos de reaireación y viento ---
            "kai": "O'Connor-Dobbins",
            "kawindmethod": "None",

            # --- LÍNEA 16: Coeficientes de reaireación del usuario ---
            "reaa": 3.93, # alpha
            "reab": 0.5,  # beta
            "reac": 1.5,  # gamma
        }
        # self.rates_dict = {
        #     # --- Estequiometria (Stoichiometry) ---
        #     "vss":  .1,          # vi:    Velocidad de sedimentacion de solidos inorganicos (m/d)
        #     "mgC":  40,          #        Relacion carbono (gC/gC)
        #     "mgN":  7.2,         #        Relacion nitrogeno (gN/gN)
        #     "mgP":  1,           #        Relacion fosforo (gP/gP)
        #     "mgD":  100,         #        Relacion peso seco (gD/gD)
        #     "mgA":  1,           #        Relacion clorofila (gA/gA)
        #     # --- Oxigeno (Oxygen) ---
        #     "tka":  1.024,       # qa:    Correccion de temperatura para reareacion
        #     "roc":  2.69,        # roc:   O2 consumido por oxidacion de carbono (gO2/gC)
        #     "ron":  4.57,        # ron:   O2 consumido por nitrificacion de NH4 (gO2/gN)
        #     # Modelos de inhibicion/mejora por oxigeno (saturation_types define el modelo por componente)
        #     "Ksocf": .6,         # Ksocf: Param. inhibicion O2 - oxidacion CBOD rapido (L/mgO2), Exponential
        #     "Ksona": .6,         # Ksona: Param. inhibicion O2 - nitrificacion (L/mgO2), Exponential
        #     "Ksodn": .6,         # Ksodn: Param. mejora O2 - desnitrificacion (L/mgO2), Exponential
        #     "Ksop":  .6,         # Ksop:  Param. inhibicion O2 - resp. fitoplancton (L/mgO2), Exponential
        #     "Ksob":  .6,         # Ksob:  Param. mejora O2 - resp. alga bentonica (L/mgO2), Exponential
        #     # --- CBOD lento (Slow CBOD) ---
        #     "khc":   0,          # khc:   Tasa de hidrolisis CBOD lento (/d)
        #     "tkhc":  1.07,       # qhc:   Correccion de temperatura para khc
        #     "kdcs":  0,          # kdcs:  Tasa de oxidacion CBOD lento (/d)
        #     "tkdcs": 1.047,      # qdcs:  Correccion de temperatura para kdcs
        #     # --- CBOD rapido (Fast CBOD) ---
        #     "kdc":   .09,        # kdc:   Tasa de oxidacion CBOD rapido (/d)
        #     "tkdc":  1.047,      # qdc:   Correccion de temperatura para kdc
        #     # --- Nitrogeno organico (Organic N) ---
        #     "khn":   .015,       # khn:   Tasa de hidrolisis N organico (/d)
        #     "tkhn":  1.07,       # qhn:   Correccion de temperatura para khn
        #     "von":   .0005,      # von:   Velocidad de sedimentacion N organico (m/d)
        #     # --- Amonio (Ammonium) ---
        #     "kn":    .08,        # kna:   Tasa de nitrificacion (/d)
        #     "tkn":   1.07,       # qna:   Correccion de temperatura para kn
        #     # --- Nitrato (Nitrate) ---
        #     "ki":    .1,         # kdn:   Tasa de desnitrificacion (/d)
        #     "tki":   1.07,       # qdn:   Correccion de temperatura para ki
        #     "vdi":   .8,         # vdi:   Coef. transferencia desnitrificacion sedimento (m/d)
        #     "tvdi":  1.07,       # qdi:   Correccion de temperatura para vdi
        #     # --- Fosforo organico (Organic P) ---
        #     "khp":   .03,        # khp:   Tasa de hidrolisis P organico (/d)
        #     "tkhp":  1.07,       # qhp:   Correccion de temperatura para khp
        #     "vop":   .001,       # vop:   Velocidad de sedimentacion P organico (m/d)
        #     # --- Fosforo inorganico (Inorganic P) ---
        #     "vip":   .8,         # vip:   Velocidad de sedimentacion P inorganico (m/d)
        #     "kspi":  1,          # kspi:  Constante media saturacion O2 atenuacion P sedimento (mgO2/L)
        #     "Kdpi":  1000,       # Kdpi:  Coeficiente de sorcion P inorganico (L/mgD)
        #     # --- Fitoplancton (Phytoplankton) ---
        #     "kga":   3.8,        # kgp:   Tasa maxima de crecimiento fitoplancton (/d)
        #     "tkga":  1.07,       # qgp:   Correccion de temperatura para kga
        #     "krea":  .15,        # krp:   Tasa de respiracion fitoplancton (/d)
        #     "tkrea": 1.07,       # qrp:   Correccion de temperatura para krea
        #     "kexa":  .3,         # kep:   Tasa de excrecion fitoplancton (/d)
        #     "tkexa": 1.07,       # qdp:   Correccion de temperatura para kexa
        #     "kdea":  .1,         # kdp:   Tasa de muerte fitoplancton (/d)
        #     "tkdea": 1.07,       # qdp:   Correccion de temperatura para kdea
        #     "ksn":   100,        # ksPp:  Constante media saturacion N externo fitoplancton (ugN/L)
        #     "ksp":   10,         # ksNp:  Constante media saturacion P externo fitoplancton (ugP/L)
        #     "ksc":   .000013,    # ksCp:  Constante media saturacion carbono inorganico fitoplancton (mol/L)
        #     "Isat":  250,        # KLp:   Constante de luz fitoplancton - modelo Half saturation (langley/d)
        #     "khnx":  25,         # khnxp: Preferencia de amonio fitoplancton (ugN/L)
        #     "va":    0,          # va:    Velocidad de sedimentacion fitoplancton (m/d)
        #     # Cuotas internas de nutrientes fitoplancton (Droop model)
        #     "NINpmin":  .8,      # q0Np:  Cuota minima interna de N fitoplancton (mgN/mgA)
        #     "NIPpmin":  1.07,    # q0Pp:  Cuota minima interna de P fitoplancton (mgP/mgA)
        #     "NINpupmax": 1,      # rmNp:  Tasa maxima de captacion interna N fitoplancton (mgN/mgA/d)
        #     "NIPpupmax": 1,      # rmPp:  Tasa maxima de captacion interna P fitoplancton (mgP/mgA/d)
        #     # --- Alga bentonica (Bottom Algae) ---
        #     "typeF":   "Zero-order",  # Modelo de crecimiento alga bentonica (Zero-order / First-order)
        #     "kgaF":    200,      # Cgb:   Tasa maxima de crecimiento alga bentonica (mgA/m2/d)
        #     "tkgaF":   1.07,     # qgb:   Correccion de temperatura para kgaF
        #     "abmax":   1000,     # ab,max: Capacidad de carga modelo First-order (mgA/m2)
        #     "kreaF":   .2,       # krb:   Tasa de respiracion alga bentonica (/d)
        #     "tkreaF":  1.07,     # qrb:   Correccion de temperatura para kreaF
        #     "kexaF":   .12,      # keb:   Tasa de excrecion alga bentonica (/d)
        #     "tkexaF":  1.07,     # qdb:   Correccion de temperatura para kexaF
        #     "kdeaF":   .1,       # kdb:   Tasa de muerte alga bentonica (/d)
        #     "tkdeaF":  1.07,     # qdb:   Correccion de temperatura para kdeaF
        #     "ksnF":    300,      # ksPb:  Constante media saturacion N externo alga bentonica (ugN/L)
        #     "kspF":    100,      # ksNb:  Constante media saturacion P externo alga bentonica (ugP/L)
        #     "kscF":    .000013,  # ksCb:  Constante media saturacion carbono inorganico alga bentonica (mol/L)
        #     "Isatf":   100,      # KLb:   Constante de luz alga bentonica - Half saturation (langley/d)
        #     "khnxF":   25,       # khnxb: Preferencia de amonio alga bentonica (ugN/L)
        #     # Cuotas internas de nutrientes alga bentonica (Droop model)
        #     "kai":        .72,   # q0N:   Cuota minima interna de N alga bentonica (mgN/mgA)
        #     "rea_extras": [72, 5, .9, .13],  # [rmN, rmP, KqN, KqP]: tasa max captacion N/P e internas half-sat N/P
        #                                      #   rmN=72 (mgN/mgA/d), rmP=5 (mgP/mgA/d), KqN=0.9 (mgN/mgA), KqP=0.13 (mgP/mgA)
        #     # --- Detrito / POM (Detritus) ---
        #     "kdt":   .23,        # kdt:   Tasa de disolucion detrito (/d)
        #     "tkdt":  1.07,       # qdt:   Correccion de temperatura para kdt
        #     "ffast": 1,          # Ff:    Fraccion de disolucion que va a CBOD rapido
        #     "vdt":   .008,       # vdt:   Velocidad de sedimentacion detrito (m/d)
        #     # --- Otros / Placeholders ---
        #     "xdum": [0, 0, 0, 0, 0, 0],  # Valores dummy (patogenos, pH, etc. no activos)
        #     "kawindmethod": .1,  #        Parametro efecto del viento en reareacion
        #     # --- Modelos de saturacion por oxigeno (orden: CBOD ox., nitrif., denitrif., resp. fito., resp. alga bent., crec. fito., crec. alga bent.) ---
        #     "saturation_types": [
        #         "Exponential",    # CBOD oxidation
        #         "Exponential",    # Nitrification
        #         "Exponential",    # Denitrification
        #         "Exponential",    # Phyto respiration
        #         "Exponential",    # Bottom algae respiration
        #         "Half saturation",# Phyto growth
        #         "Half saturation" # Bottom algae growth
        #     ],
        #     # --- Reareacion (Reaeration) ---
        #     "reaeration_methods": ["O'Connor-Dobbins", "None"],  # [metodo global, metodo viento]
        #     "reaa": 3.93,        # α:     Coeficiente de reareacion usuario - alpha
        #     "reab": .5,          # β:     Coeficiente de reareacion usuario - beta
        #     "reac": 1.5,         # γ:     Coeficiente de reareacion usuario - gamma
        #     # --- Constituyentes conservativos i, ii, iii ---
        #     "consts": [
        #         {"kconst": 0, "tkconst": 1, "vconst": 0},  # Constituyente i:   tasa reaccion (/d), corr. temp., sedimentacion (m/d)
        #         {"kconst": 0, "tkconst": 1, "vconst": 0},  # Constituyente ii
        #         {"kconst": 0, "tkconst": 1, "vconst": 0},  # Constituyente iii
        #     ],
        # }

        # Boundary data
        self.boundary_dict = {
            "dlstime": 0,
            "DownstreamBoundary": False,
            "nHw": 0,
            "headwaters": []
        }

        # Hydraulics data
        self.hydraulics_data_dict = {
            "nhydda": 0,
            "data": []
        }

        # Diel data
        self.diel_dict = {"ndiel": 5, "idiel": 1, "ndielstat": 0, "stations": [0]}

    def generar_reach_rates_default(self, n: int) -> Dict[str, Any]:
        """
        Genera un diccionario de reach_rates con valores por defecto.

        Args:
            n: Número de tramos

        Returns:
            Diccionario de reach_rates
        """
        return {
            "nr": n,
            "reaches": [
                {
                    # Reaeration
                    "kaaa":      None,   # Metodo de reareacion prescrito por tramo
                    # ISS: Inorganic suspended solids
                    "vss_rch":   None,   # vi:  Velocidad de sedimentacion solidos inorganicos (m/d)
                    # Slow CBOD
                    "khc_rch":   None,   # khc: Tasa de hidrolisis CBOD lento (/d)
                    "kdcs_rch":  None,   # kdcs: Tasa de oxidacion CBOD lento (/d)
                    # Fast CBOD
                    "kdc_rch":   None,   # kdc: Tasa de oxidacion CBOD rapido (/d)
                    # Organic N
                    "khn_rch":   None,   # khn: Tasa de hidrolisis N organico (/d)
                    "von_rch":   None,   # von: Velocidad de sedimentacion N organico (m/d)
                    # Ammonium
                    "kn_rch":    0.001,  # kna: Tasa de nitrificacion (/d)
                    # Nitrate
                    "ki_rch":    None,   # kdn: Tasa de desnitrificacion (/d)
                    "vdi_rch":   None,   # vdi: Coef. transferencia desnitrificacion sedimento (m/d)
                    # Organic P
                    "khp_rch":   None,   # khp: Tasa de hidrolisis P organico (/d)
                    "vop_rch":   None,   # vop: Velocidad de sedimentacion P organico (m/d)
                    # Inorganic P
                    "vip_rch":   None,   # vip: Velocidad de sedimentacion P inorganico (m/d)
                    # Phytoplankton
                    "kga_rch":   None,   # kgp: Tasa maxima de crecimiento fitoplancton (/d)
                    "krea_rch":  None,   # krp: Tasa de respiracion fitoplancton (/d)
                    "kexa_rch":  None,   # kep: Tasa de excrecion fitoplancton (/d)
                    "kdea_rch":  None,   # kdp: Tasa de muerte fitoplancton (/d)
                    "va_rch":    None,   # va:  Velocidad de sedimentacion fitoplancton (m/d)
                    # Bottom Algae
                    "kgaF_rch":  None,   # Cgb: Tasa maxima de crecimiento alga bentonica (/d)
                    "kreaF_rch": None,   # krb: Tasa de respiracion alga bentonica (/d)
                    "kexaF_rch": None,   # keb: Tasa de excrecion alga bentonica (/d)
                    "kdeaF_rch": None,   # kdb: Tasa de muerte alga bentonica (/d)
                    # Detritus (POM)
                    "kdt_rch":   None,   # kdt: Tasa de disolucion detrito (/d)
                    "vdt_rch":   None,   # vdt: Velocidad de sedimentacion detrito (m/d)
                    "ffast_rch": None,   # Ff:  Fraccion de disolucion que va a CBOD rapido
                }
                for _ in range(n)
            ]
        }

    def generar_reach_rates_custom(self, n: int,
                                   kaaa_list: List = None,
                                   khc_list: List = None,
                                   kdcs_list: List = None,
                                   kdc_list: List = None,
                                   khn_list: List = None,
                                   kn_list: List = None,
                                   ki_list: List = None,
                                   khp_list: List = None,
                                   kdt_list: List = None) -> Dict[str, Any]:
        """
        Genera un diccionario de reach_rates personalizado.

        Args:
            n: Número de tramos
            kaaa_list: Lista de métodos de reaireación
            khc_list: Lista de tasas de hidrólisis de C
            kdcs_list: Lista de tasas de descomposición particulada
            kdc_list: Lista de tasas de oxidación disuelta
            khn_list: Lista de tasas de hidrólisis de Norg
            kn_list: Lista de tasas de nitrificación
            ki_list: Lista de tasas de inhibición por OD
            khp_list: Lista de tasas de hidrólisis de Porg
            kdt_list: Lista de tasas de descomposición de detrito

        Returns:
            Diccionario de reach_rates
        """
        # Valores por defecto si no se proporcionan
        if kaaa_list is None: kaaa_list = [None] * n
        if khc_list is None: khc_list = [None] * n
        if kdcs_list is None: kdcs_list = [None] * n
        if kdc_list is None: kdc_list = [None] * n
        if khn_list is None: khn_list = [None] * n
        if kn_list is None: kn_list = [None] * n
        if ki_list is None: ki_list = [None] * n
        if khp_list is None: khp_list = [None] * n
        if kdt_list is None: kdt_list = [None] * n

        # Validación
        listas = [kaaa_list, khc_list, kdcs_list, kdc_list, khn_list,
                  kn_list, ki_list, khp_list, kdt_list]
        if any(len(lst) != n for lst in listas):
            raise ValueError("Todas las listas deben tener longitud igual a n")

        reaches = []
        for i in range(n):
            tramo = {
                # Reaeration
                "kaaa":      kaaa_list[i],   # Metodo de reareacion prescrito por tramo
                # ISS
                "vss_rch":   None,           # vi:  Velocidad de sedimentacion solidos inorganicos (m/d)
                # Slow CBOD
                "khc_rch":   khc_list[i],    # khc: Tasa de hidrolisis CBOD lento (/d)
                "kdcs_rch":  kdcs_list[i],   # kdcs: Tasa de oxidacion CBOD lento (/d)
                # Fast CBOD
                "kdc_rch":   kdc_list[i],    # kdc: Tasa de oxidacion CBOD rapido (/d)
                # Organic N
                "khn_rch":   khn_list[i],    # khn: Tasa de hidrolisis N organico (/d)
                "von_rch":   None,           # von: Velocidad de sedimentacion N organico (m/d)
                # Ammonium
                "kn_rch":    kn_list[i],     # kna: Tasa de nitrificacion (/d)
                # Nitrate
                "ki_rch":    ki_list[i],     # kdn: Tasa de desnitrificacion (/d)
                "vdi_rch":   None,           # vdi: Coef. transferencia desnitrificacion sedimento (m/d)
                # Organic P
                "khp_rch":   khp_list[i],    # khp: Tasa de hidrolisis P organico (/d)
                "vop_rch":   None,           # vop: Velocidad de sedimentacion P organico (m/d)
                # Inorganic P
                "vip_rch":   None,           # vip: Velocidad de sedimentacion P inorganico (m/d)
                # Phytoplankton
                "kga_rch":   None,           # kgp: Tasa maxima de crecimiento fitoplancton (/d)
                "krea_rch":  None,           # krp: Tasa de respiracion fitoplancton (/d)
                "kexa_rch":  None,           # kep: Tasa de excrecion fitoplancton (/d)
                "kdea_rch":  None,           # kdp: Tasa de muerte fitoplancton (/d)
                "va_rch":    None,           # va:  Velocidad de sedimentacion fitoplancton (m/d)
                # Bottom Algae
                "kgaF_rch":  None,           # Cgb: Tasa maxima de crecimiento alga bentonica (/d)
                "kreaF_rch": None,           # krb: Tasa de respiracion alga bentonica (/d)
                "kexaF_rch": None,           # keb: Tasa de excrecion alga bentonica (/d)
                "kdeaF_rch": None,           # kdb: Tasa de muerte alga bentonica (/d)
                # Detritus (POM)
                "kdt_rch":   kdt_list[i],    # kdt: Tasa de disolucion detrito (/d)
                "vdt_rch":   None,           # vdt: Velocidad de sedimentacion detrito (m/d)
                "ffast_rch": None,           # Ff:  Fraccion de disolucion que va a CBOD rapido
            }
            reaches.append(tramo)

        return {
            "nr": n,
            "reaches": reaches
        }

    def actualizar_header(self, **kwargs):
        """
        Actualiza valores del header.

        Ejemplo:
            config.actualizar_header(rivname="Nuevo Rio", tf=10)
        """
        self.header_dict.update(kwargs)

    def actualizar_rates(self, **kwargs):
        """
        Actualiza valores de las tasas cinéticas.

        Ejemplo:
            config.actualizar_rates(kn=0.1, ki=0.2)
        """
        self.rates_dict.update(kwargs)

    def actualizar_light(self, **kwargs):
        """
        Actualiza parámetros de luz.

        Ejemplo:
            config.actualizar_light(PAR=0.5, kep=0.3)
        """
        self.light_dict.update(kwargs)