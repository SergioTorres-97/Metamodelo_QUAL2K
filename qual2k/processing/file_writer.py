import os
from typing import Dict, Any, List, Union


class Q2KFileWriter:
    """
    Escribe archivos .q2k en el formato requerido por QUAL2K.
    """

    @staticmethod
    def format_number(value: Union[float, int, str]) -> str:
        """
        Convierte un número a formato FORTRAN/VBA compatible con QUAL2K.

        Args:
            value: Valor numérico a formatear

        Returns:
            str: Número formateado
        """
        s = str(value).replace('e', 'E')

        if s.startswith('0.'):
            s = s[1:]
        elif s.startswith('-0.'):
            s = '-' + s[2:]

        return s

    @staticmethod
    def safe_value(value: Any, default: Union[int, str] = -99999,
                   preserve_empty: bool = False) -> Union[float, int, str]:
        """
        Convierte valores vacíos, nulos o NaN a un valor por defecto.

        Args:
            value: Valor a procesar
            default: Valor por defecto
            preserve_empty: Si True, mantiene cadenas vacías

        Returns:
            Valor procesado o valor por defecto
        """
        if value is None:
            return "" if preserve_empty else default

        if isinstance(value, str) and not value.strip():
            return "" if preserve_empty else default

        try:
            if float(value) != float(value):
                return "" if preserve_empty else default
        except (ValueError, TypeError):
            pass

        return value

    @staticmethod
    def format_value(value: Any, preserve_empty: bool = False) -> str:
        """
        Formatea un valor para escritura en archivo Q2K.

        Args:
            value: Valor a formatear
            preserve_empty: Si mantener valores vacíos

        Returns:
            str: Valor formateado
        """
        val = Q2KFileWriter.safe_value(value, preserve_empty=preserve_empty)

        if val == "":
            return ""
        if isinstance(val, str):
            return val

        try:
            return f"{float(val):.15g}".replace('e', 'E')
        except (ValueError, TypeError):
            return str(val) if preserve_empty else "-99999"

    @staticmethod
    def write_line_segments(f, values: List, items_per_line: int = 10) -> None:
        """
        Escribe valores en múltiples líneas.

        Args:
            f: Archivo abierto para escritura
            values: Lista de valores a escribir
            items_per_line: Número de elementos por línea
        """
        for i in range(0, len(values), items_per_line):
            segment = values[i:i + items_per_line]
            f.write(",".join(segment) + "\n")

    def write_header(self, f, header: Dict[str, Any]) -> None:
        """Escribe el encabezado del archivo .q2k"""
        f.write(f"\"{header['version']}\"\n")
        f.write(f"\"{header['rivname']}\",\"{header['filename']}\","
                f"\"{header['filedir']}\",\"{header['applabel']}\"\n")
        f.write(f"{header['xmon']},{header['xday']},{header['xyear']}\n")
        f.write(f"{self.format_number(header['timezonehour'])},"
                f"{self.format_number(header['pco2'])},"
                f"{self.format_number(header['dtuser'])},"
                f"{self.format_number(header['tf'])},"
                f"\"{header['IMeth']}\",\"{header['IMethpH']}\"\n")

    def write_reach_data(self, f, reach_block: Dict[str, Any]) -> None:
        """Escribe el bloque de datos de tramos"""
        f.write(f"{reach_block['nr']},{reach_block['nHw']},{reach_block['ne']}\n")

        for r in reach_block["reaches"]:
            vals = [
                f"\"{r['rlab1']}\"", f"\"{r['rlab2']}\"", f"\"{r['rname']}\"",
                r["xrup"], r["xrdn"], r["numElm"],
                r["elev1"], r["elev2"],
                r["latd"], r["latm"], r["lats"],
                r["lond"], r["lonm"], r["lons"],
                self.safe_value(r["Q"]),
                r["BB"], r["SS1"], r["SS2"], r["s"], r["nm"],
                r["alp1"], r["bet1"], r["alp2"], r["bet2"],
                r["Ediff"], r["Frsed"], r["Frsod"],
                r["SODspec"], r["JCH4spec"], r["JNH4spec"], r["JSRPspec"],
                f"\"{r['weirType']}\"",
                r["Hweir"], r["Bweir"],
                r["adam"], r["bdam"], r["evap"]
            ]
            f.write(",".join(self.format_number(v) for v in vals) + "\n")

    def write_light_data(self, f, light: Dict[str, Any]) -> None:
        """Escribe el bloque de datos de luz y sedimentos"""
        f.write(f"{self.format_number(light['PAR'])},"
                f"{self.format_number(light['kep'])},"
                f"{self.format_number(light['kela'])},"
                f"{self.format_number(light['kenla'])},"
                f"{self.format_number(light['kess'])},"
                f"{self.format_number(light['kepom'])}\n")

        f.write(f"\"{light['solarMethod']}\","
                f"{self.format_number(light['nfacBras'])},"
                f"{self.format_number(light['atcRyanStolz'])},"
                f"\"{light['longatMethod']}\",\"{light['fUwMethod']}\"\n")

        f.write(f"{self.format_number(light['Hsed'])},"
                f"{self.format_number(light['alphas'])},"
                f"{self.format_number(light['rhos'])},"
                f"{self.format_number(light['rhow'])},"
                f"{self.format_number(light['Cps'])},"
                f"{self.format_number(light['Cpw'])}\n")

        f.write(f"\"{light['SedComp']}\"\n")

    def write_point_sources(self, f, ps_block: Dict[str, Any]) -> None:
        """Escribe el bloque de fuentes puntuales"""
        f.write(f"{ps_block['npt']}\n")

        for src in ps_block["sources"]:
            f.write(f"\"{src['PtName']}\","
                    f"{src['PtHwID'] - 1},"
                    f"{self.format_number(src['xptt'])},"
                    f"{self.format_number(src['Qptta'])},"
                    f"{self.format_number(src['Qptt'])},"
                    f"{self.format_number(src['TepttMean'])},"
                    f"{self.format_number(src['TepttAmp'])},"
                    f"{self.format_number(src['TepttMaxTime'])}\n")

            for c in src["constituents"]:
                f.write(f"{self.format_number(c['mean'])},"
                        f"{self.format_number(c['amp'])},"
                        f"{self.format_number(c['maxtime'])}\n")

    def write_diffuse_sources(self, f, diff_block: Dict[str, Any]) -> None:
        """Escribe el bloque de fuentes difusas"""
        f.write(f"{diff_block['ndiff']}\n")

        if diff_block["ndiff"] == 0:
            return

        for src in diff_block["sources"]:
            f.write(f"\"{src['DiffName']}\","
                    f"{src['DiffHwID'] - 1},"
                    f"{self.format_number(src['xdup'])},"
                    f"{self.format_number(src['xddn'])},"
                    f"{self.format_number(src['Qdifa'])},"
                    f"{self.format_number(src['Qdif'])},"
                    f"{self.format_number(src['Tedif'])}\n")

            for c in src["constituents"]:
                f.write(f"{self.format_number(c['value'])}\n")

            f.write(f"{self.format_number(src['pHind'])}\n")

    def write_rates_general(self, f, r: Dict[str, Any]) -> None:
        """Escribe el bloque de tasas cinéticas generales estrictamente posicional"""
        fmt = self.format_number

        # Línea 1: Sólidos inorgánicos y Estequiometría
        f.write(f"{fmt(r['vss'])},{fmt(r['mgC'])},{fmt(r['mgN'])},"
                f"{fmt(r['mgP'])},{fmt(r['mgD'])},{fmt(r['mgA'])}\n")

        # Línea 2: Reaireación térmica y O2 estequiométrico
        f.write(f"{fmt(r['tka'])},{fmt(r['roc'])},{fmt(r['ron'])}\n")

        # Línea 3: Inhibición O2, CBOD lento, CBOD rápido y N Orgánico
        f.write(f"{fmt(r['Ksocf'])},{fmt(r['Ksona'])},{fmt(r['Ksodn'])},"
                f"{fmt(r['Ksop'])},{fmt(r['Ksob'])},{fmt(r['khc'])},"
                f"{fmt(r['tkhc'])},{fmt(r['kdcs'])},{fmt(r['tkdcs'])},"
                f"{fmt(r['kdc'])},{fmt(r['tkdc'])},{fmt(r['khn'])},"
                f"{fmt(r['tkhn'])},{fmt(r['von'])}\n")

        # Línea 4: Amonio, Nitrato, P Orgánico y P Inorgánico
        f.write(f"{fmt(r['kn'])},{fmt(r['tkn'])},{fmt(r['ki'])},"
                f"{fmt(r['tki'])},{fmt(r['vdi'])},{fmt(r['tvdi'])},"
                f"{fmt(r['khp'])},{fmt(r['tkhp'])},{fmt(r['vop'])},"
                f"{fmt(r['vip'])},{fmt(r['kspi'])},{fmt(r['Kdpi'])}\n")

        # Línea 5: Fitoplancton (Tasas principales y luz)
        f.write(f"{fmt(r['kga'])},{fmt(r['tkga'])},{fmt(r['krea'])},"
                f"{fmt(r['tkrea'])},{fmt(r['kexa'])},{fmt(r['tkexa'])},"
                f"{fmt(r['kdea'])},{fmt(r['tkdea'])},{fmt(r['ksn'])},"
                f"{fmt(r['ksp'])},{fmt(r['ksc'])},{fmt(r['Isat'])}\n")

        # Línea 6: Fitoplancton y Algas Bentónicas (Parte 1) - Nota: typeF en comillas y abmax al final
        f.write(f"{fmt(r['khnx'])},{fmt(r['va'])},\"{r['typeF']}\","
                f"{fmt(r['kgaF'])},{fmt(r['tkgaF'])},{fmt(r['kreaF'])},"
                f"{fmt(r['tkreaF'])},{fmt(r['kexaF'])},{fmt(r['tkexaF'])},"
                f"{fmt(r['kdeaF'])},{fmt(r['abmax'])}\n")

        # Línea 7: Algas Bentónicas (Parte 2) y Detrito (POM)
        f.write(f"{fmt(r['tkdeaF'])},{fmt(r['ksnF'])},{fmt(r['kspF'])},"
                f"{fmt(r['kscF'])},{fmt(r['Isatf'])},{fmt(r['khnxF'])},"
                f"{fmt(r['kdt'])},{fmt(r['tkdt'])},{fmt(r['ffast'])},"
                f"{fmt(r['vdt'])}\n")

        # Línea 8: Fitoplancton - Consumo de lujo (Luxury uptake)
        f.write(f"{fmt(r['NINpmin'])},{fmt(r['NIPpmin'])},"
                f"{fmt(r['NINpupmax'])},{fmt(r['NIPpupmax'])},"
                f"{fmt(r['KqNp'])},{fmt(r['KqPp'])}\n")

        # Línea 9: Algas Bentónicas - Consumo de lujo (Luxury uptake)
        f.write(f"{fmt(r['NINbmin'])},{fmt(r['NIPbmin'])},"
                f"{fmt(r['NINbupmax'])},{fmt(r['NIPbupmax'])},"
                f"{fmt(r['KqNb'])},{fmt(r['KqPb'])}\n")

        # Línea 10: Patógenos
        f.write(f"{fmt(r['kpath'])},{fmt(r['tkpath'])},"
                f"{fmt(r['vpath'])},{fmt(r['aPath'])}\n")

        # Líneas 11, 12, 13: Constituyentes genéricos
        for c in r["consts"]:
            f.write(f"{fmt(c['kconst'])},{fmt(c['tkconst'])},"
                    f"{fmt(c['vconst'])}\n")

        # Línea 14: Tipos de saturación (Modelos de luz y O2 en texto con comillas)
        f.write(",".join(f"\"{x}\"" for x in r["saturation_types"]) + "\n")

        # Línea 15: Métodos de reaireación y viento (Texto con comillas)
        f.write(f"\"{r['kai']}\",\"{r['kawindmethod']}\"\n")

        # Línea 16: Coeficientes de reaireación del usuario
        f.write(f"{fmt(r['reaa'])},{fmt(r['reab'])},{fmt(r['reac'])}\n")
        
    # def write_rates_general(self, f, r: Dict[str, Any]) -> None:
        # """Escribe el bloque de tasas cinéticas generales"""
        # fmt = self.format_number

        # # Línea 1: Estequiometría
        # f.write(f"{fmt(r['vss'])},{fmt(r['mgC'])},{fmt(r['mgN'])},"
        #         f"{fmt(r['mgP'])},{fmt(r['mgD'])},{fmt(r['mgA'])}\n")

        # # Línea 2: Reaireación
        # f.write(f"{fmt(r['tka'])},{fmt(r['roc'])},{fmt(r['ron'] * 1000)}\n")

        # # Línea 3: Sedimento y materia orgánica
        # f.write(f"{fmt(r['Ksocf'])},{fmt(r['Ksona'])},{fmt(r['Ksodn'])},"
        #         f"{fmt(r['Ksop'])},{fmt(r['Ksob'])},{fmt(r['khc'])},"
        #         f"{fmt(r['tkhc'])},{fmt(r['kdcs'])},{fmt(r['tkdcs'])},"
        #         f"{fmt(r['kdc'])},{fmt(r['tkdc'])},{fmt(r['khn'])},"
        #         f"{fmt(r['tkhn'])},{fmt(r['von'])}\n")

        # # Línea 4: Nitrógeno y fósforo
        # f.write(f"{fmt(r['kn'])},{fmt(r['tkn'])},{fmt(r['ki'])},"
        #         f"{fmt(r['tki'])},{fmt(r['vdi'])},{fmt(r['tvdi'])},"
        #         f"{fmt(r['khp'])},{fmt(r['tkhp'])},{fmt(r['vop'])},"
        #         f"{fmt(r['vip'])},{fmt(r['kspi'])},{fmt(r['Kdpi'])}\n")

        # # Línea 5: Algas
        # f.write(f"{fmt(r['kga'])},{fmt(r['tkga'])},{fmt(r['krea'])},"
        #         f"{fmt(r['tkrea'])},{fmt(r['kexa'])},{fmt(r['tkexa'])},"
        #         f"{fmt(r['kdea'])},{fmt(r['tkdea'])},{fmt(r['ksn'])},"
        #         f"{fmt(r['ksp'])},{fmt(r['ksc'])},{fmt(r['Isat'])}\n")

        # # Línea 6: Algas flotantes
        # f.write(f"{fmt(r['khnx'])},{fmt(r['va'])},\"{r['typeF']}\","
        #         f"{fmt(r['kgaF'])},{fmt(r['tkgaF'])},{fmt(r['kreaF'])},"
        #         f"{fmt(r['tkreaF'])},{fmt(r['kexaF'])},{fmt(r['tkexaF'])},"
        #         f"{fmt(r['kdeaF'])},{fmt(r['abmax'])}\n")

        # # Línea 7: Detritus
        # f.write(f"{fmt(r['tkdeaF'])},{fmt(r['ksnF'])},{fmt(r['kspF'])},"
        #         f"{fmt(r['kscF'])},{fmt(r['Isatf'])},{fmt(r['khnxF'])},"
        #         f"{fmt(r['kdt'])},{fmt(r['tkdt'])},{fmt(r['ffast'])},"
        #         f"{fmt(r['vdt'])}\n")

        # # Línea 8: Valores dummy
        # f.write(",".join(fmt(x) for x in r["xdum"]) + "\n")

        # # Línea 9: Reaireación extras
        # f.write(f"{fmt(r['kai'])},{fmt(r['kawindmethod'])},"
        #         f"{','.join(fmt(x) for x in r['rea_extras'])}\n")

        # # Línea 10: Límites de nutrientes
        # f.write(f"{fmt(r['NINpmin'])},{fmt(r['NIPpmin'])},"
        #         f"{fmt(r['NINpupmax'])},{fmt(r['NIPpupmax'])}\n")

        # # Líneas 11-13: Constituyentes genéricos
        # for c in r["consts"]:
        #     f.write(f"{fmt(c['kconst'])},{fmt(c['tkconst'])},"
        #             f"{fmt(c['vconst'])}\n")

        # # Línea 14: Tipos de saturación
        # f.write(",".join(f"\"{x}\"" for x in r["saturation_types"]) + "\n")

        # # Línea 15: Métodos de reaireación
        # f.write(",".join(f"\"{x}\"" for x in r["reaeration_methods"]) + "\n")

        # # Línea 16: Parámetros de reaireación
        # f.write(f"{fmt(r['reaa'])},{fmt(r['reab'])},{fmt(r['reac'])}\n")

    def write_reach_rates(self, f, reach_rates: Dict[str, Any]) -> None:
        """Escribe el bloque de tasas específicas por tramo"""
        rate_keys = [
            "kaaa", "vss_rch", "khc_rch", "kdcs_rch", "kdc_rch",
            "khn_rch", "von_rch", "kn_rch", "ki_rch", "vdi_rch",
            "khp_rch", "vop_rch", "vip_rch", "kga_rch", "krea_rch",
            "kexa_rch", "kdea_rch", "va_rch", "kgaF_rch", "kreaF_rch",
            "kexaF_rch", "kdeaF_rch", "kdt_rch", "vdt_rch", "ffast_rch"
        ]

        for r in reach_rates["reaches"]:
            values = [self.format_number(self.safe_value(r.get(k))) for k in rate_keys]
            f.write(",".join(values) + "\n")

    def write_boundary_data(self, f, b: Dict[str, Any]) -> None:
        """Escribe el bloque de datos de frontera"""
        f.write(f"{self.format_number(b['dlstime'])}\n")

        boundary_flag = ".TRUE.\n" if b["DownstreamBoundary"] else ".FALSE.\n"
        f.write(boundary_flag)

        if b.get("nHw", 0) == 0:
            return

    def write_headwaters_q2k(self, f, hw_block: Dict[str, Any]) -> None:
        """Escribe el bloque de cabeceras"""
        nHw = hw_block.get("nHw", 0)
        if nHw == 0:
            return

        for hw in hw_block["headwaters"]:
            line_parts = [
                hw["begRch"],
                f"\"{hw['NameHw']}\"",
                self.format_value(hw["QHw"]),
                self.format_value(hw["elevHw"]),
                f"\"{hw['weirTypeHw']}\"",
                self.format_value(hw["HweirHw"]),
                self.format_value(hw["BweirHw"]),
                self.format_value(hw["alp1Hw"]),
                self.format_value(hw["bet1Hw"]),
                self.format_value(hw["alp2Hw"]),
                self.format_value(hw["bet2Hw"]),
                self.format_value(hw["sHw"]),
                self.format_value(hw["nmHw"]),
                self.format_value(hw["bbHw"]),
                self.format_value(hw["ss1Hw"]),
                self.format_value(hw["ss2Hw"]),
                self.format_value(hw["ediffHw"]),
                self.format_value(hw["adamHw"]),
                self.format_value(hw["bdamHw"])
            ]
            f.write(",".join(str(p) for p in line_parts) + "\n")

            # Temperatura
            temp_values = [self.format_value(v, True) for v in hw["TeHw"]]
            f.write(",".join(temp_values) + ",\"\"\n")

            # 19 constituyentes
            for c in hw["cHw"]:
                const_values = [self.format_value(v, True) for v in c]
                f.write(",".join(const_values) + ",\"\"\n")

            # pH
            ph_values = [self.format_value(v, True) for v in hw["pHHw"]]
            f.write(",".join(ph_values) + ",\"\"\n")

    def write_meteorological_data_q2k(self, f, meteo: Dict[str, Any]) -> None:
        """Escribe los datos meteorológicos"""
        nr = meteo.get("nr", 1)
        met_vars = ["shadeHH", "TaHH", "TdHH", "UwHH", "ccHH"]

        for var_name in met_vars:
            for i in range(nr):
                values = [self.format_value(v) for v in meteo[var_name][i]]
                f.write(",".join(values) + ",\"\"\n")

    def write_temperature_data_q2k(self, f, temp_data: Dict[str, Any]) -> None:
        """Escribe el bloque de datos de temperatura observados"""
        nteda = temp_data.get("nteda", 0)
        f.write(f"{nteda}\n")

        for row in temp_data.get("data", []):
            parts = [
                str(int(row["tedaHwID"]) - 1),
                self.format_value(row["xteda"]),
                self.format_value(row["tedaav"]),
                self.format_value(row["tedamn"]),
                self.format_value(row["tedamx"])
            ]
            f.write(",".join(parts) + "\n")

    def write_hydraulics_data_q2k(self, f, hyd_data: Dict[str, Any]) -> None:
        """Escribe el bloque de datos hidráulicos observados"""
        nhydda = hyd_data.get("nhydda", 0)
        f.write(f"{nhydda}\n")

        for row in hyd_data.get("data", []):
            parts = [
                str(int(row["hyddaHwID"]) - 1),
                self.format_value(row["xhydda"]),
                self.format_value(row["Qdata"]),
                self.format_value(row["Hdata"]),
                self.format_value(row["Udata"]),
                self.format_value(row["Travdata"])
            ]
            f.write(",".join(parts) + "\n")

    def write_wqdata_q2k(self, f, wqdata: Dict[str, Any]) -> None:
        """Escribe el bloque de datos de calidad de agua observados"""
        CONSTITUENT_ORDER = [
            "Cond", "ISS", "DO", "CBODs", "CBODf", "Norg", "NH4", "NO3",
            "Porg", "Inorg_P", "Phyto", "Detr", "Pathogens", "Alk",
            "Constituent_i", "Constituent_ii", "Constituent_iii", "pH",
            "Bot_Alg", "TN", "TP", "TSS", "NH3", "Sat_data", "SOD_data",
            "Sediment_1", "Sediment_2", "Sediment_3", "CBODu", "TOC", "TKN"
        ]

        f.write(f"\"{wqdata['sheet_name']}\",{wqdata['nwqd']}\n")

        for st in wqdata.get("stations", []):
            f.write(f"{int(st['cwqHwID']) - 1},{self.format_value(st['dist'])}\n")

            values = [self.format_value(st["constituents"].get(c, -99999))
                      for c in CONSTITUENT_ORDER]

            self.write_line_segments(f, values, items_per_line=10)

        f.write("\"WQ Data Min\",0\n")
        f.write("\"WQ Data Max\",0\n")

    def write_diel_block_q2k(self, f, diel: Dict[str, Any]) -> None:
        """Escribe el bloque de simulación diurna"""
        ndiel = diel.get("ndiel", 5)
        idiel = diel.get("idiel", 1)
        ndielstat = diel.get("ndielstat", 0)
        stations = diel.get("stations", [0])

        f.write(f"{self.format_number(ndiel)},{self.format_number(idiel)}\n")
        f.write(f"{self.format_number(ndielstat)}\n")
        f.write("\"MULTSTATION DIEL\"\n")

        if ndielstat > 0 and stations:
            for st in stations:
                f.write(f"\"STATION\",{self.format_number(st)}\n")
        else:
            f.write("\"STATION\",0\n")

        f.write("\"END MULTSTATION DIEL\"\n")

    def create_q2k_file(self, filepath: str, data: Dict[str, Any]) -> None:
        """
        Crea un archivo QUAL2K completo (.q2k).

        Args:
            filepath: Ruta del archivo a crear
            data: Diccionario con todos los datos organizados por bloques
        """
        with open(filepath, 'w') as f:
            if 'header' in data:
                self.write_header(f, data['header'])
            if 'reach_data' in data:
                self.write_reach_data(f, data['reach_data'])
            if 'light_data' in data:
                self.write_light_data(f, data['light_data'])
            if 'point_sources' in data:
                self.write_point_sources(f, data['point_sources'])
            if 'diffuse_sources' in data:
                self.write_diffuse_sources(f, data['diffuse_sources'])
            if 'rates_general' in data:
                self.write_rates_general(f, data['rates_general'])
            if 'reach_rates' in data:
                self.write_reach_rates(f, data['reach_rates'])
            if 'boundary_data' in data:
                self.write_boundary_data(f, data['boundary_data'])
            if 'headwaters' in data:
                self.write_headwaters_q2k(f, data['headwaters'])
            if 'meteorological' in data:
                self.write_meteorological_data_q2k(f, data['meteorological'])
            if 'temperature_data' in data:
                self.write_temperature_data_q2k(f, data['temperature_data'])
            if 'hydraulics_data' in data:
                self.write_hydraulics_data_q2k(f, data['hydraulics_data'])
            if 'wq_data' in data:
                self.write_wqdata_q2k(f, data['wq_data'])
            if 'diel' in data:
                self.write_diel_block_q2k(f, data['diel'])

    @staticmethod
    def create_message(header_dict: Dict[str, Any]) -> None:
        """
        Crea el archivo message.DAT necesario para la ejecución.

        Args:
            header_dict: Diccionario con datos del header
        """
        q2k_path = os.path.join(header_dict["filedir"],
                                f"{header_dict['filename']}.q2k")
        out_path = os.path.join(header_dict["filedir"],
                                f"{header_dict['filename']}.out")
        dat_path = os.path.join(header_dict["filedir"], "message.DAT")

        os.makedirs(os.path.dirname(dat_path), exist_ok=True)

        with open(dat_path, "w") as f:
            f.write(f"\"{q2k_path}\"\n")
            f.write(f"\"{out_path}\"\n")