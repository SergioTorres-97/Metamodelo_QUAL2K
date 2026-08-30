"""
Módulo auxiliar para cálculo de métricas de evaluación del modelo.
Este módulo proporciona la función kge() que se utiliza en el análisis de resultados.
"""
import numpy as np

def kge(obs, sim):
    """
    Calcula el coeficiente de eficiencia de Kling-Gupta (KGE).

    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)

    donde:
    - r: coeficiente de correlación de Pearson
    - alpha: ratio de desviaciones estándar (sigma_sim / sigma_obs)
    - beta: ratio de medias (mu_sim / mu_obs)

    Args:
        obs: Array o Serie de valores observados
        sim: Array o Serie de valores simulados

    Returns:
        float: Valor de KGE (rango: -∞ a 1, siendo 1 el valor óptimo)
    """
    # Convertir a arrays numpy
    obs = np.array(obs)
    sim = np.array(sim)

    # Eliminar NaN y valores infinitos
    mask = ~(np.isnan(obs) | np.isnan(sim) | np.isinf(obs) | np.isinf(sim))
    obs = obs[mask]
    sim = sim[mask]

    # Verificar que hay datos suficientes
    if len(obs) < 2:
        return np.nan

    # Calcular componentes
    # Correlación de Pearson
    r = np.corrcoef(obs, sim)[0, 1]

    # Ratio de desviaciones estándar
    std_obs = np.std(obs, ddof=1)
    std_sim = np.std(sim, ddof=1)
    if std_obs == 0:
        alpha = np.nan
    else:
        alpha = std_sim / std_obs

    # Ratio de medias
    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    if mean_obs == 0:
        beta = np.nan
    else:
        beta = mean_sim / mean_obs

    # Calcular KGE
    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        return np.nan

    kge_value = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge_value


def nse(obs, sim):
    """
    Calcula el coeficiente de eficiencia de Nash-Sutcliffe (NSE).

    NSE = 1 - [sum((obs - sim)^2) / sum((obs - mean(obs))^2)]

    Args:
        obs: Array o Serie de valores observados
        sim: Array o Serie de valores simulados

    Returns:
        float: Valor de NSE (rango: -∞ a 1, siendo 1 el valor óptimo)
    """
    obs = np.array(obs)
    sim = np.array(sim)

    mask = ~(np.isnan(obs) | np.isnan(sim) | np.isinf(obs) | np.isinf(sim))
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) < 2:
        return np.nan

    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)

    if denominator == 0:
        return np.nan

    nse_value = 1 - (numerator / denominator)

    return nse_value


def rmse(obs, sim):
    """
    Calcula la raíz del error cuadrático medio (RMSE).

    RMSE = sqrt(mean((obs - sim)^2))

    Args:
        obs: Array o Serie de valores observados
        sim: Array o Serie de valores simulados

    Returns:
        float: Valor de RMSE
    """
    obs = np.array(obs)
    sim = np.array(sim)

    mask = ~(np.isnan(obs) | np.isnan(sim) | np.isinf(obs) | np.isinf(sim))
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) < 1:
        return np.nan

    rmse_value = np.sqrt(np.mean((obs - sim) ** 2))

    return rmse_value


def pbias(obs, sim):
    """
    Calcula el sesgo porcentual (PBIAS).

    PBIAS = 100 * [sum(obs - sim) / sum(obs)]

    Args:
        obs: Array o Serie de valores observados
        sim: Array o Serie de valores simulados

    Returns:
        float: Valor de PBIAS (%)
    """
    obs = np.array(obs)
    sim = np.array(sim)

    mask = ~(np.isnan(obs) | np.isnan(sim) | np.isinf(obs) | np.isinf(sim))
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) < 1:
        return np.nan

    sum_obs = np.sum(obs)
    if sum_obs == 0:
        return np.nan

    pbias_value = 100 * np.sum(obs - sim) / sum_obs

    return pbias_value