def calculate_temperature_cost(temperature_c, baseline_temp=40.0, temp_factor=0.01):
    if temperature_c is None:
        return 0.0

    extra_temp = max(0.0, temperature_c - baseline_temp)
    return extra_temp * temp_factor


def calculate_gcf(
    power_watt,
    training_time_seconds,
    memory_percent,
    temperature_c=None,
    carbon_factor=0.000417,
    memory_factor=0.001
):
    """
    Simplified Green Carbon Footprint score.

    Formula idea:
    GCF = (P × T × Cf) + (M × Mf) + Tc

    P  = power watt
    T  = training time in hours
    Cf = carbon factor
    M  = memory percentage
    Mf = memory factor
    Tc = temperature cost
    """

    training_time_hours = training_time_seconds / 3600

    energy_carbon_cost = power_watt * training_time_hours * carbon_factor
    memory_cost = memory_percent * memory_factor
    temperature_cost = calculate_temperature_cost(temperature_c)

    gcf_score = energy_carbon_cost + memory_cost + temperature_cost

    return gcf_score