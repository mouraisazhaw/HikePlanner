def calculate_calories_burned(uphill, downhill, length_3d, moving_time):
    """
    Verbesserte Berechnung des Kalorienverbrauchs basierend auf Höhenmetern aufwärts und abwärts,
    der 3D-Länge der Route und der Bewegungszeit.

    :param uphill: Höhenmeter aufwärts
    :param downhill: Höhenmeter abwärts
    :param length_3d: Gesamtlänge der Route in 3D
    :param moving_time: Bewegungszeit in Sekunden
    :return: Ungefährer Kalorienverbrauch in Kalorien
    """
    # Kalorienverbrauchsfaktoren
    base_calories_per_hour = 300  # Grundverbrauch pro Stunde, abhängig von der Aktivität
    uphill_factor = 0.2  # Zusätzliche Kalorien pro Meter Aufstieg
    downhill_factor = 0.1  # Zusätzliche Kalorien pro Meter Abstieg
    distance_factor = 0.03  # Kalorien pro Meter basierend auf der 3D-Distanz

    # Umrechnen der Bewegungszeit von Sekunden in Stunden
    moving_time_hours = moving_time / 3600

    # Kalorienverbrauch berechnen
    calories_burned = (base_calories_per_hour * moving_time_hours) + \
                      (uphill * uphill_factor) + \
                      (downhill * downhill_factor) + \
                      (length_3d * distance_factor)

    return calories_burned
