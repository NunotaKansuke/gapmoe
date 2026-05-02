import numpy as np

def get_time_list():
    time_list = [[],[]]
    # values from lcbinint.c
    time_list[0].append(1546.70833) #2000/01/03 05:00:00.0  (UT)
    time_list[0].append(1913.87500) #2001/01/04 09:00:00.0  (UT)
    time_list[0].append(2277.08333) #2002/01/02 14:00:00.0  (UT)
    time_list[0].append(2643.70833) #2003/01/04 05:00:00.0  (UT)
    time_list[0].append(3009.25000) #2004/01/04 18:00:00.0  (UT)
    time_list[0].append(3372.54167) #2005/01/02 01:00:00.0  (UT)
    time_list[0].append(3740.16667) #2006/01/04 16:00:00.0  (UT)
    time_list[0].append(4104.33333) #2007/01/03 20:00:00.0  (UT)
    time_list[0].append(4468.50000) #2008/01/03 00:00:00.0  (UT)
    time_list[0].append(4836.12500) #2009/01/04 15:00:00.0  (UT)
    time_list[0].append(5199.50000) #2010/01/03 00:00:00.0  (UT)
    time_list[0].append(5565.29167) #2011/01/03 19:00:00.0  (UT)
    time_list[0].append(5931.54167) #2012/01/05 01:00:00.0  (UT)
    time_list[0].append(6294.70833) #2013/01/02 05:00:00.0  (UT)
    time_list[0].append(6662.00000) #2014/01/04 12:00:00.0  (UT)
    time_list[0].append(7026.79167) #2015/01/04 07:00:00.0  (UT)
    time_list[0].append(7390.45833) #2016/01/02 23:00:00.0  (UT)
    time_list[0].append(7758.08333) #2017/01/04 14:00:00.0  (UT)
    time_list[0].append(8121.75000) #2018/01/03 06:00:00.0  (UT)
    time_list[0].append(8486.70833) #2019/01/03 05:00:00.0  (UT)
    time_list[0].append(8853.83333) #2020/01/05 08:00:00.0  (UT)
    
    time_list[1].append(1623.81597) #2000/03/20 07:35:00.0  (UT)
    time_list[1].append(1989.06319) #2001/03/20 13:31:00.0  (UT)
    time_list[1].append(2354.30278) #2002/03/20 19:16:00.0  (UT)
    time_list[1].append(2719.54167) #2003/03/21 01:00:00.0  (UT)
    time_list[1].append(3084.78403) #2004/03/20 06:49:00.0  (UT)
    time_list[1].append(3450.02292) #2005/03/20 12:33:00.0  (UT)
    time_list[1].append(3815.26806) #2006/03/20 18:26:00.0  (UT)
    time_list[1].append(4180.50486) #2007/03/21 00:07:00.0  (UT)
    time_list[1].append(4545.74167) #2008/03/20 05:48:00.0  (UT)
    time_list[1].append(4910.98889) #2009/03/20 11:44:00.0  (UT)
    time_list[1].append(5276.23056) #2010/03/20 17:32:00.0  (UT)
    time_list[1].append(5641.47292) #2011/03/20 23:21:00.0  (UT)
    time_list[1].append(6006.71806) #2012/03/20 05:14:00.0  (UT)
    time_list[1].append(6371.95972) #2013/03/20 11:02:00.0  (UT)
    time_list[1].append(6737.20625) #2014/03/20 16:57:00.0  (UT)
    time_list[1].append(7102.44792) #2015/03/20 22:45:00.0  (UT)
    time_list[1].append(7467.68750) #2016/03/20 04:30:00.0  (UT)
    time_list[1].append(7832.93611) #2017/03/20 10:28:00.0  (UT)
    time_list[1].append(8198.17708) #2018/03/20 16:15:00.0  (UT)
    time_list[1].append(8563.41528) #2019/03/20 21:58:00.0  (UT)
    time_list[1].append(8928.65903) #2020/03/20 03:49:00.0  (UT)
  
    return time_list

def get_peri_vernal(tref):
    if tref > 2450000:
        tref -= 2450000

    time_list = get_time_list()
    ind = np.argmin(np.abs(np.array(time_list[0]) - tref))

    return 2450000 + time_list[0][ind], 2450000 + time_list[1][ind]

def solve_kep_eq(M, e = 0.0167):
    """ 
    Solve Kepler equation by Newton method 
    to calculate the eccentric anomly for a given mean anomaly. 
    Arguments: 
        M: mean anomaly
        e: eccentricity (default: 0.0167).
    Returns:
        E: eccentric anomaly
    """
    E = M #initial guess
    func = E - e * np.sin(E) - M

    while np.abs(func) > 1e-8:
        func = E - e * np.sin(E) - M
        df = 1.0 - e * np.cos(E)
        E -=  func / df #Eq.(A.54)

    return E

def get_north_east(RA, Dec):
    """ 
    Calculate the North and East unit vectors 
    in the event sky plane based on Eqs. A.54, A.55, A.56

    Arguments:
        RA: Right Ascension of the event in degrees
        Dec: Declination of the event in degrees

    Returns:
        sky_north: The unit vector pointing North in the event sky plane
        sky_east: The unit vector pointing East in the event sky plane
    """

    lambda0  = RA * np.pi / 180
    beta0  = Dec * np.pi / 180

    earth_north = np.array([0.0, 0.0, 1.0])
    event = np.array([np.cos(lambda0) * np.cos(beta0),
                       np.sin(lambda0) * np.cos(beta0),
                       np.sin(beta0)]) #Eq. A.54

    sky_east = np.cross(earth_north, event) #Eq. A.55
    sky_east /= np.sqrt(np.dot(sky_east, sky_east)) #Eq. A.55
    sky_north = np.cross(event, sky_east) #Eq. A.56

    return sky_north, sky_east

def eclip_frame_unit_vector(t_peri, t_vernal, theta = 23.44, 
                    ecc = 0.0167 ,period = 365.25636):
    """
    Calculate the unit vectors of the ecliptic coordinate frame (x, y) 
    in the equatorial frame using Eqs. A.59, and A.60
    
 (optional): The orbital eccentricity of Earth (default is 0.0167)
        period (optional): The orbital period of Earth in days (default is 365.25636)
    
    Returns:
        x_ec: The unit vector of the x-axis of the ecliptic frame in the equatorial frame (x, y, z)
        y_ec: The unit vector of the y-axis of the ecliptic frame in the equatorial frame (x, y, z)
    """
    theta = theta * np.pi / 180
    V_spring = np.array([1.0, 0.0, 0.0])
    V_summer = np.array([0.0, np.cos(theta), np.sin(theta)])
    
    M_ver = 2 * np.pi * (t_vernal - t_peri) / period
    E_ver = solve_kep_eq(M_ver)
    cosf_ver = (np.cos(E_ver) - ecc) / (1 - ecc * np.cos(E_ver))
    sinf_ver = np.sqrt(1 - cosf_ver**2)
    x_ec = V_spring * cosf_ver - V_summer * sinf_ver
    y_ec = V_spring * sinf_ver + V_summer * cosf_ver

    return x_ec, y_ec

def get_sun_proj(time, peri, x_ec, y_ec, 
           sky_north, sky_east, ecc = 0.0167, period = 365.25636):
    """ 
    Calculate the North and East components 
    of the Sun's position vector
    projected onto the event sky plane using Eq. A.61, A.62.

    Arguments:
        time: The current time (in Julian Date or an appropriate time unit)
        peri: The time of perihelion passage (in the same time unit as `time`)
        x_ec: The unit vector of the x-axis of the ecliptic frame 
              in the equatorial frame (x, y, z)
        y_ec: The unit vector of the y-axis of the ecliptic frame 
              in the equatorial frame (x, y, z)
        sky_north: The unit vector pointing North in the event sky plane
        east: The unit vector pointing East in the event sky plane
        ecc (optional): The orbital eccentricity (default is 0.0167)
        period (optional): The orbital period of Earth in days 
                           (default is 365.25636)

    Returns:
        np.array: The North and East components of 
                  the Sun's position vector in the event sky plane 
                  as a 2-element array: [se, sn].
    """
    M = 2 * np.pi * (time - peri) / period
    E = solve_kep_eq(M) 
    x_components = - (np.cos(E) - ecc) #Eq. A.61
    y_components = - (np.sin(E) * np.sqrt(1.0 - ecc**2)) #Eq. A.61
    sun = x_ec * x_components + y_ec * y_components #Eq. A.61
    sE, sN = np.dot(sun, sky_north), np.dot(sun, sky_east) #Eq. 62

    return np.array([sE, sN])

def get_sun_velocity(time, peri, x_ec, y_ec,
           sky_north, sky_east, dt = 0.1):
    """ 
    Calculate the velocity of the Sun's position projected onto the event sky plane.

    Arguments:
        time: The current time (in Julian Date or an appropriate time unit)
        peri: The time of perihelion passage (in the same time unit as `time`)
        x_ec: The unit vector of the x-axis of the ecliptic frame in the equatorial frame (x, y, z)
        y_ec: The unit vector of the y-axis of the ecliptic frame in the equatorial frame (x, y, z)
        sky_north: The unit vector pointing North in the event sky plane
        sky_east: The unit vector pointing East in the event sky plane
        dt (optional): The time step for numerical differentiation (default is 0.1)

    Returns:
        np.array: The velocity of the Sun's position projected onto 
        the event sky plane as a 2-element array representing 
        the components in the North and East directions: [v_north, v_east].
    """
    sun_minus = get_sun_proj(
            time-dt, peri,
            x_ec, y_ec, 
            sky_north, sky_east)
    sun_plus = get_sun_proj(
            time+dt, peri,
            x_ec, y_ec, 
            sky_north, sky_east)

    sun_velocity = 0.5 * (sun_plus - sun_minus) / dt
 
    return sun_velocity

def calc_vSun(tref, RA, Dec):
    """ 
    Calculate the annual parallax effect in the tau and beta directions.

    Arguments:
        tref: The reference time (in Julian Date or an appropriate time unit).
        RA: The Right Ascension of the event in degrees.
        Dec: The Declination of the event in degrees.

    Returns:
        np.array: The velocity of the Earth's position projected onto 
        the event sky plane as a 2-element array representing 
        the components in the North and East directions: [v_north, v_east] AU / year.
    """
    t_peri, t_vernal = get_peri_vernal(tref)
    x_ec, y_ec = eclip_frame_unit_vector(t_peri, t_vernal)
    sky_north, sky_east = get_north_east(RA, Dec)
    dt = 0.1

    sun_minus = get_sun_proj(
            tref-dt, t_peri,
            x_ec, y_ec, 
            sky_north, sky_east)
    sun_plus = get_sun_proj(
            tref+dt, t_peri,
            x_ec, y_ec, 
            sky_north, sky_east)

    sun_velocity = 0.5 * (sun_plus - sun_minus) / dt
    return sun_velocity * 365.25

def hms_string_to_degrees(hms_string):
    h, m, s = map(float, hms_string.split(":"))
    hours = h + m / 60 + s / 3600
    degrees = hours * 360 / 24
    return degrees

def dms_string_to_degrees(dms_string):
    sign = -1 if dms_string.startswith("-") else 1
    dms_string = dms_string.lstrip("+-")
    d, m, s = map(float, dms_string.split(":"))
    degrees = d + m / 60 + s / 3600
    return sign * degrees
