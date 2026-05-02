import numpy as np

def calc_vEarth(t_jd, ra_deg, dec_deg):
    # === JPL要素（C++と同じ） ===
    a0 = 1.00000261
    adot = 0.00000562
    e0 = 0.01671123
    edot = -0.00004392
    inc0 = -0.00001531
    incdot = -0.01294668
    L0 = 100.46457166
    Ldot = 35999.37244981
    om0 = 102.93768193
    omdot = 0.32327364
    eps0 = 23.439281
    deg = np.pi / 180

    # === 時刻補正（ユリウス世紀） ===
    if t_jd < 2450000:
        t_jd += 2450000
    T = (t_jd - 2451545.0) / 36525.0

    # === 軌道要素 ===
    a = a0 + adot * T
    e = e0 + edot * T
    inc = (inc0 + incdot * T) * deg
    L = (L0 + Ldot * T) * deg
    om = (om0 + omdot * T) * deg
    eps = eps0 * deg
    M = L - om
    M %= 2 * np.pi

    # === Kepler方程式を解く ===
    E = M  # 初期推定
    for _ in range(100):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if abs(dE) < 1e-10:
            break

    # === 地球の位置（AU）と速度（AU/day）===
    cosE, sinE = np.cos(E), np.sin(E)
    r = a * (1 - e * cosE)
    x = a * (cosE - e)
    y = a * np.sqrt(1 - e**2) * sinE

    # === 公転速度（平面上）===
    n = Ldot * deg / 36525  # 平均運動 [rad/day]
    vx = -a * sinE * n / (1 - e * cosE)
    vy = a * np.sqrt(1 - e**2) * cosE * n / (1 - e * cosE)

    # === 3次元変換（傾斜・近日点） ===
    cos_om = np.cos(om)
    sin_om = np.sin(om)
    cos_inc = np.cos(inc)
    sin_inc = np.sin(inc)

    # 地球の位置（不要だが残す）
    ex = x * cos_om - y * sin_om
    ey = x * sin_om * cos_inc + y * cos_om * cos_inc
    ez = x * sin_om * sin_inc + y * cos_om * sin_inc

    # 地球の速度ベクトル
    vx3 = vx * cos_om - vy * sin_om
    vy3 = vx * sin_om * cos_inc + vy * cos_om * cos_inc
    vz3 = vx * sin_om * sin_inc + vy * cos_om * sin_inc

    # 地球から見た太陽の速度（逆向き）
    vsun_ec = -np.array([vx3, vy3, vz3]) # 黄道座標系 (ecliptic)
    
    R = np.array([
    [1, 0, 0],
    [0,  np.cos(eps), -np.sin(eps)],
    [0,  np.sin(eps),  np.cos(eps)]
    ])
    vsun_eq = R @ vsun_ec

    # === RA/Dec方向から LOS・東・北ベクトルを構成 ===
    ra = ra_deg * deg
    dec = dec_deg * deg
    
    earth_north = np.array([0,0,1])

    los = np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ])
    
    east = np.cross(earth_north, los)
    east /= np.sqrt(np.dot(east, east))
    north = np.cross(los, east)            

    # === 投影 ===
    v_north = np.dot(vsun_eq, north)
    v_east  = np.dot(vsun_eq, east)

    return v_north * 365.25, v_east*365.25 # AU / year

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
