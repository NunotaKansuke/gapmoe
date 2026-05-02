#!/opt/local/bin/perl
# convert mu_hel (b,l) -> mu_geo (N,E)
# made by N.Koshimoto of 2018 01 02
# ref: NASA note p.137-138, parallax in lcbinint.c

unless(@ARGV>=1){
  print "usage:\n";
  print "\tperl ../Tool/mu_Hlb2mu_Gne.pl -muhelEN -2.53 8.28 -Dl 2.00 -Ds 8.0 -t0 6573.05 -RADec 17:58:42.85 -29:23:53.66 -tE 19.96\n"; # Sumi+16 Table13 P-+Kpkg model -> confirm!
  print "\tperl ../Tool/mu_Hlb2mu_Gne.pl -muhelEN 5.63 4.87 -Dl 4.0 -Ds 8.5 -t0 3491.877 -RADec 18:06:05.32 -30:43:57.5 -tE 41.8\n"; # Batista+15 -> unconfirm,,, 
  exit;
}
for ($i=0;$i<@ARGV;$i++){
  $gl          = $ARGV[$i+1] if $ARGV[$i] eq "-lb"; # Galactic l 
  $gb          = $ARGV[$i+2] if $ARGV[$i] eq "-lb"; # Galactic b
  $RA          = $ARGV[$i+1] if $ARGV[$i] eq "-RADec"; # Equatorial alpha
  $Dec         = $ARGV[$i+2] if $ARGV[$i] eq "-RADec"; # Equatorial delta
  $muhell      = $ARGV[$i+1] if $ARGV[$i] eq "-muhellb"; # mu_{rel,hel,l}
  $muhelb      = $ARGV[$i+2] if $ARGV[$i] eq "-muhellb"; # mu_{rel,hel,b}
  $muhelE      = $ARGV[$i+1] if $ARGV[$i] eq "-muhelEN"; # mu_{rel,hel,l}
  $muhelN      = $ARGV[$i+2] if $ARGV[$i] eq "-muhelEN"; # mu_{rel,hel,b}
  $Dl          = $ARGV[$i+1] if $ARGV[$i] eq "-Dl"; # Dl for pi_rel
  $Ds          = $ARGV[$i+1] if $ARGV[$i] eq "-Ds"; # Ds for pi_rel
  $pirel       = $ARGV[$i+1] if $ARGV[$i] eq "-pirel"; # pirel
  $t0          = $ARGV[$i+1] if $ARGV[$i] eq "-t0"; # HJD - 2450000
  $tE          = $ARGV[$i+1] if $ARGV[$i] eq "-tE"; # 
  $useNE       = 1           if $ARGV[$i] eq "-useNE"; # 
  $NOPG       = 1           if $ARGV[$i] eq "-NOPG"; # 
}
$info = `pwd`;
$path = "/Users/koshimoto/work" if $info =~ /Users/;
$path = "/moao29_4/koshimoa/binfit/work" if $info =~ /koshimoa/;
($gl, $gb) = eq2ga($RA,$Dec) if ($RA && $Dec);
unless ($gl && $gb){
  print "Need (l,b) to project onto the sky!\n";
  print "Use -lb option\n";
	exit;
}
printf "gal_l= %9.4f\n",$gl;
printf "gal_b= %9.4f\n",$gb;
$pirel = 1/$Dl - 1/$Ds if $Ds && $Dl && ! $pirel; # mas = AU/kpc
#----From http://ned.ipac.caltech.edu/forms/calculator.html --------
#  Input parameters:
#
#  System: Equatorial Equinox: J2000.0
#  Observation epoch: 2016  # I don't know what this is
#  RA or Longitude: 0
#  DEC or Latitude: 90
#  PA (East of North): 0.0
#  Output Parameters:
#
#  System: Galactic Equinox: J2000.0
#
#  Input: Equatorial J2000.0
#
#  RA or Longitude   DEC or Latitude   PA(East of North)
#  0.00000000        90.00000000       0.000000     
#  00h00m00.00000s   +90d00m00.0000s
#
#  Output: Galactic 
#  122.93202404      27.12845461       12.859493   
#################################################### 
# 銀河座標で、地球を原点、銀河中心方向にx軸、銀河北極にz軸、右手座標になるようにy軸(銀河回転方向になる)をとる
$PI = atan2(1,1)*4;
$lNP = 122.9320; # galactic l of equatorial north pole
$bNP =  27.1284; # galactic b of equatorial north pole
$cosl   = cos($gl/180 * $PI);
$sinl   = sin($gl/180 * $PI);
$cosb   = cos($gb/180 * $PI);
$sinb   = sin($gb/180 * $PI);
$coslNP = cos($lNP/180 * $PI);
$sinlNP = sin($lNP/180 * $PI);
$cosbNP = cos($bNP/180 * $PI);
$sinbNP = sin($bNP/180 * $PI);
@nvector = ($cosb*$cosl,$cosb*$sinl,$sinb); # unit vector along line of sight 
@galNP   = (0,0,1);                         # unit vector toward galactic north pole
@eqNP = ($cosbNP*$coslNP,$cosbNP*$sinlNP,$sinbNP); # unit vector toward equatorial north pole
printf "   n = (%7.4f, %7.4f, %7.4f)\n",$nvector[0],$nvector[1],$nvector[2];
printf "eqNP = (%7.4f, %7.4f, %7.4f)\n",$eqNP[0],$eqNP[1],$eqNP[2];

# Make Unit Vectors along (b, l) and along (N, E) on the target sky
@elvector = cross(\@galNP,\@nvector); # p x n
$magtmp = dot(\@elvector,\@elvector);
map {$_ /= sqrt($magtmp)} @elvector; # normalize
@ebvector = cross(\@nvector,\@elvector); # eb = n x el
printf "  el = (%7.4f, %7.4f, %7.4f), mag= %.5f\n",$elvector[0],$elvector[1],$elvector[2],$elvector[0]**2+$elvector[1]**2+$elvector[2]**2;
#printf "el = (%7.4f, %7.4f, %7.4f)\n",-$sinl,$cosl,0;
printf "  eb = (%7.4f, %7.4f, %7.4f), mag= %.5f\n",$ebvector[0],$ebvector[1],$ebvector[2],$ebvector[0]**2+$ebvector[1]**2+$ebvector[2]**2;
#printf "eb = (%7.4f, %7.4f, %7.4f)\n",-$cosl*$sinb,-$sinl*$sinb,$cosb;

@eEvector = cross(\@eqNP,\@nvector); # p x n
$magtmp = dot(\@eEvector,\@eEvector); 
map {$_ /= sqrt($magtmp)} @eEvector; # normalize
@eNvector = cross(\@nvector,\@eEvector); # eN = n x eE
printf "  eE = (%7.4f, %7.4f, %7.4f), mag= %.5f\n",$eEvector[0],$eEvector[1],$eEvector[2],$eEvector[0]**2+$eEvector[1]**2+$eEvector[2]**2;
printf "  eN = (%7.4f, %7.4f, %7.4f), mag= %.5f\n",$eNvector[0],$eNvector[1],$eNvector[2],$eNvector[0]**2+$eNvector[1]**2+$eNvector[2]**2;

# Calc -PA (negative position angle, from l to E or from b to N. A positive PA should be defined from N to b, maybe
$cosPA = dot(\@elvector,\@eEvector);
@crosstmp = cross(\@elvector,\@eEvector);
$sinPA = dot(\@nvector,\@crosstmp);
$sinPA *= -1;
$PA = atan2($sinPA,$cosPA); # atan2(Y,X) -PA, radian, l to E
printf "-PA (from b to N) = %9.4f deg.\n",180*$PA/$PI;
printf "cos(-PA) = %.7f\n",$cosPA;
printf "sin(-PA) = %.7f\n",$sinPA;

if ($muhell && $muhelb){ # muhel(l,b) -> muhel(N,E)
  $muhelN =  $cosPA*$muhelb + $sinPA*$muhell;
  $muhelE = -$sinPA*$muhelb + $cosPA*$muhell;
	$muhel = sqrt($muhelb**2+$muhell**2);
	# $muhelb =  $cosPA*$muhelN - $sinPA*$muhelE; # inverse 
	# $muhell =  $sinPA*$muhelN + $cosPA*$muhelE; # 
}elsif($muhelE && $muhelN){
	$muhelb =  $cosPA*$muhelN - $sinPA*$muhelE; # inverse 
	$muhell =  $sinPA*$muhelN + $cosPA*$muhelE; # 
	$muhel = sqrt($muhelb**2+$muhell**2);
}
printf "mu_hel(b,l)= (%8.4f, %8.4f), mu_hel= %.5f\n",$muhelb,$muhell,sqrt($muhelb**2+$muhell**2);
printf "mu_hel(N,E)= (%8.4f, %8.4f), mu_hel= %.5f\n",$muhelN,$muhelE,sqrt($muhelN**2+$muhelE**2);

if ($t0){ # calc v_Earth(N,E)
	$EarthAxis = 23.44; # 84381.406 arcsec at 2000
	$AU2km = 149597870.7;
	$ecc = 0.0167;
	$period = 365.25636;  #恒星年
  ($ra1,$ra2,$ra3) = split(":",$RA);
  ($dec1,$dec2,$dec3) = split(":",$Dec);
  $alpha  = 15* (abs($ra1) + $ra2/60 + $ra3/3600)*$ra1/abs($ra1); #degree
  $delta  = (abs($dec1) + $dec2/60 + $dec3/3600)*$dec1/abs($dec1);
	$t0 -= 2450000 if $t0 > 2450000;
  ($yr,$peri,$vernal) = peri_vernal($t0);
	print "t0=$t0 year= $yr, perihelion= $peri, vernal= $vernal\n";
  # 赤道座標系で、地球を原点、春分点にx軸、天の北極にz軸、右手座標になるようにy軸(夏の方向になる)をとる
	$spring[0]=1.0; # 春分点のx
  $spring[1]=0.0; # y
  $spring[2]=0.0; # z
  $summer[0]=0.0; # 夏至点のx
  $summer[1]=cos($EarthAxis/180 * $PI); # y 
  $summer[2]=sin($EarthAxis/180 * $PI); # z 
  $offset = $vernal - $peri; #近日点->春分点の時刻差. periとvernalはperihelion_vernalで2020年までは得られる
  $phi = (1.0 - $offset/$period)*2.0*$PI; # 春分点(今のx軸)->近日点のmean anomaly M
  $psi = getpsi($phi,$ecc);  # phiに対応するeccentric anomaly Eを求める
  $costh = (cos($psi) - $ecc)/(1-$ecc*cos($psi)); # Eからcos T (Tはtrue anomaly, 焦点から見た近点からの角度)
  $sinth = -sqrt(1-$costh*$costh);  # 春分点 -> 近日点の角度は > pi だからsinはマイナス
  for ($i=0;$i<=2;$i++){
    $xpos[$i] = $spring[$i]*$costh + $summer[$i]*$sinth; # 春分点x, 夏至点yの座標系(黄道座標系)を、角度thだけ回転(つまりxposは近点の方向)
    $ypos[$i] =-$spring[$i]*$sinth + $summer[$i]*$costh; # yposは近点から90度進んだ方向(黄道面に沿って)
  }
  $north[0] = 0.0;  # 赤道座標でのNorth pole x
  $north[1] = 0.0;  # y
  $north[2] = 1.0;  # z
  $rad[0] = cos($alpha/180 * $PI)*cos($delta/180 * $PI);  # イベントの位置 x
  $rad[1] = sin($alpha/180 * $PI)*cos($delta/180 * $PI);  # y 
  $rad[2] = sin($delta/180 * $PI);                        # z
  @east = cross(\@north,\@rad); # east = NP x rad, rad(イベントの)方向のsky上の, east方向
  $magtmp = dot(\@east,\@east);
  map {$_ /= sqrt($magtmp)} @east; # normalize
  @north = cross(\@rad,\@east); # north = rad x east
  # Calc position at t0 + dt
	$dt = 0.01;
	$phi = ($t0+$dt - $peri)/$period*2.0*$PI;  # 時刻t0 + 1での地球(太陽)のmean anomaly M
  $psi = getpsi($phi,$ecc);  # t0時刻のeccentric anomaly E
  $qn2 = 0.0;
  $qe2 = 0.0;
  for ($i=0;$i<=2;$i++){
     $sun[$i] = $xpos[$i]*(cos($psi)-$ecc) +  # SSDの式(2.41), 太陽の半径1の球内での位置. (球面上ではなく、楕円を考えてる)
              $ypos[$i]*sin($psi)*sqrt(1.0-$ecc*$ecc); 
     $qn2 = $qn2 + $sun[$i]*$north[$i];  # 太陽の位置ベクトルのイベント方向のskyのnorth成分
     $qe2 = $qe2 + $sun[$i]*$east[$i];   # おなじくeast成分
  }
  # Calc position at t0 - dt
	$phi = ($t0-$dt - $peri)/$period*2.0*$PI;  # 時刻t0 + 1での地球(太陽)のmean anomaly M
  $psi = getpsi($phi,$ecc);  # t0時刻のeccentric anomaly E
  $qn1 = 0.0;
  $qe1 = 0.0;
  for ($i=0;$i<=2;$i++){
     $sun[$i] = $xpos[$i]*(cos($psi)-$ecc) +  # SSDの式(2.41), 太陽の半径1の球内での位置. (球面上ではなく、楕円を考えてる)
              $ypos[$i]*sin($psi)*sqrt(1.0-$ecc*$ecc); 
     $qn1 = $qn1 + $sun[$i]*$north[$i];  # 太陽の位置ベクトルのイベント方向のskyのnorth成分
     $qe1 = $qe1 + $sun[$i]*$east[$i];   # おなじくeast成分
  }
  $vn0 = -($qn2-$qn1)/2/$dt;  # 0.2日間での太陽の位置ベクトルの射影の差を0.2日で割って、1日あたりの速度にしてる
  $ve0 = -($qe2-$qe1)/2/$dt;  # AU / day  地球の速度は太陽の-1倍
  $vn0km = $vn0*$AU2km/3600/24;
	$ve0km = $ve0*$AU2km/3600/24;
	printf "v_Earth(N,E) [km/s]  = ( %8.4f , %8.4f ), v_Earth= %.5f\n",$vn0km,$ve0km,sqrt($vn0km**2+$ve0km**2);
  $vb0km = $cosPA*$vn0km - $sinPA*$ve0km;
  $vl0km = $sinPA*$vn0km + $cosPA*$ve0km;
	printf "v_Earth(b,l) [km/s]  = ( %8.4f , %8.4f ), v_Earth= %.5f\n",$vb0km,$vl0km,sqrt($vb0km**2+$vl0km**2);
  $vn0 *= 365.25; # AU / year
  $ve0 *= 365.25; # AU / year
  printf "v_Earth(N,E) [AU/yr] = ( %8.4f , %8.4f ), v_Earth= %.5f\n",$vn0,$ve0,sqrt($vn0**2+$ve0**2);
  $vb0 = $cosPA*$vn0 - $sinPA*$ve0;
  $vl0 = $sinPA*$vn0 + $cosPA*$ve0;
	 printf "v_Earth(b,l) [AU/yr] = ( %8.4f , %8.4f ), v_Earth= %.5f\n",$vb0,$vl0,sqrt($vb0**2+$vl0**2);
}
if ($muhell && $muhelb && $t0 && $pirel){ # determine mugeo(N,E)
  $mugeoN = $muhelN - $vn0*$pirel;
  $mugeoE = $muhelE - $ve0*$pirel;
  $mugeob = $cosPA*$mugeoN - $sinPA*$mugeoE;
  $mugeol = $sinPA*$mugeoN + $cosPA*$mugeoE;
	printf "pirel [mas] = %.3f\n", $pirel;
  printf "mu_geo(N,E) [mas/yr] = (%7.4f, %7.4f), mu_geo= %.5f\n",$mugeoN,$mugeoE,sqrt($mugeoN**2+$mugeoE**2);
	$vtildhell = $muhell/$pirel * $AU2km / 365.25 / 24 / 3600; # mas/yr -> km/sec
	$vtildhelb = $muhelb/$pirel * $AU2km / 365.25 / 24 / 3600; # mas/yr -> km/sec
  $vtildhelN =  $cosPA*$vtildhelb + $sinPA*$vtildhell;  # b,l -> N,E
  $vtildhelE = -$sinPA*$vtildhelb + $cosPA*$vtildhell;  # b,l -> N,E
	$vtildgeol = $mugeol/$pirel * $AU2km / 365.25 / 24 / 3600; # mas/yr -> km/sec
	$vtildgeob = $mugeob/$pirel * $AU2km / 365.25 / 24 / 3600; # mas/yr -> km/sec
  $vtildgeoN =  $cosPA*$vtildgeob + $sinPA*$vtildgeol;  # b,l -> N,E
  $vtildgeoE = -$sinPA*$vtildgeob + $cosPA*$vtildgeol;  # b,l -> N,E
  printf "v_tild,hel(N,E) [km/s] = (%6.1f, %6.1f), vtild,hel= %.1f\n",$vtildhelN,$vtildhelE,sqrt($vtildhelN**2+$vtildhelE**2);
  printf "v_tild,hel(b,l) [km/s] = (%6.1f, %6.1f), vtild,hel= %.1f\n",$vtildhelb,$vtildhell,sqrt($vtildhelb**2+$vtildhell**2);
  printf "v_tild,geo(N,E) [km/s] = (%6.1f, %6.1f), vtild,geo= %.1f\n",$vtildgeoN,$vtildgeoE,sqrt($vtildgeoN**2+$vtildgeoE**2);
  printf "v_tild,geo(b,l) [km/s] = (%6.1f, %6.1f), vtild,geo= %.1f\n",$vtildgeob,$vtildgeol,sqrt($vtildgeob**2+$vtildgeol**2);
  $vtildhel = sqrt($vtildhelN**2+$vtildhelE**2);
}
if ($tE && $mugeoN && $mugeoE && $pirel){ # determine mugeo(N,E)
  $mugeo = sqrt($mugeoN**2 + $mugeoE**2); 
  $thetaE = $mugeo * $tE / 365.25; #mas
  $piE = $pirel/$thetaE; # mas/mas
	$piEN = $piE* $mugeoN/$mugeo;
	$piEE = $piE* $mugeoE/$mugeo;
	printf "(piEN, piEE) = (%.4f, %.4f), piE = %.4f\n",$piEN,$piEE,$piE;
}


########################################################
sub dot{
  local (*a, *b) = @_;
  return "OutOfRange" unless @a == 3;
  return "OutOfRange" unless @b == 3;
  return ($a[0]*$b[0]+$a[1]*$b[1]+$a[2]*$b[2]);
}
##################################################################
sub cross{
  local (*a, *b) = @_;
  return "OutOfRange" unless @a == 3;
  return "OutOfRange" unless @b == 3;
  return ($a[1]*$b[2]-$a[2]*$b[1], $a[2]*$b[0]-$a[0]*$b[2], $a[0]*$b[1]-$a[1]*$b[0]);
}
##################################################################
sub peri_vernal{
  local $tfix = $_[0];
  my @peri; my @vernal;
	# values from lcbinint.c
  $peri[0]   = 1546.70833; #2000/01/03 05:00:00.0  (UT)
  $peri[1]   = 1913.87500; #2001/01/04 09:00:00.0  (UT)
  $peri[2]   = 2277.08333; #2002/01/02 14:00:00.0  (UT)
  $peri[3]   = 2643.70833; #2003/01/04 05:00:00.0  (UT)
  $peri[4]   = 3009.25000; #2004/01/04 18:00:00.0  (UT)
  $peri[5]   = 3372.54167; #2005/01/02 01:00:00.0  (UT)
  $peri[6]   = 3740.16667; #2006/01/04 16:00:00.0  (UT)
  $peri[7]   = 4104.33333; #2007/01/03 20:00:00.0  (UT)
  $peri[8]   = 4468.50000; #2008/01/03 00:00:00.0  (UT)
  $peri[9]   = 4836.12500; #2009/01/04 15:00:00.0  (UT)
  $peri[10]  = 5199.50000; #2010/01/03 00:00:00.0  (UT)
  $peri[11]  = 5565.29167; #2011/01/03 19:00:00.0  (UT)
  $peri[12]  = 5931.54167; #2012/01/05 01:00:00.0  (UT)
  $peri[13]  = 6294.70833; #2013/01/02 05:00:00.0  (UT)
  $peri[14]  = 6662.00000; #2014/01/04 12:00:00.0  (UT)
  $peri[15]  = 7026.79167; #2015/01/04 07:00:00.0  (UT)
  $peri[16]  = 7390.45833; #2016/01/02 23:00:00.0  (UT)
  $peri[17]  = 7758.08333; #2017/01/04 14:00:00.0  (UT)
  $peri[18]  = 8121.75000; #2018/01/03 06:00:00.0  (UT)
  $peri[19]  = 8486.70833; #2019/01/03 05:00:00.0  (UT)
  $peri[20]  = 8853.83333; #2020/01/05 08:00:00.0  (UT)

  $vernal[0]  = 1623.81597; #2000/03/20 07:35:00.0  (UT)
  $vernal[1]  = 1989.06319; #2001/03/20 13:31:00.0  (UT)
  $vernal[2]  = 2354.30278; #2002/03/20 19:16:00.0  (UT)
  $vernal[3]  = 2719.54167; #2003/03/21 01:00:00.0  (UT)
  $vernal[4]  = 3084.78403; #2004/03/20 06:49:00.0  (UT)
  $vernal[5]  = 3450.02292; #2005/03/20 12:33:00.0  (UT)
  $vernal[6]  = 3815.26806; #2006/03/20 18:26:00.0  (UT)
  $vernal[7]  = 4180.50486; #2007/03/21 00:07:00.0  (UT)
  $vernal[8]  = 4545.74167; #2008/03/20 05:48:00.0  (UT)
  $vernal[9]  = 4910.98889; #2009/03/20 11:44:00.0  (UT)
  $vernal[10] = 5276.23056; #2010/03/20 17:32:00.0  (UT)
  $vernal[11] = 5641.47292; #2011/03/20 23:21:00.0  (UT)
  $vernal[12] = 6006.71806; #2012/03/20 05:14:00.0  (UT)
  $vernal[13] = 6371.95972; #2013/03/20 11:02:00.0  (UT)
  $vernal[14] = 6737.20625; #2014/03/20 16:57:00.0  (UT)
  $vernal[15] = 7102.44792; #2015/03/20 22:45:00.0  (UT)
  $vernal[16] = 7467.68750; #2016/03/20 04:30:00.0  (UT)
  $vernal[17] = 7832.93611; #2017/03/20 10:28:00.0  (UT)
  $vernal[18] = 8198.17708; #2018/03/20 16:15:00.0  (UT)
  $vernal[19] = 8563.41528; #2019/03/20 21:58:00.0  (UT)
  $vernal[20] = 8928.65903; #2020/03/20 03:49:00.0  (UT)
  my $dperimin=9999999999;
	my ($year,$perihelion,$vernalEquinox,$signmin);
	for (my $i=0;$i<21;$i++){
	  my $dperi = abs($peri[$i] - $tfix);
		my $sign = ($tfix - $peri[$i] > 0) ? 1 : -1;
		if ($dperi < $dperimin){
       $dperimin      = $dperi;
			 $signmin       = $sign*$dperi;
       $year          = 2000 + $i;
       $perihelion    = $peri[$i];
       $vernalEquinox = $vernal[$i]; 
		}
	}
	printf ("%d days after perihelion (~ 3 Jan)\n",$signmin);
  my $month = ($signmin > 0) ? 1+$signmin/30 : 12 + $signmin/30; 
	printf "-> month: %.2f\n",$month;
	return ($year,$perihelion,$vernalEquinox);
}
##################################################################
sub getpsi{  # Kepler方程式をNewton法で解いてMean anomalyから Eccentric anomalyを計算
				local ($phi, $ecc) = @_;
				my $psi= $phi + abs(sin($phi))/sin($phi) * 0.85 * $ecc; # Solar System Dynamics(SSD)の式(2.64)によると、E_0 = M + sign(sinM)*k*e がbetter. k = 0.85が推奨されている
				for (my $i=1;$i<=4;$i++){  # Newton法は2次収束だから、4回繰り返すと、E = Mとするより16桁くらい精度がよくなる。たぶん
					my $fun = $psi - $ecc*sin($psi); # E_i - e*sin E_i
					my $dif = $phi - $fun; # dif = f (E_i) = M - E_i + e*sinE_i
					my $der = 1.0 - $ecc*cos($psi); # der = -f'(E_i) = 1 - e*cosE_i
					$psi = $psi + $dif/$der;  # E_i+1 = E_i - f(E_i)/f'(E_i) 
				}  
				return $psi;
}  
##################################################################
sub eq2ga{  # ../Tool-osaka/eq2ga.pl からコピペ 
  #
  # convert equatorial cooridinate to galactic coordinate
  #
  #  made by D.Suzuki on 2011 12 19
  #  reference => http://plain.isas.jaxa.jp/~ebisawa/TEACHING/2007Komaba/2007Komaba/node16.html
  #
  #
  my $PI = 3.141592653589793;#23846264338327950288
  local ($ra, $dec) = @_;
  my ($rah, $ram, $ras) = split("\:", $ra);
  $rah = $rah + $ram/60 + $ras/3600;
  my $rad = $rah *15;

  my ($decd, $decm, $decs) = split("\:", $dec);
  $decd = $decd + $decm/60 + $decs/3600 unless  $decd =~ /-/;
  $decd = $decd - $decm/60 - $decs/3600 if      $decd =~ /-/ ;

	$rad  = $rad * ($PI/180);
  $decd = $decd * ($PI/180);

## convert RA Dec to (X, Y, Z) -----------

  my $x_eq = cos($decd) * cos($rad);
  my $y_eq = cos($decd) * sin($rad);
  my $z_eq = sin($decd);

##-----------------------------------------

## matrix
  my $a = -0.0548755; my $b = -0.873437; my $c = -0.483835;
  my $d = 0.49411   ; my $e = -0.44483 ; my $f = 0.746982;
  my $g = -0.867666 ; my $h = -0.198076; my $i = 0.455984;

## convert (X, Y, Z) to (X_ga, Y_ga, Z_ga) --

  my $x_ga = ($a * $x_eq) + ($b * $y_eq) + ($c * $z_eq);
  my $y_ga = ($d * $x_eq) + ($e * $y_eq) + ($f * $z_eq);
  my $z_ga = ($g * $x_eq) + ($h * $y_eq) + ($i * $z_eq);

##-------------------------------------------

## convert (X_ga, Y_ga, Z_ga) to galactic longitude, galactic latitude(l, b) -- 

  my $r_ga = sqrt($x_ga * $x_ga + $y_ga * $y_ga);
  my $frac_yx = $y_ga/$x_ga;
  my $frac_zr = $z_ga/$r_ga;

  
  my $l_ga = atan2($y_ga, $x_ga);
  my $b_ga = atan2($z_ga, $r_ga);

  $l_ga = $l_ga * (180/$PI);
  $b_ga = $b_ga * (180/$PI);
##-----------------------------------------------------------------------------
  return ($l_ga, $b_ga);
}
