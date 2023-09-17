"""
Import modules
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import mastcasjobs
import os
import getpass
import io

from PIL import Image

from astropy import units as u
from astropy.coordinates import SkyCoord, Angle, match_coordinates_sky
from astropy.table import Table
from astropy.io import fits
from astroquery.vizier import Vizier

Vizier.ROW_LIMIT = -1


"""
___________________________________________________________
FUNCTION TO CHECK IF SOURCE IS INSIDE GALAXY
___________________________________________________________
"""
def check_in_ellipse(x, y, center_x, center_y, semi_major, semi_minor, angle, tol=1):
    """
    Function to check if point or array of points are in a defined ellipse
    
    x = ra of points
    y = dec of points
    center_x = ra of ellipse center
    center_y = dec of ellipse center
    semi_major = semi major axis in deg
    semi_minor = semi minor axis in deg
    angle = rotated angle of ellipse
    tol = i.e. 1.1 adds 10% of radius size to axis as a 
          "fudge factor"
    
    returns array of bools for whether or not is in ellipse
    """
    angle = np.pi/2 + np.deg2rad(angle) 
    #turn points into skycoords this works for a list or single point for testpoints
    center = SkyCoord(ra=center_x*u.deg, dec=center_y*u.deg)
    testpoint = SkyCoord(ra=x*u.deg, dec=y*u.deg)
    #transform the test_points to a ref frame with center of frame at the ellipse center
    testpoint_trans = testpoint.transform_to(center.skyoffset_frame())
    #getting the diff values, the lat and lon are the differences since the center is ell
    dx = testpoint_trans.lon.to(u.deg).value
    dy = testpoint_trans.lat.to(u.deg).value
    #including rotation of ellipse
    x_rot = np.cos(angle) * dx - np.sin(angle) * dy
    y_rot = np.sin(angle) * dx + np.cos(angle) * dy
    semi_major = (semi_major/2) * tol
    semi_minor = (semi_minor/2) * tol
    #get array of inside ellipse
    is_inside = (x_rot**2 / semi_major**2 + y_rot**2 / semi_minor**2) < 1
    #return result
    return is_inside


"""
___________________________________________________________
CATALOGUE SEARCH FUNCTIONS FOR VIZIER
___________________________________________________________
"""
def search_hleda(ra, dec, rad):
    co_ord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    rad = rad*u.deg
    catalog='VII/237/pgc'
    result = Vizier.query_region(co_ord, radius=rad,
    catalog=catalog, frame='icrs')
    
    cols = ['PGC', 'RA', 'DEC', 'RAJ2000',
    'DEJ2000', 'OType', 'MType', 'logD25', 'logR25', 'PA', 'ANames']
    
    if len(result) == 0:
        return pd.DataFrame(columns=cols)
    else:
        result = result[0].to_pandas()
        
    result["RA"] = np.zeros(len(result))
    result["DEC"] = np.zeros(len(result))
    for idx, row in result.iterrows():
        result.at[idx, "RA"] = Angle(row.RAJ2000.split()[0]+"h"+
        row.RAJ2000.split()[1]+"m"+row.RAJ2000.split()[2]+"s").deg
        
        result.at[idx, "DEC"] = Angle(row.DEJ2000.split()[0]+"d"+
        row.DEJ2000.split()[1]+"m"+row.DEJ2000.split()[2]+"s").deg
        
    result=result[cols]
    result["semi_major"] = (0.1*(10**(result.logD25))/60) #in degrees
    result["R25"] = (10**(result.logR25))
    result["semi_minor"] = result.semi_major/result.R25 #in degrees

    return result


def square_search_hleda(ra, dec, width, height):
    co_ord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    width = width*u.degree
    height = height*u.degree
    catalog='VII/237/pgc'
    result = Vizier.query_region(co_ord, width=width, height=height,
    catalog=catalog, frame='icrs')
    
    cols = ['PGC', 'RA', 'DEC', 'RAJ2000',
    'DEJ2000', 'OType', 'MType', 'logD25', 'logR25', 'PA', 'ANames']
    
    if len(result) == 0:
        return pd.DataFrame(columns=cols)
    else:
        result = result[0].to_pandas()
        
    result["RA"] = np.zeros(len(result))
    result["DEC"] = np.zeros(len(result))
    for idx, row in result.iterrows():
        result.at[idx, "RA"] = Angle(row.RAJ2000.split()[0]+"h"+
        row.RAJ2000.split()[1]+"m"+row.RAJ2000.split()[2]+"s").deg
        
        result.at[idx, "DEC"] = Angle(row.DEJ2000.split()[0]+"d"+
        row.DEJ2000.split()[1]+"m"+row.DEJ2000.split()[2]+"s").deg
        
    result=result[cols]
    result["semi_major"] = (0.1*(10**(result.logD25))/60) #in degrees
    result["R25"] = (10**(result.logR25))
    result["semi_minor"] = result.semi_major/result.R25 #in degrees

    return result


def square_search_apass(ra, dec, width, height):
    co_ord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    width = width*u.degree
    height = height*u.degree
    catalog='apass'
    result = Vizier.query_region(co_ord, width=width, height=height, catalog=catalog,
    frame='icrs')
    cols = ['RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000', 'Field', 'nobs', 'mobs',
       'B-V', 'e_B-V', 'Vmag', 'e_Vmag', 'Bmag', 'e_Bmag', 'g_mag', 'e_g_mag',
       'r_mag', 'e_r_mag', 'i_mag', 'e_i_mag']
    if len(result) == 0:
        return pd.DataFrame(columns=cols)
    else:
        result = result[0].to_pandas().drop("recno", axis=1)        
    return result

def search_apass(ra,dec,rad):
    co_ord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    rad = rad*u.deg
    catalog='apass'
    result = Vizier.query_region(co_ord, radius=rad, catalog=catalog,
    frame='icrs')
    cols = ['RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000', 'Field', 'nobs', 'mobs',
       'B-V', 'e_B-V', 'Vmag', 'e_Vmag', 'Bmag', 'e_Bmag', 'g_mag', 'e_g_mag',
       'r_mag', 'e_r_mag', 'i_mag', 'e_i_mag']
    if len(result) == 0:
        return pd.DataFrame(columns=cols)
    else:
        result = result[0].to_pandas().drop("recno", axis=1)        
    return result

def search_tycho(ra,dec,rad):
    co_ord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    rad = rad*u.deg
    catalog='tyc2'
    result = Vizier.query_region(co_ord, radius=rad, catalog=catalog,
    frame='icrs')
    cols = ['RA_ICRS_', 'DE_ICRS_','pmRA',
    'pmDE', 'BTmag', 'VTmag', 'HIP','TYC1', 'TYC2', 'TYC3']
    if len(result) == 0:
        return pd.DataFrame(columns=cols)
    else:
        result = result[0].to_pandas()
    result = result[cols]
    return result

def square_search_tycho(ra, dec, width, height):
    co_ord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    width = width*u.degree
    height = height*u.degree
    catalog='tyc2'
    result = Vizier.query_region(co_ord, width=width, height=height, catalog=catalog,
    frame='icrs')
    cols = ['RA_ICRS_', 'DE_ICRS_','pmRA',
    'pmDE', 'BTmag', 'VTmag', 'HIP','TYC1', 'TYC2', 'TYC3']
    if len(result) == 0:
        return pd.DataFrame(columns=cols)
    else:
        result = result[0].to_pandas()
    result = result[cols]
    return result

def search_bright_stars(ra, dec, rad):
    
    apass = search_apass(ra, dec, rad)
    tycho = search_tycho(ra, dec, rad)
    
    cols = ['key_0', 'AP_RA', 'AP_DEC', 'e_RAJ2000', 'e_DEJ2000', 'Field', 'nobs',
       'mobs', 'B-V', 'e_B-V', 'Vmag', 'e_Vmag', 'Bmag', 'e_Bmag', 'g_mag',
       'e_g_mag', 'r_mag', 'e_r_mag', 'i_mag', 'e_i_mag', 'TY_RA', 'TY_DEC',
       'pmRA', 'pmDE', 'BTmag', 'VTmag', 'HIP', 'TYC1', 'TYC2', 'TYC3', 'RA',
       'Dec']
    
    if len(apass) == 0 and len(tycho) == 0:
        return pd.DataFrame(columns=cols)
    
    elif len(apass) == 0:
        bright_stars = tycho.rename(columns={"RA_ICRS_": "RA", "DE_ICRS_": "Dec"})
        bright_stars["duplicated"] = bright_stars.duplicated(subset=["RA","Dec"])
        dupes = bright_stars.query("duplicated == True").index
        bright_stars = bright_stars.drop(dupes)
        return bright_stars

    elif len(tycho) == 0:
        bright_stars = apass.rename(columns={"RAJ2000": "RA", "DEJ2000": "Dec"})
        bright_stars["duplicated"] = bright_stars.duplicated(subset=["RA","Dec"])
        dupes = bright_stars.query("duplicated == True").index
        bright_stars = bright_stars.drop(dupes)
        return bright_stars
    
    else:
    
        # tycho_match = SkyCoord(ra=tycho.RA_ICRS_*u.degree, dec=tycho.DE_ICRS_*u.degree)
        tycho_match = SkyCoord(ra=tycho.RA_ICRS_, dec=tycho.DE_ICRS_, unit=u.degree)

        # apass_match = SkyCoord(ra=apass.RAJ2000*u.degree, dec=apass.DEJ2000*u.degree)
        apass_match = SkyCoord(ra=apass.RAJ2000, dec=apass.DEJ2000, unit=u.degree)

        idx, sep2d, dist3d = match_coordinates_sky(tycho_match, apass_match) #match results

        tycho = tycho.set_index(idx)
        bright_stars = pd.merge(apass, tycho, left_index=True, right_on=tycho.index,
                                how="left").reset_index(drop=True)

        # bright_stars["duplicated"] = bright_stars.duplicated(subset=["RAJ2000","DEJ2000"])
        # dupes = bright_stars.query("duplicated == True").index
        # bright_stars = bright_stars.drop(dupes)

        bright_stars = bright_stars.rename(columns={"RA_ICRS_": "TY_RA", "DE_ICRS_": "TY_DEC",
                                     "RAJ2000": "AP_RA", "DEJ2000": "AP_DEC"})
        
        bright_stars["RA"] = np.where(np.isnan(bright_stars.AP_RA),
                                      bright_stars.TY_RA, bright_stars.AP_RA)
        bright_stars["Dec"] = np.where(np.isnan(bright_stars.AP_DEC),
                                      bright_stars.TY_DEC, bright_stars.AP_DEC)
        
        bright_stars["duplicated"] = bright_stars.duplicated(subset=["RA","Dec"])
        dupes = bright_stars.query("duplicated == True").index
        bright_stars = bright_stars.drop("duplicated", axis=1)
        
    return bright_stars

def square_search_bright_stars(ra, dec, width, height):
    
    apass = square_search_apass(ra, dec, width, height)
    tycho = square_search_tycho(ra, dec, width, height)
    
    cols = ['key_0', 'AP_RA', 'AP_DEC', 'e_RAJ2000', 'e_DEJ2000', 'Field', 'nobs',
       'mobs', 'B-V', 'e_B-V', 'Vmag', 'e_Vmag', 'Bmag', 'e_Bmag', 'g_mag',
       'e_g_mag', 'r_mag', 'e_r_mag', 'i_mag', 'e_i_mag', 'TY_RA', 'TY_DEC',
       'pmRA', 'pmDE', 'BTmag', 'VTmag', 'HIP', 'TYC1', 'TYC2', 'TYC3', 'RA',
       'Dec']
    
    if len(apass) == 0 and len(tycho) == 0:
        return pd.DataFrame(columns=cols)
    
    elif len(apass) == 0:
        bright_stars = tycho.rename(columns={"RA_ICRS_": "RA", "DE_ICRS_": "Dec"})
        bright_stars["duplicated"] = bright_stars.duplicated(subset=["RA","Dec"])
        dupes = bright_stars.query("duplicated == True").index
        bright_stars = bright_stars.drop(dupes)
        return bright_stars

    elif len(tycho) == 0:
        bright_stars = apass.rename(columns={"RAJ2000": "RA", "DEJ2000": "Dec"})
        bright_stars["duplicated"] = bright_stars.duplicated(subset=["RA","Dec"])
        dupes = bright_stars.query("duplicated == True").index
        bright_stars = bright_stars.drop(dupes)
        return bright_stars
    
    else:
    
        tycho_match = SkyCoord(ra=tycho.RA_ICRS_*u.degree, dec=tycho.DE_ICRS_*u.degree)
        apass_match = SkyCoord(ra=apass.RAJ2000*u.degree, dec=apass.DEJ2000*u.degree)

        idx, sep2d, dist3d = match_coordinates_sky(tycho_match, apass_match) #match results

        tycho = tycho.set_index(idx)
        bright_stars = pd.merge(apass, tycho, left_index=True, right_on=tycho.index,
                                how="left").reset_index(drop=True)

        # bright_stars["duplicated"] = bright_stars.duplicated(subset=["RAJ2000","DEJ2000"])
        # dupes = bright_stars.query("duplicated == True").index
        # bright_stars = bright_stars.drop(dupes)

        bright_stars = bright_stars.rename(columns={"RA_ICRS_": "TY_RA", "DE_ICRS_": "TY_DEC",
                                     "RAJ2000": "AP_RA", "DEJ2000": "AP_DEC"})
        
        bright_stars["RA"] = np.where(np.isnan(bright_stars.AP_RA),
                                      bright_stars.TY_RA, bright_stars.AP_RA)
        bright_stars["Dec"] = np.where(np.isnan(bright_stars.AP_DEC),
                                      bright_stars.TY_DEC, bright_stars.AP_DEC)
        
        bright_stars["duplicated"] = bright_stars.duplicated(subset=["RA","Dec"])
        dupes = bright_stars.query("duplicated == True").index
        bright_stars = bright_stars.drop("duplicated", axis=1)
        
        return bright_stars

"""
___________________________________________________________
CATALOG SEARCH FUNCTIONS FOR MAST CASJOBS
___________________________________________________________
"""
#call this to sign in to enable mastcasjobs functions
def mastcasjobs_init():
    if not os.environ.get('CASJOBS_USERID'):
        os.environ['CASJOBS_USERID'] = input('Enter Casjobs username:')
    if not os.environ.get('CASJOBS_PW'):
        os.environ['CASJOBS_PW'] = getpass.getpass('Enter Casjobs password:')

def mastQuery(request, json_return=False):
    """Perform a MAST query.

    Parameters
    ----------
    request (dictionary): The MAST request json object
    
    Returns the text response or (if json_return=True) the json response
    """
    
    url = "https://mast.stsci.edu/api/v0/invoke"

    # Encoding the request as a json string
    requestString = json.dumps(request)
    
    # make the query
    r = requests.post(url, data=dict(request=requestString))
    
    # raise exception on error
    r.raise_for_status()
    
    if json_return:
        return r.json()
    else:
        return r.text


def resolve(name):
    """Get the RA and Dec for an object using the MAST name resolver
    
    Parameters
    ----------
    name (str): Name of object

    Returns RA, Dec tuple with position"""

    resolverRequest = {'service':'Mast.Name.Lookup',
                       'params':{'input':name,
                                 'format':'json'
                                },
                      }
    resolvedObject = mastQuery(resolverRequest, json_return=True)
    # The resolver returns a variety of information about the resolved object, 
    # however for our purposes all we need are the RA and Dec
    try:
        objRa = resolvedObject['resolvedCoordinate'][0]['ra']
        objDec = resolvedObject['resolvedCoordinate'][0]['decl']
    except IndexError as e:
        raise ValueError("Unknown object '{}'".format(name))
    return (objRa, objDec)


def search_circ_region(ra, dec, rad, table_name=None, task_name="My Query", keep_table=True):
    rad = rad*60
    ps_jobs = mastcasjobs.MastCasJobs(context="PanSTARRS_DR2")
    
    if table_name == None:
        if keep_table == True:
            raise TypeError("table_name must be provided if keep_table is True")
        table_name = "_temp"
        ps_jobs.drop_table_if_exists(table_name)

    
    if table_name in ps_jobs.list_tables():
        raise NameError(f"Table \'{table_name}\' Already Exists")

    ps_query = f"""SELECT s.objID, s.raMean, s.decMean,
        s.gKronMag, s.gPSFMag, s.gKronMagErr, s.gPSFMagErr, gExtNSigma,
        s.rKronMag, s.rPSFMag, s.rKronMagErr, s.rPSFMagErr, rExtNSigma,
        s.iKronMag, s.iPSFMag, s.iKronMagErr, s.iPSFMagErr, iExtNSigma,
        s.zKronMag, s.zPSFMag, s.zKronMagErr, s.zPSFMagErr, zExtNSigma,
        s.yKronMag, s.yPSFMag, s.yKronMagErr, s.yPSFMagErr, yExtNSigma,
        s.nDetections
        from fGetNearbyObjEq({ra},{dec},{rad}) nb
        inner join StackObjectView s on s.objid=nb.objid and s.primaryDetection=1
        INTO {table_name}
        """
    
    ps_job_id = ps_jobs.submit(ps_query, task_name=f"{task_name}")
    ps_jobs.monitor(ps_job_id)
    ps_df = ps_jobs.fast_table(table_name).to_pandas()
    
    if keep_table == False:
        ps_jobs.drop_table_if_exists(table_name)

    return ps_df
    

def quick_search_circ_region(ra, dec, rad, task_name="My Query"):
    rad = rad*60
    ps_jobs = mastcasjobs.MastCasJobs(context="PanSTARRS_DR2")

    ps_query = f"""SELECT s.objID, s.raMean, s.decMean,
        s.gKronMag, s.gPSFMag, s.gKronMagErr, s.gPSFMagErr, gExtNSigma,
        s.rKronMag, s.rPSFMag, s.rKronMagErr, s.rPSFMagErr, rExtNSigma,
        s.iKronMag, s.iPSFMag, s.iKronMagErr, s.iPSFMagErr, iExtNSigma,
        s.zKronMag, s.zPSFMag, s.zKronMagErr, s.zPSFMagErr, zExtNSigma,
        s.yKronMag, s.yPSFMag, s.yKronMagErr, s.yPSFMagErr, yExtNSigma,
        s.nDetections
        from fGetNearbyObjEq({ra},{dec},{rad}) nb
        inner join StackObjectView s on s.objid=nb.objid and s.primaryDetection=1
        """
    
    ps_df = ps_jobs.quick(ps_query, task_name=f"{task_name}").to_pandas()

    return ps_df

def retrieve_table(table_name):
    """
    Function to retrieve table name from PS1 mastcasjobs
    """
    ps_jobs = mastcasjobs.MastCasJobs(context="PanSTARRS_DR2")
    ps_df = ps_jobs.fast_table(table_name).to_pandas()
    return ps_df


def search_rect_region(ra1, ra2, dec1, dec2, table_name, task_name="My Query"):
    
    ps_jobs = mastcasjobs.MastCasJobs(context="PanSTARRS_DR2")
    
    if table_name in ps_jobs.list_tables():
        raise NameError(f"Table \'{table_name}\' Already Exists")

    ps_query = f"""SELECT s.objID, s.raMean, s.decMean,
        s.gKronMag, s.gPSFMag, s.gKronMagErr, s.gPSFMagErr, gExtNSigma,
        s.rKronMag, s.rPSFMag, s.rKronMagErr, s.rPSFMagErr, rExtNSigma,
        s.iKronMag, s.iPSFMag, s.iKronMagErr, s.iPSFMagErr, iExtNSigma,
        s.zKronMag, s.zPSFMag, s.zKronMagErr, s.zPSFMagErr, zExtNSigma,
        s.yKronMag, s.yPSFMag, s.yKronMagErr, s.yPSFMagErr, yExtNSigma,
        s.nDetections
        from fGetObjFromRect({ra1},{ra2},{dec1},{dec2}) nb
        inner join StackObjectView s on s.objid=nb.objid and s.primaryDetection=1
        INTO {table_name}"""
    
    ps_job_id = ps_jobs.submit(ps_query, task_name=f"{task_name}")
    ps_jobs.monitor(ps_job_id)
    ps_df = ps_jobs.fast_table(table_name).to_pandas()

    return ps_df

#custom function
def get_mast_table(table_name):
    return mastcasjobs.MastCasJobs().fast_table(table_name).to_pandas()


def proximity_search(ralist, declist, point, rad):
    """
    Function to search and return all points within a radius of skycoords
    
    ralist = list of right ascension of points
    declist = list of declinations of points
    point = point the search will be located around, tuple or list
    rad = radius in degrees of search
    
    returns: indexes of the matching points
    """
    #converting point and ra,dec to skycoords
    point = SkyCoord(point[0]*u.deg, point[1]*u.deg)
    # coords = SkyCoord(ralist*u.deg, declist*u.deg)
    coords = SkyCoord(ralist, declist, unit=u.deg)
    #calculating separation of point vs all coords in this list
    seplist = coords.separation(point)
    #this is a boolean array whether or not each instance is in or out of the radius
    in_rad = seplist.deg <= rad
    #this gets the index of all the nonzero points (i.e. true, inside rad)
    inds_in_rad = in_rad.nonzero()[0]
    return inds_in_rad

"""
___________________________________________________________
FUNCTION TO PLOT CUTOUTS VIA ONLINE QUERY
___________________________________________________________
"""

def getimages(ra,dec,filters="grizy"):
    
    """Query ps1filenames.py service to get a list of images
    
    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """
    
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    table = Table.read(url, format='ascii')
    return table


def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):
    
    """Get URL for images in the table
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """
    
    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra,dec,filters=filters)
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size}&format={format}")
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    return url

def getcolorim(ra, dec, size=240, output_size=None, filters="grizy", format="jpg"):
    
    """Get color image at a sky position
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png")
    Returns the image
    """
    
    if format not in ("jpg","png"):
        raise ValueError("format must be jpg or png")
    url = geturl(ra,dec,size=size,filters=filters,output_size=output_size,format=format,color=True)
    r = requests.get(url)
    im = Image.open(io.BytesIO(r.content))
    return im

#custom function
def plot_cutouts(ra, dec, nrows=1, ncols=1, size=240, figsize=(6,6)):
    if type(ra) == pd.core.series.Series:
        ra=ra.reset_index(drop=True)
        dec=dec.reset_index(drop=True)
    else:
        ra = [ra]
        dec = [dec]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    if ncols*nrows > 1:
        axs = axs.flatten()
    else:
        axs  = [axs]
        
    for i in range(nrows*ncols):
        cim = getcolorim(ra[i], dec[i], size=size)
        axs[i].imshow(cim, origin="upper")
        axs[i].tick_params(axis='both', which='both',
        bottom=False, top=False, left=False, right=False, labelbottom=False)
        axs[i].set_yticklabels([])
        axs[i].set_xticklabels([])
        axs[i].set_title(f"Loc: {ra[i]}\n{dec[i]}", fontsize=10)
    plt.tight_layout()
    return

def get_fits_image(ra,dec, size=240, filters="r"):
    fitsurl = geturl(ra, dec, size=size, filters=filters, format="fits")
    fh = fits.open(fitsurl[0])
    fim = fh[0].data
    fim[np.isnan(fim)] = 0.0
    return fim

"""
___________________________________________________________

___________________________________________________________
"""

def getImageTable(tra, tdec, size=240, filters="grizy", format="fits", imagetypes="stack"):
    
    """Query ps1filenames.py service for multiple positions to get a list of images
    This adds a url column to the table to retrieve the cutout.
     
    tra, tdec = list of positions in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    format = data format (options are "fits", "jpg", or "png")
    imagetypes = list of any of the acceptable image types.  Default is stack;
        other common choices include warp (single-epoch images), stack.wt (weight image),
        stack.mask, stack.exp (exposure time), stack.num (number of exposures),
        warp.wt, and warp.mask.  This parameter can be a list of strings or a
        comma-separated string.
 
    Returns pandas dataframe with the results
    """
    pd.set_option("display.max_colwidth", 10000)
    
    ps1filename = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    fitscut = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
    
    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    # if imagetypes is a list, convert to a comma-separated string
    if not isinstance(imagetypes,str):
        imagetypes = ",".join(imagetypes)

    # put the positions in an in-memory file object
    cbuf = io.StringIO()
    cbuf.write('\n'.join(["{} {}".format(ra, dec) for (ra, dec) in zip(tra,tdec)]))
    cbuf.seek(0)
    # use requests.post to pass in positions as a file
    r = requests.post(ps1filename, data=dict(filters=filters, type=imagetypes),
        files=dict(file=cbuf))
    r.raise_for_status()
    tab = Table.read(r.text, format="ascii")
 
    urlbase = "{}?size={}&format={}".format(fitscut,size,format)
    tab["url"] = ["{}&ra={}&dec={}&red={}".format(urlbase,ra,dec,filename)
            for (filename,ra,dec) in zip(tab["filename"],tab["ra"],tab["dec"])]
    
    tab = tab.to_pandas()
    return tab


def readFitsImage(url):
    """Gets header and image data from fits files retrieved from urls in pandas
    dataframe from function getImageTable. Returns header and image data.
    """
    r = requests.get(url)
    memory_file = io.BytesIO(r.content)
    
    with fits.open(memory_file) as hdulist:
        data = hdulist[0].data
        header = hdulist[0].header
    
    return header, data


