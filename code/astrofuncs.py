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

#this makes the astropy vizier queries return an unlimited number of rows.
Vizier.ROW_LIMIT = -1


"""
___________________________________________________________
FUNCTIONS TO MASK OBJECTS IN GALAXIES OR AROUND BRIGHT STARS
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
