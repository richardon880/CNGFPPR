# Objectives

 - Implement first pass masking  (check if |RA1-RA2|<0.5 before calling SkyCoor to get exact offset) -- Complete

 - Compare catalog to MGC. Completness (do we find all the galaxies in MGC), purity (is there a lot of crap in our catalog that is not in MGC and which is not real), accuracy (do our z agree with MGC, and are the MGC z within 95% of the PDF).

 - x-match MGC and PS1 cat, what is the reason for things that are in former but not latter and vice versa. Look at cutouts.

 - Make several  notebooks explaining testing of each module in  code, should be understandable by someone new to the project

 - Make standalone get_z.py, which runs from commandline and takes RA, Dec as arguments, returns CSV catalog of galaxies with CDF for redshift

 - Update Overleaf.
