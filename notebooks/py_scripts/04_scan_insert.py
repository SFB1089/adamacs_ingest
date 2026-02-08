# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Scanning Session Ingestion

# ## Setup

# ### Connect to the database

# If you are don't have your login information, contact the administrator.
#
# Using local config file (see [01_pipeline_activation](../01_pipeline_activation.ipynb)):

# +
import os
# change to the upper level folder to detect dj_local_conf.json
from pathlib import Path
if Path.cwd().name == 'notebooks':
    os.chdir('..')
import datajoint as dj; dj.conn()

from adamacs.pipeline import subject, session, surgery, scan
from adamacs import utility
from adamacs.ingest import session as isess
sub, lab, protocol, line, mutation, user, project, subject_genotype, subject_death = (
    subject.Subject(), subject.Lab(), subject.Protocol(), subject.Line(), 
    subject.Mutation(), subject.User(), subject.Project(), subject.SubjectGenotype(), 
    subject.SubjectDeath())
# -

# Manual entry:

# +
# Manual Entry
import datajoint as dj; import getpass
dj.config['database.host'] = os.environ.get('DJ_HOST', 'localhost')        # Put the server name between these apostrophe
dj.config['database.user'] = os.environ.get('DJ_USER', 'dj_user')             # Put your user name between these apostrophe
dj.config['database.password'] = getpass.getpass()  # Put your password in the prompt
dj.conn()

from adamacs.pipeline import subject, session, surgery, scan
from adamacs import utility
from adamacs.ingest import session as isess
sub, lab, protocol, line, mutation, user, project, subject_genotype, subject_death = (
    subject.Subject(), subject.Lab(), subject.Protocol(), subject.Line(), 
    subject.Mutation(), subject.User(), subject.Project(), subject.SubjectGenotype(), 
    subject.SubjectDeath())
# -

# ## Ingesting scan

# Ingest all scans associated with a given session ID.

scan.ScanPath()

isess.ingest_session_scan('sess9FB2LN5C', verbose=True)

session.Session * session.SessionDirectory

key='scan9FB2LN5C'
(scan.Scan & f'scan_id=\"{key}\"')

scan.ScanInfo.populate()

# ##### Some placeholders for equipment and location during development

scan.ScanInfo()

scan.ScanInfo.Field()

# Note the relative path below:

#temporary step - insert placeholder values
equipment_placeholder = "Equipment"
location_placeholder = "Location"
from adamacs.pipeline import Equipment, Location
Equipment.insert1({'scanner' : equipment_placeholder}, skip_duplicates=True)
Location.insert1({'anatomical_location': location_placeholder}, skip_duplicates=True) 

