# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Basic Querying Template
#
# Connect to the database with config file:

import os
from pathlib import Path
if Path.cwd().name == 'notebooks':
    os.chdir('..')
import datajoint as dj; dj.conn()
from adamacs.pipeline import subject, session, surgery, session, behavior, equipment, \
                             imaging, scan, train, model

# Manually connect:

import datajoint as dj; import getpass
dj.config['database.host'] = os.environ.get('DJ_HOST', 'localhost')
dj.config['database.user'] = os.environ.get('DJ_USER', 'dj_user')
dj.config['database.password'] = getpass.getpass() # enter the password securily
dj.conn()
from adamacs.pipeline import subject, session, surgery, session, behavior, equipment, \
                             imaging, scan, train, model

# ### How many mice?

query = subject.Subject()
query.fetch().size

# ### How many scans per mouse?

query = session.Session() * scan.Scan() & 'subject = "WEZ-8701"'
query.fetch().size

query

