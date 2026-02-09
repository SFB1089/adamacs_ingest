import logging
import warnings

logging.getLogger("datajoint").setLevel(logging.WARNING)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from adamacs.notebook_runtime import bootstrap_ingest_notebook

ctx = bootstrap_ingest_notebook(verbose=False)
repo_root = ctx.repo_root

import datajoint as dj

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

# # Querying Guide

# Database access with config file:

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

# Manual entry database access:

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

# ### Get all Sessions and Scans of an Animal

