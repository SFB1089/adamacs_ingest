import datajoint as dj
from .. import db_prefix

schema = dj.schema(db_prefix + 'subject')


# -------------- Table declarations --------------


@schema
class Lab(dj.Manual):
    definition = """
    lab             : varchar(8)   # short lab name, pyrat labid
    ---
    lab_name        : varchar(255)
    institution     : varchar(255)
    address         : varchar(255)
    """


@schema
class User(dj.Lookup):
    definition = """
    user_id        : int
    ---
    name           : varchar(32)
    shorthand=''   : varchar(32) # TR: added name_FirstInitialSurname shorthand
    initials=''    : varchar(2)  # Update after pyrat ingestion
    email=''       : varchar(32) # TR: for completeness' sake also email    
    -> [nullable] Lab
    """


@schema
class Protocol(dj.Manual):
    definition = """
    # PyRAT licence number and title
    protocol                        : varchar(32)
    ---
    protocol_description=''         : varchar(255)
    """


@schema
class Line(dj.Manual):
    definition = """
    # animal line 
    line                        : int  # strain_id within PyRAT. Not name_id seen in GUI
    ---
    line_name=''                : varchar(64)
    is_active                   : enum('active','inactive','unknown')  # TODO BUGFIX expects float for unknown reason
    """


@schema
class Mutation(dj.Manual):
    definition = """
    # The mutations of animal lines
    -> Line
    mutation_id                 : int
    ---
    description=''              : varchar(32)
    """


@schema
class Project(dj.Lookup):
    definition = """
    project                 : varchar(32)
    ---
    project_description=''  : varchar(1024)
    """


@schema
class Subject(dj.Manual):
    definition = """
    # Animal Subject
    # Our Animals are not uniquely identified by their ID
    # because different labs use different animal facilities.

    subject                 : varchar(16)     # PyRat import uses this for earmark value
    ---
    earmark                 : varchar(16)     #
    sex                     : enum('M', 'F', 'U')  # Geschlecht
    birth_date              : varchar(32)          # Geb.
    death_date = null       : varchar(32)          # Gest.
    generation=''           : varchar(64)     # Generation (F2 in example sheet)
    parent_ids              : tinyblob        # dict of parent_sex: parent_eartag
    -> User.proj(owner_id='user_id')          # Besitzer
    -> User.proj(responsible_id='user_id')    # Verantwortlicher
    -> Line                                   # Linie / Stamm
    -> Protocol
    """

@schema 
class SubjectRspace(dj.Manual):
    definition = """
    -> Subject
    ---
    rspace_subject_id: varchar(32) # id of rspace subject folder
    """
 

@schema
class SubjectGenotype(dj.Manual):
    definition = """
    -> Subject
    -> Mutation
    ---
    genotype        : enum('wt/wt', 'wt/tg', 'tg/wt', 'tg/tg')
    """


@schema
class SubjectDeath(dj.Manual):
    definition = """
    -> Subject
    ---
    death_date      : date
    cause           : varchar(255)
    """


lab_key = 'Rose'  # Short, unique identifier for the lab. Maximum 8 characters. Example: 'Rose'.
lab_name = 'Circut Mechanisms of Behavior'  # A longer, more descriptive name for the laboratory.
institution = 'Institute for Experimental Epileptology and Cognition Research (IEECR)'  # The institution the laboratory belongs to.
address = 'Venusberg-Campus 1, 53127 Bonn'  # The postal address of the laboratory.

Lab.insert1((lab_key, lab_name, institution, address), skip_duplicates=True)

user_data = [{'user_id': 1, 'name': 'Rose Tobias', 'initials': 'TR', 'shorthand': 'tobiasr', 'email': 'trose@uni-bonn.de', 'lab': 'Rose'},
        {'user_id': 2, 'name': 'Kück Laura', 'initials': 'LK', 'shorthand': 'laurak', 'email': 'laura.kueck@ukbonn.de', 'lab': 'Rose'},
        {'user_id': 3, 'name': 'Krasilshchikova Natalia', 'initials': 'NK', 'shorthand': 'nataliak', 'email': ' nkra1@uni-bonn.de', 'lab': 'Rose'},
        {'user_id': 4, 'name': 'Bühler Daniel', 'initials': 'DB', 'shorthand': 'danielb', 'email': 'Db247@uni-bonn.de', 'lab': 'Rose'},
        {'user_id': 5, 'name': 'Luxem Kevin', 'initials': 'KL', 'shorthand': 'kevinl', 'email': 'luxemk@uni-bonn.de', 'lab': 'Rose'},
        {'user_id': 6, 'name': 'Jung Jisoo', 'initials': 'JJ', 'shorthand': 'jisooj', 'email': 'jjun1@uni-bonn.de', 'lab': 'Rose'},
        {'user_id': 7, 'name': 'Narayanamurthy Rukhmani', 'initials': 'RN', 'shorthand': 'rukhun', 'email': 'rnar@uni-bonn.de', 'lab': 'Rose'},
        {'user_id': 8, 'name': 'Kremers Leon', 'initials': 'LE', 'shorthand': 'leonk', 'email': 'leon.kremers@uni-bonn.de', 'lab': 'Rose'},
        {'user_id': 9, 'name': 'Asma Mekhnache', 'initials': 'AM', 'shorthand': 'asmam', 'email': 's4asmekh@uni-bonn.de', 'lab': 'Rose'},
        {'user_id': 10, 'name': 'Aelton Araujo', 'initials': 'AA', 'shorthand': 'aeltona', 'email': 'aara@uni-bonn.de', 'lab': 'Rose'},
        {'user_id': 11, 'name': 'Shanquian Ma', 'initials': 'SM', 'shorthand': 'shanm', 'email': 'shanqianma@gmail.com', 'lab': 'Rose'},
        {'user_id': 12, 'name': 'Yoshiki Hatashita', 'initials': 'YH', 'shorthand': 'yoshikih', 'email': 'y.hatashita3@gmail.com', 'lab': 'Rose'},
        {'user_id': 13, 'name': 'Kevin Hoogers', 'initials': 'KH', 'shorthand': 'kevinh', 'email': 'hoogerskevin@gmail.com', 'lab': 'Rose'}]
User.insert(user_data, skip_duplicates=True) 

project_data = [{'project': 'hpc-repstab', 'project_description': 'hpc-representational-stability' },
        {'project': 'vc-lgn-repstab', 'project_description': 'vc-lgn-representational-stability'},
        {'project': 'rsc-functop', 'project_description': 'rsc-functional-topography'},
        {'project': 'rsc-hpc', 'project_description': 'rsc-hippocampal'},
        {'project': 'sc-lgn-actvis', 'project_description': 'sc-lgn-active-vision'},
        {'project': 'ATN', 'project_description': 'ATN-functional-characterization'},
        {'project': 'rsc-latent', 'project_description': 'rsc-contextual-multimodal-latent-ss'},
        {'project': 'V1-oddball', 'project_description': 'v1-oddball-predicitve-prior'},
        {'project': 'uncertainty-repstab', 'project_description': 'uncertainty-neuromodulation-stability'},
        {'project': 'dummy', 'project_description': 'dummy'}
        ]
Project.insert(project_data, skip_duplicates=True) 