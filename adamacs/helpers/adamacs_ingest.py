# Tobias Rose 2023: Routine ingest helpers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, HTML
from natsort import natsorted, ns
import re
from tqdm.autonotebook import tqdm
import pathlib
from datetime import datetime
from adamacs.pipeline import subject, session, scan, equipment, surgery, event, trial, imaging, behavior, model, imaging
from adamacs.ingest import session as isess
from adamacs.helpers import stack_helpers as sh
from adamacs.ingest import behavior as ibe
from adamacs.utility import *
from element_interface.utils import find_full_path
from adamacs.paths import get_dlc_root_data_dir
import traceback

sub, lab, protocol, line, mutation, user, project, subject_genotype, subject_death = (
    subject.Subject(), subject.Lab(), subject.Protocol(), subject.Line(), 
    subject.Mutation(), subject.User(), subject.Project(), subject.SubjectGenotype(), 
    subject.SubjectDeath()
    )

def select_sessions(AvailableSessionDirB, do_population = False, rspace_upload = False, ingest_opt = 'trigger'):
    
    # Personal default values
    user_defaults = get_user_defaults(AvailableSessionDirB)
    usercam_defaults = get_user_cam_defaults(AvailableSessionDirB)

    # extract lookup tables
    Project = project.fetch('project')
    Equipment = equipment.Equipment().fetch('scanner')
    Recording_Location = surgery.AnatomicalLocation().fetch('anatomical_location')
    SessionNotes = session.SessionNote.fetch('session_note')
    s2pparm = imaging.ProcessingParamSet.fetch("paramset_idx", "paramset_desc")
    session_dirs_ingested = session.SessionDirectory.fetch('session_dir')
    scan_dirs_ingested = scan.ScanPath.fetch('path')
    IngestedSessionDirA = get_session_dir_key_from_dir(scan_dirs_ingested)
    # ScanDirArrayingested = get_scan_dir_key_from_dir(scan_dirs_ingested)
    DLCModels = model.Model.fetch("model_name")

    # Define the widgets
    session_dropdowns = []
    session_checkboxes = []
    for i, session_list in enumerate(AvailableSessionDirB):

        # TODO - these things should be queried and set on the scan level, not session level

        # Dropdowns for Project, Recording Location, Equipment

        current_session = get_session_key_from_dir([session_list])
        current_subject = get_subject_key_from_dir([session_list])

        # check if directory session is already ingested - if yes: populate GUI with table values. If no: use user defaults
        query = session.Session() & f'session_id = "{current_session[0]}"'
        count = len(query.fetch('session_id'))

        # get all sessions of current animal that are already ingested
        subject_sessions = session.SessionSameSite().proj("session_id") * session.Session.proj("subject") & f'subject = "{current_subject[0]}"'
        subject_sessions_array = subject_sessions.fetch("session_id")

        if count > 0:
            # POPULATE PRESELECTION FROM DATABASE
            # get the project associated with a session
            query = session.ProjectSession() & f'session_id = "{current_session[0]}"'
            try:
                project_dropdown_value = query.fetch1("project")
            except Exception as e:
                print(f"Error occurred while checking project table: {e}")
                print(current_session[0])
            # get the location associated with a session
            query = session.Session() * scan.ScanLocation() & f'session_id = "{current_session[0]}"'
            location_dropdown_value = query.fetch("anatomical_location")
            # get the equipment associated with a session
            query = session.Session() * scan.Scan() & f'session_id = "{current_session[0]}"'
            equipment_dropdown_value = query.fetch("scanner")
            # get the note associated with a session
            query = session.SessionNote() & f'session_id = "{current_session[0]}"'
            session_note_textbox_value = query.fetch("session_note")
            if len(session_note_textbox_value) == 0:
                session_note_textbox_value = ["none"]
            # get the current SessionSameSite
            query = session.SessionSameSite() & f'session_id = "{current_session[0]}"'
            subject_session_dropdown_value = query.fetch("same_site_id")
            # check if manually curated
            key = dict(session_id = current_session[0])
            try:
                # curated = (imaging.Curation & key).fetch("manual_curation").astype("bool")
                curated = (imaging.Curation & key).fetch("manual_curation", order_by = "-curation_time")[0] # only fetch latest curation (when having data with multiple parameter runs)
            except Exception as e:
                # print(f"Error occurred while checking curation table: {e}")
                curated = False
            try:
                DLCModels_dropdown_value = (model.PoseEstimationTaskNew & key).fetch("model_name")[0]
                if len(DLCModels_dropdown_value) == 0:
                    DLCModels_dropdown_value = 'dummy'
            except Exception as e:
                # print(f"Error occurred while checking DLCModels table: {e}")
                DLCModels_dropdown_value = 'dummy'

            try:
                DLCModels_2_dropdown_value = (model.PoseEstimationTaskNew & key).fetch("model_name")[1]
                if len(DLCModels_2_dropdown_value) == 0:
                    DLCModels_2_dropdown_value = 'dummy'
            except Exception as e:
                # print(f"Error occurred while checking DLCModels table: {e}")
                DLCModels_2_dropdown_value = 'dummy'

            try:
                DLCModels_3_dropdown_value = (model.PoseEstimationTaskNew & key).fetch("model_name")[2]
                if len(DLCModels_3_dropdown_value) == 0:
                    DLCModels_3_dropdown_value = 'dummy'
            except Exception as e:
                # print(f"Error occurred while checking DLCModels table: {e}")
                DLCModels_3_dropdown_value =  'dummy'

        else:
            # POPULATE PRESELECTION FROM USER DEFAULTS
            project_dropdown_value = Project[user_defaults[i][0]]
            location_dropdown_value = [Recording_Location[user_defaults[i][1]]]
            equipment_dropdown_value = [Equipment[user_defaults[i][2]]]
            subject_session_dropdown_value = current_session
            subject_sessions_array = np.append(subject_sessions_array, current_session)
            session_note_textbox_value = ["none"]
            curated = False
            DLCModels_dropdown_value =  DLCModels[user_defaults[i][4]]
            DLCModels_2_dropdown_value =  DLCModels[user_defaults[i][5]]
            DLCModels_3_dropdown_value =  DLCModels[user_defaults[i][6]]
        
        project_dropdown = widgets.Dropdown(options=Project, value=project_dropdown_value, description="Project:")
        location_dropdown = widgets.Dropdown(options=Recording_Location, value=location_dropdown_value[0], description="Location:")
        equipment_dropdown = widgets.Dropdown(options=Equipment, value=equipment_dropdown_value[0], description="Setup:")
        s2pparms_dropdown = widgets.Dropdown(options=s2pparm[1], value=s2pparm[1][user_defaults[i][3]], description="s2p parm:")
        try:
            subject_session_dropdown = widgets.Dropdown(options=subject_sessions_array, value=subject_session_dropdown_value[0], description="same site as:")
        except Exception as e:
            print(f"Error occurred while checking subject_sessions table for samesite: {e}")
            subject_session_dropdown = widgets.Dropdown(options=current_session[0], value=current_session[0], description="same site as:")

        # print(session_list)

        # print(DLCModels_dropdown_value)

        DLCModels_dropdown = widgets.Dropdown(options=DLCModels, value=DLCModels_dropdown_value, description="DLCModel 1:")
        DLCModels_2_dropdown = widgets.Dropdown(options=DLCModels, value=DLCModels_2_dropdown_value, description="DLCModel 2:")
        DLCModels_3_dropdown = widgets.Dropdown(options=DLCModels, value=DLCModels_3_dropdown_value, description="DLCModel 3:")

        session_note_textbox = widgets.Text(value=session_note_textbox_value[0], description='Session comment:')
       
        
        session_dropdowns.append((project_dropdown, location_dropdown, equipment_dropdown, s2pparms_dropdown, subject_session_dropdown, DLCModels_dropdown, DLCModels_2_dropdown, DLCModels_3_dropdown, session_note_textbox))

        
        # Checkbox for Process - check to commit for ingest and processing
        session_checkbox = widgets.Checkbox(description='run?', layout=widgets.Layout(width='auto'))
        # session_checkboxes.append(session_checkbox)

        # Checkbox for Curation - check if manually curated
        curation_checkbox = widgets.Checkbox(description='curated?', layout=widgets.Layout(width='auto'))
        session_checkboxes.append((session_checkbox, curation_checkbox))
    
    # Display the widgets
    output = widgets.Output()

    with output:
        now = datetime.now()
        # Display the Sessions labels and associated dropdowns and checkboxes
        hbox_list = []
        for i, session_list in enumerate(AvailableSessionDirB): #unique_directory_strings(SessionDirA, SessionDirB)
            # Create an HBox to hold the label and associated dropdowns and checkbox
            hbox = widgets.HBox()
            hbox.children = [
                widgets.Label(value=session_list + ':', layout=widgets.Layout(width='1800px')), 
                session_dropdowns[i][0],
                session_dropdowns[i][1],
                session_dropdowns[i][2],
                session_dropdowns[i][3],
                session_dropdowns[i][4],
                session_dropdowns[i][5],
                session_dropdowns[i][6],
                session_dropdowns[i][7],
                session_dropdowns[i][8],
                session_dropdowns[i][9],
                session_checkboxes[i][0],
                session_checkboxes[i][1]
            ]
            if session_list in unique_directory_strings(IngestedSessionDirA, AvailableSessionDirB):
                hbox.children[0].value = '*' + session_list + ':'
                hbox.children[10].value = True
                hbox.children[11].value = False
            else:
                hbox.children[10].value = False
                hbox.children[11].value = False
                
                hbox.children[1].disabled = False
                hbox.children[2].disabled = False
                hbox.children[3].disabled = False
                hbox.children[4].disabled = False
                hbox.children[5].disabled = False
                hbox.children[6].disabled = False
                hbox.children[7].disabled = False
                hbox.children[8].disabled = False
                hbox.children[9].disabled = False
            if curated: #sigh - this really needs to be on scan level - currently just takes the first scan of the session
                hbox.children[11].value = True
                
            hbox_list.append(hbox)

        vbox = widgets.VBox(hbox_list, layout=widgets.Layout(flex='0 0 auto', overflow_y='scroll'))
        # Display the commit button
        commit_button = widgets.Button(description='Commit', layout=widgets.Layout(width='auto'))
        display(vbox, commit_button)

        # Define the callback function for the commit button
        def commit_button_clicked(b):
            selected_sessions = [AvailableSessionDirB[i] for i in range(len(AvailableSessionDirB)) if session_checkboxes[i][0].value]
            selected_projects = [session_dropdowns[i][0].value for i in range(len(AvailableSessionDirB)) if session_checkboxes[i][0].value]
            selected_locations = [session_dropdowns[i][1].value for i in range(len(AvailableSessionDirB)) if session_checkboxes[i][0].value]
            selected_equipment = [session_dropdowns[i][2].value for i in range(len(AvailableSessionDirB)) if session_checkboxes[i][0].value]
            selected_s2pparms = [session_dropdowns[i][3].index for i in range(len(AvailableSessionDirB)) if session_checkboxes[i][0].value]
            selected_same_site = [session_dropdowns[i][4].value for i in range(len(AvailableSessionDirB)) if session_checkboxes[i][0].value]
            selected_DLCmodel = [session_dropdowns[i][5].value for i in range(len(AvailableSessionDirB)) if session_checkboxes[i][0].value]
            selected_DLCmodel_2 = [session_dropdowns[i][6].value for i in range(len(AvailableSessionDirB)) if session_checkboxes[i][0].value]
            selected_DLCmodel_3 = [session_dropdowns[i][7].value for i in range(len(AvailableSessionDirB)) if session_checkboxes[i][0].value]
            entered_session_note = [session_dropdowns[i][8].value for i in range(len(AvailableSessionDirB)) if session_checkboxes[i][0].value]

            selected_scans = get_scan_key_from_dir(selected_sessions)
            selected_sessions = get_session_key_from_dir(selected_sessions)

            selected_s2pparms_index = s2pparm[0][selected_s2pparms]
            selected_s2pparms_text = s2pparm[1][selected_s2pparms]

            output.clear_output()

            
            with output:
                # OUTPUT FUNCTION: PERFORM INGESTIONS AND JOBS BASED ON SELECTION

                # Ingest selected sessions here
                populate_settings =  {'display_progress': False, 'suppress_errors': True} 

                ## SESSION processing
                # ToDo: Make a separate CURATION PROCESSING loop - and Exclude CURATION-CHECKED AND UNCHECKED HERE. 
                # If BOTH run and curation are clicked, generate a LOAD processing task, a manual curation task and corresponding populations
                for i, sessi in enumerate(tqdm(selected_sessions, desc='Current Session')):
                    isess.ingest_session_scan(sessi, verbose=False, project_key=selected_projects[i], equipment_key=selected_equipment[i], location_key=selected_locations[i], software_key='ScanImage')
                    # update / insert session info based on user choice from above
                    try: #now update1
                        session.SessionNote.update1({'session_id': sessi, 'session_note': entered_session_note[i]})
                    except:
                        session.SessionNote.insert1({'session_id': sessi, 'session_note': entered_session_note[i]})
                    
                    session.SessionSameSite.update1({'session_id': sessi, 'same_site_id': selected_same_site[i]})
                    
                    # update scaninfo based on user choice from above
                   
                    ## SCAN Processing 
                    # get the scans associated with a session
                    query_all = session.Session() * scan.Scan() & f'session_id = "{sessi}"'
                    scans_to_process = query_all.fetch("scan_id")
                    
                    for j, scansi in enumerate(scans_to_process):
                        try:
                            scan.Scan.update1({'session_id': sessi, 'scan_id': scansi, 'scan_notes': entered_session_note[i]})
                        except:
                            scan.Scan.insert1({'session_id': sessi, 'scan_id': scansi, 'scan_notes': entered_session_note[i]})
                        try:
                            scan.ScanLocation.update1({'session_id': sessi, 'scan_id': scansi, 'anatomical_location': selected_locations[i]})
                        except: 
                            scan.ScanLocation.insert1({'session_id': sessi, 'scan_id': scansi, 'anatomical_location': selected_locations[i]})
                        
                        query = scan.ScanPath() & 'scan_id = "' + scansi + '"'
                        dir_proc = query.fetch('path')[0]

                        # push scan to ProcessingTask
                        # TODO: handle multiscan concatenation from here?
                        if selected_s2pparms_text[i] != 'dummy': #dummy is a placeholder for no processing
                            imaging.ProcessingTask.insert1((sessi, scansi, selected_s2pparms_index[i], dir_proc, ingest_opt), skip_duplicates=True)
                        

                            
                    scan.ScanInfo.populate(query_all, **populate_settings) 
                    
                    ## AUX Processing
                    for j, scansi in enumerate(scans_to_process):
                                            # get the equipment/experiment class from userfunction_consolidate_files associated with a scan

                        print('- - - -')
                        print('- Ingesting AUX and STIM for scan:', scansi)
                        try:
                            aux_setup_typestr = (scan.ScanInfo() & 'scan_id = "' + scansi + '"').fetch("userfunction_info")[0]    
                        except Exception as e:
                            aux_setup_typestr = "bench2p_Oddball_V2"
                            print(f"Error occurred: {e}")
                            print(f'Failed to fetch aux_setup_typestr for scan = "{scansi}"')
                            print('setting aux_setup_typestr:', aux_setup_typestr)
                            print('but this wont be enough')
                            continue

                        try:
                            ibe.ingest_aux(sessi,scansi,verbose=False, aux_setup_type=aux_setup_typestr) # ingests AUX into event.Event and event.BehaviorRecording
                        except Exception as e:
                            print(f"Error occurred: {e}")
                            print(f'Failed to ingest aux for scan = "{scansi}"')
                            
                        try:
                            
                            if aux_setup_typestr == "bench2p" or aux_setup_typestr == "bench2p_kine" or aux_setup_typestr == "bench2p_oddball" or aux_setup_typestr == "bench2p_lineartrack" or aux_setup_typestr == "openfield" or aux_setup_typestr == "bench2p_Oddball_V2":
                                # print('bench2p - detected stimuli and trials:')
                                stimuli = (event.Event &  'scan_id = "' + scansi + '"' & 'event_type LIKE "%;%"').fetch("event_type")
                                trials = len(set(stimuli))
                                trial_stims = {x: list(stimuli).count(x) for x in stimuli}
                                # print(trial_stims)
                                
                                ibe.get_and_ingest_trial_times(scansi, aux_setup_typestr) 

                                # print('bench2p - Ingesting running wheel data')
                                
                            if aux_setup_typestr == "mini2p1_openfield":
                                # print('openfield - detected stimuli and trials:')
                                # stimuli = (event.Event &  'scan_id = "' + scansi + '"' & 'event_type LIKE "%;%"').fetch("event_type")
                                # trials = len(set(stimuli))
                                # trial_stims = {x: list(stimuli).count(x) for x in stimuli}
                                # print(trial_stims)
                                
                                ibe.get_and_ingest_trial_times(scansi, aux_setup_typestr) 

                                # print('openfield - Ingesting IMU data')
                        except Exception as e:
                            print(f"Error occurred: {e}")
                            print(f'Failed to ingest stim data for scan = "{scansi}"')
                            
                        try:
                            print('- Ingesting BPOD for scan:', scansi) #TR23 bench2p will also have bpod
                            ibe.ingest_bpod(sessi,scansi,verbose=False, aux_setup_type=aux_setup_typestr)
                        except Exception as e:
                            print(f"Error occurred: {e}")
                            print(f'Failed to ingest bpod data for scan = "{scansi}"')        
                    
                    ## Task processing and population
                    try:
                        if "bench2p" in aux_setup_typestr or "bench2p_kine" in aux_setup_typestr:
                            print('- Ingesting Treadmill for scan:', scansi)
                            ingestquery = (event.BehaviorRecording &  f'scan_id = "{scansi}"')
                            behavior.TreadmillRecording.populate(ingestquery, **populate_settings)
                        if "mini2p1" in aux_setup_typestr:
                            print('- Ingesting HARP IMU for scan:', scansi)
                            ingestquery = (event.BehaviorRecording &  f'scan_id = "{scansi}"')
                            behavior.HarpRecording.populate(ingestquery, **populate_settings)
                            behavior.CamSyncRecording.populate(ingestquery, **populate_settings)
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        print(f'Failed to populate HARP/Treadmill data for scan = "{scansi}"')  
                    ## Optitrack processing and population
                    try:
                        # Skip if no OptiTrack data is expected for this setup type
                        if "openfield" in aux_setup_typestr or "mini2p1" in aux_setup_typestr:
                           
                            # Initialize Mocap table if not already populated
                            # mocap.Mocap.insert1({
                            #     "mocap_name": "motive_raw_data",
                            #     "description": "The unprocessed export from Motive, with markers and rigid bodies",
                            # }, skip_duplicates=True)
                            
                            search_str, camera = 'ROS', 'mocap'
                            scan_key = (scan.Scan & f'scan_id = "{scansi}"').fetch('KEY')[0]
                            scan_path = (scan.ScanPath() & scan_key).fetch("path")[0]
                            
                            # Find OptiTrack files (CSV or TAK)
                            # mocapfiles = list(pathlib.Path(scan_path).glob(f"*{search_str}*.csv"))
                            # if not mocapfiles:
                            mocapfiles = list(pathlib.Path(scan_path).glob(f"*{search_str}*.tak*"))
                            
                            if mocapfiles:
                                mocappath = str(mocapfiles[0])
                                
                                # Insert MocapRecording
                                key = scan_key.copy()
                                key.update({'camera': camera})
                                mocap.MocapRecording.insert1(key, skip_duplicates=True)
                                
                                # Insert file and populate recording info
                                key.update({'file_path': mocappath, 'file_id': 0})
                                mocap.MocapRecording.File.insert1(key, ignore_extra_fields=True, skip_duplicates=True)
                                mocap.MocapRecordingInfo.populate(key, **populate_settings)
                                
                                # Insert task and populate data
                                key = scan_key.copy()
                                key.update({'mocap_name': 'motive_raw_data'})
                                mocap.MotionCaptureTask.insert1(key, ignore_extra_fields=True, skip_duplicates=True)
                                mocap.MotionCapture.populate(key, **populate_settings)
                                print(f"- Populated OptiTrack data for scan: {scansi}")
                    except Exception as e:
                        print(f"Failed to populate OptiTrack data for scan = {scansi}: {e}")
                    
                    ## Cascade processing and population
                    try:
                        indicator = (subject.Subject * session.Session()  * subject.Line()  & f'scan_id = "{scansi}"').fetch1('line_name')
                        print(f"Indicator: {indicator}")

                        if 'GCaMP8s' in indicator:
                            modelname = 'GC8s_EXC_30Hz_smoothing25ms_high_noise'
                        elif 'GCaMP6s' in indicator:
                            modelname = 'Global_EXC_30Hz_smoothing50ms_high_noise'

                        print(f"Model name: {modelname}")

                        insertkey = (imaging.Fluorescence * imaging.ProcessingParamSet.proj('processing_method') * imaging.ActivityExtractionMethod
                            & f'scan_id = "{scansi}"'
                            & f'paramset_idx = {selected_s2pparms_index[i]}'
                            & f'curation_id = 1'
                            & 'extraction_method = "cascade_inference"').fetch1()


                        insertkey['model_name'] = modelname

                        imaging.ActivityCascadeTask.insert1(insertkey, ignore_extra_fields=True, skip_duplicates=True)
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        print(f'Failed to populate CASCADE data for scan = "{scansi}"')  
                    
                    ## DLC processing and population
                    if aux_setup_typestr == "mini2p1_openfield" or aux_setup_typestr == "openfield" or "bench2p" in aux_setup_typestr:
                        try:
                            # if selected_DLCmodel[i] != "dummy":
                                ## OLD MODEL TABLE
                                # for j, scansi in enumerate(scans_to_process):
                                    
                                #     aux_setup_typestr = (scan.ScanInfo() & 'scan_id = "' + scansi + '"').fetch("userfunction_info")[0]

                                #     print('- - - -')
                                #     print('DLC task insertion & pose estimation:', scansi)
                                    
                                #     # insert TOP movie into model table
                                #     scan_key = (scan.Scan & f'scan_id = "{scansi}"').fetch('KEY')[0] 
                                #     moviepath = str(list(pathlib.Path((scan.ScanPath() & scan_key).fetch("path")[0]).glob("*top*.mp4*"))[0])
                                    
                                #     key = {'session_id': scan_key["session_id"],
                                #         'recording_id': scan_key["scan_id"], 
                                #         'camera': "mini2p1_top", # Currently 'scanner' due to in equipment tables
                                #         }
                                #     model.VideoRecording.insert1(key, skip_duplicates=True)
                                #     key.update({'file_path': moviepath,
                                #                 'file_id': 0})  #INCREMENT FILE_ID WITH CAM NUMBER?
                                    
                                #     model.VideoRecording.File.insert1(key, ignore_extra_fields=True, skip_duplicates=True)
                                    
                                #     key =  (model.VideoRecording & f'recording_id="{scansi}"').fetch1('KEY')
                                #     key.update({'model_name': selected_DLCmodel[i], 'task_mode': 'trigger'})        
                                    
                                #     model.PoseEstimationTask.insert_estimation_task(key, key["model_name"], analyze_videos_params={'save_as_csv':True, 'dynamic':(True,.5,60)}) # dynamic cropping
                            
                            if selected_DLCmodel[i] != "dummy":
                                ## NEW MODEL TABLE
                                for j, scansi in enumerate(scans_to_process):

                                    aux_setup_typestr = (scan.ScanInfo() & 'scan_id = "' + scansi + '"').fetch("userfunction_info")[0]

                                    print('- - - -')
                                     
                                    
                                    try:
                                        search_str = str.split(selected_DLCmodel[i], ';')[2]
                                                                            
                                        search_str = search_str.replace(" ", "")
                                        
                                        camera = usercam_defaults[i][0]
                                        
                                    except:
                                        print('Old model name without search string - assuming top')
                                        search_str = "top"
                                        camera = "mini2p1_top"

                                    
                                    print('DLC1 task insertion & pose estimation:', scansi)
                                    
                                    # insert TOP movie into model table
                                    scan_key = (scan.Scan & f'scan_id = "{scansi}"').fetch('KEY')[0]                                                                         
                                    moviepath = str(list(pathlib.Path((scan.ScanPath() & scan_key).fetch("path")[0]).glob(f"*{search_str}*.mp4*"))[0])

                                    recid = scan_key["scan_id"] + '_' + camera

                                    key = {'session_id': scan_key["session_id"],
                                        'scan_id': scan_key["scan_id"], 
                                        'recording_id': recid,
                                        'camera': camera, # Currently 'scanner' due to in equipment tables
                                    }

                                    model.VideoRecordingNew.insert1(key, skip_duplicates=True)
                                    if aux_setup_typestr == "mini2p1_openfield":
                                        model.DLCliveRecording.insert1(key, skip_duplicates=True)

                                    key.update({'file_path': moviepath,
                                                'file_id': 0})  #INCREMENT FILE_ID WITH CAM NUMBER?
                                    
                                    model.VideoRecordingNew.File.insert1(key, ignore_extra_fields=True, skip_duplicates=True)
                                    if aux_setup_typestr == "mini2p1_openfield":
                                        model.DLCliveRecording.File.insert1(key, ignore_extra_fields=True, skip_duplicates=True)
                                    
                                    
                                    key =  (model.VideoRecordingNew & f'recording_id="{recid}"').fetch1('KEY')
                                    key.update({'model_name': selected_DLCmodel[i], 'task_mode': 'trigger'})        
                                    
                                    model.PoseEstimationTaskNew.insert_estimation_task(key, key["model_name"], analyze_videos_params={'save_as_csv':True, 'dynamic':(True,.5,60)}) # dynamic cropping          
                                    
                                    if aux_setup_typestr == "mini2p1_openfield":
                                        key.update({'model_name': selected_DLCmodel[i], 'task_mode': 'load'}) 
                                        model.DLCLivePoseEstimationTask.insert1(key, ignore_extra_fields=True, skip_duplicates=True) 
                                    
                            if selected_DLCmodel_2[i] != "dummy":
                                
                                ## NEW MODEL TABLE
                                for j, scansi in enumerate(scans_to_process):
                                    
                                    aux_setup_typestr = (scan.ScanInfo() & 'scan_id = "' + scansi + '"').fetch("userfunction_info")[0]
                                    search_str = str.split(selected_DLCmodel_2[i], ';')[2]
                                    search_str = search_str.replace(" ", "")

                                    camera = usercam_defaults[i][1]

                                    print('- - - -')
                                     
                                    print('DLC2 task insertion & pose estimation:', scansi)
                                    
                                    # insert TOP movie into model table
                                    scan_key = (scan.Scan & f'scan_id = "{scansi}"').fetch('KEY')[0] 
                                    moviepath = str(list(pathlib.Path((scan.ScanPath() & scan_key).fetch("path")[0]).glob(f"*{search_str}*.mp4*"))[0])
                                    
                                    recid = scan_key["scan_id"] + '_' + camera

                                    key = {'session_id': scan_key["session_id"],
                                        'scan_id': scan_key["scan_id"], 
                                        'recording_id': recid,
                                        'camera': camera, # Currently 'scanner' due to in equipment tables
                                    }

                                    model.VideoRecordingNew.insert1(key, skip_duplicates=True)
                                    key.update({'file_path': moviepath,
                                                'file_id': 0})  #INCREMENT FILE_ID WITH CAM NUMBER?
                                    
                                    model.VideoRecordingNew.File.insert1(key, ignore_extra_fields=True, skip_duplicates=True)
                                    
                                    key =  (model.VideoRecordingNew & f'recording_id="{recid}"').fetch1('KEY')
                                    key.update({'model_name': selected_DLCmodel_2[i], 'task_mode': 'trigger'})        
                                    
                                    model.PoseEstimationTaskNew.insert_estimation_task(key, key["model_name"], analyze_videos_params={'save_as_csv':True, 'dynamic':(True,.5,60)}) # dynamic cropping          

                            if selected_DLCmodel_3[i] != "dummy":
                                ## NEW MODEL TABLE
                                for j, scansi in enumerate(scans_to_process):
                                    
                                    aux_setup_typestr = (scan.ScanInfo() & 'scan_id = "' + scansi + '"').fetch("userfunction_info")[0]
                                    search_str = str.split(selected_DLCmodel_3[i], ';')[2]
                                    search_str = search_str.replace(" ", "")

                                    camera = usercam_defaults[i][2]

                                    print('- - - -')
                                     
                                    print('DLC3 task insertion & pose estimation:', scansi)
                                    
                                    if "eye1" in search_str:
                                        print('eye1 detected - replacing with eye2')
                                        search_str = search_str.replace("eye1", "eye2")
                                    
                                    # insert TOP movie into model table
                                    scan_key = (scan.Scan & f'scan_id = "{scansi}"').fetch('KEY')[0] 
                                    moviepath = str(list(pathlib.Path((scan.ScanPath() & scan_key).fetch("path")[0]).glob(f"*{search_str}*.mp4*"))[0])
                                    
                                    recid = scan_key["scan_id"] + '_' + camera

                                    key = {'session_id': scan_key["session_id"],
                                        'scan_id': scan_key["scan_id"], 
                                        'recording_id': recid,
                                        'camera': camera, # Currently 'scanner' due to in equipment tables
                                    }

                                    model.VideoRecordingNew.insert1(key, skip_duplicates=True)
                                    

                                    
                                    key.update({'file_path': moviepath,
                                                'file_id': 0})  #INCREMENT FILE_ID WITH CAM NUMBER?
                                    
                                    model.VideoRecordingNew.File.insert1(key, ignore_extra_fields=True, skip_duplicates=True)

                                    
                                    key =  (model.VideoRecordingNew & f'recording_id="{recid}"').fetch1('KEY')
                                    key.update({'model_name': selected_DLCmodel_3[i], 'task_mode': 'trigger'})        
                                    
                                    model.PoseEstimationTaskNew.insert_estimation_task(key, key["model_name"], analyze_videos_params={'save_as_csv':True, 'dynamic':(True,.5,60)}) # dynamic cropping          
                                


                            if do_population:
                                model.RecordingInfoNew.populate(query_all, **populate_settings)
                                model.PoseEstimationNew.populate(query_all, **populate_settings)
                        except Exception as e:
                            tb = traceback.format_exc()
                            print(f"Error occurred: {e}")
                            print(f'Failed to populate DLC data for scan = "{scansi}"') 
                            print(f"Error on line {tb.split('line ')[1].split(',')[0]}")
                    
                    if rspace_upload:
                        try: 
                            print('- - - -')
                            print('Make RSpace entries', scansi)       
                            ## Make RSpace entries 
                            query = session.Session() * subject.User() & f'session_id = "{sessi}"'
                            animalID = query.fetch("subject")[0]
                            date = query.fetch("session_datetime")[0].strftime("%Y-%m-%d")
                            userID = query.fetch("initials")[0]
                            sessionID = query.fetch("session_id")[0]
                            fetchtable = session.Session() * scan.ScanPath() * session.SessionUser() * session.ProjectSession() * session.SessionNote() * session.SessionSameSite() * scan.Scan() *scan.ScanInfo() & f'session_id = "{sessi}"'
                            # include stimulus / task info
                            
                            make_rspace_session_document(animalID, sessi, date, userID, fetchtable)
                        except Exception as e:
                            print(f"Error occurred: {e}")
                            print(f'Failed to upload Rspace data for session = "{sessi}"') 

                
                    

                # RUN PRIMARY POPULATION TASKS (S2P)
                
                if do_population:
                    
                    print("\n---- Populate imported and computed imaging tables ----")
                    
                    try:
                        imaging.Processing.populate(**populate_settings)

                        try:
                            for i, sessi in enumerate(set(selected_sessions)):
                                # get the scans associated with a session
                                query = session.Session() * scan.Scan() & f'session_id = "{sessi}"'
                                scans_to_process = query.fetch("scan_id")

                                for j, scansi in enumerate(scans_to_process):
                                    scan_key = (scan.Scan() & f'scan_id = "{scansi}"').fetch1('KEY')
                                    processing_key = (imaging.Processing & scan_key).fetch1('KEY')
                                    # manual_curation = session_checkboxes[i][1].value
                                    manual_curation = False # TODO: make this a checkbox
                                    # do a new Curation task
                                    imaging.Curation().create1_from_processing_task(key, is_curated=manual_curation)

                                    # imaging.Curation().create1_from_processing_task({'session_id': sessi, 'scan_id': scansi, "paramset_idx": selected_s2pparms_index[i], "manual_curation": session_checkboxes[i][1].value})
                                    #load the scan, session s2p parameter index from the proccessing table (the first s2p run)
                                    
                                    # set manual curation to TRUE
                                    
                                    
                        except Exception as e:
                            print(f"Error occurred: {e}")
                            print(f'No curation entries found for {sessi}')                      

                        imaging.MotionCorrection.populate(**populate_settings)

                        imaging.Segmentation.populate(**populate_settings)

                        imaging.MaskClassification.populate(**populate_settings)

                        imaging.Fluorescence.populate(**populate_settings)

                        imaging.Activity.populate(**populate_settings)

                        print("\n---- Successfully completed workflow ----")

                        ## Append RSpace Template figures
                        # TODO: make movies and upload as well 
                        if rspace_upload:
                            for i, sessi in enumerate(set(selected_sessions)):
                                # get the scans associated with a session
                                query = session.Session() * scan.Scan() & f'session_id = "{sessi}"'
                                scans_to_process = query.fetch("scan_id")
                                for j, scansi in enumerate(scans_to_process):
                                    session_key = (session.Session & f'session_id = "{sessi}"').fetch('KEY')[0]
                                    scan_key = (scan.Scan & f'scan_id = "{scansi}"').fetch('KEY')[0]
                                    curation_key = (imaging.Curation & scan_key & 'curation_id=1').fetch1('KEY')
                                    print(curation_key)
                                    rspace_id = (session.SessionRspace & f'session_id = "{sessi}"').fetch('rspace_id')[0]
                            
                                    make_overview_figures_rspace(curation_key, rspace_id)
                                    
                                    print('making movie')
                                    make_overview_movies_rspace(curation_key, rspace_id)
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        print(f'IMAGING POPULATION DID NOT RUN FOR {sessi}')
            
                # Log output
                # date_time_string = now.strftime('%Y-%m-%d_%H-%M-%S')
                # file_path = f"/home/tobiasr/adamacs/notebooks/logs/{date_time_string}ingestion_log.log"

                # with open(file_path, "a") as f:
                #     for out in output.outputs:
                #         if 'text' in out:
                #             f.write(out['text'])  

    # Attach the callback function to the commit button
    commit_button.on_click(commit_button_clicked)



    # Display the output
    display(output)
        




def make_rspace_session_document(animalID, sessionID, date, userID, fetchtable):
    # find 'Experiments' folder and get ID
    folders = api.list_folder_tree()
    experiments_ids = []
    for record in folders['records']:
        if record['name'] == 'Experiments':
            experiments_ids.append(record['id'])

    # find folders under 'Experiments' folder
    names_and_ids = []
    subfolders = api.list_folder_tree(experiments_ids[0])
    names_and_ids.append([(record['name'], record['id']) for record in subfolders['records']])

    # find matches with animal IDs
    match_found = any(animalID in item[0] for sublist in names_and_ids for item in sublist)

    # if found, find folder id matching that animal
    if match_found:
        save_folder_id = [x[1] for sublist in names_and_ids for x in sublist if animalID in x[0]][0]
    else:
        # if not generate the directory
        new_folder = api.create_folder(name=f"{userID}_{animalID}", parent_folder_id=experiments_ids[0], notebook=True)
        save_folder_id = new_folder['id']


    # find matches with document IDs

    # find all documents in the animal folder
    all_animal_documents = api.list_folder_tree(save_folder_id)
    doc_names_and_ids = []
    doc_names_and_ids.append([(record['name'], record['id']) for record in all_animal_documents['records']])
    # find matches with sessionID
    match_found = any(sessionID in item[0] for sublist in doc_names_and_ids for item in sublist)

    if match_found:
        ids = [id for string, id in doc_names_and_ids[0] if sessionID in string]
        new_doc = {"id": ids[0]}
    else:
        new_doc = api.create_document(name=f"{date}_{sessionID}", parent_folder_id=save_folder_id)

    # create RSpace document
    df = fetchtable.fetch(format='frame')
    html_table = df.to_html()
    content = html_table
    api.append_content(new_doc['id'], content)

    # ingest document in rspace table
    session.SessionRspace.insert1({'session_id': sessionID, 'rspace_id': new_doc['id']}, skip_duplicates=True)
    # ingest animal folder id in subject Rspace table
    subject.SubjectRspace.insert1({'subject': animalID, 'rspace_subject_id': save_folder_id}, skip_duplicates=True)
    # session.SessionRspace.insert1({'rspace_id': new_doc['id'], 'rspace_url': new_doc['_links'][0]['link']})



def get_session_dir_key_from_dir(directory):
    return [path.split('/')[-1] for path in directory]
     
def get_scan_dir_key_from_dir(directory):
    return [path.split('/')[-1] for path in directory]

def get_session_key_from_dir(string):
    result = [re.search(r'sess\S+', item).group(0) for item in string]
    return result

def get_user_initials_from_dir(string):
    result = [name[:2] for name in string]
    return result

def get_subject_key_from_dir(string):
    result = [item.split("_")[1] for item in string]
    return result

def get_date_key_from_dir(directory):
    return directory.split("_")[-3]

def get_scan_key_from_dir(string):
    result = [re.search(r'scan\S+_', item).group(0)[:-1] for item in string]
    return result

def unique_directory_strings(dirs1, dirs2):
    set1 = set(dirs1)
    set2 = set(dirs2)
    common_dirs = list(set1.intersection(set2))
    unique_dirs = list(set(set1.union(set2)) - set(common_dirs))
    return unique_dirs

def get_user_defaults(directory):
    JJ = [4, 7, 7, 0, 2, 8, 8]
    RN = [2, 2, 0, 0, 7, 8, 8]
    TR = [2, 2, 0, 0, 7, 8, 8]
    LK = [2, 2, 0, 3, 7, 8, 8]
    DB = [4, 0, 1, 0, 8, 8, 8]
    NK = [7, 5, 1, 13, 11, 12, 12]
    LE = [9, 9, 0, 7, 10, 8, 8]
    AM = [8, 8, 0, 7, 7, 8, 8]
    AA = [8, 8, 0, 7, 7, 8, 8]
    SM = [5, 8, 7, 0, 2, 8, 8]
    YH = [9, 8, 0, 7, 10, 8, 8]
    KH = [8, 9, 0, 7, 10, 8, 8]
    
    user_initials = get_user_initials_from_dir(directory)

    user_arrays = {}
    for initial in user_initials:
        values = locals()[initial]
        user_arrays[initial] = values

    new_array = [user_arrays[initial] for initial in user_initials]
    return new_array

def get_user_cam_defaults(directory):
    JJ = ['mini2p1_top', 'mini2p1_eye_left', 'mini2p1_eye_right']
    RN = ['bench2p_face', 'bench2p_body', 'bench2p_back']
    TR = ['mini2p1_top', 'mini2p1_eye_left', 'mini2p1_eye_right']
    LK = ['bench2p_face', 'bench2p_body', 'bench2p_back']
    DB = ['bench2p_face', 'bench2p_body', 'bench2p_back']
    NK = ['mini2p1_top', 'mini2p1_eye_left', 'mini2p1_eye_right']
    LE = ['bench2p_face', 'bench2p_body', 'bench2p_back']
    AM = ['bench2p_face', 'bench2p_body', 'bench2p_back']
    AA = ['bench2p_face', 'bench2p_body', 'bench2p_back']
    SM = ['mini2p1_top', 'mini2p1_eye_left', 'mini2p1_eye_right']
    YH = ['bench2p_face', 'bench2p_body', 'bench2p_back']
    KH = ['bench2p_face', 'bench2p_body', 'bench2p_back']

    
    user_initials = get_user_initials_from_dir(directory)

    user_arrays = {}
    for initial in user_initials:
        values = locals()[initial]
        user_arrays[initial] = values

    new_array = [user_arrays[initial] for initial in user_initials]
    return new_array


def make_overview_figures_rspace(curation_key, rspace_id):
    # Figure Style settings for notebook.

    import matplotlib as mpl
    mpl.rcParams.update({
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'figure.subplot.wspace': .01,
        'figure.subplot.hspace': .01,
        'figure.figsize': (18, 13),
        'ytick.major.left': False,
        'xtick.major.bottom': False
    })
    jet = mpl.cm.get_cmap('jet')
    jet.set_bad(color='k')

    ref_image = (imaging.MotionCorrection.Summary & curation_key & 'field_idx=0').fetch1('ref_image')
    average_image = (imaging.MotionCorrection.Summary & curation_key & 'field_idx=0').fetch1('average_image')
    correlation_image = (imaging.MotionCorrection.Summary & curation_key & 'field_idx=0').fetch1('correlation_image')
    max_proj_image = (imaging.MotionCorrection.Summary & curation_key & 'field_idx=0').fetch1('max_proj_image')
    
    plt.ioff()
    plt.subplot(1, 4, 1)
    plt.imshow(ref_image, cmap='gray', )
    plt.title("Reference Image for Registration");

    plt.subplot(1, 4, 2)
    plt.imshow(average_image, cmap='gray')
    plt.title("Registered Image, Mean Projection");

    plt.subplot(1, 4, 3)
    plt.imshow(max_proj_image, cmap='gray')
    plt.title("Registered Image, Max Projection")

    plt.subplot(1, 4, 4)
    plt.imshow(correlation_image, cmap='gray')
    plt.title("Registered Image, Correlation Map")

    tmpdir = dj.config['custom'].get('suite2p_fast_tmp')[0]
    session_id = curation_key['session_id']
    scan_id = curation_key['scan_id']
    save_path = os.path.join(tmpdir, f'{session_id}_{scan_id}_templates.png')
    plt.savefig(save_path)
    plt.clf()

    # print(f'rspace_id' {rspace_id})
    # print(f'save_path' {save_path})
    # print(f'session_id' {session_id})
    print('bypassing upload')
    # append_rspace_session_image(rspace_id, save_path, captionimage='Suite2p Templates ' + session_id)   
    plt.ion()

    
def make_overview_movies_rspace(curation_key, rspace_id):
    # params_key = (imaging.ProcessingParamSet & 'paramset_idx = "4"').fetch('KEY')
    # reg_tiffs_available = (imaging.ProcessingParamSet & params_key).fetch("params")[0]['reg_tif']
    from scipy.ndimage import mean
    import tifffile
    path = (scan.ScanPath & curation_key).fetch1("path") + ("/suite2p/plane0/reg_tif")

    # path = '/datajoint-data/data/jisooj/RN_OPI-1681_2023-02-15_scan9FGLEFJ3_sess9FGLEFJ3/suite2p_exp9FGLEFJ3/suite2p/plane0/reg_tif'
    # Get a list of all tiff files in the folder
    tiff_files = [os.path.join(path, f) for f in natsorted(os.listdir(path)) if f.endswith('.tif')]

    # print(tiff_files)

    # Load each tiff stack into a list of numpy arrays
    stacks = []
    for f in tiff_files:
        with tifffile.TiffFile(f) as tif:
            # Get the number of pages in the file
            num_pages = len(tif.pages)
            
            # Create a numpy array to store all pages
            stack = np.zeros((num_pages,) + tif.pages[0].shape, dtype=tif.pages[0].dtype)
            
            # Iterate over the pages and store them in the array
            for i, page in enumerate(tif.pages):
                stack[i] = page.asarray()

        stacks.append(stack)

    # Concatenate the stacks into a single numpy array along the z-axis
    volume = np.concatenate(stacks, axis=0)

    # delete registration tiff
    for f in tiff_files:
        os.remove(f) 
    
    ### moving average filter
    # Create a running Z mean projection of the volume

    runav = 20
    # running_z_projection = uniform_filter_mt(volume, size=(runav,xyrunav,xyrunav))
    running_z_projection = sh.rolling_average_filter(volume, runav)

    session_id = curation_key['session_id']
    scan_id = curation_key['scan_id']

    filename = os.path.join(path, 'registered_movie_' + session_id + '_' + scan_id + '_' + str(runav) + '_frame_runningaverage2' + '.mp4')

    fps = 120   # frames per second - 120 default
    p1 = 2       # percentile scaling low - 1 default
    p2 = 99.998  # percentile scaling high - 99.995 default

    rescaled_image_8bit = sh.make_stack_movie(running_z_projection, filename, fps, p1, p2)

    tmpdir = dj.config['custom'].get('suite2p_fast_tmp')[0]


    ### movie uload does not work yet
    # append_rspace_session_image(rspace_id, filename, captionimage='registered movie ' + scan_id)   

def append_rspace_session_image(rspace_id, filename, captionimage='no caption'):
    with open(filename, 'rb') as f:
        uploaded_image = api.upload_file(f, caption=captionimage)
        
        content = f"""
        <p>Suite2p templates of dataset {filename}
        <p>
        <fileId={uploaded_image['id']}>
        """
        api.append_content(rspace_id, content)
        
        print(f"uploaded image id = {uploaded_image['id']}")