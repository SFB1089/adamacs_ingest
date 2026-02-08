#!/usr/bin/env python
# coding: utf-8

# Tobias Rose 2020
# THIS IS JUST FOR INTERIM QUICK EXTRACTION - THE DATAJOINT PIPELINE IS ALREADY MORE ADVANCED
# TR 10/2021 - updated function syntax (header parsing much simpler. Switch to Json in the future, still
# TR 11/2022 - updated parts to handle our new naming scheme 

import numpy as np
from pathlib import Path
import glob
import os
import sys
import tifffile
from suite2p import default_ops, run_s2p
ops = default_ops() # populates ops with the default options

# option to import from github folder
sys.path.insert(0, 'C:/Users/trose/Documents/GitHub/suite2p')
from shutil import copy, rmtree
from tqdm import tqdm
#from helpers import parse_SI_header as pSI
from ScanImageTiffReader import ScanImageTiffReader
# from line_profiler import LineProfiler

def batch_s2p( runops ):
    "Batch run for s2p'ing through experiments. Settings are currently for mini2p (to be optimized)"

    # populates s2p ops with the default options

    # Unpack option list
    exp = runops.get('exp')
    concat = runops.get('concat')
    main_root = runops.get('main_root')

    adata = runops.get('adata')
    ftiff = runops.get('ftiff')
    ftemp = runops.get('ftemp')
    nonrigid = runops.get('nonrigid')
    use_builtin_classifier = runops.get('use_builtin_classifier')
    classifier_path = runops.get('classifier_path')
    darkframes = runops.get('darkframes')
    readfiles = runops.get('readfiles')

    # make subfolders
    ftemp_tmp = os.path.join(ftemp, 'suite2p')
    Path(ftemp_tmp).mkdir(parents=True, exist_ok=True)
    print('- - - - - - - -')
    print('Made directory (ftemp_tmp) ' + ftemp_tmp)

    # Concatenation case folder settings -> s2p root folder is the concat tiff folder
    if concat:
        froot = ftiff
        fsave_tmp =  os.path.join(froot, 'suite2p_exp')
        fsave = fsave_tmp

        #rmtree(ftemp_tmp)
        #print('removed ' + ftemp_tmp)

        Path(fsave_tmp).mkdir(parents=True, exist_ok=True)
        print('Made directory (fsave_tmp) ' + fsave_tmp)
        print('- - - - - - - -')
        print('Concatenating all experiments in single tiff folder')

        bad_frames = np.linspace(0, darkframes-1, darkframes)
        np.save(fsave_tmp + '/bad_frames.npy', bad_frames)
        np.save(ftemp_tmp + '/bad_frames.npy', bad_frames)

        files_fullpath = []
        files_name = []
        for val in exp:
            files = list(Path(main_root).rglob('scan'+val+'*.tif')) #recursive search over main_root
            files = sorted(files)
            files = files[0:readfiles]
            for n,f in enumerate(files):
                copy(files[n], froot)
                targetstring = [str(files[n]), str(froot)]
                print('copied {} to {}'.format(*targetstring))
                files_fullpath.append(str(files[n]))
                files_name.append(f.name)
        files_fullpath = sorted(files_fullpath)
        files_name = sorted(files_name)

        info = si_info( files_fullpath )

        newops = {
            'exp': exp,
            'files_name': files_name,
            'main_root': main_root,
            'adata': adata,
            'ftemp': ftemp,
            'ftiff': ftiff,
            'froot': froot,
            'fsave': fsave,
            'level': info.get('level'),
            'channels': info.get('channels'),
            'framerate': info.get('framerate'),
            'nonrigid': nonrigid,
            'use_builtin_classifier': use_builtin_classifier,
            'classifier_path': classifier_path
            }
        


        db = s2p_ownops(newops)
        opsEnd = run_s2p(ops=ops, db=db)

        rmtree(ftemp_tmp)
        print('removed ' + ftemp_tmp)

    else:
        for ids, val in tqdm(enumerate(exp)):

            files_fullpath = []
            files_name = []

            files = list(Path(main_root).rglob('scan'+val+'*.tif')) #recursive
            files = files[0:readfiles]

            froot = os.path.join(os.path.dirname(files[0]))

            fsave_tmp = os.path.join(froot, 'suite2p_exp' + exp[ids])
            fsave = fsave_tmp

            if runops['freshrun']:
                print('freshrun')
                try:
                    rmtree(fsave_tmp)
                    print('removed ' + fsave_tmp)
                except:
                    print('')

            Path(fsave_tmp).mkdir(parents=True, exist_ok=True)
            print('Made directory (fsave_tmp) ' + fsave_tmp)

            print('- - - - - - - -')
            print('Processing exp' + val)
            print('- - - - - - - -')
            print('Folders used')
            print('Fast disk (ftemp) ' + ftemp)
            print('Data folder (froot) ' + froot)
            print('Save path (fsave) ' + fsave)
            print('- - - - - - - -')

            for n,f in enumerate(files):
                #files[n] = os.path.basename(f)
                files_fullpath.append(str(files[n]))
                files_name.append(f.name)

            files_fullpath = sorted(files_fullpath)
            files_name = sorted(files_name)

            bad_frames = np.linspace(0, darkframes-1, darkframes)
            np.save(fsave_tmp + '/bad_frames.npy', bad_frames)
            np.save(ftemp_tmp + '/bad_frames.npy', bad_frames)

            info = si_info( files_fullpath )

            newops = {
                'exp': exp,
                'files_name': files_name,
                'main_root': main_root,
                'adata': adata,
                'ftemp': ftemp,
                'ftiff': ftiff,
                'froot': froot,
                'fsave': fsave,
                'level': info.get('level'),
                'channels': info.get('channels'),
                'framerate': info.get('framerate'),
                'nonrigid': nonrigid,
                'use_builtin_classifier': use_builtin_classifier,
                'classifier_path': classifier_path
                }
            
            db = s2p_ownops(newops)
            opsEnd = run_s2p(ops=ops, db=db)
            
            return opsEnd, db 
            
            #rmtree(ftemp_tmp)
            #print('removed ' + ftemp_tmp)

def s2p_clean( runops ):

    print('- - - - - - - - -')
    print('S2P CLEANUP')
    ftemp_tmp = runops.get('ftemp')

    sys.stdout.write("remove " + ftemp_tmp + " ?")
    yes = {'yes','y', 'ye', ''}
    no = {'no','n'}
    choice = input().lower()

    if choice in yes:
        try:
            rmtree(ftemp_tmp)
            print('removed content of ' + ftemp_tmp)
        except:
            print("folder not found or files in use")
    if runops.get('concat'):
        ftiff = runops.get('ftiff')

        sys.stdout.write("remove " + ftiff + " ?")
        choice = input().lower()

        if choice in yes:
            try:
                rmtree(ftiff)
                print('removed content of ' + ftiff)
            except:
                print("folder not found or files in use")

def si_info( files_fullpath ):
    with ScanImageTiffReader(files_fullpath[0]) as reader:
        # header_slice = (reader.description(0))
        mov_dim = reader.shape()
        header = reader.metadata()


    splitlist = header.splitlines()
    SIheader = []
    for key_value in splitlist:
        if key_value == '':
            break
        key, value = key_value.split(' = ', 1)
        # print(key)
        if not SIheader or key in SIheader[-1]:
            SIheader.append({})
        SIheader[-1][key] = value
    SIheader = SIheader[0]
    
    level = int(SIheader["SI.hStackManager.actualNumSlices"])
    zoom = float(SIheader["SI.hRoiManager.scanZoomFactor"])
    framerate = float(SIheader["SI.hRoiManager.scanFrameRate"])
    try:
        channels = len(eval(SIheader["SI.hChannels.channelSave"].replace(" ",','))) #TR21: leaving eval here as example to get the channel array
    except:
        channels = 1   #TR23: cheap trick for single channel. 
    volumes = int(SIheader["SI.hStackManager.numVolumes"])
    frames =int(SIheader["SI.hStackManager.framesPerSlice"])
    frames_per_file = SIheader["SI.hScan2D.logFramesPerFile"]


    # level = pSI.parse_SI_header_level(header)
    # zoom = pSI.parse_SI_header_zoom(header)
    # framerate = pSI.parse_SI_header_FrameRate(header)
    # channels = pSI.parse_SI_header_Channels(header)
    # volumes = pSI.parse_SI_header_Volumes(header)
    # frames = pSI.parse_SI_header_Frames(header)
    # frames_per_file = pSI.parse_SI_header_FramesPerFile(header)


    si_info = {
        'dims': mov_dim,
        'level': level,
        'zoom': zoom,
        'framerate': framerate,
        'channels': channels,
        'volumes': volumes,
        'frames': frames,
        'frames_per_file': frames_per_file,
    }
    return si_info

def s2p_ownops(runops):
    db = {        
          # file input/output settings
        'look_one_level_down': False,  # whether to look in all subfolders when searching for tiffs
        'data_path': [runops.get('froot')],  # a list of folders with tiffs
        'fast_disk': runops.get('ftemp'),  # used to store temporary binary file, defaults to save_path0
        'delete_bin': False,  # whether to delete binary file after processing
        'mesoscan': False,  # for reading in scanimage mesoscope files
        'bruker': False,  # whether or not single page BRUKER tiffs!
        'bruker_bidirectional': False, # bidirectional multiplane in bruker: 0, 1, 2, 2, 1, 0 (True) vs 0, 1, 2, 0, 1, 2 (False)
        'h5py': [],  # take h5py as input (deactivates data_path)
        'h5py_key': 'data',  #key in h5py where data array is stored
        'nwb_file': '', # take nwb file as input (deactivates data_path)
        'nwb_driver': '', # driver for nwb file (nothing if file is local)
        'nwb_series': '', # TwoPhotonSeries name, defaults to first TwoPhotonSeries in nwb file
        'save_path0': runops.get('fsave'),
        'save_folder': [],
        'subfolders': [],
        'move_bin': True,  # if 1, and fast_disk is different than save_disk, binary file is moved to save_disk

        # main settings
        'nplanes' : runops.get('level'),  # each tiff has these many planes in sequence
        'nchannels' : runops.get('channels'),  # each tiff has these many channels per plane
        'functional_chan' : 1,  # this channel is used to extract functional ROIs (1-based)
        'tau':  1.2,  # this is the main parameter for deconvolution
        'fs': runops.get('framerate') / runops.get('level'),  # sampling rate (PER PLANE e.g. for 12 plane recordings it will be around 2.5)
        'force_sktiff': False, # whether or not to use scikit-image for tiff reading
        'frames_include': -1,
        'multiplane_parallel': False, # whether or not to run on server
        'ignore_flyback': [],

        # output settings
        'preclassify': 0.,  # apply classifier before signal extraction with probability 0.3
        'save_mat': True,  # whether to save output as matlab files
        'save_NWB': False,  # whether to save output as NWB file
        'combined': True,  # combine multiple planes into a single result /single canvas for GUI
        'aspect': 1.0,  # um/pixels in X / um/pixels in Y (for correct aspect ratio in GUI)

        # bidirectional phase offset
        'do_bidiphase': True,
        'bidiphase': 0,
        'bidi_corrected': False, #TR: this means if it is _already_ bidi-corrected. Will not run again if true

        # registration settings
        'do_registration': 1,  # whether to register data (2 forces re-registration)
        'two_step_registration': True,
        'keep_movie_raw': True,
        'nimg_init': 1000,  # subsampled frames for finding reference image
        'batch_size': 8000,  # number of frames per batch (needs to be 8000 or less if writing standard 16bit registered tiffs! Otherwise can be higher)
        'maxregshift': 0.1,  # max allowed registration shift, as a fraction of frame max(width and height)
        'align_by_chan' : 1,  # when multi-channel, you can align by non-functional channel (1-based)
        'reg_tif': True,  # whether to save registered tiffs
        'reg_tif_chan2': False,  # whether to save channel 2 registered tiffs
        'subpixel' : 10,  # precision of subpixel registration (1/subpixel steps)
        'smooth_sigma_time': 1,  # gaussian smoothing in time
        'smooth_sigma': 1.15,  # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
        'th_badframes': 1.0,  # this parameter determines which frames to exclude when determining cropping - set it smaller to exclude more frames
        'norm_frames': True, # normalize frames when detecting shifts
        'force_refImg': False, # if True, use refImg stored in ops if available
        'pad_fft': False,
        
        # non rigid registration settings
        'nonrigid': runops.get('nonrigid'),  # whether to use nonrigid registration
        'block_size': [128, 128],  # block size to register (** keep this a multiple of 2 **)
        'snr_thresh': 1.2,  # if any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing
        'maxregshiftNR': 5,  # maximum pixel shift allowed for nonrigid, relative to rigid

        # 1P settings
        '1Preg': False,  # whether to perform high-pass filtering and tapering
        'spatial_hp': 42,  # window for spatial high-pass filtering before registration
        'spatial_hp_reg': 42,  # window for spatial high-pass filtering before registration
        'spatial_hp_detect': 25,  # window for spatial high-pass filtering for neuropil subtraction before detection
        'pre_smooth': 0,  # whether to smooth before high-pass filtering before registration
        'spatial_taper': 40,  # how much to ignore on edges (important for vignetted windows, for FFT padding do not set BELOW 3*ops['smooth_sigma'])

        # cell detection settings
        'roidetect': True,  # whether or not to run ROI extraction
        'spikedetect': True,  # whether or not to run spike deconvolution
        'anatomical_only': 0, # use cellpose masks from mean image (no functional segmentation)
        'cellprob_threshold': 0.0, # cellprob_threshold for cellpose (if anatomical_only > 1)
        'flow_threshold': 1.5, # flow_threshold for cellpose (if anatomical_only > 1)
        'sparse_mode': True,  # whether or not to run sparse_mode
        'diameter': 0,  # if anatomical_only, use diameter for cellpose, if 0 estimate diameter
        'spatial_scale': 12,  # 0: multi-scale; 1: 6 pixels, 2spatial_hp_cp: 12 pixels, 3: 24 pixels, 4: 48 pixels
        'spatial_hp_cp': 0, # (int, default: 0) Window for spatial high-pass filtering of image to be used for cellpose.
        'connected': True,  # whether or not to keep ROIs fully connected (set to 0 for dendrites)
        'nbinned': 9000,  # max number of binned frames for cell detection
        'max_iterations': 30,  # maximum number of iterations to do cell detection
        'threshold_scaling': 1.0,  # adjust the automatically determined threshold by this scalar multiplier
        'max_overlap': 0.75,  # cells with more overlap than this get removed during triage, before refinement
        'high_pass': 100,  # running mean subtraction with window of size 'high_pass' (use low values for 1P)
        'denoise': True, # denoise binned movie for cell detection in sparse_mode

        # classification parameters
        'soma_crop': True, # crop dendrites for cell classification stats like compactness
        # ROI extraction parameters
        'neuropil_extract': True, # whether or not to extract neuropil; if False, Fneu is set to zero
        'inner_neuropil_radius': 2,  # number of pixels to keep between ROI and neuropil donut
        'min_neuropil_pixels': 350,  # minimum number of pixels in the neuropil
        'lam_percentile': 50., # percentile of lambda within area to ignore when excluding cell pixels for neuropil extraction
        'allow_overlap': False,  # pixels that are overlapping are thrown out (False) or added to both ROIs (True)
        'use_builtin_classifier': runops.get('use_builtin_classifier'),  # whether or not to use built-in classifier for cell detection (overrides
                                         # classifier specified in classifier_path if set to True)
        'classifier_path': runops.get('classifier_path'), # path to classifier
        
        # channel 2 detection settings (stat[n]['chan2'], stat[n]['not_chan2'])
        'chan2_thres': 0.65,  # minimum for detection of brightness on channel 2

        # deconvolution settings
        'baseline': 'maximin',  # baselining mode (can also choose 'prctile')
        'win_baseline': 60.,  # window for maximin
        'sig_baseline': 10.,  # smoothing constant for gaussian filter
        'prctile_baseline': 8.,  # optional (whether to use a percentile baseline)
        'neucoeff': .7,  # neuropil coefficient
        

        # List of tiffs to be loaded
        'tiff_list': runops.get('files_name') # list of tiffs in folder * data_path *!
      }
    return db
