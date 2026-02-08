import datajoint as dj
from ..pipeline import imaging, db_prefix, scan
import numpy as np
import matplotlib.pyplot as plt
import importlib
from element_interface.utils import find_full_path, dict_to_uuid, find_root_directory
from element_session import session_with_id as session
from adamacs.pipeline  import scan, imaging
from element_calcium_imaging.scan import get_imaging_root_data_dir, get_processed_root_data_dir, get_scan_image_files, get_scan_box_files, get_nd2_files

import os
import re
import sys
import pathlib

import torch
import skimage.io as skio  
import glob
from tqdm import tqdm
from datetime import datetime


#TR 2024

import random

schema = dj.schema(db_prefix + 'denoising')

@schema
# class PupilEllipseParameter(dj.Lookup):
#   """Set hyperparameters for pupil ellipse fitting, mainly for preprocessing"""
#   definition = """
#   parameter_id: int # unique id for parameters
#   ---
#   likelihood_thres: float # likelihood threshold for good tracking
#   exclude_ir_std: float # define a boundary by mean and std to separate good tracking and bad tracking
#   ellipticity_thres: float # for computing camera center by elliptic frames
#   description: varchar(255) # short description of the parameters
#   """

#   contents = [(1, 0.5, 5, 0.85, "default, free-moving eye cam")]

class DenoisingMethod(dj.Lookup):
    definition = """  #  Method, package, analysis suite used for processing of calcium imaging data (e.g. Suite2p, CaImAn, etc.)
    denoising_method: char(24)
    ---
    denoising_method_desc: varchar(100)
    """

    contents = [('support', 'SUPPORT denoising'),
                ('deepcadrt', 'DeepCAD-RT denoising'),
                ('deepinterpolation', 'DeepInterpolation denoising'),]

@schema
class DenoisingProcessingParamSet(dj.Lookup):
    definition = """  #  Parameter set used for Denoising
    denoise_paramset_idx:  smallint
    ---
    -> imaging.ProcessingParamSet
    -> DenoisingMethod
    denoise_paramset_desc: varchar(128)
    denoise_param_set_hash: uuid
    unique index (denoise_param_set_hash)
    denoise_params: longblob  # dictionary of all applicable parameters
    """

    @classmethod
    def insert_new_params(cls, paramset_idx, denoising_method: str, denoise_paramset_idx: int,
                          denoise_paramset_desc: str, denoise_params: dict):
        param_dict = {'paramset_idx': paramset_idx,
                      'denoising_method': denoising_method,
                      'denoise_paramset_idx': denoise_paramset_idx,
                      'denoise_paramset_desc': denoise_paramset_desc,
                      'denoise_params': denoise_params,
                      'denoise_param_set_hash': dict_to_uuid(denoise_params)}
        q_param = cls & {'denoise_param_set_hash': param_dict['denoise_param_set_hash']}

        if q_param:  # If the specified param-set already exists
            pname = q_param.fetch1('denoise_paramset_idx')
            if pname == denoise_paramset_idx:  # If the existed set has the same name: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError(
                    'The specified param-set already exists - name: {}'.format(pname))
        else:
            cls.insert1(param_dict)

@schema
class DenoisingTask(dj.Manual):
    definition = """  # Manual table for defining a processing task ready to be run
    -> scan.Scan
    -> DenoisingProcessingParamSet
    ---
    processing_output_dir: varchar(255)         #  output directory of the processed scan relative to root data directory
    task_mode='load': enum('load', 'trigger')   # 'load': load computed analysis results, 'trigger': trigger computation
    """

    @classmethod
    def infer_output_dir(cls, scan_key,  relative=False, mkdir=False):
        image_locators = {'NIS': get_nd2_files, 'ScanImage': get_scan_image_files, 'Scanbox': get_scan_box_files}
        image_locator = image_locators[(scan.Scan & scan_key).fetch1('acq_software')]

        scan_dir = find_full_path(get_imaging_root_data_dir(), image_locator(scan_key)[0]).parent
        root_dir = find_root_directory(get_imaging_root_data_dir(), scan_dir)
        
        paramset_key = DenoisingProcessingParamSet.fetch1()
        processed_dir = pathlib.Path(get_processed_root_data_dir())
        output_dir = (processed_dir
                / scan_dir.relative_to(root_dir)
                / f'{paramset_key["denoising_method"]}_{paramset_key["denoise_paramset_idx"]}')

        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir.relative_to(processed_dir) if relative else output_dir

    @classmethod
    def auto_generate_entries(cls, scan_key, task_mode):
        """
        Method to auto-generate DenoisingTask entries for a particular Scan using a default paramater set.
        """

        # default_denoise_paramset_idx = os.environ.get('DEFAULT_denoise_paramset_idx', 0)
        
        default_paramset_idx = os.environ.get('DEFAULT_PARAMSET_IDX', 1)
        
        
        # output_dir = cls.infer_output_dir(scan_key, relative=False, mkdir=True)
        output_dir = (scan.ScanPath & scan_key).fetch1('path')

        # method = (DenoisingProcessingParamSet & {'paramset_idx': default_paramset_idx}).fetch1('denoising_method')
        
        # try:
        #     if method == 'suite2p':
        #         from element_interface import suite2p_loader
        #         loaded_dataset = suite2p_loader.Suite2p(output_dir)
        #     elif method == 'caiman':
        #         from element_interface import caiman_loader
        #         loaded_dataset = caiman_loader.CaImAn(output_dir)
        #     else:
        #         raise NotImplementedError('Unknown/unimplemented method: {}'.format(method))
        # except FileNotFoundError:
        #     task_mode = 'trigger'
        # else:
        #     task_mode = 'load'

        cls.insert1({
            **scan_key, 'denoise_paramset_idx': default_paramset_idx,
            'processing_output_dir': output_dir, 'task_mode': task_mode})

@schema
class Denoising(dj.Computed):
    definition = """  # Denoising Procedure
    -> DenoisingTask
    ---
    denoising_time     : datetime  # time of generation of this set of denoised results
    """

    # Run processing only on Scan with ScanInfo inserted
    @property
    def key_source(self):
        return DenoisingTask & scan.ScanInfo

    def make(self, key):
        task_mode = (DenoisingTask & key).fetch1('task_mode')

        output_dir = (DenoisingTask & key).fetch1('processing_output_dir')
        output_dir = find_full_path(get_imaging_root_data_dir(), output_dir).as_posix()
        # if not output_dir:
        #     output_dir = DenoisingTask.infer_output_dir(key, relative=True, mkdir=True)
        #     # update processing_output_dir
        #     DenoisingTask.update1({**key, 'processing_output_dir': output_dir.as_posix()})
        
        # method = "support" # hardcoded method for now
       
        if task_mode == 'load': # not implemented yet
            print('Not implemented')
            # method, imaging_dataset = get_loader_result(key, DenoisingTask)
            # if method == 'suite2p':
            #     if (scan.ScanInfo & key).fetch1('nrois') > 0:
            #         raise NotImplementedError(f'Suite2p ingestion error - Unable to handle'
            #                                   f' ScanImage multi-ROI scanning mode yet')
            #     suite2p_dataset = imaging_dataset
            #     key = {**key, 'processing_time': suite2p_dataset.creation_time}
            # elif method == 'caiman':
            #     caiman_dataset = imaging_dataset
            #     key = {**key, 'processing_time': caiman_dataset.creation_time}
            # else:
            #     raise NotImplementedError('Unknown method: {}'.format(method))
        elif task_mode == 'trigger':

            method = (DenoisingTask * DenoisingProcessingParamSet * DenoisingMethod & key).fetch1('denoising_method')
            output_dir_denoise = output_dir + '/' + method
            
            if not os.path.isdir(output_dir_denoise):
                os.makedirs(output_dir_denoise)
            
            if method == 'support':
                # Path to the cloned repository
                repo_path = "/home/backup_user/SUPPORT" #TR25: change to more common install location....
                # Add the repository path to the Python path
                sys.path.append(repo_path)
                
                
                from src.train import train
                from src.utils.dataset import DatasetSUPPORT_test_stitch
                from model.SUPPORT import SUPPORT
                
                from suite2p.registration import register
                from suite2p import default_ops 
                
                # import suite2p

                denoise_params = (DenoisingTask * DenoisingProcessingParamSet & key).fetch1('denoise_params')
                new_key = key.copy()
                new_key['paramset_idx'] = (DenoisingProcessingParamSet & new_key).fetch1('paramset_idx')
                s2p_params = (imaging.ProcessingParamSet & new_key).fetch1('params')
                denoise_params['save_path0'] = output_dir_denoise
          
                # if concatenate == 'indiv':
                image_files = (DenoisingTask * scan.Scan * scan.ScanInfo * scan.ScanInfo.ScanFile & key).fetch('file_path')
                image_files = sorted([find_full_path(get_imaging_root_data_dir(), image_file) for image_file in image_files])
                # elif concatenate == 'concat':
                #     # Removing the "scan_id" key from the dictionary
                #     keynoscan = key.copy()
                #     del keynoscan['scan_id']            
                #     image_files = (DenoisingTask * scan.Scan * scan.ScanInfo * scan.ScanInfo.ScanFile & keynoscan).fetch('file_path')
                #     image_files = [find_full_path(get_imaging_root_data_dir(), image_file) for image_file in image_files]
                # elif concatenate == 'consame':
                #     # Removing the "scan_id" key from the dictionary and setting session id to same_site.
                #     keynoscansamesite = key.copy()
                #     del keynoscansamesite['scan_id'] 
                #     keynoscansamesite['same_site_id'] = keynoscansamesite.pop('session_id')           
                #     image_files = (DenoisingTask * scan.Scan * scan.ScanInfo * scan.ScanInfo.ScanFile * session.SessionSameSite & keynoscansamesite).fetch('file_path')
                #     image_files = [find_full_path(get_imaging_root_data_dir(), image_file) for image_file in image_files]

                # input_format = pathlib.Path(image_files[0]).suffix
                # suite2p_params['input_format'] = input_format[1:]
                
                support_paths = {
                    'data_path': [image_files[0].parent.as_posix()],
                    'tiff_list': [f.as_posix() for f in image_files]
                }
                
                # do rigid registration first

                # extract suite2p parameters from the database

                ops = default_ops()  # Load the default operations dictionary as background for further changes
                ops.update(s2p_params)  # Update the default ops with the user-defined ops

                # further user opts
                ops.update({
                    "nonrigid": False,  # Disable non-rigid registration for rigid-only
                    "batch_size": 100000,  # Process frames in batches - set to large number to process all frames at once. GPU server has 1.5TB of RAM
                    "do_bidiphase": True,  # Enable bidiphase correction
                    "bidiphase": 0,  # Initial phase shift guess (set to 0 for estimation)
                    "bidi_corrected": False,  # Indicates bidiphase correction is not yet applied
                    "reg_tif": False, # Dont Save registered tiffs
                    "save_path0": output_dir,  # Output path for registered data
                    "reg_tif": False
                })

                # load data 
                print('Loading data...')
                image_data = np.concatenate([skio.imread(file).astype(np.int16) for file in image_files], axis=0)
                nchannels = (scan.ScanInfo & key).fetch1('nchannels')
                image_data = image_data[::nchannels, :, :]  # Reduce the stack by taking every nth channel
                # image_data = skio.imread(data_file).astype(np.float32)
                
                # Perform registration
                print('rigid pre-registration before denoising...')
                refImg, rmin, rmax, meanImg, rigid_offsets, _, _, _, _, _, _ = register.registration_wrapper(
                    f_reg=image_data, f_raw=image_data, ops=ops
                )

                # Denoise
                # find good gpu here and store as denoise_params["device"]
                gputouse = select_gpu()
                device = torch.device(f"cuda:{gputouse}" if torch.cuda.is_available() else "cpu")
                denoise_params["device"] = device
                
                # Initialize the model
                model = SUPPORT(
                    in_channels=denoise_params["patch_size"][0], 
                    mid_channels=denoise_params["unet_channels"], 
                    depth=denoise_params["depth"],
                    blind_conv_channels=denoise_params["blind_conv_channels"], 
                    one_by_one_channels=denoise_params["one_by_one_channels"], 
                    last_layer_channels=denoise_params["last_layer_channels"], 
                    bs_size=denoise_params["bs_size"], 
                    bp=denoise_params["bp_mode"]
                ).to(denoise_params["device"])

                # Load the model state
                
                state_dict = torch.load(denoise_params['model_file'], map_location=denoise_params["device"])
                model.load_state_dict(state_dict)

                # Prepare the data
                demo_tif = torch.from_numpy(image_data).type(torch.FloatTensor).to(denoise_params["device"])
                demo_tif = demo_tif[:, :, :]

                # Create the dataset and dataloader
                testset = DatasetSUPPORT_test_stitch(demo_tif, patch_size=denoise_params["patch_size"], patch_interval=denoise_params["patch_interval"])
                testloader = torch.utils.data.DataLoader(testset, batch_size=denoise_params["batch_size"])

                # Validate the model
                print('denoising...')
                denoised_stack = validate(testloader, model)
                # print(denoised_stack.shape)

                # Extract the final stack
                # finalstack = denoised_stack[(model.in_channels-1)//2:-(model.in_channels-1)//2, :, :]
                
                # put second channel black frames in if needed for as long as not both channels are denoised
                if nchannels > 1:
                    finalstack = np.zeros((denoised_stack.shape[0] * nchannels, denoised_stack.shape[1], denoised_stack.shape[2]), dtype=denoised_stack.dtype)
                    finalstack[::nchannels] = denoised_stack
                else:
                    finalstack = denoised_stack
                
                # Pad the final stack with black frames to match the size of the original image data
                # finalstack_padded = pad_with_black_frames(image_data, finalstack)
                
                print('saving...')
                savefile = output_dir_denoise + '/' + image_files[0].stem + '_denoised_stack.tif'
                print(savefile)
                skio.imsave(savefile, finalstack.astype(np.int16), metadata={'axes': 'TYX'})
                
                
                key = {**key, 'denoising_time': datetime.now()}

            elif method == 'caiman':
                from element_interface.run_caiman import run_caiman

                tiff_files = (DenoisingTask * scan.Scan * scan.ScanInfo * scan.ScanInfo.ScanFile & key).fetch('file_path')
                tiff_files = [find_full_path(get_imaging_root_data_dir(), tiff_file).as_posix() for tiff_file in tiff_files]

                params = (DenoisingTask * DenoisingProcessingParamSet & key).fetch1('params')
                sampling_rate = (DenoisingTask * scan.Scan * scan.ScanInfo & key).fetch1('fps')

                ndepths = (DenoisingTask * scan.Scan * scan.ScanInfo & key).fetch1('ndepths')

                is3D = bool(ndepths > 1)
                if is3D:
                    raise NotImplementedError('Caiman pipeline is not capable of analyzing 3D scans at the moment.')
                run_caiman(file_paths=tiff_files, parameters=params, sampling_rate=sampling_rate, output_dir=output_dir, is3D=is3D)

                _, imaging_dataset = imaging.get_loader_result(key, DenoisingTask)
                caiman_dataset = imaging_dataset
                key['processing_time'] = caiman_dataset.creation_time

        else:
            raise ValueError(f'Unknown task mode: {task_mode}')

        self.insert1(key, skip_duplicates=True)


# # ---------------- HELPER FUNCTIONS ----------------


# # SUPPORT test function on select GPU

# def validate(test_dataloader, model):
#     """
#     Validate a model with test data.
#     """
#     with torch.no_grad():
#         model.eval()
#         denoised_stack = np.zeros(test_dataloader.dataset.noisy_image.shape, dtype=np.float32)
        
#         for _, (noisy_image, _, single_coordinate) in enumerate(tqdm(test_dataloader, desc="validate")):
#             noisy_image = noisy_image.to(next(model.parameters()).device)  # Ensure the input is on the same device as the model
#             noisy_image_denoised = model(noisy_image)
#             T = noisy_image.size(1)
            
#             for bi in range(noisy_image.size(0)): 
#                 stack_start_w = int(single_coordinate['stack_start_w'][bi])
#                 stack_end_w = int(single_coordinate['stack_end_w'][bi])
#                 patch_start_w = int(single_coordinate['patch_start_w'][bi])
#                 patch_end_w = int(single_coordinate['patch_end_w'][bi])

#                 stack_start_h = int(single_coordinate['stack_start_h'][bi])
#                 stack_end_h = int(single_coordinate['stack_end_h'][bi])
#                 patch_start_h = int(single_coordinate['patch_start_h'][bi])
#                 patch_end_h = int(single_coordinate['patch_end_h'][bi])

#                 stack_start_s = int(single_coordinate['init_s'][bi])
                
#                 denoised_stack[stack_start_s+(T//2), stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
#                     = noisy_image_denoised[bi].squeeze()[patch_start_h:patch_end_h, patch_start_w:patch_end_w].cpu()

#         # Move std_image and mean_image to CPU before converting to NumPy
#         denoised_stack = denoised_stack * test_dataloader.dataset.std_image.cpu().numpy() + test_dataloader.dataset.mean_image.cpu().numpy()

#         return denoised_stack


# SUPPORT test function on select GPU
def validate(test_dataloader, model):
    """
    Validate a model with test data.
    """
    with torch.no_grad():
        model.eval()
        denoised_stack = np.zeros(test_dataloader.dataset.noisy_image.shape, dtype=np.float32)
        
        for _, (noisy_image, _, single_coordinate) in enumerate(tqdm(test_dataloader, desc="validate")):
            noisy_image = noisy_image.to(next(model.parameters()).device)  # Ensure the input is on the same device as the model
            noisy_image_denoised = model(noisy_image)
            T = noisy_image.size(1)
            
            for bi in range(noisy_image.size(0)): 
                stack_start_w = int(single_coordinate['stack_start_w'][bi])
                stack_end_w = int(single_coordinate['stack_end_w'][bi])
                patch_start_w = int(single_coordinate['patch_start_w'][bi])
                patch_end_w = int(single_coordinate['patch_end_w'][bi])

                stack_start_h = int(single_coordinate['stack_start_h'][bi])
                stack_end_h = int(single_coordinate['stack_end_h'][bi])
                patch_start_h = int(single_coordinate['patch_start_h'][bi])
                patch_end_h = int(single_coordinate['patch_end_h'][bi])

                stack_start_s = int(single_coordinate['init_s'][bi])
                
                denoised_stack[stack_start_s+(T//2), stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = noisy_image_denoised[bi].squeeze()[patch_start_h:patch_end_h, patch_start_w:patch_end_w].cpu()

        # Move std_image and mean_image to CPU before converting to NumPy
        denoised_stack = denoised_stack * test_dataloader.dataset.std_image.cpu().numpy() + test_dataloader.dataset.mean_image.cpu().numpy()

        return denoised_stack
    
# Function to find the file with the maximum iteration number in the given directory
def find_max_iteration_file(directory):
    files = [
        f for f in os.listdir(directory)
        if (match := re.match(r"model_(\d+)\.pth", f))
    ]
    return os.path.join(directory, max(files, key=lambda x: int(re.search(r"model_(\d+)\.pth", x).group(1)))) if files else None


def pad_with_black_frames(image_data, finalstack):
    """
    Pads finalstack with black frames to match the size of image_data.

    Args:
        image_data (np.ndarray): The original image data stack (T, Y, X).
        finalstack (np.ndarray): The denoised image stack (T, Y, X).

    Returns:
        np.ndarray: Padded finalstack.

    Raises:
        ValueError: If the difference in size is not symmetric.
    """
    # Calculate the difference in size along the temporal (T) dimension
    diff = image_data.shape[0] - finalstack.shape[0]

    if diff % 2 != 0:
        raise ValueError("The difference in size is not symmetric; cannot pad evenly.")

    # Calculate the number of black frames to pad on each side
    pad_size = diff // 2

    # Pad the temporal dimension with black frames (zeros)
    padded_stack = np.pad(finalstack, ((pad_size, pad_size), (0, 0), (0, 0)), mode='constant', constant_values=0)

    return padded_stack


# -------------- function for GPU server use! TR2024

# Step 1: Check available GPUs and their loads
def select_gpu():
    # Get all GPUs and their usage status
    import GPUtil
    gpus = GPUtil.getGPUs()
    available_gpus = [gpu.id for gpu in gpus if gpu.load == 0]

    # Step 2: Select a GPU
    if available_gpus:
        # Randomly select a GPU with no load
        selected_gpu = random.choice(available_gpus)
    else:
        # If all GPUs are under load, default to GPU 0
        selected_gpu = 0

    return selected_gpu

# @schema
# class Denoising(dj.Computed):
#     """Denoise imaging data"""
#     definition = """
#     -> PupilEllipseParameter
#     -> PoseEstimationNew
#     ---
#     ellipse_dict: longblob # dictionary of ellipse fitting results
#     cam_center: longblob # camera center in pixel coordinates
#     eye_center: longblob # eye center in pixel coordinates
#     scale: longblob # scaling factor from pixel to rotation angle
#     theta: longblob # eye rotation in radians at azimuth direction
#     phi: longblob # eye rotation in elevation direction
#     diameter: longblob # long axis of the fitted ellipse (equals pupil diameter)
#     """
#     # def plot_preprocess(self, key):
#     #     # IR exlusion
#     #     id_bad_ir=np.isnan(ir_xg)
#     #     bad_ir_fr = np.where(id_bad_ir)[0]

#     #     bad_p_fr=[]
#     #     for i, (x, y) in enumerate(zip(p_x, p_y)):
#     #         if np.count_nonzero(~np.isnan(x))<5:
#     #             bad_p_fr.append(i)
          
#     #     print("# frames of bad IR tracking:", len(bad_ir_fr),", # frames of bad pupil tracking:", len(bad_p_fr),'intersecting # frames', len(np.intersect1d(bad_ir_fr,bad_p_fr)))

#     #     plt.plot(ir_x[~ir_bad],ir_y[~ir_bad],'o', alpha=0.6,markersize=1,label='selected tracking')
#     #     plt.plot(ir_x[ir_bad],ir_y[ir_bad],'o', alpha=0.6,markersize=1, label='excluded tracking')
#     #     plt.plot(irx_mean,iry_mean,'o',label='mean position')
#     #     plt.legend()
#     #     plt.title('tracking dots for IR reflection')
      
#     def make(self, key):
        
#     #scansi = key['scan_id']
#     #scan_key = (scan.Scan & f'scan_id = "{scansi}"').fetch('KEY')[0]

#     #dlc_scan_key = (pupil.DLC() & key).fetch1('dlc_scan_key')


#         # load all the tracking dots
#         bodypart = ['nose_corner', 'lateral_corner', 'IR', 'pupil_left',
#                     'pupil_left_lower','pupil_left_up','pupil_lower','pupil_right',
#                     'pupil_right_lower','pupil_right_up','pupil_upper']
#         x_all = []
#         y_all = []
#         llh_all = []
#         # key = eye_dlc_scan_key and include parameters
#         for bp in bodypart:
#           x = (PoseEstimationNew.BodyPartPosition & key & f"body_part='{bp}'").fetch('x_pos')
#           y = (PoseEstimationNew.BodyPartPosition & key & f"body_part='{bp}'").fetch('y_pos')
#           llh = (PoseEstimationNew.BodyPartPosition & key & f"body_part='{bp}'").fetch('likelihood')
#           x_all = np.concatenate((x_all,x))
#           y_all = np.concatenate((y_all,y))
#           llh_all = np.concatenate((llh_all,llh))

#         x_all = np.vstack(x_all).T
#         y_all = np.vstack(y_all).T
#         llh_all = np.vstack(llh_all).T

#         # get parameters from PupilEllipseParameter table
#             # set a cutoff for poor fitting points in DLC model
#         thres_llh = (PupilEllipseParameter & key).fetch1('likelihood_thres') # 0.5 #0.8
#         thres_ir = (PupilEllipseParameter & key).fetch1('exclude_ir_std')
#         thres_ellipticity = (PupilEllipseParameter & key).fetch1('ellipticity_thres')

#         # set points lower than thres to NaN
#         mask = llh_all < thres_llh
#         # Set the elements in x_all and y_all to NaN where the mask is True
#         x_thr, y_thr = np.copy(x_all), np.copy(y_all)
#         x_thr[mask] = np.nan
#         y_thr[mask] = np.nan

#         # assign dots coord to corresponding name
#         ncor_x = x_thr[:,0]
#         ncor_y = y_thr[:,0]
#         lcor_x = x_thr[:,1]
#         lcor_y = y_thr[:,1]
#         ir_x = x_thr[:,2]
#         ir_y = y_thr[:,2]
#         p_x = x_thr[:,3:]
#         p_y = y_thr[:,3:]

#         # define a boundary by mean and std to separate good tracking and bad tracking
#         irx_mean = np.nanmean(ir_x)
#         iry_mean = np.nanmean(ir_y)
#         irx_std = np.nanstd(ir_x)
#         iry_std = np.nanstd(ir_y)
#         ir_std = np.linalg.norm([irx_std,iry_std])
#         # set threshold as 5*std, still need to validata with other dataset
#         ir_bad=(ir_x-irx_mean)**2+(ir_y-iry_mean)**2>(thres_ir*ir_std)**2 #49 frames

#         # exclude off position IR dots
#         ir_xg, ir_yg = np.copy(ir_x), np.copy(ir_y)
#         ir_xg[ir_bad] = np.nan
#         ir_yg[ir_bad] = np.nan

#         # center pupil tracking by IR reflection
#         pc_x = p_x-ir_xg[:,None]
#         pc_y = p_y-ir_yg[:,None] # if IR is invalid, the pupil tracking is invalid

#         ### eye center estimation is needed for gaze reconstruction 
#         ### search how
#         # centered by IR reference point in each frame
#         ncor_cx = ncor_x - ir_xg[:None]
#         ncor_cy = ncor_y - ir_yg[:None]
#         lcor_cx = lcor_x - ir_xg[:None]
#         lcor_cy = lcor_y - ir_yg[:None]

#         # estimate eye center by 2 eye corners, 
#         eye_center_cx_ = (ncor_cx + lcor_cx)/2 
#         eye_center_cy_ = (ncor_cy + lcor_cy)/2

#         eye_center_cx = np.nanmean(eye_center_cx_)
#         eye_center_cy = np.nanmean(eye_center_cy_)
#         eye_center_cx,eye_center_cy
#         eye_cent = np.array([eye_center_cx,eye_center_cy])

#         def fit_ellipse(x, y):
#             # NaN need to be removed for SVD convergence
#             x0=x[~np.isnan(x)]
#             y0=y[~np.isnan(y)]
            
#             meanX = np.nanmean(x)
#             meanY = np.nanmean(y)
            
#             # remove bias
#             x = x0 - meanX
#             y = y0 - meanY
            
#             # Estimation of the conic equation
#             X = np.array([x**2, x*y, y**2, x, y])
#             X = np.stack(X).T
#             try:
#                 param = np.dot(np.sum(X, axis=0), np.linalg.pinv(np.matmul(X.T,X)))

            
#                 # #least square method to fit ellipse
#                 # b = np.ones(8)
#                 # param = np.linalg.lstsq(X, b, rcond=None)[0].squeeze() # very similar results
                
#                 # Extract parameters from the conic equation
#                 a, b, c, d, e = param[0], param[1], param[2], param[3], param[4]
                
#                 if b**2-4*a*c > 0: 
#                     print('given dots does not form a valid ellipse for frame',i)
                
#                 # Eigen decomp
#                 Q = np.array([[a, b/2],[b/2, c]])
#                 eig_val, eig_vec = np.linalg.eig(Q)
                
#                 # Get angle to long axis
#                 if eig_val[0] < eig_val[1]:
#                     angle_to_x = np.arctan2(eig_vec[1,0], eig_vec[0,0])
#                 else:
#                     angle_to_x = np.arctan2(eig_vec[1,1], eig_vec[0,1])
                
#                 angle_from_x = angle_to_x
#                 orientation_rad = 0.5 * np.arctan2(b, (c-a))
#                 cos_phi = np.cos(orientation_rad)
#                 sin_phi = np.sin(orientation_rad)
                
#                 a, b, c, d, e = [a*cos_phi**2 - b*cos_phi*sin_phi + c*sin_phi**2,
#                                 0,
#                                 a*sin_phi**2 + b*cos_phi*sin_phi + c*cos_phi**2,
#                                 d*cos_phi - e*sin_phi,
#                                 d*sin_phi + e*cos_phi]
                
#                 meanX, meanY = [cos_phi*meanX - sin_phi*meanY,
#                                 sin_phi*meanX + cos_phi*meanY]
                
#                 # Check if conc expression represents an ellipse
#                 test = a*c

#                 # select tracking frames only when fitted valid ellipse, and
#                 # IR is valid and at least 5 pupil dots are available
#                 if test > 0 and np.count_nonzero(~np.isnan(x))>=5:
                
#                     # Make sure coefficients are positive
#                     if a<0:
#                         a, c, d, e = [-a, -c, -d, -e]
                
#                     # Final ellipse parameters
#                     X0 = meanX - d/2/a
#                     Y0 = meanY - e/2/c
#                     F = 1 + (d**2)/(4*a) + (e**2)/(4*c)
#                     a = np.sqrt(F/a)
#                     b = np.sqrt(F/c)
#                     long_axis = 2*np.maximum(a,b)
#                     short_axis = 2*np.minimum(a,b)
                
#                     # Rotate axes backwards to find center point of
#                     # original tilted ellipse
#                     R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
#                     P_in = R @ np.array([[X0],[Y0]])
#                     X0_in = P_in[0][0]
#                     Y0_in = P_in[1][0]
                
#                     # Organize parameters in dictionary to return
#                     ellipse_dict = {
#                         'X0':X0,
#                         'Y0':Y0,
#                         'F':F,
#                         'a':a,
#                         'b':b,
#                         'long_axis':long_axis/2,
#                         'short_axis':short_axis/2,
#                         'angle_to_x':angle_to_x,
#                         'angle_from_x':angle_from_x,
#                         'cos_phi':cos_phi,
#                         'sin_phi':sin_phi,
#                         'X0_in':X0_in,
#                         'Y0_in':Y0_in,
#                         'phi':orientation_rad
#                     }
                
#                 else:
#                     # If the conic equation didn't return an ellipse, do not
#                     # return any real values and fill the dictionary with NaNs.
#                     dict_keys = ['X0','Y0','F','a','b','long_axis',
#                                     'short_axis','angle_to_x','angle_from_x',
#                                     'cos_phi','sin_phi','X0_in','Y0_in','phi']
#                     dict_vals = list(np.ones([len(dict_keys)]) * np.nan)
#                     ellipse_dict = dict(zip(dict_keys, dict_vals))
#                 return ellipse_dict
#             except np.linalg.LinAlgError as e:
#                 print("Error occurred during computation:", e, "for frame",i) 

#         # fitting ellipse, looping for all the valid and invalid frames
#         ellipse_dict_all, X0_in_all, Y0_in_all, long_axis_all, short_axis_all, angle_all = [], [], [], [], [], []
#         for i, (x, y) in enumerate(zip(pc_x, pc_y)):
#             ellipse_dic = fit_ellipse(x,y)
#             ellipse_dict_all.append(ellipse_dic)

#             X0_in_all.append(ellipse_dic['X0_in'])
#             Y0_in_all.append(ellipse_dic['Y0_in'])
#             long_axis_all.append(ellipse_dic['long_axis'])
#             short_axis_all.append(ellipse_dic['short_axis'])
#             angle_all.append(ellipse_dic['angle_from_x'])

#         X0_in_all = np.array(X0_in_all)
#         Y0_in_all = np.array(Y0_in_all)
#         long_axis_all = np.array(long_axis_all)
#         short_axis_all = np.array(short_axis_all)
#         angle_all = np.array(angle_all)

#         nice_mask=[]
#         for i, (x, y) in enumerate(zip(pc_x, pc_y)):
#             if np.count_nonzero(~np.isnan(x))>=7:
#                 if short_axis_all[i]/long_axis_all[i] < thres_ellipticity: # 0.85 is the threshold from Neill's code
#                     nice_mask.append(i)
                
#         # calculate camera center with selected frames
#         A = np.vstack([np.cos(angle_all[nice_mask]),np.sin(angle_all[nice_mask])])
#         b = np.expand_dims(np.diag(A.T @ np.squeeze(np.array([X0_in_all[nice_mask],Y0_in_all[nice_mask]]))), axis=1)
#         cam_cent = np.linalg.inv(A @ A.T) @ A @ b
#         # print(f"{len(nice_mask)} frames are selected to estimate camera center, estimated result: {cam_cent[0,0],cam_cent[1,0]}")

#         ellipticity = np.array(short_axis_all[nice_mask]) / np.array(long_axis_all[nice_mask])
#         scale = np.sum(np.sqrt(1 - (ellipticity)**2) *                       \
#         (np.linalg.norm(np.array([X0_in_all[nice_mask],Y0_in_all[nice_mask]]).T - cam_cent.T, axis=1)))       \
#         / np.sum(1 - (ellipticity)**2)


#         # Horizontal orientation (THETA)
#         # valid range (-1,1)
#         theta = np.arcsin((X0_in_all - cam_cent[0]) / scale) # negative because video captured upside down??? do i need?? right eye need negative, left eye no need

#         # Vertical orientation (PHI)
#         phi = np.arcsin((Y0_in_all - cam_cent[1]) / np.cos(theta) / scale) # no need negative because origin is at upper corner, increasing downwards



#         self.insert1({**key, 'ellipse_dict': ellipse_dict_all, 'cam_center': cam_cent, 'eye_center': eye_cent, 'scale': scale, 'theta': theta, 'phi': phi, 'diameter': long_axis_all})


# '''
# @schema
# class GazeReconstruction3D(dj.Computed):
#   definition = """
#   -> PupilEllipseFitting # theta, phi, cam_center, eye_center
#   -> IMUtracking # yaw, pitch, roll
#   ---
#   gaze_x: longblob
#   gaze_y: longblob
#   gaze_z: longblob
#   """


# @schema
# class GazeReconstruction2D(dj.Computed):
#   definition = """
#   -> PupilEllipseFitting # theta, phi, cam_center, eye_center
#   -> DLCLivePoseEstimation # only consider yaw movement
#   ---
#   gaze_x: longblob
#   gaze_y: longblob
#   gaze_z: longblob
#   """

#   '''
