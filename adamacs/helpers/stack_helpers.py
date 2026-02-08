#!/usr/bin/env python
# coding: utf-8

# Tobias Rose 2023

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter
from scipy.signal import convolve
import cv2
import ipywidgets as widgets
from IPython.display import display
from natsort import natsorted, ns
import concurrent.futures
from multiprocessing import Pool, cpu_count
import imageio
import imageio.plugins.ffmpeg as ffmpeg
import os
from tqdm import tqdm
from dask import compute
from dask import delayed
import matplotlib.pyplot as plt
import dask.array as da
from datetime import datetime, timedelta
import bisect
from adamacs.pipeline import subject, session, equipment, surgery, event, trial, imaging, behavior, scan, model,  analysis, denoising

def calculate_dFF(Fall, Fneu_all, framerate, event_module, scan_key, 
                  neuropil_factor=0.3, smoothing_window_seconds=120, percentile=15, 
                  darkframe_correction=True, shuttertime=0.09):
    """
    Calculate ΔF/F0 for calcium imaging traces with darkframe correction, neuropil correction, and baseline estimation.
    
    This function performs the standard calcium imaging preprocessing pipeline:
    1. Darkframe correction: Subtract mean dark signal from both F and F_neuropil
    2. Neuropil correction: dF = (F - neuropil_factor * F_neuropil) + median correction
    3. Baseline (F0) estimation using percentile filtering
    4. ΔF/F0 calculation: (dF - F0) / F0
    
    Parameters:
    -----------
    Fall : np.ndarray
        Fluorescence traces, shape (n_cells, n_frames)
    Fneu_all : np.ndarray
        Neuropil fluorescence traces, shape (n_cells, n_frames)
    framerate : float
        Imaging frame rate in Hz (frames per second)
    event_module : module
        DataJoint event module for accessing Event table (e.g., from adamacs.pipeline import event)
    scan_key : dict
        DataJoint scan key containing session_id and scan_id
    neuropil_factor : float, default=0.7
        Factor for neuropil correction. Typical values: 0.7 (default in Suite2p)
    smoothing_window_seconds : float, default=60
        Window size in seconds for baseline (F0) estimation using percentile filtering
    percentile : int, default=15
        Percentile value for baseline estimation. Lower values give lower baselines.
        Typical values: 8-15
    darkframe_correction : bool, default=True
        Whether to perform darkframe correction before other processing
    shuttertime : float, default=0.09
        Shutter time offset in seconds for darkframe calculation
    
    Returns:
    --------
    dFF : np.ndarray
        Normalized fluorescence change (ΔF/F0), shape (n_cells, n_frames)
    dF : np.ndarray
        Neuropil-corrected fluorescence, shape (n_cells, n_frames)
    F0 : np.ndarray
        Estimated baseline fluorescence, shape (n_cells, n_frames)
    mean_darksignal : float or None
        Mean dark signal value (if darkframe_correction=True), None otherwise
        
    Example:
    --------
    >>> from adamacs.pipeline import event
    >>> 
    >>> # Fetch fluorescence data
    >>> traces = (imaging.Fluorescence.Trace & key).fetch(as_dict=True, order_by="mask")
    >>> Fall = np.vstack([trace['fluorescence'] for trace in traces])
    >>> Fneu_all = np.vstack([trace['neuropil_fluorescence'] for trace in traces])
    >>> framerate = (scan.ScanInfo & key).fetch1('fps')
    >>> 
    >>> # Calculate dFF with darkframe correction
    >>> dFF, dF, F0, mean_darksignal = calculate_dFF(
    ...     Fall, Fneu_all, framerate, event, key,
    ...     neuropil_factor=0.7, 
    ...     smoothing_window_seconds=60, 
    ...     percentile=15,
    ...     darkframe_correction=True
    ... )
    """
    from scipy.ndimage import percentile_filter
    from joblib import Parallel, delayed
    
    # Darkframe correction
    mean_darksignal = None
    if darkframe_correction:
        mean_darksignal = calculate_mean_darksignal(event_module, Fall, scan_key, shuttertime=shuttertime)
        Fall = Fall - mean_darksignal
        Fneu_all = Fneu_all - mean_darksignal
    
    # Calculate smoothing window size in frames
    smoothing_window = int(framerate * smoothing_window_seconds)
    
    # Neuropil correction with median offset to avoid negative values
    dF = (Fall - neuropil_factor * Fneu_all) + (np.nanmedian(Fneu_all, axis=1, keepdims=True) * neuropil_factor)
    
    # Define baseline (F0) computation function
    def compute_F0_single(trace_dF, smoothing_window, percentile):
        """Compute baseline for a single trace using percentile filtering"""
        return percentile_filter(trace_dF, percentile, size=smoothing_window)
    
    # Parallelize F0 computation across all traces
    F0 = np.array(Parallel(n_jobs=-1)(
        delayed(compute_F0_single)(dF[i, :], smoothing_window, percentile) 
        for i in range(dF.shape[0])
    ))
    
    # Calculate ΔF/F0 (normalized fluorescence change)
    dFF = (dF - F0) / F0
    
    return dFF, dF, F0, mean_darksignal


# Function to calculate the mean darksignal of the mean of all traces for a given scan and shuttertime
def calculate_mean_darksignal(event_module, traces, scan_key, shuttertime=0.09):
    """
    Calculate the mean darksignal of the mean of all traces for a given scan and shuttertime.
    Args:
        traces: list or array of fluorescence traces (each trace is 1D array or dict with 'fluorescence' key)
        scan_key: DataJoint key for the scan
        shuttertime: float, shutter time offset (default: 0.09)
    Returns:
        mean_darksignal: float, mean darksignal value
    """
    # Get darkframe times
    try:
        darkframe_shutter_start = (event.Event() & "event_type='shutter'" & scan_key).fetch('event_start_time')
        darkframe_shutter_end = (event.Event() & "event_type='shutter'" & scan_key).fetch('event_end_time')
        darkframetimes = [float(darkframe_shutter_end[0]) + shuttertime, float(darkframe_shutter_start[1]) - shuttertime]
        twoptimestamps = (event.Event() & 'event_type LIKE "%2p_frames%"' & scan_key).fetch('event_start_time')
        darkframes = get_closest_timestamps(darkframetimes, twoptimestamps.astype('float'))
    except Exception:
        darkframes = [12, 24]  # Hardcoding! Problem: importing event already in imaging is problematic
        print("event.Event() not available, using hardcoded darkframes")
        print("darkframes: ", darkframes)

    # Stack traces if needed
    if isinstance(traces[0], dict) and 'fluorescence' in traces[0]:
        traces_stack = np.vstack([tr['fluorescence'] for tr in traces])
    else:
        traces_stack = np.vstack(traces)
    average_trace = np.mean(traces_stack, axis=0)
    mean_darksignal = np.mean(average_trace[darkframes[0]:darkframes[-1]])
    
    print("Mean darksignal: ", mean_darksignal)
    print("Darkframes used: ", darkframes)
    print('Event darkframe times: ', darkframetimes)
    
    return mean_darksignal
# Example usage:
# mean_darksignal = calculate_mean_darksignal(traces, scan_key)


# Function to find the closest timestamp in a series
def get_closest_timestamps(series, target_timestamp):
    # List to store the indices
    indices = []

    # For each timestamp in series1, find the closest timestamp in series2 and get its index
    for t1 in series:
        closest_index = closest_timestamp(target_timestamp, t1)
        indices.append(closest_index)
    return indices

# Function to find closest timestamp
def closest_timestamp(series, target_timestamp):
    index = bisect.bisect_left(series, target_timestamp)
    if index == 0:
        return 0
    if index == len(series):
        return len(series)-1
    before = series[index - 1]
    after = series[index]
    if after - target_timestamp < target_timestamp - before:
       return index
    else:
       return index-1


# Define a function to display the volume with a slider
def display_volume_z(volume, scale, scalemin = 1, scalemax = 99.99):
    z_max = volume.shape[0] - 1
    if scale:
        vmin = np.percentile(volume[:500,:,:],scalemin)  
        vmax = np.percentile(volume[:500,:,:],scalemax)
    else:
        vmin = 0
        vmax = np.max(volume)
    def display_volume_z(z=0):
        plt.imshow(volume[z], cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.show()
    widgets.interact(display_volume_z, z=widgets.IntSlider(min=0, max=z_max, step=1, value=0))



def convert_id_to_datetime(id_str):
    # Convert base-36 encoded string to decimal
    decimal_value = int(id_str, 36)
    
    # Divide by 10^6 to match the MATLAB representation if necessary
    # Adjust this step based on how the ID string relates to MATLAB's serial date number
    matlab_serial_date_number = decimal_value / 10**6
    
    # MATLAB's serial date number starts from 'January 0, 0000'
    # Python's datetime starts from 'January 1, 1970'
    # Calculate the offset between MATLAB's and Python's start dates
    # MATLAB's datenum for January 1, 1970 is 719529
    offset = matlab_serial_date_number - 719529
    
    # Convert the offset to a datetime object
    start_date = datetime(1970, 1, 1)
    result_date = start_date + timedelta(days=offset)
    
    # Format the datetime object to the desired format
    date_str = result_date.strftime('%B %d, %Y %H:%M:%S.%f')[:-3]  # Trim microseconds to 3 digits
    
    return date_str



# Function to return a delayed read operation for each frame
def get_delayed_frame(reader, i):
    return delayed(reader.get_data)(i)

# Function to create a lazy Dask array of video frames
def create_lazy_video_array(video_path):
    reader = imageio.get_reader(video_path, 'ffmpeg')
    n_frames = reader.count_frames()
    lazy_frames = [get_delayed_frame(reader, i) for i in range(n_frames)]
    
    # Convert list of delayed objects to a Dask array
    # Note: You need to know frame shape (height, width, channels) in advance
    frame_shape = reader.get_data(0).shape  # Get shape from the first frame
    lazy_array = [da.from_delayed(frame, shape=frame_shape, dtype='uint8') for frame in lazy_frames]
    video_array = da.stack(lazy_array, axis=0)
    return video_array

# Display function adapted for use with the lazy Dask array
def display_video_with_widgets(video_path):
    video_array = create_lazy_video_array(video_path)
    z_max = video_array.shape[0] - 1
    
    def display_volume_z(z=0):
        frame = video_array[z].compute()  # Compute the frame for display
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
        plt.show()
    
    widgets.interact(display_volume_z, z=widgets.IntSlider(min=0, max=z_max, step=1, value=0))


def rolling_average_filter(image_stack, kernel_size):
    image_stack = image_stack.astype('float32')
    
    # Define the kernel for the rolling average filter
    kernel = np.ones((kernel_size, 1, 1)) / kernel_size
    
    # Apply the rolling average filter along the z-axis
    filtered_image_stack = np.apply_along_axis(lambda m: np.convolve(m, kernel.flatten(), mode='same'), axis=0, arr=image_stack)
    
    return filtered_image_stack

def convolve_chunk(chunk_kernel_tuple):
    chunk, kernel = chunk_kernel_tuple
    return convolve(chunk, kernel, mode='same')

def rolling_average_filter_mt(image_stack, kernel_size):
    # Define the kernel for the rolling average filter
    kernel = np.ones((kernel_size, 1, 1)) / kernel_size

    # Define the overlap size
    overlap = kernel_size

    # Split the image stack into overlapping chunks
    chunks = []
    for i in range(0, image_stack.shape[0] - overlap, image_stack.shape[0] // cpu_count()):
        chunk_start = max(0, i - overlap)
        chunk_end = min(image_stack.shape[0], i + image_stack.shape[0] // cpu_count() + overlap)
        chunk = image_stack[chunk_start:chunk_end, :, :]
        chunks.append(chunk)

    # Process each chunk in parallel
    with Pool(processes=cpu_count()) as pool:
        filtered_chunks = pool.map(convolve_chunk, [(chunk, kernel) for chunk in chunks])

    # Trim the overlapping regions from the filtered chunks
    trimmed_chunks = []
    for i in range(len(filtered_chunks)):
        if i == 0:
            trimmed_chunk = filtered_chunks[i][overlap:, :, :]
        elif i == len(filtered_chunks) - 1:
            trimmed_chunk = filtered_chunks[i][:-overlap, :, :]
        else:
            trimmed_chunk = filtered_chunks[i][overlap:-overlap, :, :]
        trimmed_chunks.append(trimmed_chunk)

    # Concatenate the trimmed chunks in the original order
    filtered_image_stack = np.concatenate(trimmed_chunks, axis=0)

    return filtered_image_stack

# Rescale the image using multithreading
def rescale_image(running_z_projection, p1, p99):
    rescaled_image = (running_z_projection - p1) / (p99 - p1)
    rescaled_image[rescaled_image < 0] = 0
    rescaled_image[rescaled_image > 1] = 1

    # rescaled_image_8bit = cv2.convertScaleAbs(rescaled_image * 255 / np.max(rescaled_image))
    rescaled_image_8bit = cv2.convertScaleAbs(rescaled_image * 255 / 1)

    return rescaled_image_8bit

def process_image_part(running_z_projection, p1, p99, start, end):
    # Process a part of the image
    rescaled_image_part = rescale_image(running_z_projection[start:end], p1, p99)

    return rescaled_image_part

def rescale_image_multithreaded(running_z_projection, p1, p99):
    # Split the image into smaller parts
    num_pixels = len(running_z_projection)
    chunk_size = num_pixels // cpu_count()
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(cpu_count())]
    chunks[-1] = (chunks[-1][0], num_pixels)

    # Process each part of the image using a separate thread
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_chunk = {executor.submit(process_image_part, running_z_projection, p1, p99, start, end): i for i, (start, end) in enumerate(chunks)}
        results = [None] * cpu_count()
        for future in concurrent.futures.as_completed(future_to_chunk):
            index = future_to_chunk[future]
            results[index] = future.result()

    # Combine the results
    rescaled_image = np.concatenate(results)

    return rescaled_image


def make_stack_movie(running_z_projection, filename, fpsset=120, p1set=1, p2set=99.995):

    codecset = 'libx264'

    # Create an imageio VideoWriter object to write the video
    writer = imageio.get_writer(filename, fps=fpsset, codec=codecset, output_params=['-crf', '18', '-preset', 'slow'])

    # Calculate the 1st and 99th percentile
    p1, p99 = np.percentile(running_z_projection[:500,:,:], (p1set, p2set))

    # rescale to 8 bit
    rescaled_image_8bit = rescale_image_multithreaded(running_z_projection, p1, p99)

    for page in rescaled_image_8bit:
        writer.append_data(page)

    # Close the video writer
    writer.close()

    print(filename)

    return rescaled_image_8bit


def make_runningaverage_movie(path, curation_key, runav = 30, p1 = 2, p2 = 99.9, session_id = "XXXX", scan_id = "XXXX", num_frames = None, fps = 120):
    # params_key = (imaging.ProcessingParamSet & 'paramset_idx = "4"').fetch('KEY')
    # reg_tiffs_available = (imaging.ProcessingParamSet & params_key).fetch("params")[0]['reg_tif']
    from scipy.ndimage import mean
    import tifffile
    

    # path = '/datajoint-data/data/jisooj/RN_OPI-1681_2023-02-15_scan9FGLEFJ3_sess9FGLEFJ3/suite2p_exp9FGLEFJ3/suite2p/plane0/reg_tif'
    # Get a list of all tiff files in the folder
    tiff_files = [os.path.join(path, f) for f in natsorted(os.listdir(path)) if f.endswith('.tif')]

    # print(tiff_files)

    # Load each tiff stack into a list of numpy arrays
    stacks = []
    # for f in tiff_files:
    #     with tifffile.TiffFile(f) as tif:
    #         # Get the number of pages in the file
    #         num_pages = len(tif.pages)
            
    #         # Create a numpy array to store all pages
    #         stack = np.zeros((num_pages,) + tif.pages[0].shape, dtype=tif.pages[0].dtype)
            
    #         # Iterate over the pages and store them in the array
    #         for i, page in enumerate(tif.pages):
    #             stack[i] = page.asarray()



    for f in tqdm(tiff_files, desc="Loading registered tiff files"):
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

    # Define the number of frames


    # Select the frames
    if num_frames is not None:
        volume = volume[:num_frames]

    # delete registration tiff
    # for f in tiff_files:
    #     os.remove(f) 
    
    ### moving average filter
    # Create a running Z mean projection of the volume

    # runav = 10
    # running_z_projection = uniform_filter_mt(volume, size=(runav,xyrunav,xyrunav))
        
    # print('Making Moving Average Movie with running average of ' + str(runav) + ' frames - using only ' volume.shape[0] + ' frames - this may take a while')
    
    running_z_projection = rolling_average_filter(volume, runav)


    try:
        session_id = curation_key['session_id']
        scan_id = curation_key['scan_id']
    except:
        print('no session_id or scan_id')
        
    filename = os.path.join(path, 'registered_movie_' + session_id + '_' + scan_id + '_' + str(runav) + '_frame_runningaverage_' + str(fps) + 'fps.mp4')


    # p1 = 2       # percentile scaling low - 1 default
    # p2 = 99.998  # percentile scaling high - 99.995 default


    rescaled_image_8bit = make_stack_movie(running_z_projection, filename, fps, p1, p2)

    # return rescaled_image_8bit
    # tmpdir = dj.config['custom'].get('suite2p_fast_tmp')[0]

def plot_auxdata_overview(scan_key, downsample_factor=1000, height=800, width=1000, aux_setup_type=None, time_range_seconds=(0, 60)):
    """
    Efficiently load and plot all analog and digital auxdata from a scan using interactive Plotly.
    
    Parameters:
    -----------
    scan_key : dict
        DataJoint scan key {'session_id': ..., 'scan_id': ...}
    downsample_factor : int, default=1000
        Factor by which to downsample the data (1 = no downsampling, 1000 = 1000x downsampling)
    height : int, default=800
        Figure height in pixels
    width : int, default=1000
        Figure width in pixels
    aux_setup_type : str, optional
        Auxiliary setup type. If None, will try to fetch from database
    time_range_seconds : tuple of (float, float), default=(0, 60)
        Time range in seconds to load as (start_time, end_time)
        Default loads first 60 seconds. Set to None to load entire recording
        Examples: (0, 10) for first 10 seconds, (30, 60) for seconds 30-60
        
    Returns:
    --------
    fig : plotly figure
        Interactive Plotly figure object
    auxdata_info : dict
        Dictionary containing loaded data information
    """
    from pywavesurfer import ws
    from element_interface.utils import find_full_path
    from adamacs.paths import get_experiment_root_data_dir
    from adamacs.ingest import behavior as ibe
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    import numpy as np
    
    # Set Plotly renderer for VS Code
    pio.renderers.default = "plotly_mimetype+notebook"
    
    # Get aux file path
    try:
        aux_path_relative = (event.BehaviorRecording.File & scan_key & "filepath LIKE '%.h5%'").fetch("filepath")[0]
        auxpath = find_full_path(get_experiment_root_data_dir(), aux_path_relative)
    except:
        print(f"⚠️ No aux file: {scan_key['scan_id']}")
        return None, None
    
    # Get aux setup type if not provided
    if aux_setup_type is None:
        try:
            aux_setup_type = (scan.ScanInfo() & scan_key).fetch1("userfunction_info")
        except:
            aux_setup_type = "unknown"
    
    # Load aux file
    try:
        curr_file = ws.loadDataFile(filename=str(auxpath), format_string='double')
        sweep = [x for x in curr_file.keys() if 'sweep' in x][0]
    except Exception as e:
        print(f"⚠️ Load error: {scan_key['scan_id']}")
        return None, None
    
    # Extract data
    analog_scans = curr_file[sweep]['analogScans']
    analog_labels = curr_file['header']['AIChannelNames']
    digital_labels = curr_file['header']['DIChannelNames'] 
    sr = curr_file['header']['AcquisitionSampleRate'][0][0]
    
    # Demultiplex digital data using optimized function
    try:
        digital_scans = ibe.demultiplex_fast(curr_file[sweep]['digitalScans'][0], len(digital_labels))
    except Exception as e:
        digital_scans = None
    
    # Apply time range filtering if specified
    total_samples = analog_scans.shape[1]
    if time_range_seconds is not None:
        start_time, end_time = time_range_seconds
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Ensure indices are within bounds
        start_sample = max(0, start_sample)
        end_sample = min(total_samples, end_sample)
        
        if start_sample >= end_sample:
            print(f"⚠️ Invalid time range: start ({start_time}s) >= end ({end_time}s)")
            return None, None
        
        # Check if requested end time exceeds recording duration
        max_duration = total_samples / sr
        if end_time > max_duration:
            print(f"📏 Requested end time ({end_time:.1f}s) exceeds recording duration ({max_duration:.1f}s)")
            print(f"   Loading available data: {start_time:.1f}-{max_duration:.1f}s")
            end_time = max_duration
            duration_str = f"{start_time:.1f}-{max_duration:.1f}s"
        else:
            duration_str = f"{start_time:.1f}-{end_time:.1f}s"
        
        # Slice the data
        analog_scans = analog_scans[:, start_sample:end_sample]
        if digital_scans is not None:
            digital_scans = digital_scans[:, start_sample:end_sample]
            
        # Adjust timebase to start from the specified start time
        timebase_offset = start_time
    else:
        # Load entire recording when explicitly set to None
        timebase_offset = 0
        duration_str = f"{total_samples / sr:.1f}s (full)"
    
    # Create timebase (downsampled) with offset
    timebase = (np.arange(0, analog_scans.shape[1], downsample_factor) / sr) + timebase_offset
    
    # Determine channel counts
    n_analog = analog_scans.shape[0]
    n_digital = digital_scans.shape[0] if digital_scans is not None else 0
    total_channels = n_analog + n_digital
    
    if total_channels == 0:
        return None, None
    
    # Create subplot figure with shared x-axis
    subplot_titles = []
    for i in range(n_analog):
        subplot_titles.append(f"AI {i}: {analog_labels[i]}")
    if digital_scans is not None:
        for i in range(n_digital):
            subplot_titles.append(f"DI {i}: {digital_labels[i]}")
    
    fig = make_subplots(
        rows=total_channels, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=subplot_titles,
        row_heights=[1] * total_channels
    )
    
    # Define color palette for better visualization
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot analog channels
    for i in range(n_analog):
        data_downsampled = analog_scans[i][::downsample_factor]
        timebase_trimmed = timebase[:len(data_downsampled)]
        
        fig.add_trace(
            go.Scatter(
                x=timebase_trimmed,
                y=data_downsampled,
                mode='lines',
                name=f"AI {i}: {analog_labels[i]}",
                line=dict(color=colors[i % len(colors)], width=1),
                hovertemplate=f"<b>AI {i}: {analog_labels[i]}</b><br>" +
                             "Time: %{x:.3f} s<br>" +
                             "Value: %{y:.4f}<br>" +
                             "<extra></extra>"
            ),
            row=i+1, col=1
        )
    
    # Plot digital channels  
    if digital_scans is not None:
        for i in range(n_digital):
            data_downsampled = digital_scans[i][::downsample_factor]
            timebase_digital = timebase[:len(data_downsampled)]
            
            fig.add_trace(
                go.Scatter(
                    x=timebase_digital,
                    y=data_downsampled,
                    mode='lines',
                    name=f"DI {i}: {digital_labels[i]}",
                    line=dict(color=colors[(n_analog + i) % len(colors)], width=1),
                    hovertemplate=f"<b>DI {i}: {digital_labels[i]}</b><br>" +
                                 "Time: %{x:.3f} s<br>" +
                                 "Value: %{y}<br>" +
                                 "<extra></extra>"
                ),
                row=n_analog+i+1, col=1
            )
            
            # Set y-axis range for digital channels
            fig.update_yaxes(range=[-0.1, 1.1], row=n_analog+i+1, col=1)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{scan_key["scan_id"]} - Interactive Aux Data Overview<br>' +
                 f'<sub>Setup: {aux_setup_type} | SR: {sr:.1f} Hz | ' +
                 f'Downsample: {downsample_factor}x | Duration: {duration_str}</sub>',
            x=0.5,
            font=dict(size=16)
        ),
        height=height,
        width=width,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        ),
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Update x-axis (only bottom subplot gets label)
    fig.update_xaxes(title_text="Time (s)", row=total_channels, col=1)
    
    # Update y-axes with channel labels
    for i in range(n_analog):
        fig.update_yaxes(title_text=f"AI {i}", row=i+1, col=1)
    if digital_scans is not None:
        for i in range(n_digital):
            fig.update_yaxes(title_text=f"DI {i}", row=n_analog+i+1, col=1)
    
    # Add crossfilter (zoom/pan synchronization)
    fig.update_layout(
        xaxis=dict(rangeslider=dict(visible=False)),
        dragmode='zoom'
    )
    
    # Prepare return info
    auxdata_info = {
        'scan_id': scan_key['scan_id'],
        'aux_setup_type': aux_setup_type,
        'sample_rate': sr,
        'downsample_factor': downsample_factor,
        'n_analog_channels': n_analog,
        'n_digital_channels': n_digital,
        'analog_labels': analog_labels.tolist() if hasattr(analog_labels, 'tolist') else analog_labels,
        'digital_labels': digital_labels.tolist() if hasattr(digital_labels, 'tolist') else digital_labels,
        'duration_seconds': analog_scans.shape[1] / sr,
        'time_range_seconds': time_range_seconds,
        'total_duration_seconds': total_samples / sr,
        'aux_file_path': str(auxpath)
    }

    # Minimal output (only print if successful)
    print(f"📊 {scan_key['scan_id']}: {duration_str}, {n_analog}AI/{n_digital}DI channels")
    
    return fig, auxdata_info