#!/usr/bin/env python
# coding: utf-8

# Tobias Rose 2023

import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import re


def plot_event_trial_start_times(event_data, trial_data, scan_id="", behavior_notes=""):
    # Set the Seaborn style to dark
    sns.set(style="dark")

    # Set the style to 'dark_background'
    plt.style.use('dark_background')

    # Preparing event data for plotting
    event_times_by_type = {}
    for data in event_data:
        etype = data['event_type']
        etime = data['event_start_time']
        if etype not in event_times_by_type:
            event_times_by_type[etype] = []
        event_times_by_type[etype].append(etime)

    # Integrating trial start times into event data
    for data in trial_data:
        ttype = data['trial_type']
        ttime = data['trial_start_time']
        # Use a prefix like 'trial-' to distinguish trial types from event types
        ttype_key = f'trial-{ttype}'
        if ttype_key not in event_times_by_type:
            event_times_by_type[ttype_key] = []
        event_times_by_type[ttype_key].append(ttime)

    # Define the HUSL color palette (one color per event/trial type)
    colors = sns.husl_palette(len(event_times_by_type), h=0.5, s=0.8, l=0.7)

    # Plotting
    plt.figure(figsize=(20, 20))
    for i, (etype, etimes) in enumerate(event_times_by_type.items()):
        plt.eventplot(etimes, lineoffsets=i, linelengths=0.8, colors=[colors[i]], label=etype)

    plt.yticks(range(len(event_times_by_type)), list(event_times_by_type.keys()))
    plt.xlabel('Event Start Time')
    plt.ylabel('Event/Trial Type')
    title_suffix = []
    if scan_id:
        title_suffix.append(str(scan_id))
    if behavior_notes:
        notes = str(behavior_notes).replace("\n", " ").strip()
        title_suffix.append(notes[:100])
    title = "Event and Trial Start Times Plot"
    if title_suffix:
        title += " | " + " | ".join(title_suffix)
    plt.title(title)
    # plt.legend()
    # plt.show()
    return(plt)
 

def plot_event_trial_start_times_zoom(event_data, trial_data, time_start=None, time_end=None, 
                                     figsize=(20, 20), line_height=0.8, enable_zoom=True,
                                     scan_id="", behavior_notes=""):
    """
    Enhanced version of plot_event_trial_start_times with zoom functionality.
    
    Parameters:
    -----------
    event_data : list of dict
        Event data with 'event_type' and 'event_start_time' keys
    trial_data : list of dict  
        Trial data with 'trial_type' and 'trial_start_time' keys
    time_start : float, optional
        Start time for zoom window (seconds)
    time_end : float, optional
        End time for zoom window (seconds)
    figsize : tuple, optional
        Figure size (width, height)
    line_height : float, optional
        Height of event lines in the plot
    enable_zoom : bool, optional
        Enable interactive zoom functionality
        
    Returns:
    --------
    matplotlib.pyplot
        The pyplot object for further customization
    """
    # Set the Seaborn style to dark
    sns.set(style="dark")
    plt.style.use('dark_background')

    # Preparing event data for plotting
    event_times_by_type = {}
    all_times = []
    
    for data in event_data:
        etype = data['event_type']
        # etime = data['event_start_time']
        etime = float(data['event_start_time'])
        all_times.append(etime)
        if etype not in event_times_by_type:
            event_times_by_type[etype] = []
        event_times_by_type[etype].append(etime)

    # Integrating trial start times into event data
    for data in trial_data:
        ttype = data['trial_type']
        ttime = data['trial_start_time']
        all_times.append(ttime)
        # Use a prefix like 'trial-' to distinguish trial types from event types
        ttype_key = f'trial-{ttype}'
        if ttype_key not in event_times_by_type:
            event_times_by_type[ttype_key] = []
        event_times_by_type[ttype_key].append(ttime)

    # Determine time window
    if all_times:
        data_start = min(all_times)
        data_end = max(all_times)
        
        # Use provided zoom window or default to full data range
        if time_start is None:
            time_start = data_start
        if time_end is None:
            time_end = data_end
            
        # Ensure zoom window is within data bounds
        time_start = max(time_start, data_start)
        time_end = min(time_end, data_end)
    else:
        time_start = 0
        time_end = 1

    # Filter events to zoom window
    filtered_event_times_by_type = {}
    for etype, etimes in event_times_by_type.items():
        filtered_times = [t for t in etimes if time_start <= t <= time_end]
        if filtered_times:  # Only include types with events in the zoom window
            filtered_event_times_by_type[etype] = filtered_times

    if not filtered_event_times_by_type:
        print(f"No events found in time window [{time_start:.2f}, {time_end:.2f}]")
        return plt

    # Define the HUSL color palette (one color per event/trial type)
    colors = sns.husl_palette(len(filtered_event_times_by_type), h=0.5, s=0.8, l=0.7)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plotting
    for i, (etype, etimes) in enumerate(filtered_event_times_by_type.items()):
        ax.eventplot(etimes, lineoffsets=i, linelengths=line_height, 
                    colors=[colors[i]], label=etype, linewidths=2)

    # Customize plot
    ax.set_yticks(range(len(filtered_event_times_by_type)))
    ax.set_yticklabels(list(filtered_event_times_by_type.keys()))
    ax.set_xlabel('Event Start Time (s)')
    ax.set_ylabel('Event/Trial Type')
    
    # Set zoom window
    ax.set_xlim(time_start, time_end)
    
    # Enhanced title with zoom info
    if time_start != data_start or time_end != data_end:
        duration = time_end - time_start
        title = f'Event and Trial Start Times Plot (Zoom: {time_start:.2f}s - {time_end:.2f}s, Duration: {duration:.2f}s)'
    else:
        title = 'Event and Trial Start Times Plot (Full Range)'
    title_suffix = []
    if scan_id:
        title_suffix.append(str(scan_id))
    if behavior_notes:
        notes = str(behavior_notes).replace("\n", " ").strip()
        title_suffix.append(notes[:100])
    if title_suffix:
        title += " | " + " | ".join(title_suffix)
    ax.set_title(title)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add zoom controls and info text
    if enable_zoom:
        # Add text box with zoom info and instructions
        zoom_info = f'Time window: [{time_start:.2f}, {time_end:.2f}]s\n'
        zoom_info += f'Duration: {time_end - time_start:.2f}s\n'
        zoom_info += f'Events shown: {sum(len(times) for times in filtered_event_times_by_type.values())}\n'
        zoom_info += 'Use matplotlib toolbar to pan/zoom'
        
        ax.text(0.02, 0.98, zoom_info, transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="darkblue", alpha=0.7),
               verticalalignment='top', fontsize=10, color='white')
        
        # Enable interactive navigation - use %matplotlib widget in Jupyter for best results
        # Configure matplotlib for interactivity
        try:
            import matplotlib
            # For Jupyter notebooks, ensure we use widget backend
            if 'ipykernel' in matplotlib.get_backend().lower():
                plt.ion()  # Turn on interactive mode
            # Enable toolbar
            fig.canvas.toolbar_visible = True if hasattr(fig.canvas, 'toolbar_visible') else None
        except:
            pass  # Fallback gracefully if backend doesn't support it
    
    # Add legend if not too many items
    if len(filtered_event_times_by_type) <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
    
    return plt


def plot_event_trial_start_times_zoom_plotly(event_data, trial_data, time_start=None, time_end=None, 
                                             height=800, width=1200):
    """
    Interactive Plotly version of plot_event_trial_start_times_zoom with full zoom/pan capabilities.
    
    Parameters:
    -----------
    event_data : list of dict
        Event data with 'event_type' and 'event_start_time' keys
    trial_data : list of dict  
        Trial data with 'trial_type' and 'trial_start_time' keys
    time_start : float, optional
        Start time for initial zoom window (seconds)
    time_end : float, optional
        End time for initial zoom window (seconds)
    height : int, optional
        Plot height in pixels
    width : int, optional
        Plot width in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.colors import qualitative
    except ImportError:
        print("Plotly not available. Install with: pip install plotly")
        return None

    # Prepare event data for plotting
    event_times_by_type = {}
    all_times = []
    
    for data in event_data:
        etype = data['event_type']
        etime = data['event_start_time']
        all_times.append(etime)
        if etype not in event_times_by_type:
            event_times_by_type[etype] = []
        event_times_by_type[etype].append(etime)

    # Integrate trial start times into event data
    for data in trial_data:
        ttype = data['trial_type']
        ttime = data['trial_start_time']
        all_times.append(ttime)
        ttype_key = f'trial-{ttype}'
        if ttype_key not in event_times_by_type:
            event_times_by_type[ttype_key] = []
        event_times_by_type[ttype_key].append(ttime)

    if not all_times:
        print("No events or trials to plot")
        return None

    # Determine time range
    data_start = min(all_times)
    data_end = max(all_times)
    
    if time_start is None:
        time_start = data_start
    if time_end is None:
        time_end = data_end

    # Create figure
    fig = go.Figure()
    
    # Define high-contrast, bright colors that are visible on dark background
    bright_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#85929E', '#A569BD',
        '#5DADE2', '#58D68D', '#F4D03F', '#EB984E', '#AED6F1'
    ]
    
    # Plot each event type with enhanced visibility
    for i, (etype, etimes) in enumerate(event_times_by_type.items()):
        color = bright_colors[i % len(bright_colors)]
        
        # Create multiple traces for better visibility
        y_pos = [i] * len(etimes)
        
        # Add main event markers (large circles)
        fig.add_trace(go.Scatter(
            x=etimes,
            y=y_pos,
            mode='markers',
            marker=dict(
                color=color,
                size=12,
                symbol='circle',
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            name=etype,
            hovertemplate=f'<b>{etype}</b><br>Time: %{{x:.3f}}s<br>Count: {len(etimes)} events<extra></extra>',
            showlegend=True
        ))
        
        # Add vertical tick marks for precise timing
        for etime in etimes:
            fig.add_shape(
                type="line",
                x0=etime, y0=i-0.3, x1=etime, y1=i+0.3,
                line=dict(color=color, width=3, dash='solid'),
                opacity=0.7
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Interactive Event and Trial Timeline<br><sub>Zoom: {time_start:.2f}s - {time_end:.2f}s</sub>',
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Time (s)', font=dict(color='white', size=14)),
            range=[time_start, time_end],
            showgrid=True,
            gridcolor='rgba(128,128,128,0.5)',
            gridwidth=1,
            tickfont=dict(color='white', size=12),
            linecolor='white',
            linewidth=2,
            mirror=True
        ),
        yaxis=dict(
            title=dict(text='Event/Trial Type', font=dict(color='white', size=14)),
            tickvals=list(range(len(event_times_by_type))),
            ticktext=list(event_times_by_type.keys()),
            showgrid=True,
            gridcolor='rgba(128,128,128,0.5)',
            gridwidth=1,
            tickfont=dict(color='white', size=12),
            linecolor='white',
            linewidth=2,
            mirror=True,
            range=[-0.5, len(event_times_by_type) - 0.5]
        ),
        height=height,
        width=width,
        template='plotly_dark',
        plot_bgcolor='rgba(20,20,20,1)',
        paper_bgcolor='rgba(0,0,0,1)',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(40,40,40,0.8)',
            bordercolor='white',
            borderwidth=1,
            font=dict(color='white', size=12)
        ),
        margin=dict(r=220, l=80, t=100, b=80),  # Extra margins for better layout
        font=dict(color='white', size=12)
    )
    
    # Add rangeslider for easy navigation
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="linear"
        )
    )
    
    return fig


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
