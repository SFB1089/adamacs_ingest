import datajoint as dj
from . import subject
from element_deeplabcut.model import PoseEstimationNew
from adamacs.schemas.mocap import MotionCapture
from adamacs.schemas.virtual_markers_optitrack import RigidMouseTracking
from ..pipeline import model, db_prefix, event
#from adamacs.pipeline import subject, session, equipment, surgery, event, trial, imaging, behavior, model, scan
from adamacs.schemas import subject
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm.notebook import tqdm
from tqdm import tqdm

from scipy.interpolate import interp1d, Akima1DInterpolator
# The numerical core of the gaze chain lives in a module that declares nothing, so the
# DISK-imputed sibling chain in adamacs/schemas/disk.py runs exactly this code. See that
# module's docstring for why it cannot simply be functions on this one.
from adamacs.helpers.eye_geometry import (  # noqa: F401  (re-exported for callers)
    interpolate_fill_gaps, resample_eyecam_to_optitrack, upsample_eye_trace,
    rot_x, rot_y, rot_z, side_sign_from_recording_id, eye_camera_event_type,
    compute_gaze_vectors)
schema = dj.schema(db_prefix + 'pupil_tracking')

@schema
class PupilEllipseParameter(dj.Lookup):
  """Set hyperparameters for pupil ellipse fitting, mainly for preprocessing"""
  definition = """
  parameter_id: int # unique id for parameters
  ---
  likelihood_thres: float # likelihood threshold for good tracking
  exclude_ir_std: float # define a boundary by mean and std to separate good tracking and bad tracking
  ellipticity_thres: float # for computing camera center by elliptic frames
  description: varchar(255) # short description of the parameters
  """

  contents = [(1, 0.5, 5, 0.85, "default, free-moving eye cam")]


@schema
class PupilEllipseImageFitParameter(dj.Lookup):
    """Extension of PupilEllipseParameter: a row here switches PupilEllipseFitting.make() to the image rim fit
    (adamacs/helpers/pupil_image_fit.py) for that parameter set, for head-fixed face-cam recordings only.
    Parameter sets without a row here run the DLC-marker fit exactly as before.
    Rows are inserted explicitly (no contents), e.g. parameter_id 10 = "image rim fit (v2)"."""
    definition = """
    -> PupilEllipseParameter
    ---
    fit_source                        : enum('image_rim')
    guide_source                      : enum('dlc', 'disk_headfixed', 'disk_else_dlc')  # markers that place the rays
    disk_model_name = ''              : varchar(64)   # DISK head-fixed imputation used as guide (disk_* sources)
    disk_parameter_id = null          : int
    disk_imputation_paramset_id = null : int
    n_rays                            : smallint
    r_min                             : double        # ray start, x guide radius R0
    r_max                             : double        # ray end, x R0
    r_step                            : double        # px
    blur_sigma                        : double        # px, Gaussian on the grayscale frame
    ransac_draws                      : smallint
    inlier_tol                        : double        # px, normalised radial residual
    min_inlier_frac                   : double        # accept (v2)
    min_rim_step_rel                  : double        # accept: x session median rim step
    axis_min                          : double        # accept: x session median semi-axis
    axis_max                          : double
    tc_dist_px                        : double        # temporal check: distance from the rolling median
    tc_window_s                       : double        # rolling-median window
    tc_max_brief_s                    : double        # excursions shorter than this are rejected
    nice_min_inlier_frac              : double        # camera-centre frames: accepted and inlier_frac >= this
    angle_estimator                   : enum('production', 'eye_model_3d')  # theta/phi: production camera-centre estimator or pooled 3D eye model
    e3d_k0 = null                     : double        # eye model: prior R / fissure length (1.25 mm / 3 mm)
    e3d_k_sd = null                   : double        # prior SD of R / fissure
    e3d_prior_weight = null           : double        # prior weight x sqrt(n frames)
    e3d_huber_k = null                : double        # IRLS Huber threshold
    e3d_irls_iter = null              : smallint
    e3d_n_frames = null               : int           # random subset of accepted frames for the fit
    corner_llh = null                 : double        # fissure: per-frame corners above this likelihood ...
    corner_min_frac = null            : double        # ... if at least this fraction of frames has both
    static_corner_llh = null          : double        # else static corners: session medians over frames above this
    static_min_frames = null          : int
    seed                              : int           # RANSAC generator seed (one stream, sequential over frames)
    code_version                      : varchar(80)
    gate_min_px = null                : double        # v3 spatial gate: reject kept centres > max(gate_min_px, gate_r0_mult*R0) from their median; null = no gate (v2)
    gate_r0_mult = null               : double
    fissure_prior = null              : enum('session', 'animal_median')  # eye-model prior length; null = 'session' (v2)
    fissure_min_px = null             : double        # valid per-session fissure range for the animal median
    fissure_max_px = null             : double
    e3d_ms_deg = null                 : double        # multi-start grid of the mean gaze offset, +- deg; null = single start (v2)
    e3d_ms_n = null                   : smallint      # n x n starts (plus the v2 single start)
    r_max_guide = null                : double        # v4: rays end at max(r_max*R0, r_max_guide*rolling guide radius); null = r_max*R0 (v3)
    guide_r_window_s = null           : double        # v4: rolling-median window of the per-frame guide radius, s
    guide_r_engage = null             : double        # v4: guide-scaled rays only where the rolling guide radius > this x R0
    r_max_retry = null                : double        # v4: refit failed frames (inlier_frac < min) with rays to this x max(R0, guide radius); null = no retry
    weak_inlier_frac = null           : double        # v4: fits below this inlier fraction need corroboration by confident fits nearby in time; null = off (v3)
    weak_ref_min_inlier = null        : double        # v4: confident = v2-accepted with inlier_frac >= this
    weak_ref_window_s = null          : double        # v4: +- window of the confident-fit reference (rolling median), s
    weak_ref_dist_max = null          : double        # v4: weak fit centre within this x R0 of the reference centre ...
    weak_ref_axis_min = null          : double        # v4: ... and semi-major / reference semi-major within [min, max]
    weak_ref_axis_max = null          : double
    """

@schema
class PupilEllipseFitting(dj.Computed):
    """fit pupil dots from DLC into ellipse"""
    definition = """
    -> PupilEllipseParameter
    -> PoseEstimationNew
    ---
    ellipse_dict: longblob # dictionary of ellipse fitting results
    cam_center: longblob # camera center in pixel coordinates
    eye_center: longblob # eye center in pixel coordinates
    scale: longblob # scaling factor from pixel to rotation angle
    theta: longblob # eye rotation in radians at azimuth direction (left/right = negative/positive), centered by eye center
    phi: longblob # eye rotation in elevation direction
    diameter: longblob # long axis of the fitted ellipse (equals pupil diameter)
    """    
    # pupil_center: longblob # pupil center in pixel coordinates

    class ImageFit(dj.Part):
        """Provenance and QC of an image rim fit (parameter sets with a PupilEllipseImageFitParameter row)."""
        definition = """
        -> master
        ---
        guide_used          : enum('dlc', 'disk_headfixed')
        fit_origin          : enum('video', 'cache')   # cache = precomputed per-frame fits of the same code version
        cache_file = ''     : varchar(512)
        video_file          : varchar(512)
        n_frames_markers    : int unsigned
        n_frames_video      : int unsigned
        n_frames_fitted     : int unsigned              # frames with a finite rim fit
        fps                 : float
        r0                  : float                     # guide radius, px
        rim_step_median     : float                     # session median rim step, gray levels
        axis_median         : float                     # session median semi-axis, px
        frac_accepted_v2    : float
        frac_accepted       : float                     # v2 accept and temporally consistent
        n_brief_jumps       : int unsigned              # frames rejected by the temporal check
        n_nice_frames       : int unsigned              # frames used for the camera centre
        angle_estimator     : enum('production', 'eye_model_3d')
        e3d_cx = null       : double                    # eye model: rotation-centre projection, px
        e3d_cy = null       : double
        e3d_r = null        : double                    # eye model: pupil rotation radius, px (= scale)
        fissure_px = null   : double                    # palpebral fissure length used for the prior
        fissure_source = 'none' : enum('per_frame', 'static', 'none', 'animal_median', 'rig_median')
        theta_offset = null : double                    # rad: theta + theta_offset = azimuth relative to the camera axis
        phi_offset = null   : double                    # rad: phi + phi_offset = elevation relative to the camera axis
        n_frames_eye_model = null : int unsigned
        code_version        : varchar(80)
        fissure_session_px = null : double              # v3: this recording's own fissure measurement (source per fissure_session_source)
        fissure_session_source = null : enum('per_frame', 'static', 'none')
        fissure_n_sessions = null : smallint unsigned   # v3: sessions in the animal (or rig) median
        n_frames_rejected_spatial = null : int unsigned # v3: kept frames rejected by the spatial gate
        gate_px = null      : float                     # v3: spatial gate radius used, px
        e3d_objective = null : double                   # v3: robust objective of the chosen eye-model solution
        e3d_objective_single = null : double            # v3: objective of the v2 single-start solution
        e3d_objective_far = null : double               # v3: best objective among solutions > 5 deg from the chosen offset
        e3d_n_starts = null : smallint unsigned
        n_frames_long_rays = null : int unsigned        # v4: frames fitted with guide-scaled rays longer than r_max*R0
        n_frames_rejected_weak = null : int unsigned    # v4: weak fits (inlier_frac < weak_inlier_frac) not corroborated by confident fits
        n_frames_retry = null : int unsigned            # v4: frames whose fit comes from the long-ray retry
        """

    @property
    def key_source(self):
        """All parameter sets x pose estimations, as before; image-fit parameter sets only for face-cam recordings."""
        ks = PupilEllipseParameter.proj() * PoseEstimationNew.proj()
        img = PupilEllipseImageFitParameter.proj()
        return (ks - img) + (ks & img & 'recording_id LIKE "%face%"')

    # def plot_preprocess(self, key):
    #     # IR exlusion
    #     id_bad_ir=np.isnan(ir_xg)
    #     bad_ir_fr = np.where(id_bad_ir)[0]

    #     bad_p_fr=[]
    #     for i, (x, y) in enumerate(zip(p_x, p_y)):
    #         if np.count_nonzero(~np.isnan(x))<5:
    #             bad_p_fr.append(i)
          
    #     print("# frames of bad IR tracking:", len(bad_ir_fr),", # frames of bad pupil tracking:", len(bad_p_fr),'intersecting # frames', len(np.intersect1d(bad_ir_fr,bad_p_fr)))

    #     plt.plot(ir_x[~ir_bad],ir_y[~ir_bad],'o', alpha=0.6,markersize=1,label='selected tracking')
    #     plt.plot(ir_x[ir_bad],ir_y[ir_bad],'o', alpha=0.6,markersize=1, label='excluded tracking')
    #     plt.plot(irx_mean,iry_mean,'o',label='mean position')
    #     plt.legend()
    #     plt.title('tracking dots for IR reflection')
    def fit_ellipse(self, i, x, y):
        # NaN need to be removed for SVD convergence
        x0=x[~np.isnan(x)]
        y0=y[~np.isnan(y)]
        
        meanX = np.nanmean(x)
        meanY = np.nanmean(y)
        
        # remove bias
        x = x0 - meanX
        y = y0 - meanY
        
        # Estimation of the conic equation
        X = np.array([x**2, x*y, y**2, x, y])
        X = np.stack(X).T
        try:
            param = np.dot(np.sum(X, axis=0), np.linalg.pinv(np.matmul(X.T,X)))

        
            # #least square method to fit ellipse
            # b = np.ones(8)
            # param = np.linalg.lstsq(X, b, rcond=None)[0].squeeze() # very similar results
            
            # Extract parameters from the conic equation
            a, b, c, d, e = param[0], param[1], param[2], param[3], param[4]
            
            if b**2-4*a*c > 0: 
                print('given dots does not form a valid ellipse for frame',i)
            
            # Eigen decomp
            Q = np.array([[a, b/2],[b/2, c]])
            eig_val, eig_vec = np.linalg.eig(Q)
            
            # Get angle to long axis
            if eig_val[0] < eig_val[1]:
                angle_to_x = np.arctan2(eig_vec[1,0], eig_vec[0,0])
            else:
                angle_to_x = np.arctan2(eig_vec[1,1], eig_vec[0,1])
            
            angle_from_x = angle_to_x
            orientation_rad = 0.5 * np.arctan2(b, (c-a))
            cos_phi = np.cos(orientation_rad)
            sin_phi = np.sin(orientation_rad)
            
            a, b, c, d, e = [a*cos_phi**2 - b*cos_phi*sin_phi + c*sin_phi**2,
                            0,
                            a*sin_phi**2 + b*cos_phi*sin_phi + c*cos_phi**2,
                            d*cos_phi - e*sin_phi,
                            d*sin_phi + e*cos_phi]
            
            meanX, meanY = [cos_phi*meanX - sin_phi*meanY,
                            sin_phi*meanX + cos_phi*meanY]
            
            # Check if conc expression represents an ellipse
            test = a*c

            # select tracking frames only when fitted valid ellipse, and
            # IR is valid and at least 5 pupil dots are available
            if test > 0 and np.count_nonzero(~np.isnan(x))>=5:
            
                # Make sure coefficients are positive
                if a<0:
                    a, c, d, e = [-a, -c, -d, -e]
            
                # Final ellipse parameters
                X0 = meanX - d/2/a
                Y0 = meanY - e/2/c
                F = 1 + (d**2)/(4*a) + (e**2)/(4*c)
                a = np.sqrt(F/a)
                b = np.sqrt(F/c)
                long_axis = 2*np.maximum(a,b)
                short_axis = 2*np.minimum(a,b)
            
                # Rotate axes backwards to find center point of
                # original tilted ellipse
                R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
                P_in = R @ np.array([[X0],[Y0]])
                X0_in = P_in[0][0]
                Y0_in = P_in[1][0]

                # Organize parameters in dictionary to return
                ellipse_dict = {
                    'X0':X0,
                    'Y0':Y0,
                    'F':F,
                    'a':a,
                    'b':b,
                    'long_axis':long_axis/2,
                    'short_axis':short_axis/2,
                    'angle_to_x':angle_to_x,
                    'angle_from_x':angle_from_x,
                    'cos_phi':cos_phi,
                    'sin_phi':sin_phi,
                    'X0_in':X0_in,
                    'Y0_in':Y0_in,
                    'phi':orientation_rad
                }
            
            else:
                # If the conic equation didn't return an ellipse, do not
                # return any real values and fill the dictionary with NaNs.
                dict_keys = ['X0','Y0','F','a','b','long_axis',
                                'short_axis','angle_to_x','angle_from_x',
                                'cos_phi','sin_phi','X0_in','Y0_in','phi']
                dict_vals = list(np.ones([len(dict_keys)]) * np.nan)
                ellipse_dict = dict(zip(dict_keys, dict_vals))
            return ellipse_dict
        except np.linalg.LinAlgError as e:
            print("Error occurred during computation:", e, "for frame",i) 

    def make(self, key):
        # load all the tracking dots
        bodypart = ['nose_corner', 'lateral_corner', 'IR', 'pupil_left',
                    'pupil_left_lower','pupil_left_up','pupil_lower','pupil_right',
                    'pupil_right_lower','pupil_right_up','pupil_upper']
        x_all = []
        y_all = []
        llh_all = []

        for bp in bodypart:
            x = (PoseEstimationNew.BodyPartPosition & key & f"body_part='{bp}'").fetch('x_pos')
            y = (PoseEstimationNew.BodyPartPosition & key & f"body_part='{bp}'").fetch('y_pos')
            llh = (PoseEstimationNew.BodyPartPosition & key & f"body_part='{bp}'").fetch('likelihood')     

            x_all = np.concatenate((x_all,x))
            y_all = np.concatenate((y_all,y))
            llh_all = np.concatenate((llh_all,llh))

        x_all = np.vstack(x_all).T
        y_all = np.vstack(y_all).T
        llh_all = np.vstack(llh_all).T

        # get parameters from PupilEllipseParameter table
            # set a cutoff for poor fitting points in DLC model
        thres_llh = (PupilEllipseParameter & key).fetch1('likelihood_thres') # 0.5 #0.8
        thres_ir = (PupilEllipseParameter & key).fetch1('exclude_ir_std')
        thres_ellipticity = (PupilEllipseParameter & key).fetch1('ellipticity_thres')

        # set points lower than thres to NaN
        mask = llh_all < thres_llh
        # Set the elements in x_all and y_all to NaN where the mask is True
        x_thr, y_thr = np.copy(x_all), np.copy(y_all)
        x_thr[mask] = np.nan
        y_thr[mask] = np.nan

        # assign dots coord to corresponding name
        ncor_x = x_thr[:,0]
        ncor_y = y_thr[:,0]
        lcor_x = x_thr[:,1]
        lcor_y = y_thr[:,1]
        ir_x = x_thr[:,2]
        ir_y = y_thr[:,2]
        p_x = x_thr[:,3:]
        p_y = y_thr[:,3:]

        # define a boundary by mean and std to separate good tracking and bad tracking
        irx_mean = np.nanmean(ir_x)
        iry_mean = np.nanmean(ir_y)
        irx_std = np.nanstd(ir_x)
        iry_std = np.nanstd(ir_y)
        ir_std = np.linalg.norm([irx_std,iry_std])
        # set threshold as 5*std, still need to validata with other dataset
        ir_bad=(ir_x-irx_mean)**2+(ir_y-iry_mean)**2>(thres_ir*ir_std)**2 

        # exclude off position IR dots
        ir_xg, ir_yg = np.copy(ir_x), np.copy(ir_y)
        ir_xg[ir_bad] = np.nan
        ir_yg[ir_bad] = np.nan

        # assume camera is following IR reflection
        # center pupil tracking by IR reflection
        pc_x = p_x-ir_xg[:,None]
        pc_y = p_y-ir_yg[:,None] # if IR is invalid, the pupil tracking is invalid



        # image rim fit instead of the marker fit when the parameter set has an extension row
        image_cfg = (PupilEllipseImageFitParameter & key).fetch(as_dict=True)
        image_cfg = image_cfg[0] if image_cfg else None
        image_info = None
        if image_cfg is not None:
            results, ellipse_dict_all, image_nice, image_info = self._image_rim_frames(
                key, image_cfg, x_thr, y_thr, ir_xg, ir_yg, thres_ellipticity)
        else:
            # select tracking frames only when IR is valid and at least 6 pupil dots are available
            nd = 6 ### can be put in parameter table!
            track_fr=[]
            for i, (x, y) in enumerate(zip(pc_x, pc_y)):
                if np.count_nonzero(~np.isnan(x))>=nd:
                    track_fr.append(i)
            print(len(track_fr), ' frames out of ', len(pc_x), ' frames is selected for ', nd, ' dots threshold')

            fields = ["a", "b", "X0_in", "Y0_in", "long_axis", "short_axis", "angle_from_x", "X0+ir", "Y0+ir"]
            results = {f: [] for f in fields}
            ellipse_dict_all = []
            for i, (x, y) in enumerate(tqdm(zip(pc_x, pc_y), total=len(pc_x), desc="fitting ellipses")):
                if i in track_fr: 
                    ellipse_dic = self.fit_ellipse(i, x, y)
                    ellipse_dic['X0+ir'] = ellipse_dic['X0'] + ir_xg[i]
                    ellipse_dic['Y0+ir'] = ellipse_dic['Y0'] + ir_yg[i]
                else: ellipse_dic = {f: np.nan for f in fields}
                # collect values
                for f in fields:
                    results[f].append(ellipse_dic[f])
                ellipse_dict_all.append(ellipse_dic)

        # convert lists to arrays for camera center estimation
        X0_in_all    = np.array(results["X0_in"])
        Y0_in_all    = np.array(results["Y0_in"])
        long_axis_all  = np.array(results["long_axis"])
        short_axis_all = np.array(results["short_axis"])
        angle_all    = np.array(results["angle_from_x"])


        if image_cfg is not None:
            nice_mask = image_nice
        else:
            nice_mask=[]
            for i, (x, y) in enumerate(zip(pc_x, pc_y)):
                if np.count_nonzero(~np.isnan(x))>=7:
                    if short_axis_all[i]/long_axis_all[i] < thres_ellipticity: # 0.85 is the threshold from Neill's code
                        nice_mask.append(i)
                
        if image_cfg is not None and image_cfg["angle_estimator"] == "eye_model_3d":
            cam_center, eye_center, scale, theta, phi, e3d_info = self._eye_model_angles(
                key, image_cfg, results, x_all, y_all, llh_all)
            image_info.update(e3d_info)
        else:
            # calculate camera center with selected frames
            A = np.vstack([np.cos(angle_all[nice_mask]),np.sin(angle_all[nice_mask])])
            b = np.expand_dims(np.diag(A.T @ np.squeeze(np.array([X0_in_all[nice_mask],Y0_in_all[nice_mask]]))), axis=1)
            cam_cent = np.linalg.inv(A @ A.T) @ A @ b
            cam_cent = cam_cent.flatten()
            # print(f"{len(nice_mask)} frames are selected to estimate camera center, estimated result: {cam_cent[0,0],cam_cent[1,0]}")

            ellipticity = np.array(short_axis_all[nice_mask]) / np.array(long_axis_all[nice_mask])
            scale = np.sum(np.sqrt(1 - (ellipticity)**2) *                       \
            (np.linalg.norm(np.array([X0_in_all[nice_mask],Y0_in_all[nice_mask]]).T - cam_cent.T, axis=1)))       \
            / np.sum(1 - (ellipticity)**2)

        
            # Horizontal orientation (THETA)
            theta_cam = np.arcsin((X0_in_all - cam_cent[0]) / scale) # Here looking to left is negative; looking to right is positive ### shan: double check the positive negative direction
            # Vertical orientation (PHI)
            phi_cam = np.arcsin((Y0_in_all - cam_cent[1]) / np.cos(theta_cam) / scale) # no need negative because origin is at upper corner, increasing downwards

            ### eye_center will be the mean of all pupil center (tested and very close to cam_center)
            ### ideally should calib for camera optic axis and use cam_center to estimate theta and phi, and then further get 3D gaze
            eye_cent = np.column_stack((np.nanmean(X0_in_all), np.nanmean(Y0_in_all)))
            eye_cent = eye_cent.flatten()


            horizontal_offset = np.arcsin((eye_cent[0] - cam_cent[0]) / scale)
            vertical_offset = np.arcsin((eye_cent[1] - cam_cent[1]) / np.cos(horizontal_offset) / scale)

            # calculate theta and phi relative to eye center
            theta = theta_cam - horizontal_offset
            phi = phi_cam - vertical_offset

            # uncenter coordinates from IR centered into original coordinates
            cam_center = np.column_stack((cam_cent[0]+ir_xg, cam_cent[1]+ir_yg))
            pupil_center = np.column_stack((X0_in_all + ir_xg, Y0_in_all + ir_yg))
            eye_center = np.column_stack((eye_cent[0] + ir_xg, eye_cent[1] + ir_yg))

        # if modify possible, add pupil_center to the table
        self.insert1({**key, 'ellipse_dict': ellipse_dict_all, 'cam_center': cam_center, 'eye_center': eye_center, 'scale': scale, 'theta': theta, 'phi': phi, 'diameter': long_axis_all})
        if image_info is not None:
            image_info.setdefault("angle_estimator", "production")
            self.ImageFit.insert1({**key, **image_info, 'n_nice_frames': len(nice_mask)})

    def _eye_model_angles(self, key, cfg, results, x_all, y_all, llh_all):
        """theta / phi from the pooled 3D eye model (adamacs/helpers/pupil_image_fit.py, fit_eye_model) on the accepted
        image-fit ellipses, in the production output layout: theta / phi relative to the eye centre (= mean pupil
        position, as in production), scale = pupil rotation radius R (px), cam_center = the rotation-centre projection
        c and eye_center = the mean pupil position, both constant per recording (the eye model works in camera pixels,
        not IR-centred). The offsets returned in e3d_info give the angles relative to the camera axis:
        theta + theta_offset, phi + phi_offset (comparable across sessions of a head-fixed animal).
        Prior on R from the palpebral fissure: DLC nose/lateral corners (columns 0 and 1 of the marker arrays) of this
        recording ('session', v2) or, with fissure_prior='animal_median' (v3), the median over the animal's recordings
        already in this table for the parameter set plus this one (order-dependent for recordings populated later; the
        update pass recomputes all rows with the full set)."""
        from adamacs.helpers import pupil_image_fit as pif
        p = {k: cfg[k] for k in pif.EYE_MODEL_DEFAULTS}
        L_s, L_src = pif.fissure_length(x_all[:, 0], y_all[:, 0], llh_all[:, 0], x_all[:, 1], y_all[:, 1], llh_all[:, 1], **p)
        L, src, n_sess = L_s, L_src, None
        if cfg.get("fissure_prior") == "animal_median":
            L, src, n_sess = _animal_fissure_prior(key, cfg, L_s, L_src)
        try:
            return pif.eye_model_from_results(results, L, src, int(cfg["seed"]), cfg, L_session=L_s, L_session_source=L_src,
                                              n_sessions=n_sess)
        except ValueError as e:
            raise ValueError(f"{key}: {e}") from e

    def _image_rim_frames(self, key, cfg, x_thr, y_thr, ir_xg, ir_yg, thres_ellipticity):
        """Per-frame image rim fit (adamacs/helpers/pupil_image_fit.py) in the result layout of the marker loop.

        Guide: DISK head-fixed markers (observed or imputed) when guide_source allows it and an imputation exists,
        else the DLC pupil markers above likelihood_thres (x_thr, y_thr: already thresholded, NaN = invalid).
        Per-frame fits are read from dj.config['custom']['pupil_image_fit_cache']/<scan_id>.npz when that file was
        made by the same code with the same guide (the V1-drift audit's DISK-guided fits; checked via the default
        parameters, the guide radius and the frame count), else computed from the face video. Rejected frames (v2
        accept rule or temporal check) get NaN in every geometric field, like frames the marker fit skips;
        inlier_frac, rim_step and the accept flags are kept.
        Field conventions follow fit_ellipse: long_axis / short_axis are SEMI-axes, angle_from_x is the long-axis
        direction in radians (image coordinates). X0_in / Y0_in are IR-centred, X0+ir / Y0+ir are the fitted center
        in camera pixels, X0 / Y0 = X0_in / Y0_in.
        Camera-centre frames: accepted and inlier_frac >= nice_min_inlier_frac and ellipticity below threshold
        (replaces the marker path's ">= 7 DLC dots").
        """
        from pathlib import Path
        import cv2
        from adamacs.helpers import pupil_image_fit as pif
        T = x_thr.shape[0]
        X, Y = x_thr[:, 3:].T, y_thr[:, 3:].T
        OK = ~np.isnan(X) & ~np.isnan(Y)
        guide_used = "dlc"
        if cfg["guide_source"] in ("disk_headfixed", "disk_else_dlc"):
            D = _disk_guide_markers(key, cfg)
            if D is None and cfg["guide_source"] == "disk_headfixed":
                raise ValueError(f"{key}: no DISK head-fixed imputation for the guide (guide_source=disk_headfixed)")
            if D is not None:
                X, Y, OK = D
                guide_used = "disk_headfixed"
        if X.shape[1] != T:
            raise ValueError(f"{key}: guide markers have {X.shape[1]} frames, DLC markers {T}")
        gx, gy, gok, R0 = pif.guide_from_markers(X, Y, OK)
        params, v4, anchor = pif.fit_settings(cfg)    # per-frame parameters; v4 columns (empty for v3); v4.1 anchor or None
        grad = pif.guide_radius(X, Y, OK, gx, gy) if v4 else None
        vkey = {k: key[k] for k in ("session_id", "scan_id", "recording_id")}
        video = (model.VideoRecordingNew.File & vkey).fetch1("file_path")
        cap = cv2.VideoCapture(str(video)); fps = float(cap.get(cv2.CAP_PROP_FPS)); n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        if not fps > 0:
            raise ValueError(f"{key}: video reports no frame rate ({video})")
        cache_dir = dj.config.get("custom", {}).get("pupil_image_fit_cache", "")
        cache = Path(cache_dir) / f"{key['scan_id']}.npz" if cache_dir else None
        res, origin = None, "video"
        if (cache is not None and cache.exists() and guide_used == "disk_headfixed"
                and v4.get("r_max_guide") is None and v4.get("r_max_retry") is None and anchor is None
                and all(np.isclose(float(params[k]), float(pif.DEFAULTS[k])) for k in params)):
            z = np.load(cache, allow_pickle=True)
            if z["res"].shape == (T, 7) and np.isclose(float(z["R0"]), R0, rtol=0, atol=1e-6):
                res, origin = z["res"].astype(np.float32), "cache"
        n_long = n_retry = None
        if res is None:
            res, vinfo = pif.fit_video(video, gx, gy, gok, R0, progress_every=50000, guide_r=grad, anchor=anchor,
                                       **params, **v4)
            n_long, n_retry = vinfo["n_frames_long_rays"], vinfo["n_frames_retry"]
        gate_p = {k: cfg.get(k) for k in pif.GATE_DEFAULTS}
        if all(v is not None for v in gate_p.values()):
            keep, acc, brief, far, gate = pif.accepted_centers_gated(res, fps, R0, {**params, **gate_p, **v4})
        else:                                                   # v2 parameter rows: no spatial gate
            (keep, acc, brief), far, gate = pif.accepted_centers(res, fps, {**params, **v4}, R0=R0), None, None
        cx = np.where(keep, res[:, 0], np.nan).astype(float); cy = np.where(keep, res[:, 1], np.nan).astype(float)
        a = np.where(keep, res[:, 2], np.nan).astype(float); b = np.where(keep, res[:, 3], np.nan).astype(float)
        ang = np.where(keep, pif.long_axis_angle_rad(res[:, 2], res[:, 3], res[:, 4]), np.nan)
        long_ax, short_ax = np.fmax(a, b), np.fmin(a, b)
        x_in, y_in = cx - ir_xg, cy - ir_yg
        results = {"a": a, "b": b, "X0_in": x_in, "Y0_in": y_in, "long_axis": long_ax, "short_axis": short_ax,
                   "angle_from_x": ang, "X0+ir": cx, "Y0+ir": cy}
        inl, step = res[:, 5].astype(float), res[:, 6].astype(float)
        ellipse_dict_all = [{"X0": x_in[i], "Y0": y_in[i], "a": a[i], "b": b[i], "long_axis": long_ax[i],
                             "short_axis": short_ax[i], "angle_to_x": ang[i], "angle_from_x": ang[i], "X0_in": x_in[i],
                             "Y0_in": y_in[i], "X0+ir": cx[i], "Y0+ir": cy[i], "inlier_frac": inl[i], "rim_step": step[i],
                             "accepted": bool(keep[i]), "accepted_v2": bool(acc[i]),
                             **({} if far is None else {"rejected_spatial": bool(far[i])})} for i in range(T)]
        with np.errstate(invalid="ignore", divide="ignore"):
            nice = (keep & (inl >= cfg["nice_min_inlier_frac"]) & np.isfinite(x_in) & np.isfinite(y_in)
                    & (short_ax / long_ax < thres_ellipticity))
        con_med, ax_med = pif.session_medians(res)
        info = dict(guide_used=guide_used, fit_origin=origin, cache_file=str(cache) if origin == "cache" else "",
                    video_file=str(video), n_frames_markers=int(T), n_frames_video=int(n_vid),
                    n_frames_fitted=int(np.isfinite(res[:, 0]).sum()), fps=fps, r0=float(R0), rim_step_median=con_med,
                    axis_median=ax_med, frac_accepted_v2=float(acc.mean()), frac_accepted=float(keep.mean()),
                    n_brief_jumps=int(brief.sum()), code_version=cfg["code_version"])
        if far is not None:
            info.update(n_frames_rejected_spatial=int(far.sum()), gate_px=float(gate))
        if v4:
            acc2 = pif.accept_v2(res, **params)
            weak = pif.weak_uncorroborated(res, acc2, R0, fps, **{**pif.V4_DEFAULTS, **v4})
            info.update(n_frames_rejected_weak=int(weak.sum()),
                        n_frames_long_rays=n_long, n_frames_retry=n_retry)
        return results, ellipse_dict_all, np.where(nice)[0].tolist(), info


def _session_fissures(parameter_id, subject=None):
    """{(session_id, scan_id, recording_id): per-session fissure px} over the parameter set's ImageFit rows (optionally
    one subject): fissure_session_px (v3) or, for v2 rows, fissure_px when its source is per_frame/static."""
    from ..pipeline import session
    q = PupilEllipseFitting.ImageFit & {"parameter_id": parameter_id}
    if subject is not None:
        q = q & (session.Session & {"subject": subject}).proj()
    rows = q.fetch("session_id", "scan_id", "recording_id", "fissure_px", "fissure_source", "fissure_session_px",
                   "fissure_session_source", as_dict=True)
    out = {}
    for r in rows:
        if r["fissure_session_source"] is not None:
            v = r["fissure_session_px"]
        else:
            v = r["fissure_px"] if r["fissure_source"] in ("per_frame", "static") else None
        out[(r["session_id"], r["scan_id"], r["recording_id"])] = v
    return out


def _animal_fissure_prior(key, cfg, L_session, L_session_source):
    """Animal-median fissure length for the eye-model prior -> (L, source, n sessions). This recording's own value
    replaces any stored one; falls back to the median over all recordings of the parameter set ('rig_median')."""
    from adamacs.helpers import pupil_image_fit as pif
    lim = {k: cfg.get(k) if cfg.get(k) is not None else pif.EYE_MODEL_V3_DEFAULTS[k] for k in ("fissure_min_px", "fissure_max_px")}
    me = (key["session_id"], key["scan_id"], key["recording_id"])
    own = L_session if L_session_source in ("per_frame", "static") else None
    vals = _session_fissures(key["parameter_id"], subject_of(key)); vals[me] = own
    L, n = pif.animal_fissure(list(vals.values()), **lim)
    if np.isfinite(L):
        return L, "animal_median", n
    allv = _session_fissures(key["parameter_id"]); allv[me] = own
    L, n = pif.animal_fissure(list(allv.values()), **lim)
    return (L, "rig_median", n) if np.isfinite(L) else (float("nan"), "none", 0)


def _disk_guide_markers(key, cfg):
    """DISK head-fixed pupil markers (observed or imputed) for the image-fit guide, or None if there is no imputation.
    adamacs.pipeline binds adamacs.schemas.disk (disk.activate with the pipeline as linking module)."""
    from adamacs.schemas import disk
    if getattr(disk, "_linking_module", None) is None:
        import adamacs.pipeline  # noqa: F401  (adamacs.pipeline activates disk; see there)
    dk = {k: key[k] for k in ("session_id", "scan_id", "recording_id", "model_name")}
    dk.update(disk_model_name=cfg["disk_model_name"], parameter_id=cfg["disk_parameter_id"],
              imputation_paramset_id=cfg["disk_imputation_paramset_id"])
    rows = (disk.HeadfixedImputation.BodyPart & dk).fetch("body_part", "x_pos", "y_pos", "sample_source")
    if len(rows[0]) == 0:
        return None
    D = {b: (x, y, s) for b, x, y, s in zip(*rows)}
    pu = ['pupil_left', 'pupil_left_lower', 'pupil_left_up', 'pupil_lower', 'pupil_right', 'pupil_right_lower', 'pupil_right_up', 'pupil_upper']
    X = np.stack([D[b][0] for b in pu]).astype(float); Y = np.stack([D[b][1] for b in pu]).astype(float)
    OK = np.isin(np.stack([D[b][2] for b in pu]), [disk.SAMPLE_SOURCE["observed"], disk.SAMPLE_SOURCE["disk"]])
    return X, Y, OK


# ---------------------------------------- for free-moving eye camera setup -----------------
@schema
class PupilEllipseParameterFreeMoving(dj.Lookup):
  """Set hyperparameters for pupil ellipse fitting, mainly for preprocessing.

  The attribute names below are the ones the live table carries. A class spelled
  ``PupilEllipseParametersFreeMoving`` stood here from 2025-12-15; because DataJoint
  has no migrations, renaming it created a second, empty table rather than renaming
  this one, and ``PupilEllipseFittingFreeMoving`` -- declared 22 minutes earlier --
  kept its foreign key on the original. Every production fit, and everything
  downstream of it, hangs off this table.
  """
  definition = """
  parameter_id: int # unique id for parameters
  ---
  llh_thres_pupil: float # likelihood threshold for good tracking
  llh_thres_ir: float # likelihood threshold for good tracking
  llh_thres_corner: float # likelihood threshold for good tracking
  pupil_min_dots: int # minimum number of pupil dots required for fitting
  exclude_ir_std: float # define a boundary by mean and std to separate good tracking and bad tracking
  ellipticity_thres: float # for computing camera center by elliptic frames
  description: varchar(255) # short description of the parameters
  """

  contents = [(1, 0.9, .7, .9, 6, 5, 0.85, "default, free-moving eye cam")]


@schema
class EyeCamTimeSource(dj.Lookup):
    definition = """
    eyecam_time_source : varchar(32)
    ---
    event_suffix : varchar(64)   # part of event_type after eye_left / eye_right
    description  : varchar(256)
    """

    contents = [
        ('ocr',         'frames',             'OCR corrected eyecam frame timestamps'),
        ('lin',         'frames_lin',         'RANSAC robust linear fit eyecam timestamps'),
        ('bonsaicsv',   'frames_bonsaicsv',   'Eyecam timestamps imported from Bonsai CSV'),
    ]


@schema
class PupilEllipseFittingFreeMoving(dj.Computed):
    """fit pupil dots from DLC into ellipse"""
    definition = """
    -> PupilEllipseParameterFreeMoving
    -> PoseEstimationNew
    ---
    ellipse_dict: longblob # dictionary of ellipse fitting results
    cam_center: longblob # camera center in pixel coordinates
    eye_center: longblob # eye center in pixel coordinates
    pupil_center: longblob # pupil center in pixel coordinates for each frame
    scale: longblob # scaling factor from pixel to rotation angle
    theta: longblob # eye rotation in radians at azimuth direction (left/right = negative/positive), centered by eye center
    phi: longblob # eye rotation in elevation direction
    diameter: longblob # long axis of the fitted ellipse (equals pupil diameter)
    """

    @property
    def key_source(self):
        """Eye recordings only, crossed with the parameter sets.

        Without this the default key source is every PoseEstimationNew row crossed with
        every parameter set. Measured on 2026-09-15 that is 6 x 4,779 = 28,674 candidates,
        of which 28,278 are not yet fit. 2,797 of the 4,779 pose estimations (58.5%) come
        from topcam, side-cam and other-user models carrying none of the eleven markers
        make() indexes by fixed column position, so a bare populate() would raise on
        roughly 16,782 of them and flood ~jobs. Until now nothing stopped that structurally;
        safety rested on every caller remembering to scope its own restriction.

        The restriction is to models that actually carry the markers this table reads --
        the corneal reflection, both eye corners, and at least one pupil marker. Anything
        narrower than that, a particular cohort or parameter set, belongs in the populate()
        restriction rather than here.

        This mirrors the same fix applied to PupilFrameQuality in b84c3be. It is a
        Python-side property only: the declared table is unchanged, so no migration and no
        effect on the 396 rows already stored.
        """
        bp = model.Model.BodyPart
        eye_models = (dj.U("model_name") & (bp & 'body_part = "IR"')) \
            & (dj.U("model_name") & (bp & 'body_part = "nose_corner"')) \
            & (dj.U("model_name") & (bp & 'body_part = "lateral_corner"')) \
            & (dj.U("model_name") & (bp & 'body_part like "pupil%"'))
        return (PoseEstimationNew & eye_models) * PupilEllipseParameterFreeMoving

    def fit_ellipse(self, i, x, y):
        # NaN need to be removed for SVD convergence
        x0=x[~np.isnan(x)]
        y0=y[~np.isnan(y)]
        
        meanX = np.nanmean(x)
        meanY = np.nanmean(y)
        
        # remove bias
        x = x0 - meanX
        y = y0 - meanY
        
        # Estimation of the conic equation
        X = np.array([x**2, x*y, y**2, x, y])
        X = np.stack(X).T
        try:
            param = np.dot(np.sum(X, axis=0), np.linalg.pinv(np.matmul(X.T,X)))

        
            # #least square method to fit ellipse
            # b = np.ones(8)
            # param = np.linalg.lstsq(X, b, rcond=None)[0].squeeze() # very similar results
            
            # Extract parameters from the conic equation
            a, b, c, d, e = param[0], param[1], param[2], param[3], param[4]
            
            if b**2-4*a*c > 0: 
                print('given dots does not form a valid ellipse for frame',i)
            
            # Eigen decomp
            Q = np.array([[a, b/2],[b/2, c]])
            eig_val, eig_vec = np.linalg.eig(Q)
            
            # Get angle to long axis
            if eig_val[0] < eig_val[1]:
                angle_to_x = np.arctan2(eig_vec[1,0], eig_vec[0,0])
            else:
                angle_to_x = np.arctan2(eig_vec[1,1], eig_vec[0,1])
            
            angle_from_x = angle_to_x
            orientation_rad = 0.5 * np.arctan2(b, (c-a))
            cos_phi = np.cos(orientation_rad)
            sin_phi = np.sin(orientation_rad)
            
            a, b, c, d, e = [a*cos_phi**2 - b*cos_phi*sin_phi + c*sin_phi**2,
                            0,
                            a*sin_phi**2 + b*cos_phi*sin_phi + c*cos_phi**2,
                            d*cos_phi - e*sin_phi,
                            d*sin_phi + e*cos_phi]
            
            meanX, meanY = [cos_phi*meanX - sin_phi*meanY,
                            sin_phi*meanX + cos_phi*meanY]
            
            # Check if conc expression represents an ellipse
            test = a*c

            # select tracking frames only when fitted valid ellipse, and
            # IR is valid and at least 5 pupil dots are available
            if test > 0 and np.count_nonzero(~np.isnan(x))>=5:
            
                # Make sure coefficients are positive
                if a<0:
                    a, c, d, e = [-a, -c, -d, -e]
            
                # Final ellipse parameters
                X0 = meanX - d/2/a
                Y0 = meanY - e/2/c
                F = 1 + (d**2)/(4*a) + (e**2)/(4*c)
                a = np.sqrt(F/a)
                b = np.sqrt(F/c)
                long_axis = 2*np.maximum(a,b)
                short_axis = 2*np.minimum(a,b)
            
                # Rotate axes backwards to find center point of
                # original tilted ellipse
                R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
                P_in = R @ np.array([[X0],[Y0]])
                X0_in = P_in[0][0]
                Y0_in = P_in[1][0]

                # Organize parameters in dictionary to return
                ellipse_dict = {
                    'X0':X0,
                    'Y0':Y0,
                    'F':F,
                    'a':a,
                    'b':b,
                    'long_axis':long_axis/2,
                    'short_axis':short_axis/2,
                    'angle_to_x':angle_to_x,
                    'angle_from_x':angle_from_x,
                    'cos_phi':cos_phi,
                    'sin_phi':sin_phi,
                    'X0_in':X0_in,
                    'Y0_in':Y0_in,
                    'phi':orientation_rad
                }
            
            else:
                # If the conic equation didn't return an ellipse, do not
                # return any real values and fill the dictionary with NaNs.
                dict_keys = ['X0','Y0','F','a','b','long_axis',
                                'short_axis','angle_to_x','angle_from_x',
                                'cos_phi','sin_phi','X0_in','Y0_in','phi']
                dict_vals = list(np.ones([len(dict_keys)]) * np.nan)
                ellipse_dict = dict(zip(dict_keys, dict_vals))
            return ellipse_dict
        except np.linalg.LinAlgError as e:
            print("Error occurred during computation:", e, "for frame",i) 

    def make(self, key):
        # load all the tracking dots
        bodypart = ['nose_corner', 'lateral_corner', 'IR', 'pupil_left',
                    'pupil_left_lower','pupil_left_up','pupil_lower','pupil_right',
                    'pupil_right_lower','pupil_right_up','pupil_upper']
        x_all = []
        y_all = []
        llh_all = []

        for bp in bodypart:
            x = (PoseEstimationNew.BodyPartPosition & key & f"body_part='{bp}'").fetch('x_pos')
            y = (PoseEstimationNew.BodyPartPosition & key & f"body_part='{bp}'").fetch('y_pos')
            llh = (PoseEstimationNew.BodyPartPosition & key & f"body_part='{bp}'").fetch('likelihood')     

            x_all = np.concatenate((x_all,x))
            y_all = np.concatenate((y_all,y))
            llh_all = np.concatenate((llh_all,llh))

        x_all = np.vstack(x_all).T
        y_all = np.vstack(y_all).T
        llh_all = np.vstack(llh_all).T

        # get parameters from PupilEllipseParameterFreeMoving table
        # set a cutoff for poor fitting points in DLC model
        # One fetch rather than six. The live table spells its thresholds
        # llh_thres_* / ellipticity_thres, so they are unpacked onto the names
        # used throughout the rest of this method.
        params = (PupilEllipseParameterFreeMoving & key).fetch1()
        pupil_min_dots = params['pupil_min_dots']
        thres_llh_pupil = params['llh_thres_pupil']
        thres_llh_ir = params['llh_thres_ir']
        thres_llh_corner = params['llh_thres_corner']
        exclude_ir_std = params['exclude_ir_std']
        thres_ellipticity = params['ellipticity_thres']

        # set points lower than thres to NaN
        mask_corner = llh_all[:,0:2] < thres_llh_corner
        mask_ir = llh_all[:,2] < thres_llh_ir
        mask_pupil = llh_all[:,3:] < thres_llh_pupil


        # Set the elements in x_all and y_all to NaN where the mask is True
        cor_x, cor_y = np.copy(x_all[:,0:2]), np.copy(y_all[:,0:2]) # nose (col0) and lateral(col1) corners
        cor_x[mask_corner] = np.nan
        cor_y[mask_corner] = np.nan

        ir_x, ir_y = np.copy(x_all[:,2]), np.copy(y_all[:,2])
        ir_x[mask_ir] = np.nan
        ir_y[mask_ir] = np.nan

        p_x, p_y = np.copy(x_all[:,3:]), np.copy(y_all[:,3:])
        p_x[mask_pupil] = np.nan
        p_y[mask_pupil] = np.nan

        # define a boundary by mean and std to separate good tracking and bad tracking
        irx_mean = np.nanmean(ir_x)
        iry_mean = np.nanmean(ir_y)
        irx_std = np.nanstd(ir_x)
        iry_std = np.nanstd(ir_y)
        ir_std = np.linalg.norm([irx_std,iry_std])
        # set threshold as 5*std, still need to validata with other dataset
        ir_bad=(ir_x-irx_mean)**2+(ir_y-iry_mean)**2>(exclude_ir_std*ir_std)**2 
        # exclude off position IR dots
        ir_xg, ir_yg = np.copy(ir_x), np.copy(ir_y)
        ir_xg[ir_bad] = np.nan
        ir_yg[ir_bad] = np.nan

        # assume camera is following IR reflection
        # center pupil tracking by IR reflection
        pc_x = p_x-ir_xg[:,None]
        pc_y = p_y-ir_yg[:,None] # if IR is invalid, the pupil tracking is invalid

        # select tracking frames only when IR is valid and at least 6 pupil dots are available ### can be put in parameter table!
        track_fr=[]
        for i, (x, y) in enumerate(zip(pc_x, pc_y)):
            if np.count_nonzero(~np.isnan(x)) >= pupil_min_dots:
                track_fr.append(i)
        print(len(track_fr), ' frames out of ', len(pc_x), ' frames is selected for ', pupil_min_dots, ' dots threshold')

        fields = ['X0','Y0','F','a','b','long_axis', 'short_axis','angle_to_x','angle_from_x', 'cos_phi','sin_phi','X0_in','Y0_in','phi', "X0+ir", "Y0+ir"]
        results = {f: [] for f in fields}
        ellipse_dict_all = []
        for i, (x, y) in enumerate(tqdm(zip(pc_x, pc_y), total=len(pc_x), desc="fitting ellipses")):
            if i in track_fr: 
                ellipse_dic = self.fit_ellipse(i, x, y)
                ellipse_dic['X0+ir'] = ellipse_dic['X0_in'] + ir_xg[i]
                ellipse_dic['Y0+ir'] = ellipse_dic['Y0_in'] + ir_yg[i]
            else: ellipse_dic = {f: np.nan for f in fields}
            # collect values
            for f in fields:
                results[f].append(ellipse_dic[f])
            ellipse_dict_all.append(ellipse_dic)

        # convert lists to arrays for camera center estimation
        X0_in_all    = np.array(results["X0_in"])
        Y0_in_all    = np.array(results["Y0_in"])
        long_axis_all  = np.array(results["long_axis"])
        short_axis_all = np.array(results["short_axis"])
        angle_all    = np.array(results["angle_from_x"])


        nice_mask=[]
        for i, (x, y) in enumerate(zip(pc_x, pc_y)):
            if np.count_nonzero(~np.isnan(x))>=7:
                if short_axis_all[i]/long_axis_all[i] < thres_ellipticity: # 0.85 is the threshold from Neill's code
                    nice_mask.append(i)
                
        # calculate camera center with selected frames
        A = np.vstack([np.cos(angle_all[nice_mask]),np.sin(angle_all[nice_mask])])
        b = np.expand_dims(np.diag(A.T @ np.squeeze(np.array([X0_in_all[nice_mask],Y0_in_all[nice_mask]]))), axis=1)
        cam_cent = np.linalg.inv(A @ A.T) @ A @ b
        cam_cent = cam_cent.flatten()
        # print(f"{len(nice_mask)} frames are selected to estimate camera center, estimated result: {cam_cent[0,0],cam_cent[1,0]}")

        ellipticity = np.array(short_axis_all[nice_mask]) / np.array(long_axis_all[nice_mask])
        scale = np.sum(np.sqrt(1 - (ellipticity)**2) *                       \
        (np.linalg.norm(np.array([X0_in_all[nice_mask],Y0_in_all[nice_mask]]).T - cam_cent.T, axis=1)))       \
        / np.sum(1 - (ellipticity)**2)

        
        # Horizontal orientation (THETA)
        theta_cam = np.arcsin((X0_in_all - cam_cent[0]) / scale) # Here looking to left is negative; looking to right is positive ### shan: double check the positive negative direction
        # Vertical orientation (PHI)
        phi_cam = np.arcsin((Y0_in_all - cam_cent[1]) / np.cos(theta_cam) / scale) # no need negative because origin is at upper corner, increasing downwards

        ### eye_center will be the mean of all pupil center (tested and very close to cam_center)
        ### ideally should calib for camera optic axis and use cam_center to estimate theta and phi, and then further get 3D gaze
        eye_cent = np.column_stack((np.nanmean(X0_in_all), np.nanmean(Y0_in_all)))
        eye_cent = eye_cent.flatten()


        horizontal_offset = np.arcsin((eye_cent[0] - cam_cent[0]) / scale)
        vertical_offset = np.arcsin((eye_cent[1] - cam_cent[1]) / np.cos(horizontal_offset) / scale)

        # calculate theta and phi relative to eye center
        theta = theta_cam - horizontal_offset
        phi = phi_cam - vertical_offset

        # uncenter coordinates from IR centered into original coordinates
        cam_center = np.column_stack((cam_cent[0]+ir_xg, cam_cent[1]+ir_yg))
        pupil_center = np.column_stack((X0_in_all + ir_xg, Y0_in_all + ir_yg))
        eye_center = np.column_stack((eye_cent[0] + ir_xg, eye_cent[1] + ir_yg))

        # if modify possible, add pupil_center to the table
        self.insert1({**key, 'ellipse_dict': ellipse_dict_all, 'cam_center': cam_center, 'eye_center': eye_center, 'pupil_center': pupil_center, 'scale': scale, 'theta': theta, 'phi': phi, 'diameter': long_axis_all})


# ======================================================================================
# Eye-trace upsampling and gaze geometry, as module-level functions
# ======================================================================================
#
# These carry the numerical bodies of PupilRotationOptiTrack.make() and
# GazeReconstruction3D.make(). They sit at module level so the DISK-imputed sibling chain
# in adamacs/schemas/disk.py can call exactly this code rather than reimplementing it.
# The imputed tables differ from these two only in where their theta/phi come from --
# PupilEllipseFittingImputed instead of PupilEllipseFittingFreeMoving -- so a second copy
# of the interpolation and the rotation maths would be a second thing to keep correct, and
# the two chains would be free to drift apart in exactly the way that would invalidate
# every comparison between them.
#
# The methods below now delegate here and are kept as thin wrappers, because callers
# outside this module already reach them through the class. Nothing about production
# behaviour changes: the bodies were moved, not rewritten.


def resolve_eye_event_type(recording_id, eyecam_time_source):
    """The event_type carrying this recording's eye-camera frame timestamps."""
    suffix = (EyeCamTimeSource & {'eyecam_time_source': eyecam_time_source}).fetch1('event_suffix')
    return eye_camera_event_type(recording_id, suffix)


def fetch_eye_and_optitrack_clocks(base_key, eyecam_time_source):
    """The eye-camera and OptiTrack frame timestamps for a recording, in session time.

    ``base_key`` is the recording key without ``eyecam_time_source``. Both clocks come off
    the shared aux-DAQ acquisition, so they are directly comparable.
    """
    recording_id = base_key['recording_id']
    event_key = {k: base_key[k] for k in ('session_id', 'scan_id') if k in base_key}
    event_type = resolve_eye_event_type(recording_id, eyecam_time_source)

    # order_by is load-bearing, not tidiness. The caller pairs theta[i] with t[i] by
    # POSITION and then hands the result to interp1d(assume_sorted=True). Nothing in an
    # unordered fetch guarantees chronological rows -- measured on 30 recordings they do
    # come back sorted today, so this is a latent fault rather than a live one, and the
    # sibling fetch for optitrack_frames in multi_trial_lib has always passed order_by.
    t_eyecam = (event.Event & event_key & f'event_type="{event_type}"').fetch(
        'event_start_time', order_by='event_start_time').astype(float)
    if len(t_eyecam) == 0:
        raise ValueError(f"No eye camera events found for {event_type} ({recording_id})")
    t_optitrack = (event.Event & event_key & 'event_type="optitrack_frames"').fetch(
        'event_start_time', order_by='event_start_time').astype(float)
    if len(t_optitrack) == 0:
        raise ValueError("No optitrack_frames events found for this session/scan")
    return t_eyecam, t_optitrack


def eye_side_sign(recording_id, subject_name):
    """+1 for a left eye, -1 for a right one, after the known mislabelling correction.

    recording_id says which eye this is, but for some subjects that label is known to be
    wrong: the video files were named incorrectly at acquisition, so a recording labelled
    "eye_left" actually shows the right eye. EyeSideNote records those subjects; the most
    recent note for the subject wins, so clearing the flag (or adding a newer note with
    sides_swapped=0) undoes this correction. If the underlying files and recording_ids are
    ever renamed, the note MUST be updated in the same change - otherwise the side is
    corrected twice and the error comes back.

    This deliberately corrects ONLY the eye-in-head azimuth sign. It must NOT be applied to
    resolve_eye_event_type(), whose event_type is still correct despite the wrong label:
    the video, its DLC output and its frame events all come from the same physical camera,
    so they share one self-consistent time base.
    """
    s = side_sign_from_recording_id(recording_id)

    notes = (EyeSideNote & {'subject': subject_name}).fetch(
        as_dict=True, order_by='note_date DESC', limit=1)
    if notes and notes[0]['sides_swapped']:
        s = -s
    return s


def compute_torsion(key):
    """Torsion from head pitch, through the subject's manual linear calibration.

    Despite being stored per frame, this is not a per-frame measurement: it is a linear
    function of pitch with per-animal coefficients, so it carries no information the pitch
    does not already carry.
    """
    param_coeff = (TorsionCalibManual & key).fetch1('param_coeff')
    param_intercept = (TorsionCalibManual & key).fetch1('param_intercept')
    pitch = (RigidMouseTracking & key).fetch1('pitch')
    return param_coeff * pitch + param_intercept


@schema
class PupilRotationOptiTrack(dj.Computed):
    """Upsampling results from PupilEllipseFittingFreeMoving table so that synchronized with optitracking frames"""
    definition = """
    -> PupilEllipseFittingFreeMoving
    -> EyeCamTimeSource
    
    ---
    theta_opt: longblob # interpolated upscaled horizontal orientation in radians
    phi_opt: longblob # interpolated upscaled vertical orientation in radians
    diameter_opt: longblob # interpolated upscaled pupil diameter in pixels
    """

    @property
    def key_source(self):
        return PupilEllipseFittingFreeMoving * EyeCamTimeSource

    # The four methods below delegate to the module-level functions of the same name, which
    # the DISK-imputed sibling chain in adamacs/schemas/disk.py also calls. They are kept
    # because callers outside this module reach them through the class.

    def _resolve_eye_event_type(self, recording_id, eyecam_time_source):
        return resolve_eye_event_type(recording_id, eyecam_time_source)

    def interpolate_fill_gaps(self, data, t_orig):
        return interpolate_fill_gaps(data, t_orig)

    def resample_eyecam_to_optitrack(self, data, t_orig, t_resample):
        return resample_eyecam_to_optitrack(data, t_orig, t_resample)

    def make(self, key):
        # 1) Fill gaps on original timestamp with Akima (no edge extrapolation).
        # 2) Resample linearly to the provided optitrack timestamp, preserving NaN spans.
        # * theta and phi are not treated as circular data because of the small range. so they share the linear interpolation with upsampling diameter.
        base_key = {k: v for k, v in key.items() if k != 'eyecam_time_source'}
        theta = (PupilEllipseFittingFreeMoving & base_key).fetch1('theta')
        phi = (PupilEllipseFittingFreeMoving & base_key).fetch1('phi')
        diameter = (PupilEllipseFittingFreeMoving & base_key).fetch1('diameter')

        t_eyecam, t_optitrack = fetch_eye_and_optitrack_clocks(
            base_key, key['eyecam_time_source'])

        theta_opt, phi_opt, diameter_opt = upsample_eye_trace(
            theta, phi, diameter, t_eyecam, t_optitrack)

        self.insert1({**key, 'theta_opt': theta_opt, 'phi_opt': phi_opt, 'diameter_opt': diameter_opt})



@schema
class TorsionCalibManual(dj.Manual):
    """Manually insert torsion calibration parameters for each animal"""
    definition = """
    -> subject.Subject
    ---
    param_coeff: float # slope of the linear fit of torsion angle vs pitch angle
    param_intercept: float  # intercept of the linear fit of torsion angle vs pitch angle
    """
  # Example for insertion:
  # TorsionCalibManual.insert1({'subject': ROS-2112, 'param_coeff': [0.5], 'param_intercept': [0.1]})


# from sklearn.linear_model import LinearRegression
# @schema 
# class TorsionCalibLinear(dj.Computed):
#   """Compute torsion angle and linear fit with head pitch for each animal"""
#   definition = """
#   -> PupilEllipseFitOptiTrack # from torsion calibration session, unique for each animal
#   -> RigidMouseTracking
#   ---
#   param_coeff: longblob # slope of the linear fit of torsion angle vs pitch angle
#   param_intercept: longblob # intercept of the linear fit of torsion angle vs pitch angle
#   """
#   def torsion_estimation(self, video_frames):
#      ### estimate torsion angle based on pupil boundary
#      torsion_angle = []
#      #
#      #
#      #
#      return torsion_angle
  
#   ### torsion = param_coeff * pitch + param_intercept
#   def make(self, key):
#     # fetch pitch from Optitracking
#     pitch = (Optitracking & key).fetch1('pitch')
#     torsion = self.torsion_estimation()

#     model = LinearRegression()
#     model.fit(pitch, torsion)

#     param_coeff, param_intercept = model.coef_[0], model.intercept_

#     self.insert1({**key, 'param_coeff': param_coeff.tolist(), 'param_intercept': param_intercept.tolist()})


@schema
class EyeModel(dj.Manual):
  """Manually insert the relationship of animal and its eye position"""
  definition = """
  eye_model_id:  int # unique id for each eye model
  ---
  eye_model_azimuth: float # azimuth angle of the eye in head in degrees
  eye_model_elevation: float # elevation angle of the eye in head in degrees
  reference: varchar(255) # reference for the eye model parameters or from calibration session
  """
#   Example for insertion:
#   EyeModel.insert1({'eye_model_id': 1, 'eye_model_azimuth': 60, 'eye_model_elevation': 30, 'reference': 'Sakatani & Isa, 2007' })
#   EyeModel.insert1({'eye_model_id': 2, 'eye_model_azimuth': 64, 'eye_model_elevation': 22, 'reference': 'Oommen & Stahl, 2008' })

@schema
class GazeReconstruction3D(dj.Computed):
    definition = """
    -> PupilRotationOptiTrack # theta, phi that has been upsampled, interpolated and synchronized with optitracking data
    -> RigidMouseTracking # quaternions, roll, pitch, yaw, corresponding axis = x, y, z for rotation matrix; eye position relative to pivot point
    -> EyeModel # eye model parameters for azimuth and elevation angles of eye on head; estimate from calibration session or literature
    -> TorsionCalibManual # torsion angle of the eye based on pitch, computed by linear regression (ax+b=y)
    ---
    gaze_in_head: longblob
    gaze_in_space: longblob
    torsion: longblob # torsion angle of the eye
    """

    # Delegates to the module-level rotation helpers, which the DISK-imputed sibling
    # chain in adamacs/schemas/disk.py also uses.
    def Rx(self, x):
        return rot_x(x)

    def Ry(self, y):
        return rot_y(y)

    def Rz(self, z):
        return rot_z(z)

    def make(self, key):
        # Every numerical step below lives in a module-level function, so the DISK-imputed
        # sibling GazeReconstruction3DImputed computes gaze the same way by construction.
        eye_model_azimuth = (EyeModel & key).fetch1('eye_model_azimuth')
        eye_model_elevation = (EyeModel & key).fetch1('eye_model_elevation')

        rec_id = (PupilRotationOptiTrack & key).fetch1('recording_id')
        s = eye_side_sign(rec_id, key['subject'])

        theta = (PupilRotationOptiTrack & key).fetch1('theta_opt')
        phi = (PupilRotationOptiTrack & key).fetch1('phi_opt')

        # --- head->world rotation from quaternions ---
        qx, qy, qz, qw = (RigidMouseTracking() & key).fetch1("q_x", "q_y", "q_z", "q_w")
        q_head_world_xyzw = np.column_stack([qx, qy, qz, qw])  # (n,4)

        gaze_in_head, gaze_in_space = compute_gaze_vectors(
            theta, phi, q_head_world_xyzw, eye_model_azimuth, eye_model_elevation, s)

        # --- calculate torsion angle ---
        torsion = compute_torsion(key)

        self.insert1({**key, 'gaze_in_head': gaze_in_head, 'gaze_in_space': gaze_in_space, 'torsion': torsion})

@schema
class EyeSideNote(dj.Manual):
  """Record subjects whose left/right eye labelling is known to be wrong.

  Some recordings carry the wrong eye side because the video files were named
  incorrectly at acquisition. The ingest code is generic and correct, so the
  exception belongs in the data rather than as a per-animal special case in
  the pipeline. Downstream analysis should consult `sides_swapped` instead of
  trusting the eye side encoded in `recording_id`.

  `verified` distinguishes a swap that has been confirmed independently from
  one that has so far only been reported, so that a correction is never built
  on an unchecked assumption.
  """
  definition = """
  -> subject.Subject
  note_date: date # when the labelling problem was recorded
  ---
  sides_swapped: tinyint # 1 = left and right eye labels are swapped for this subject
  affected_scope: varchar(255) # which sessions are affected, e.g. "all sessions"
  reported_by: varchar(64) # who reported the problem
  verified: tinyint # 1 = independently verified, 0 = reported only
  note: varchar(4000) # full description of the problem and its consequences
  """


def eye_sides_swapped(subject):
  """True if this subject's eye videos are stored under the wrong side.

  Reads the most recent EyeSideNote for the subject, so clearing the flag or
  adding a newer note with sides_swapped=0 switches the correction off without
  a code change. Cached per subject for the life of the process: these loaders
  are called once per trial per eye, and a note does not change mid-analysis.
  """
  cached = _SWAP_CACHE.get(subject)
  if cached is None:
    notes = (EyeSideNote & {'subject': subject}).fetch(
      as_dict=True, order_by='note_date DESC', limit=1)
    cached = bool(notes and notes[0]['sides_swapped'])
    _SWAP_CACHE[subject] = cached
  return cached


_SWAP_CACHE = {}
_SUBJECT_CACHE = {}


def subject_of(scan_key):
  """The subject for a scan_key, which often carries only session_id/scan_id."""
  if scan_key.get('subject'):
    return scan_key['subject']
  sid = scan_key['session_id']
  if sid not in _SUBJECT_CACHE:
    from ..pipeline import session
    _SUBJECT_CACHE[sid] = (session.Session & {'session_id': sid}).fetch1('subject')
  return _SUBJECT_CACHE[sid]


def stored_eye_prefix(scan_key, eye_prefix):
  """Translate an ANATOMICAL side into the side it is STORED under.

  Analysis code asks for the left or the right eye meaning the animal's actual
  left or right. For subjects whose video files were misnamed at acquisition,
  that eye sits in the recording labelled with the OTHER side, so every lookup
  that builds a recording_id or an eye event_type out of a side name has to go
  through here. Pass the anatomical side, use the returned one in the query.

  Note that this applies ONLY to quantities keyed by recording_id, i.e. those
  that come off the eye cameras. RigidMouseTracking's eye_left_global and
  eye_right_global are OptiTrack measurements and are unaffected: swapping
  those too would undo the correction rather than complete it.
  """
  if eye_prefix not in ('left', 'right'):
    raise ValueError(f"eye_prefix must be 'left' or 'right', got {eye_prefix!r}")
  if not eye_sides_swapped(subject_of(scan_key)):
    return eye_prefix
  return 'right' if eye_prefix == 'left' else 'left'
