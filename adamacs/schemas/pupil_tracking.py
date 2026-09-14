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
        self.insert1({**key, 'ellipse_dict': ellipse_dict_all, 'cam_center': cam_center, 'eye_center': eye_center, 'scale': scale, 'theta': theta, 'phi': phi, 'diameter': long_axis_all})


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

    def _resolve_eye_event_type(self, recording_id, eyecam_time_source):
        if 'eye_left' in recording_id:
            camera_type = 'mini2p1_eye_left'
        elif 'eye_right' in recording_id:
            camera_type = 'mini2p1_eye_right'
        else:
            raise ValueError(f"Cannot resolve eye side from recording_id: {recording_id}")
        suffix = (EyeCamTimeSource & {'eyecam_time_source': eyecam_time_source}).fetch1('event_suffix')
        return f"{camera_type}_{suffix}"

    def interpolate_fill_gaps(self,data, t_orig):
        t = np.asarray(t_orig, float)
        y = np.asarray(data, float)

        good = np.isfinite(y)
        if good.sum() < 2:
            raise ValueError(
                f"interpolate_fill_gaps: need ≥2 finite samples to interpolate; got {good.sum()}."
            )

        # f = Akima1DInterpolator(t[good], y[good])
        # linear interpolant on the good samples
        f = interp1d(t[good], y[good], kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=True)

        y_filled = y.copy()
        bad_idx = np.flatnonzero(~good)
        if bad_idx.size == 0:
            print("No gaps to fill.")
            return y_filled

        # fill only internal NaN runs (no edge extrapolation)
        cuts = np.where(np.diff(bad_idx) > 1)[0] + 1
        for run in np.split(bad_idx, cuts):
            i0, i1 = run[0], run[-1]
            left_ok  = (i0 - 1) >= 0 and np.isfinite(y[i0 - 1])
            right_ok = (i1 + 1) < len(y) and np.isfinite(y[i1 + 1])
            if left_ok and right_ok:
                y_filled[i0:i1+1] = f(t[i0:i1+1])
        return y_filled

    def resample_eyecam_to_optitrack(self, data, t_orig, t_resample):
        t = np.asarray(t_orig, float)
        y = np.asarray(data, float)
        t_new = np.asarray(t_resample, float)

        # linear onto t_new, no extrapolation
        f = interp1d(t, y, kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=True)
        y_new = f(t_new)

        return y_new

    def make(self, key):
        # 1) Fill gaps on original timestamp with Akima (no edge extrapolation).
        # 2) Resample linearly to the provided optitrack timestamp, preserving NaN spans.
        # * theta and phi are not treated as circular data because of the small range. so they share the linear interpolation with upsampling diameter.
        base_key = {k: v for k, v in key.items() if k != 'eyecam_time_source'}
        theta = (PupilEllipseFittingFreeMoving & base_key).fetch1('theta')
        phi = (PupilEllipseFittingFreeMoving & base_key).fetch1('phi')
        diameter = (PupilEllipseFittingFreeMoving & base_key).fetch1('diameter')
        
        recording_id = base_key['recording_id']
        event_key = {k: base_key[k] for k in ('session_id', 'scan_id') if k in base_key}
        event_type = self._resolve_eye_event_type(recording_id, key['eyecam_time_source'])

        t_eyecam = (event.Event & event_key & f'event_type="{event_type}"').fetch('event_start_time').astype(float)
        if len(t_eyecam) == 0:
            raise ValueError(f"No eye camera events found for {event_type} ({recording_id})")
        t_optitrack = (event.Event & event_key & 'event_type="optitrack_frames"').fetch('event_start_time').astype(float)
        if len(t_optitrack) == 0:
            raise ValueError("No optitrack_frames events found for this session/scan")

        theta_filled = self.interpolate_fill_gaps(theta, t_eyecam)
        phi_filled = self.interpolate_fill_gaps(phi, t_eyecam)
        diameter_filled = self.interpolate_fill_gaps(diameter, t_eyecam)

        theta_opt = self.resample_eyecam_to_optitrack(theta_filled, t_eyecam, t_optitrack)
        phi_opt = self.resample_eyecam_to_optitrack(phi_filled, t_eyecam, t_optitrack)
        diameter_opt = self.resample_eyecam_to_optitrack(diameter_filled, t_eyecam, t_optitrack)

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

    def Rx(self, x):
        return np.array([[1, 0, 0],
                        [0, np.cos(x), -np.sin(x)],
                        [0, np.sin(x), np.cos(x)]])    
    def Ry(self, y):
        return np.array([[np.cos(y), 0, np.sin(y)],
                        [0, 1, 0],
                        [-np.sin(y), 0, np.cos(y)]])
    def Rz(self, z):
        return np.array([[np.cos(z), -np.sin(z), 0],
                        [np.sin(z), np.cos(z), 0],
                        [0, 0, 1]])
    
    def make(self, key):
        # --- eye in head position ---  
        # +/- 1 for left/right eye axes
        eye_in_head_azimuth = np.radians((EyeModel & key).fetch1('eye_model_azimuth')) # pi/3 in literature
        eye_in_head_elevation = np.radians((EyeModel & key).fetch1('eye_model_elevation')) # pi/6 in literature

        rec_id = (PupilRotationOptiTrack & key).fetch1('recording_id')
        if "eye_right" in rec_id:
            s = -1
        elif "eye_left" in rec_id:
            s = 1

        R_eye_in_head = self.Rz(s * eye_in_head_azimuth) @ self.Rx(eye_in_head_elevation)  # (3,3)

        # turn static eye-in-head rotation into a Rotation object once
        R_eye_in_head_static = R.from_matrix(R_eye_in_head)

        # --- dynamic eye rotation (per frame): clockwise rightward -> -theta around z, then +phi around x ---
        # Build as a vectorized stack (n,3,3)
        theta = (PupilRotationOptiTrack & key).fetch1('theta_opt')
        phi = (PupilRotationOptiTrack & key).fetch1('phi_opt')

        # --- head->world rotation from quaternions ---
        qx, qy, qz, qw = (RigidMouseTracking() & key).fetch1("q_x","q_y","q_z","q_w")
        q_head_world_xyzw = np.column_stack([qx, qy, qz, qw]) # (n,4)
        n = min(q_head_world_xyzw.shape[0], theta.shape[0])
        q_head_world_xyzw = q_head_world_xyzw[:n]
        theta = theta[:n]
        phi = phi[:n]

        # find invalid quaternions
        invalid_mask = np.any(~np.isfinite(q_head_world_xyzw), axis=1) | np.isclose(
            np.linalg.norm(q_head_world_xyzw, axis=1), 0
        )
        invalid_idx = np.where(invalid_mask)[0]


        # also require finite theta/phi
        valid_eye = np.isfinite(theta) & np.isfinite(phi)
        valid_mask = (~invalid_mask) & valid_eye

        print("Invalid quaternion indices:", invalid_idx)

        # --- eye->world rotation per frame ---
        # allocate outputs instead of (n,3,3) rotation matrices
        eye_axis = np.array([0, 1, 0])
        gaze_in_head  = np.full((n, 3), np.nan, float)
        gaze_in_space = np.full((n, 3), np.nan, float)

        chunk_size = 100000
        total_chunks = (n + chunk_size - 1) // chunk_size
        for start in tqdm(range(0, n, chunk_size), total=total_chunks, desc="computing gaze by chunks"):
            stop = min(start + chunk_size, n)
            sl = slice(start, stop)

            mask_chunk = valid_mask[sl]

            # indices in full array for valid frames in this chunk
            idx_rel = np.nonzero(mask_chunk)[0]
            idx = idx_rel + start

            # dynamic eye rotation for this chunk
            theta_chunk = theta[idx]
            phi_chunk   = phi[idx]
            angles_chunk = np.column_stack((-theta_chunk, phi_chunk))  # (k,2) # originally theta: left = negative, but in head system left is positive

            R_eye_in_orbit = R.from_euler('zx', angles_chunk)  # dynamic eye rotation (k,)

            # Eye->head rotation per frame: R_eye_in_orbit @ R_eye_in_head
            # Rotation composition: apply rightmost first → R_eye_head = R_eye_in_orbit * R_eye_in_head_static
            R_eye_in_head_dyn = R_eye_in_orbit * R_eye_in_head_static

            # --- Gaze directions in head coords ---
            gaze_head_chunk = R_eye_in_head_dyn.apply(eye_axis)   # (k,3)
            gaze_in_head[idx] = gaze_head_chunk

            # head->world rotation from quaternions for this chunk
            q_chunk = q_head_world_xyzw[idx]  # (k,4)
            R_head_in_space_chunk = R.from_quat(q_chunk)

            # eye->world rotation: R_head_in_space @ R_eye_in_head_dyn
            R_eye_in_space_chunk = R_head_in_space_chunk * R_eye_in_head_dyn

            # --- Gaze in world coords ---
            gaze_space_chunk = R_eye_in_space_chunk.apply(eye_axis)  # (k,3)
            gaze_in_space[idx] = gaze_space_chunk

        # Normalize row-wise (Rotation.apply should already give unit vectors, but this is cheap and safe)
        norms = np.linalg.norm(gaze_in_head, axis=1, keepdims=True)
        gaze_in_head = np.divide(gaze_in_head, norms, out=gaze_in_head, where=norms != 0)
        print("gaze_in_head:", gaze_in_head.shape)

        norms_w = np.linalg.norm(gaze_in_space, axis=1, keepdims=True)
        gaze_in_space = np.divide(gaze_in_space, norms_w, out=gaze_in_space, where=norms_w != 0)
        print("gaze_in_space:", gaze_in_space.shape)

        # --- calculate torsion angle ---
        param_coeff = (TorsionCalibManual & key).fetch1('param_coeff') # (TorsionCalibLinear & key).fetch1('param_coeff') 
        param_intercept = (TorsionCalibManual & key).fetch1('param_intercept') # (TorsionCalibLinear & key).fetch1('param_intercept')
        pitch = (RigidMouseTracking & key).fetch1('pitch')
        torsion = param_coeff * pitch + param_intercept

        self.insert1({**key, 'gaze_in_head': gaze_in_head, 'gaze_in_space': gaze_in_space, 'torsion': torsion})
