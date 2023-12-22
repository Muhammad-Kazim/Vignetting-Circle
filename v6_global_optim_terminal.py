import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import scipy.signal as sgn
from scipy.optimize import minimize
from matplotlib.patches import Rectangle
import torch
import argparse
import json


def contour_points_v3(g_img, tot_pts=500):

    kernel = np.ones((15, 15), np.uint8)
    gradient = cv2.morphologyEx(g_img, cv2.MORPH_GRADIENT, kernel)
    
    blur = cv2.GaussianBlur(gradient + 20, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    gradient_img = np.zeros_like(g_img, dtype=np.uint8)

    contours, _ = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        return None
    
    cv2.drawContours(gradient_img, contours, -1, 255, 2)

    kernel = np.ones((25, 25),np.uint8)
    gradient_img = cv2.dilate(gradient_img, kernel, iterations = 1)    


    hull_img = np.zeros_like(g_img, dtype=np.uint8)

    blur = cv2.GaussianBlur(g_img, (3, 3), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 1:
        return None
    
    area_cntrs = [cv2.contourArea(cnt) for cnt in contours]
    max_cntr_index = np.argmax(area_cntrs)
    cnt = contours[max_cntr_index]

    hull = cv2.convexHull(cnt)
    cv2.drawContours(hull_img, [hull], 0, 125, 2)

    op = np.logical_and(gradient_img, hull_img).astype(np.uint8)*255

    x, y = np.where(op > 1)
    points = np.vstack([y, x]).T
        
    if points.shape[0] > tot_pts:
        points = np.take(points, list(set(np.random.randint(0, points.shape[0], size=tot_pts))), axis=0)

    return points


def center_fn(led_pos: tuple, kcx: torch.tensor=torch.tensor([1.0859473, -0.000556592144, 5.48586221e-07]), kcy: torch.tensor=torch.tensor([1.09006337, -0.000365032841, 4.51764466e-07]), 
              origin: torch.tensor=torch.tensor((2758, 1219)), unit: torch.tensor=torch.tensor((202, 186)), center_led: torch.tensor=torch.tensor((15, 15))):

    rx, ry = (led_pos[0] - center_led[0])*unit[0], (30 - led_pos[1] - center_led[1])*unit[1]
    r = torch.sqrt(rx**2 + ry**2)
    r_arr = torch.pow(r, torch.arange(3))
    
    x_est = rx*sum(kcx*r_arr) + origin[0]
    y_est = ry*sum(kcy*r_arr) + origin[1]

    return (x_est, y_est)


def axis_min_fn(led_pos: tuple, kax0: torch.tensor=torch.tensor([2499.64063531, 0.18209488, 0.03107323, 2.88452691, -0.03847153, -0.02654681, -0.12702866, -0.02022648])):
    x1, y1 = led_pos
    return kax0[1]*x1 + kax0[2]*y1 + kax0[3]*(x1**2) + kax0[4]*(y1**2) + kax0[5]*x1*y1 + kax0[6]*(x1**3) + kax0[7]*(y1**3) + kax0[0]


def axis_max_fn(led_pos: tuple, kax1: torch.tensor=torch.tensor([2765.55044151, -6.83235681, -4.38549775, 0.02671118, 0.02110941, 0.28043923])):
    x1, y1 = led_pos
    return kax1[1]*x1 + kax1[2]*y1 + kax1[3]*x1**2 + kax1[4]*y1**2 + kax1[5]*x1*y1 + kax1[0]


def angle_fn(led_pos: tuple, kang: torch.tensor=torch.tensor([-93.08141194626909, 9.42944697, 1.31815639, 0.47404313, 0.33433636, -0.74189097])):
    x1, y1 = led_pos
    ang = kang[1]*x1 + kang[2]*y1 + kang[3]*x1**2 + kang[4]*y1**2 + kang[5]*x1*y1 + kang[0]
    return ang/180*3.14


def axis_min_fn_2(led_pos: tuple, kax0: torch.tensor=torch.tensor([2499.64063531, 0.18209488, 0.03107323, 2.88452691, -0.03847153, -0.02654681, -0.12702866, -0.02022648, 0., 0.])):
    x1, y1 = led_pos
    return kax0[1]*x1 + kax0[2]*y1 + kax0[3]*(x1**2) + kax0[4]*(y1**2) + kax0[5]*x1*y1 + kax0[6]*(x1**3) + kax0[7]*(y1**3) + kax0[8]*(y1*x1**2) + kax0[9]*(x1*y1**2) + kax0[0]


def axis_max_fn_2(led_pos: tuple, kax1: torch.tensor=torch.tensor([2765.55044151, -6.83235681, -4.38549775, 0.02671118, 0.02110941, 0.28043923, 0., 0., 0., 0.])):
    x1, y1 = led_pos
    return kax1[1]*x1 + kax1[2]*y1 + kax1[3]*x1**2 + kax1[4]*y1**2 + kax1[5]*x1*y1 + kax1[6]*(x1**3) + kax1[7]*(y1**3) + kax1[8]*(y1*x1**2) + kax1[9]*(x1*y1**2) + kax1[0]


def angle_fn_2(led_pos: tuple, kang: torch.tensor=torch.tensor([-93.08141194626909, 9.42944697, 1.31815639, 0.47404313, 0.33433636, -0.74189097, 0., 0., 0., 0.])):
    x1, y1 = led_pos
    ang = kang[1]*x1 + kang[2]*y1 + kang[3]*x1**2 + kang[4]*y1**2 + kang[5]*x1*y1 + kang[6]*(x1**3) + kang[7]*(y1**3) + kang[8]*(y1*x1**2) + kang[9]*(x1*y1**2) + kang[0]
    return (ang/180)*3.14


def find_x(point, EllipseParams):

    y1 = point[1]
    A, B, C, D, E, F = EllipseParams
    bb = (B*y1 + D)/A
    cc = (C*y1**2 + E*y1 + F)/A

    temp = bb**2 - 4*cc

    garbage = torch.tensor(-8000.1, dtype=torch.float32, requires_grad=True)
    if temp > 0:
        x1, x2 = (-1*bb + torch.sqrt(temp))/2, (-1*bb - torch.sqrt(temp))/2
        return x1, x2
    elif temp == 0:
        x1, x2 = (-1*bb + torch.sqrt(temp))/2, (-1*bb - torch.sqrt(temp))/2
        return x1, x1-8000
    else:
        return garbage, garbage-1000
    

def find_all_x(points, Ellipseparams):
    xs = torch.zeros([len(points), 2])
    for i, point in enumerate(points):
        x1, x2 = find_x(point, Ellipseparams)
        xs[i, 0] = x1
        xs[i, 1] = x2
    
    return xs


def find_ellipse_params_from_extrap_fns(led_pos, 
                                        kcx: torch.tensor=torch.tensor([1.0859473, -0.000556592144, 5.48586221e-07]), 
                                        kcy: torch.tensor=torch.tensor([1.09006337, -0.000365032841, 4.51764466e-07]), 
                                        origin: torch.tensor=torch.tensor((2758, 1219)), unit: torch.tensor=torch.tensor((202, 186)),
                                        kax0: torch.tensor=torch.tensor([2499.64063531, 0.18209488, 0.03107323, 2.88452691, -0.03847153, -0.02654681, -0.12702866, -0.02022648]), 
                                        kax1: torch.tensor=torch.tensor([2765.55044151, -6.83235681, -4.38549775, 0.02671118, 0.02110941, 0.28043923]), 
                                        kang: torch.tensor=torch.tensor([-93.08141194626909, 9.42944697, 1.31815639, 0.47404313, 0.33433636, -0.74189097]),
                                        more_vars=False):
    
    if more_vars:
        a = axis_max_fn_2(led_pos, kax1)
        b = axis_min_fn_2(led_pos, kax0)
        theta = angle_fn_2(led_pos, kang)
        cx, cy = center_fn(led_pos, kcx, kcy, origin, unit)
    else:
        a = axis_max_fn(led_pos, kax1)
        b = axis_min_fn(led_pos, kax0)
        theta = angle_fn(led_pos, kang)
        cx, cy = center_fn(led_pos, kcx, kcy, origin, unit)

    A = (b**2)*torch.cos(theta)**2 + (a**2)*torch.sin(theta)**2
    B = (b**2 - a**2)*torch.sin(2*theta)
    C = (b**2)*torch.sin(theta)**2 + (a**2)*torch.cos(theta)**2
    D = -(2*A*cx + B*cy)
    E = -(2*C*cy + B*cx)
    F = A*cx**2 + C*cy**2 + B*cx*cy - (a**2)*(b**2)

    return ((a, b), (cx, cy), theta), (A, B, C, D, E, F)


def loss_img(points, extrap_points, loss_type=2):
    """
    ellipse: img with ellipse drawn on zeros
    points: all points detected by boundary method

    returns avg l2 loss for one image
    """

    x_points = points[:, 0].repeat(2, 1).T
    diff = torch.abs(x_points - extrap_points)
    
    # cords = torch.argwhere(diff < max_dist)
    loss = ((diff[range(diff.shape[0]), torch.argmin(diff, axis=1)])**loss_type).sum()
    # loss = torch.sum(diff[cords[:, 0], cords[:, 1]])
    # loss = sum([diff[x[0], x[1]] for x in cords])

    if diff.shape[0] > 0:
        loss /= diff.shape[0]
        # print(diff.shape[0]) # different for different led positions

    return loss


def loss_img_distribution(points, extrap_points):
    """
    ellipse: img with ellipse drawn on zeros
    points: all points detected by boundary method

    returns avg l2 loss for one image
    """

    x_points = points[:, 0].repeat(2, 1).T
    diff = x_points - extrap_points
    # cords = torch.argwhere(diff < max_dist)
    loss = diff[range(diff.shape[0]), torch.argmin(torch.abs(diff), axis=1)]
   
    return loss


def target_points(points, extrap_points, max_dist):
    # extrap points are only 2 x cords
    x_points = torch.tensor(points[:, 0]).repeat(2, 1).T
    diff = torch.abs(x_points - extrap_points)
    cords = torch.argwhere(diff < max_dist)
    retain_points = points[cords[:, 0].unique(), :]

    return retain_points


if __name__ == '__main__':

    ##############################################
    # READ ARGS FROM TERMINAL
    ##############################################
    parser = argparse.ArgumentParser(description='Optimize Ellipse Using Gradient Descent')
    parser.add_argument('-t', '--tot_num_pts', default=2000, type=int, help='Max number of points to sample from boundary contour')
    parser.add_argument('-m', '--max_dist_all', default=100, type=int, help='Max distance of function x to image boundary x')
    parser.add_argument('-l', '--loss_type', default=2, type=int, help='L1, L2, ... etc.')
    parser.add_argument('-n', '--n_iters', default=25, type=int, help='Total number of epochs')
    parser.add_argument('--lr1', default=1e-8, type=float, help='Learning rate of center coeffs')
    parser.add_argument('--lr1_c', default=1e-2, type=float, help='Learning rate of center bias')
    parser.add_argument('--lr2', default=1e-2, type=float, help='Learning rate of axis lengths and angles coeffs')
    parser.add_argument('--lr_scale', default=1, type=int, help='Divides all learning rate by the number')
    parser.add_argument('--scheduler', default=0, type=int, choices=[0, 1], help='0 for no and 1 for yes')
    parser.add_argument('--name_append', help='String appended to the end of the saved json file')
    
    args = parser.parse_args()


    TOT_NUM_PTS = args.tot_num_pts # sample max these number from boundary/contour of image
    MAX_DIST_ALLOWED = args.max_dist_all # |boundary of image - point on ellipse from functions| < 150, only these contour points considered
    LOSS_TYPE = args.loss_type
    N_ITERS = args.n_iters
    LR1, LR1_C, LR2 = args.lr1/args.lr_scale, args.lr1_c/args.lr_scale, args.lr2/args.lr_scale
    SCH = True if args.scheduler == 1 else False

    print(
        f'Total number of points: {TOT_NUM_PTS} \n'
        f'Max Distance Allowed: {MAX_DIST_ALLOWED} \n'
        f'Loss Type: {"L2" if LOSS_TYPE == 2 else "L1"} \n'
        f'Number of Epochs: {N_ITERS} \n'
        f'Learning rates: {LR1}, {LR1_C}, {LR2} \n'
        f'Scheduler: {SCH} \n',
        f'Append Name: {"True" if args.name_append else "False"} \n'
    )



    kcx = torch.tensor([1.0859473, -0.000556592144, 5.48586221e-07], dtype=torch.float64) # initialize coeffs tensor with polynomial fitting curves
    kcy = torch.tensor([1.09006337, -0.000365032841, 4.51764466e-07], dtype=torch.float64)
    origin = torch.tensor((2758, 1219))
    unit = torch.tensor((202, 186))
    kax0 = torch.tensor([2499.64063531, 0.18209488, 0.03107323, 2.88452691, -0.03847153, -0.02654681, -0.12702866, -0.02022648, 0., 0.], dtype=torch.float64)
    kax1 = torch.tensor([2765.55044151, -6.83235681, -4.38549775, 0.02671118, 0.02110941, 0.28043923, 0., 0., 0., 0.], dtype=torch.float64)
    kang = torch.tensor([-93.08141194626909, 9.42944697, 1.31815639, 0.47404313, 0.33433636, -0.74189097, 0., 0., 0., 0.], dtype=torch.float64)


    ################################################
    # TARGET POINTS
    ################################################

    target_points_array = torch.zeros([11, 11, TOT_NUM_PTS, 2])
    targets_len = np.zeros([11, 11]) # store total number of samples for each led location

    for x in range(10, 21):
        for y in range(10, 21):

            dir_path = "Z:/CSE\CSE-Research/Microscopy3D/CV_CSE_Collaboration/Results/CV_CSE/fpm_capture/output/2023_03_15/2023_03_15_16_02_07/"
            file_path = f"2023_03_15_16_02_07_img_shutter_05_x_{x}_y_{y}_r_0_g_1_b_0.tiff"
            
            img = cv2.imread(dir_path + file_path, cv2.IMREAD_UNCHANGED)
            _, g_img, _ = cv2.split(img)

            img_8 = cv2.convertScaleAbs(img, alpha=(255/65535))
            g_img_8 = cv2.convertScaleAbs(g_img, alpha=(255/65535))
            points = contour_points_v3(g_img_8, tot_pts=TOT_NUM_PTS) # extract points from given full-sized tiff image

            _, EllipseParams = find_ellipse_params_from_extrap_fns((x, y), kcx, kcy, origin, unit, kax0, kax1, kang, more_vars=False) # find A, B, C, D, E and F -> Ellipse Parameters -> Implicit Equation
            extrap_points = find_all_x(points, EllipseParams) # for all y cords in points, find possible corresponding 2 x cords given by the ellipse estimated through functions
            targets = target_points(points, extrap_points, MAX_DIST_ALLOWED) # keep only points that are < max_dist_allowed away from function estimated ellipse on the x-axis
            target_points_array[x-10, y-10, :targets.shape[0], :] = torch.tensor(targets)
            targets_len[x-10, y-10] = targets.shape[0]

    targets_len[17-10, 10-10] = 0
    targets_len[10-10, 17-10:] = 0
    targets_len[11-10, -1] = 0


    ##############################################
    # check l1 loss before optimization
    ##############################################

    losses = np.zeros([11, 11])
    for x in range(10, 21):
        for y in range(10, 21):

            dir_path = "Z:/CSE\CSE-Research/Microscopy3D/CV_CSE_Collaboration/Results/CV_CSE/fpm_capture/output/2023_03_15/2023_03_15_16_02_07/"
            file_path = f"2023_03_15_16_02_07_img_shutter_05_x_{x}_y_{y}_r_0_g_1_b_0.tiff"

            points = target_points_array[x-10, y-10, :int(targets_len[x-10, y-10]), :]

            _, EllipseParams = find_ellipse_params_from_extrap_fns((x, y), kcx, kcy, origin, unit, kax0, kax1, kang)
            losses[x-10, y-10] = loss_img(points, find_all_x(points, EllipseParams), loss_type=1).numpy()
            if losses[x-10, y-10] == 0.:
                losses[x-10, y-10] = np.nan
            
            # print('x:', x, 'y:', y, file_path, 'Loss:', losses[x-10, y-10])

    SUM_ABS_DIFF_BEFORE_OPT = sum(losses[targets_len.nonzero()])

    #############################################
    # GRADIENT-BASED OPTIMIZATION
    #############################################

    LOSSES_PER_ITER = []
    
    OPT_PARAMS1 = 0.0
    OPT_PARAMS1_CENTER = 0.0
    OPT_PARAMS2 = 0.0

    MIN_LOSS = 1e10
    

    params_v0 = torch.tensor([[2499.64063531, 0.18209488, 0.03107323, 2.88452691, -0.03847153, -0.02654681, -0.12702866, -0.02022648, 0., 0.],
                          [2765.55044151, -6.83235681, -4.38549775, 0.02671118, 0.02110941, 0.28043923, 0., 0., 0., 0.],
                          [-93.08141194626909, 9.42944697, 1.31815639, 0.47404313, 0.33433636, -0.74189097, 0., 0., 0., 0.]], 
                          dtype=torch.float64, requires_grad=True)
    
    # usually used: centers coeffs kx and ky are diff and soved for using linear inversion. see v7_revisit.
    params_0 = torch.tensor([[1.0859473, -0.000556592144, 5.48586221e-07],
                         [1.09006337, -0.000365032841, 4.51764466e-07]],
                         dtype=torch.float64, requires_grad=True)

    # see v7_revisit...: solved using scipy minimize and kx=ky. the curvature is more gentle.
    # params_0 = torch.tensor([[1.05975978, -3.15730950e-4, 3.30675404e-7],
    #                          [1.05975978, -3.15730950e-4, 3.30675404e-7]],
    #                          dtype=torch.float64, requires_grad=True)

                          
    
    params_0_center = torch.tensor([2758, 1219, 0], dtype=torch.float64, requires_grad=True)
    
    if SCH:
        optim = torch.optim.Adagrad([{'params': params_0, 'lr':args.lr_scale*LR1},
                                 {'params': params_0_center, 'lr':args.lr_scale*LR1_C}, 
                                 {'params': params_v0, 'lr':args.lr_scale*LR2}], lr=1e-2, lr_decay=0.9, eps=1e-12)
        
        scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=[LR1/args.lr_scale, LR1_C/args.lr_scale, LR2/args.lr_scale], 
                                                      max_lr=[args.lr_scale*LR1, args.lr_scale*LR1_C, args.lr_scale*LR2],
                                                      step_size_up=5, step_size_down=5, mode='triangular', cycle_momentum=False)
    
    else:
        optim = torch.optim.Adagrad([{'params': params_0, 'lr':LR1},
                                 {'params': params_0_center, 'lr':LR1_C}, 
                                 {'params': params_v0, 'lr':LR2}], lr=1e-2, lr_decay=0.9, eps=1e-12)
    
    for epoch in range(N_ITERS):
        optim.zero_grad()
        # optim2.zero_grad()

        loss = torch.zeros(1, dtype=torch.float64)
        for x in range(10, 21):
            for y in range(10, 21):

                if np.isnan(losses[x - 10, y - 10]) == False:
                    
                    points = target_points_array[x-10, y-10, :int(targets_len[x-10, y-10]), :]
                    _, EllipseParams = find_ellipse_params_from_extrap_fns((x, y), params_0[0], params_0[1], params_0_center, [202, 186], 
                                                                        params_v0[0], params_v0[1], params_v0[2], more_vars=False)
                    loss += loss_img(points, find_all_x(points, EllipseParams), loss_type=LOSS_TYPE)

        # calculate gradients = backward pass
        loss.backward()

        # if epoch % 1 == 0:
        # print(f'epoch {epoch}: loss = {loss.item():.2f}, {params_0}, {params_0_center}, {params_v0}')

        LOSSES_PER_ITER.append(loss.item())
        
        if loss.item() < MIN_LOSS:
            MIN_LOSS = loss.item()
            OPT_PARAMS1 = params_0.detach().clone()
            OPT_PARAMS1_CENTER = params_0_center.detach().clone()
            OPT_PARAMS2 = params_v0.detach().clone()

        optim.step()
        if SCH:
            scheduler.step()

    # print(f'Minimum Loss: {min_loss}')
    # print(f'Optimal Params: {opt_params1}, {opt_params1_center}, {opt_params2}')


    #############################################
    # L1 LOSS AFTER OPTIMIZATION
    #############################################

    losses_2 = np.zeros([11, 11])

    for x in range(10, 21):
        for y in range(10, 21):

            dir_path = "Z:/CSE\CSE-Research/Microscopy3D/CV_CSE_Collaboration/Results/CV_CSE/fpm_capture/output/2023_03_15/2023_03_15_16_02_07/"
            file_path = f"2023_03_15_16_02_07_img_shutter_05_x_{x}_y_{y}_r_0_g_1_b_0.tiff"

            points = target_points_array[x-10, y-10, :int(targets_len[x-10, y-10]), :]

            # _, EllipseParams = find_ellipse_params_from_extrap_fns((x, y))
            # _, EllipseParams = find_ellipse_params_from_extrap_fns((x, y), kcx, kcy, kax0, kax1, kang)
            _, EllipseParams = find_ellipse_params_from_extrap_fns((x, y), OPT_PARAMS1[0], OPT_PARAMS1[1], OPT_PARAMS1_CENTER, [202, 186],
                                                                    OPT_PARAMS2[0], OPT_PARAMS2[1], OPT_PARAMS2[2], more_vars=False)
            losses_2[x-10, y-10] = loss_img(points, find_all_x(points, EllipseParams), loss_type=1).detach().numpy()

            if losses_2[x-10, y-10] == 0.:
                losses_2[x-10, y-10] = np.nan
            
            # print('x:', x, 'y:', y, file_path, 'Loss:', losses_2[x-10, y-10])

    SUM_ABS_DIFF_AFTER_OPT = sum(losses_2[targets_len.nonzero()])


    #########################################
    # PLACE ALL RESULTS IN A FOLDER
    #########################################

    output_data = {
        'SUM ABS DIFF BEFORE OPTIM': SUM_ABS_DIFF_BEFORE_OPT,
        'SUM ABS DIFF AFTER OPTIM': SUM_ABS_DIFF_AFTER_OPT,
        'TRAINING LOSS': LOSSES_PER_ITER,
        'MIN LOSS': MIN_LOSS,
        'OPT PARAMS CXY COEFFS': OPT_PARAMS1.tolist(),
        'OPT PARAMS CXY BIAS': OPT_PARAMS1_CENTER.tolist(),
        'OPT PARAMS AXIS LENGTHS AND ANGLES': OPT_PARAMS2.tolist()
    }

    if args.name_append:
        with open(f'v6_based_results/{LOSS_TYPE}_{MAX_DIST_ALLOWED}_{TOT_NUM_PTS}_{N_ITERS}_{args.scheduler}_{args.lr_scale}_{args.name_append}.json', 'w') as fp:
            json.dump(output_data, fp)
    else:
        with open(f'v6_based_results/{LOSS_TYPE}_{MAX_DIST_ALLOWED}_{TOT_NUM_PTS}_{N_ITERS}_{args.scheduler}_{args.lr_scale}.json', 'w') as fp:
            json.dump(output_data, fp)