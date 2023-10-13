import numpy as np
import numexpr as ne
from scipy.signal import fftconvolve
from scipy import signal
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

def find_degree(x_heel, y_heel,x_center, y_center):
    """
    finds radians of the position of the heel in comparison to the center
    """
    return (np.arctan2(y_heel - y_center, x_heel - x_center) + np.pi)

def find_point_2degree(x_center, y_center, r, angle_rad):
    """
    find the point with a specifish angle and distance to the center point
    """
    x = x_center + r * np.cos(angle_rad)
    y = y_center + r * np.sin(angle_rad)
    return x, y

def creat_mask(angle_rad, x_center, y_center, r, degree, mask):
    """
    creates flashlight-like mask for searching
    """
    angle_lower = np.mod(angle_rad-(degree/2),2*np.pi)
    angle_upper = np.mod(angle_rad+(degree/2),2*np.pi)
    
    y_size, x_size = mask.shape   

    mesh = np.meshgrid(np.arange(x_size),np.arange(y_size))

    # Calculate the distance of the element from the center
    distance_calc ="((y - y_center)**2 + (x - x_center)**2)<=(r**2)"
    mask_distance = ne.evaluate(distance_calc, local_dict = {"x_center":x_center,"x": mesh[0],"y_center":y_center,"y":mesh[1],"r":r})
    #modulus fehlt, selbst bei nur einem Winkel funktioniert es noch nicht
    temp_angle = "(arctan2(y-y_center,x-x_center))%(2*pi)"
    mask_temp_angle = ne.evaluate(temp_angle, local_dict = {"x_center":x_center,"x": mesh[0],"y_center":y_center,"y":mesh[1], "pi":np.pi})
    angle_calc = "(angle_lower <= mask_temp_angle) & (mask_temp_angle <= angle_upper)"
    if  angle_lower> angle_upper:
        angle_temp = angle_upper
        angle_upper = angle_lower
        angle_lower = angle_temp
        angle_calc = "~ ((angle_lower <= mask_temp_angle) & (mask_temp_angle <= angle_upper))"
    mask_angle = ne.evaluate(angle_calc, local_dict = {"mask_temp_angle": mask_temp_angle, "angle_lower":angle_lower,"angle_upper":angle_upper})
    mask = ne.evaluate("a & b", local_dict = {"a":mask_distance,"b":mask_angle})
    return mask

def rotate_coord(x, y, angle):
    """
    rotate coordinates by angle, angle in rad
    """
    return x*np.cos(angle) - y*np.sin(angle), x*np.sin(angle) + y*np.cos(angle)

def create_starting_grid(centerlocx, centerlocy):
    """
    create grid of bundles mimicking the experimentally observed bundle shape
    ('horseshoe'), without equator
    """
    alpha = -0.23

    #ell_a=np.array([1.27,1.35]).mean*25.2
    #ell_b=np.array([2.18,2.38]).mean*25.2

    #ell_a = 32
    #ell_b = 55
    # weil index von numpy vertauscht ist
    yrec = np.array([a_ell*np.cos(-(alpha + (n - 1)*(np.pi - 2*alpha)/5) + np.pi)
                     for n in np.arange(1, 7)])
    xrec = np.array([b_ell*np.sin(-(alpha + (n - 1)*(np.pi - 2*alpha)/5) + np.pi)
                     for n in np.arange(1, 7)])
    xrec_rot, yrec_rot = rotate_coord(xrec, yrec, -7*np.pi/180)
    #print(xrec_rot,yrec_rot)

    recep_y = yrec_rot[:, None] + centerlocy[None, :]
    recep_x = xrec_rot[:, None] + centerlocx[None, :]
    return recep_x, recep_y, centerlocx, centerlocy




def s(time, steap, x_move):
    return (1/2*(1-np.tanh(2*steap*(time-x_move))))

def kernel_parabola(A, input_x, center_x, w_x_y, input_y, center_y):
    
    # Calculate the distance between each input coordinate and all the centers
    squared_distance = "(input_x_o_y - center_x_o_y )**2"
    squared_distance_x = ne.evaluate(squared_distance, local_dict = {"input_x_o_y": input_x[:,:,np.newaxis], "center_x_o_y":center_x})
    squared_distance_y = ne.evaluate(squared_distance, local_dict = {"input_x_o_y": input_y[:,:,np.newaxis], "center_x_o_y":center_y})
    
    # Calculate the kernel values for each center
    parabel = np.zeros(np.shape(squared_distance_y))
    parabel_dis = "(-(squared_distance_x)- (squared_distance_y)) / (w_x_y**2) + 1"
    amount_rec = squared_distance_x.shape[2]
    for i in np.arange(0,6,1):
        range_list = np.arange(i,amount_rec,6)
        parabel[:,:,range_list] = ne.evaluate(parabel_dis,local_dict={"squared_distance_x":squared_distance_x[:,:,range_list],"squared_distance_y":squared_distance_y[:,:,range_list],"w_x_y": w_x_y[i]})
    
    kernel = A * np.maximum(parabel, 0)
    # Sum the kernel values over all centers to get the final result
    #result =ne.evaluate("sum(kernel, axis=2)",local_dict = {"kernel":kernel})
    result = np.sum(kernel, axis=2)
    return result

def calc_closest_point_on_ellipse(a_ell, b_ell, point):
    """
    for a given point, calculate the closest point on the periphery
    of an ellipse with the two axes a_ell and b_ell
    - assume that the center of the ellipse is at (0, 0)
    """
    xr = np.sqrt(a_ell**2 * b_ell**2 / (b_ell**2 + a_ell**2*(point[:, :, 1]/point[:, :, 0])**2))
    yr = point[:, :, 1]/point[:, :, 0] * xr
    return np.sign(point[:, :, 0])*xr, np.sign(point[:, :, 1])*np.abs(yr)

def distance_to_exp(firstpos, pos_eval, Xint, Yint, v1int, v2int, receptor, mode='target_area'):
    """
    evaluates if pos_eval is within the correct target area/Voronoi cell
    
    """
    ## need to change v1 und v2 and a_ell and b_ell
    # mode can be 'voronoi' or 'target_area'
    goal_loc = np.zeros((nr_of_rec, 2))
    for kk in range(nr_of_rec):
        D_bundles = dist.cdist([firstpos[kk, :]],
                               [[x, y] for x, y in zip(Xint, Yint)], metric='euclid')
        goal_loc[kk, 0] = Xint[D_bundles.argmin(axis=1)]
        goal_loc[kk, 1] = Yint[D_bundles.argmin(axis=1)]
    #print(goal_loc)
    #print("x",goal_loc[:,0]==grid_x, "y",goal_loc[:,1]==grid_y)4
    #v1 und v2 vertauscht!!
    if nr_of_rec == 1 or not include_equator:
        if receptor == 1:
            #print(goal_loc)
            goal_loc += -v1int[None, :]
            #print(goal_loc)
        elif receptor == 2:
            goal_loc += v2int[None, :] - v1int[None, :]
        elif receptor == 3:
            if r3r4swap:
                goal_loc += v2int[None, :]
            else:
                goal_loc += 2*v2int[None, :] - 1*v1int[None, :]
        elif receptor == 4:
            if r3r4swap:
                goal_loc += v2int[None, :] + v1int[None, :]
            else:
                goal_loc += v2int[None, :]
        elif receptor == 5:
            goal_loc += v1int[None, :]
        elif receptor == 6:
            goal_loc += -v2int[None, :] + v1int[None, :]
    elif nr_of_rec == 9 and include_equator:
        if receptor == 1:
            goal_loc[np.array([0, 1, 5])] += -v2int[None, :]
            goal_loc[np.array([2, 3, 4, 6, 7, 8])] += v1int[None, :] - v2int[None, :]
        elif receptor == 2:
            goal_loc[np.array([0, 1, 5])] += v1int[None, :] - v2int[None, :]
            goal_loc[np.array([2, 3, 4, 6, 7, 8])] += - v2int[None, :]
        elif receptor == 3:
            goal_loc[np.array([0, 1, 5])] += 2*v1int[None, :] - 1*v2int[None, :]
            goal_loc[np.array([2, 3, 4, 6, 7, 8])] += -v1int[None, :] - 1*v2int[None, :]
        elif receptor == 4:
            goal_loc[np.array([0, 1, 5])] += v1int[None, :]
            goal_loc[np.array([2, 3, 4, 6, 7, 8])] += -v1int[None, :]
        elif receptor == 5:
            goal_loc[np.array([0, 1, 5])] += v2int[None, :]
            goal_loc[np.array([2, 3, 4, 6, 7, 8])] += -v1int[None, :] + v2int[None, :]
        elif receptor == 6:
            goal_loc[np.array([0, 1, 5])] += -v1int[None, :] + v2int[None, :]
            goal_loc[np.array([2, 3, 4, 6, 7, 8])] += v2int[None, :]
    if include_equator:
        if receptor == 1:
            ind = np.array([2, 3, 4, 6, 7, 8])
        elif receptor == 2:
            ind = np.array([1, 2, 3, 4, 7, 8])
        elif receptor == 3:
            ind = np.array([0, 1, 3, 4, 5, 8])
        elif receptor == 4:
            ind = np.array([1, 3, 4, 5, 7, 8])
        elif receptor == 5:
            ind = np.array([1, 2, 3, 4, 7, 8])
        elif receptor == 6:
            ind = np.array([3, 4, 7, 8])
    
    ind =[] #cause i dont need these cases
    if mode == 'target_area':
        correct1 = (pos_eval[:, 0] - goal_loc[:, 0] + 0.333*v1int[0])**2\
        + (pos_eval[:, 1] - goal_loc[:, 1] + 0.333*v1int[1])**2 < 1.2*b_ell**2
        if include_equator:
            correct1[ind] = (pos_eval[ind, 0] - goal_loc[ind, 0] - 0.333*v1int[0])**2\
            + (pos_eval[ind, 1] - goal_loc[ind, 1] - 0.333*v1int[1])**2 < 1.2*b_ell**2
        correct2 = np.sqrt((pos_eval[:, 0] - goal_loc[:, 0])**2/a_ell**2
                           + (pos_eval[:, 1] - goal_loc[:, 1])**2/b_ell**2) <= 1
        correct = np.logical_or(correct1, correct2)
    elif mode == 'voronoi':
        closest_bundle = np.zeros((len(goal_loc), 2))
        #print("goal", goal_loc)
        #print(a_ell,b_ell)
        #plt.scatter(Xint,Yint)
        for kk in range(len(goal_loc)):
            Xs_ell, Ys_ell = calc_closest_point_on_ellipse(b_ell, a_ell, pos_eval[None, kk, :]
                                                           - np.array([[x, y]
                                                                       for x,y in zip(Xint, Yint)])[:, None, :])
            Xs_circ, Ys_circ = calc_closest_point_on_ellipse(1.2*a_ell, 1.2*a_ell, pos_eval[None, kk, :] - np.array([[x, y] for x,y in zip(Xint - 0.333*v2int[0], Yint - 0.333*v2int[1])])[:, None, :])
            
            #plt.scatter(Xint + Xs_ell[:, 0], Yint + Ys_ell[:, 0], c="red")
            #plt.scatter(Xint - 0.333*v2int[0] + Xs_circ[:, 0], Yint - 0.333*v2int[1] + Ys_circ[:, 0], c="green")
            #plt.show()
            if include_equator:
                if kk in ind:
                    Xs_circ, Ys_circ = calc_closest_point_on_ellipse(1.2*b_ell, 1.2*b_ell, pos_eval[None, kk, :] - np.array([[x, y] for x,y in zip(Xint + 0.333*v2int[0], Yint + 0.333*v2int[1])])[:, None, :])
                D_bundles_ell = dist.cdist([pos_eval[kk, :]], [[x, y] for x, y in zip(Xint + Xs_ell[:, 0], Yint + Ys_ell[:, 0])], metric='euclid')
                D_bundles_circ = dist.cdist([pos_eval[kk, :]], [[x, y] for x, y in zip(Xint - 0.333*v1int[0] + Xs_circ[:, 0], Yint - 0.333*v1int[1] + Ys_circ[:, 0])], metric='euclid')
            else:
        
                D_bundles_ell = dist.cdist([pos_eval[kk, :]], [[x, y] for x, y in zip(Xint + Xs_ell[:, 0], Yint + Ys_ell[:, 0])], metric='euclid')
                D_bundles_circ = dist.cdist([pos_eval[kk, :]], [[x, y] for x, y in zip(Xint - 0.333*v2int[0] + Xs_circ[:, 0], Yint - 0.333*v2int[1] + Ys_circ[:, 0])], metric='euclid')
            min_ell = D_bundles_ell.argmin(axis=1) 
            min_circ = D_bundles_circ.argmin(axis=1)
            #print("Circel", min_circ,"ellipse", min_ell)
            if D_bundles_circ.min(axis=1) < D_bundles_ell.min(axis=1):
                closest_bundle[kk, 0] = Xint[min_circ]
                closest_bundle[kk, 1] = Yint[min_circ]
            else:
                closest_bundle[kk, 0] = Xint[min_ell]
                closest_bundle[kk, 1] = Yint[min_ell]
        #print(closest_bundle[6,:],goal_loc[6,:])
        correct = (np.abs(closest_bundle-goal_loc)<10).all(axis=1)
        #correct = correct[0,:]
    return correct

def creat_start(calc_density):
    # measured size of heels
    rm_heels = np.array([[3.10231051831059, 2.91492280980449, 2.04300797855269, 2.16799853146643],
                         [3.36613563287586, 3.6474344212202, 2.59092160570733, 2.49818435400807],
                         [3.26, 3.45, 2.15, 2.1],
                         [2.5838454966756, 3.26122733330694, 2.43382971588824, 1.79884449690457],
                         [2.73134911063699, 3.95170014933712, 3.89010177959303, 3.75649529229474],
                         [3.97548396049507, 3.37776310930197, 3.23239134019713, 3.02306813633871]])
    radius_heels_avg = rm_heels.mean(axis=1)*25.2
    #print(radius_heels_avg)

    # measured size of fronts
    rm_fronts = np.array([[3.39768913, 4.6265753 , 4.61201485, 4.17743499],
                          [3.11686657, 3.37643334, 3.71268422, 2.80596312],
                          [3.63754606, 4.92912784, 3.42151975, 3.4512155 ],
                          [3.01362226, 3.0008909 , 3.62965469, 3.61263049],
                          [3.4521001 , 3.91277842, 4.38606931, 4.57895422],
                          [3.99243196, 4.05693107, 4.49508072, 4.3103579 ]])
    radius_fronts_avg = rm_fronts.mean(axis=1)*25.2

    horseshoe = create_starting_grid(np.array([0]),np.array([0]))
    rows,cols = 1500, 1500 #have to be sqaured
    dat2_inter = np.zeros((rows, cols))
    POS = np.meshgrid(np.arange(rows), np.arange(cols))
    heel_pos_x = np.array([],int)
    heel_pos_y = np.array([],int)
    
    starting_pos_x, starting_pos_y = np.array([],dtype=int), np.array([],dtype=int)
    range_v2 = np.arange(6)
    v1_x = 0
    for count_v1 in np.arange(7):
        if np.mod(count_v1,2)==1:	
            v1_x = v1_x-v1[0]
        else:
            v1_x = v1_x +v1[0]
        for count_v2 in range_v2:
            x_point = v1_x+count_v2*v2[0] + 150
            y_point = count_v1*v1[1]+count_v2*v2[1] + 150
            starting_pos_x = np.append(starting_pos_x,x_point.astype(int))
            starting_pos_y = np.append(starting_pos_y,y_point.astype(int))
            horseshoe_x = (np.rint(horseshoe[0]+x_point)).astype(int)
            horseshoe_y = (np.rint(horseshoe[1]+y_point)).astype(int)
            dat2_inter[horseshoe_y,horseshoe_x] = 1
            heel_pos_x = np.append(heel_pos_x,horseshoe_x,axis = None)
            heel_pos_y = np.append(heel_pos_y,horseshoe_y,axis = None)            

    if calc_density:
        heels_desity = kernel_parabola(1, POS[0],heel_pos_x,radius_heels_avg,POS[1],heel_pos_y)
        fronts_desity = kernel_parabola(0.5, POS[0],heel_pos_x,radius_fronts_avg,POS[1],heel_pos_y)
    else:
        heels_desity = np.array([])
        fronts_desity = np.array([])
    return heels_desity,fronts_desity, heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg


def changing_pos(x_pos, y_pos, radius):
    x_max_dis = 40
    y_max_dis = 23
    x_pos = x_pos - x_max_dis
    y_pos = y_pos - y_max_dis

    x_random = np.random.normal(x_max_dis, x_max_dis*1/3)
    x_new = x_pos + x_random
    y_random = np.random.normal(y_max_dis, y_max_dis*1/3)
    y_new = y_pos + y_random
    return x_new, y_new

def random_gaus_num():
    mean = 1
    std_dev = 2/3
    random_number = np.random.normal(mean, std_dev)
    random_number = max(0, min(2, random_number))
    return random_number
    
def changing_angle(angle,flashlight_width):
    """with randome number generator"""
    angle = np.degree(angle) - flashlight_width
    angle_change = np.random.rand(1) *2
    angle_new = np.pi/180 *(angle + (flashlight_width*angle_change))
    return angle_new

def main(heels_desity, fronts_desity,heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg):
    #print(create_starting_grid(np.array([0]),np.array([0])))
    #creating starting postions
    
    #min and max values for histogram given with range = [[xmin, xmax], [ymin, ymax]]
    #bins is the controll for how blurry the graph is
    #postitions = np.histogram2d(heel_pos_y,heel_pos_x, bins = 720, range = [[0, 720], [0, 720]])

    dat2_inter = heels_desity+fronts_desity
    if making_movie:
        plt.figure(dpi=300)
        ### xlim und ylim anpassen an neue Gridgröße
        #plt.xlim(100,1300)
        #plt.ylim(100,800)
        plt.scatter(heel_pos_x,heel_pos_y, s = 1,color = "black")
        plt.imshow(dat2_inter, origin="lower",vmin = 3,vmax = 8)
        plt.colorbar()
        #plt.show()
        plt.savefig(f"{folder_path}0.png")
        plt.close()
    #variables
    circ_width_all = np.array([[76.93677250002409, 66.1581562071056, 58.64359788352946, 68.19374152821266],
                               [67.10341096706019, 63.6030367287868, 64.58480307212197, 64.68382969250712],
                               [59.99951401507621, 49.70632397768759, 59.76089748808765, 62.63689465660343],
                               [65.18759340181808, 64.6663385598332, 54.128563634672744, 58.3338061658033],
                               [73.7705462792746, 73.75319815476855, 70.71260029847282, 67.54024198457094],
                               [82.29011450458808, 70.63559641125377, 69.51033837975177, 58.19852457593135]])*angle_per
    roi_degree = np.radians(circ_width_all.mean(axis=1)) # angular width of the 'flashlight'
    roi_radius  = 25.2*np.array([4.41,3.6,5.15,3.58,4.48,4.74])
    n_fil = 10
    startangs_all = np.pi/180 * np.array([-140.6786, -64.3245, -17.25796667, 5.312072, 63.2865, 135.0751667])
    new_speed = np.array([4.04627765, 1.12029122, 0.95092257, 0.77632871, 0.94118902, 4.01210285])
    speeds = np.array([0.093, 0.053, 0.148, 0.09, 0.052, 0.077]) 
    stiff_speed = speeds*new_speed
    s_steap = 0.75
    s_x_move = 25

    histogram_input = np.zeros((42,20))
    way_matrix_x = np.zeros((42*6,21))
    way_matrix_y = np.zeros((42*6,21))
    step_size_fil = np.zeros((42*6,20))
    step_size_stiff = np.zeros((42*6,20))
    fil_matrix_x = np.zeros((42*6,10))
    fil_matrix_y = np.zeros((42*6,10))
    
    #time verändert von 20,40,1 zu 20
    for time in np.arange(20):
        if const_stiff:
            s_time = constanct_stiff
        else:
            s_time = s(time+20, s_steap, s_x_move) 
        if making_movie:
            plt.figure(dpi = 300)
            plt.imshow(dat2_inter,  interpolation='nearest',origin="lower",vmin = 3,vmax = 8)  
            plt.colorbar()     
        for R in np.arange(42*6):
            #stop the growth of the filopodia if mask is out of range of matrix
            if time>0 and (way_matrix_x[R,time] < roi_radius[np.mod(R,6)] or way_matrix_x[R,time] > rows- roi_radius[np.mod(R,6)] or way_matrix_y[R,time] < roi_radius[np.mod(R,6)] or way_matrix_y[R,time] > rows- roi_radius[np.mod(R,6)]):
                way_matrix_x[R,time+1] =way_matrix_x[R,time]
                way_matrix_y[R,time+1] =way_matrix_y[R,time]
            else:
                k_fil_scale = speeds[np.mod(R,6)]
                mask = np.zeros((rows, cols), dtype= bool) #mask_emtpy
                if time == 0: 
                    way_matrix_x[R,time] = heel_pos_x[R]
                    way_matrix_y[R,time] = heel_pos_y[R]
                    front_x, front_y = way_matrix_x[R,time],way_matrix_y[R,time]
                    angle = startangs_all[np.mod(R,6)]
                    if change_angle and R==(21*6+2):
                        angle += change
                else:
                    #last_front_x,last_front_y, front_x, front_y = way_matrix_x[R,time-1],way_matrix_y[R,time-1], way_matrix_x[R,time],way_matrix_y[R,time]
                    #angle = find_degree(last_front_x,last_front_y, front_x, front_y)
                    heel_x,heel_y, front_x, front_y = way_matrix_x[R,0],way_matrix_y[R,0], way_matrix_x[R,time],way_matrix_y[R,time]
                    angle = find_degree(heel_x, heel_y, front_x, front_y)
                heel_x, heel_y = way_matrix_x[R,0],way_matrix_y[R,0]
                ind = creat_mask(angle, front_x, front_y, roi_radius[np.mod(R,6)], roi_degree[np.mod(R,6)], mask)
                
                histog, bins = np.histogram(dat2_inter[ind], bins=10000) # histogram of the density values in the ROI
                cs = (1 - np.cumsum(histog)/np.sum(histog))**4 # cumulative density function, to the power of 4 to make it steeper - see 'inverse transform sampling' for more details
                rr = np.random.rand(n_fil) # sample n_fil = 10 random uniform [0, 1[ numbers

                ind_res = np.array([np.where(cs < rtemp)[0][0] for rtemp in rr]) # at what index does the cumulative density function cross the rr value?
                vals = ((bins[1:] + bins[:-1])/2)[ind_res] # translate the index into density value, using the middle of the bins

                # now I want to use the density values in vals and find a position in the ROI that corresponds to that density value
                D_vals = np.abs(vals[:, None] - dat2_inter[ind][None, :]) # distance of vals to all the density values in the ROI
                xfil, yfil = np.array([]), np.array([])
                xfil = np.hstack((xfil, POS[0][ind][D_vals.argmin(axis=1)])) # take the minimum distance in density and use this position (POS is a meshgrid of all possible positions, same shape as dat2_inter
                yfil = np.hstack((yfil, POS[1][ind][D_vals.argmin(axis=1)]))

                #not sure if it can be saved like that
                fil_matrix_x[R] = xfil
                fil_matrix_y[R] = yfil      

                new_fil_x = (sum(xfil- front_x)/n_fil)*k_fil_scale + front_x
                new_fil_y = (sum(yfil- front_y)/n_fil)*k_fil_scale + front_y

                if time == 0:
                    k_stiff_scale = stiff_speed[np.mod(R,6)]
                    new_stiff_x, new_stiff_y = find_point_2degree(front_x, front_y,2*roi_radius[np.mod(R,6)]*np.sin(angle)*k_stiff_scale/(3*angle), angle)

                else:
                    new_stiff_x = ((-front_x + heel_x)/(-time)) + front_x
                    new_stiff_y = ((-front_y + heel_y)/(-time)) + front_y
                step_size_stiff[R,time] = np.sqrt((front_x-new_stiff_x)**2+(front_y-new_stiff_y)**2)
                step_size_fil[R,time] = np.sqrt((front_x-new_fil_x)**2+(front_y-new_fil_y)**2)
                way_matrix_x[R,time+1] = round( s_time * new_stiff_x + (1-s_time) * new_fil_x)
                way_matrix_y[R,time+1] = round( s_time * new_stiff_y + (1-s_time) * new_fil_y)
            if making_movie:
                if np.mod(R,6) ==5:
                    plt.plot(way_matrix_x[R-5:R+1,0],way_matrix_y[R-5:R+1,0],color = "gray")
                if R == 80:
                    plt.imshow(ind,origin="lower",alpha=0.2)
                plt.plot(way_matrix_x[R,:time+1],way_matrix_y[R,:time+1],color=["blue","green","red","yellow","pink","orange"][np.mod(R,6)])
                
        #landscape doesnt work and plotting doesnt work, but at least it is quick!
        if making_movie:
            #plt.imshow(ind, alpha=0.2, cmap ="hot",interpolation='bilinear',origin="lower")
            plt.scatter(way_matrix_x[:,time+1],way_matrix_y[:,time+1], s = 1, color= "r")
            plt.scatter(way_matrix_x[:,0],way_matrix_y[:,0], s = 1,color = "black")
            plt.title(label = "Densitylandscape")
            #plt.xlim(100,1300)
            #plt.ylim(100,800)
            plt.scatter(starting_pos_x, starting_pos_y,s=3,color="black")
            #plt.show()
            plt.savefig(f"{folder_path}{time+1}.png")
            plt.close()

        fronts_desity = kernel_parabola(0.5,POS[0], way_matrix_x[:,time+1],radius_fronts_avg, POS[1],way_matrix_y[:,time+1])
        dat2_inter =heels_desity+ fronts_desity
        
    return way_matrix_x,way_matrix_y,starting_pos_x, starting_pos_y, step_size_fil, step_size_stiff #, histogram_input

def modell_distance(folder_path_1,range_1, folder_path_2, range_2):
    #heels_desity, fronts_desity,heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg=creat_start(False)
    cor_connect= np.zeros((42*6))
    middle_distance =np.zeros((42*6))
    amout_added = np.zeros((42*6))
    for fil_files in range(range_1):
        with open(f"{folder_path_1}{fil_files}.npy", 'rb') as f:
            way_matrix_x_fil = np.load(f)
            way_matrix_y_fil = np.load(f)
            grid_x = np.load(f)
            grid_y = np.load(f)
        for receptor in np.arange(1,7):
            rec_index = np.arange(receptor-1,nr_of_rec*6,6)
            first_pos = np.array(list(zip(way_matrix_x_fil[rec_index,0],way_matrix_y_fil[rec_index,0])))
            last_pos = np.array(list(zip(way_matrix_x_fil[rec_index,15],way_matrix_y_fil[rec_index,15])))
            voronoi_results = distance_to_exp(first_pos,last_pos, grid_x, grid_y, v1, v2 , receptor,"voronoi")
            cor_connect[(receptor-1)::6] = voronoi_results
        for sta_files in range(range_2):
            with open(f"{folder_path_2}{sta_files}.npy", 'rb') as f:
                way_matrix_x_sta = np.load(f)
                way_matrix_y_sta = np.load(f)
            middle_distance += [np.sqrt((way_matrix_x_fil[ind,15]-way_matrix_x_sta[ind,15])**2+(way_matrix_y_fil[ind,15]-way_matrix_y_sta[ind,15])**2) if  x else 0 for ind, x in enumerate(cor_connect)]
            amout_added += cor_connect.astype(int)
    middle_distance =np.divide(middle_distance,amout_added, out=np.zeros_like(middle_distance), where=amout_added!=0)
    #print(middle_distance)
    receptors = np.zeros(6)
    for receptor in np.arange(1,7):
        rec_index = np.arange(receptor-1,nr_of_rec*6,6)
        receptors[receptor-1] = np.mean(middle_distance[rec_index])
    return receptors

def same_modell_distance(folder_path_1,range_1):
    #heels_desity, fronts_desity,heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg=creat_start(False)
    cor_connect= np.zeros((42*6))
    middle_distance =np.zeros((42*6))
    amout_added = np.zeros((42*6))
    for fil_files in range(range_1):
        with open(f"{folder_path_1}{fil_files}.npy", 'rb') as f:
            way_matrix_x_fil = np.load(f)
            way_matrix_y_fil = np.load(f)
            grid_x = np.load(f)
            grid_y = np.load(f)
        for receptor in np.arange(1,7):
            rec_index = np.arange(receptor-1,nr_of_rec*6,6)
            first_pos = np.array(list(zip(way_matrix_x_fil[rec_index,0],way_matrix_y_fil[rec_index,0])))
            last_pos = np.array(list(zip(way_matrix_x_fil[rec_index,15],way_matrix_y_fil[rec_index,15])))
            voronoi_results = distance_to_exp(first_pos,last_pos, grid_x, grid_y, v1, v2 ,receptor,"voronoi")
            cor_connect[(receptor-1)::6] = voronoi_results
        for sta_files in range(fil_files):
            with open(f"{folder_path_1}{sta_files}.npy", 'rb') as f:
                way_matrix_x_sta = np.load(f)
                way_matrix_y_sta = np.load(f)
            middle_distance += [np.sqrt((way_matrix_x_fil[ind,15]-way_matrix_x_sta[ind,15])**2+(way_matrix_y_fil[ind,15]-way_matrix_y_sta[ind,15])**2) if  x else 0 for ind, x in enumerate(cor_connect)]
            amout_added += cor_connect.astype(int)
    middle_distance =np.divide(middle_distance,amout_added, out=np.zeros_like(middle_distance), where=amout_added!=0)
    #print(middle_distance)
    receptors = np.zeros(6)
    for receptor in np.arange(1,7):
        rec_index = np.arange(receptor-1,nr_of_rec*6,6)
        receptors[receptor-1] = np.mean(middle_distance[rec_index])
    return receptors

def length_of_step(folder_path_1,range_1):
    """
    comparing the steplength of each receptor type for the six inner bundles for each time step
    """
    #heels_desity, fronts_desity,heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg=creat_start(False)
    bundle_receptor_step = np.zeros((6,10*6,20))
    for fil_files in range(range_1):
        with open(f"{folder_path_1}test_{fil_files}.npy", 'rb') as f:
            way_matrix_x = np.load(f)
            way_matrix_y = np.load(f)
            grid_x = np.load(f)
            grid_y = np.load(f)
        bundles = np.array([14, 15, 20, 21, 26, 27])*6
        #bundle =14*6
        for receptor in np.arange(6):
            for ind, bundle in enumerate(bundles):
                bundle_receptor_step[receptor,fil_files*6+ind,:] +=np.sqrt((way_matrix_x[bundle,1:] - way_matrix_x[bundle,:-1])**2+(way_matrix_y[bundle,1:] - way_matrix_y[bundle,:-1])**2)
            bundles+=1
        #for receptor in range(6):
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axes = axes.flatten()
    for receptor in range(6):
        axes[receptor].violinplot(bundle_receptor_step[receptor,:,:])  
        axes[receptor].set_title(f'Rezeptor {receptor + 1}')"""
    see_through = [0.5,0.5,0.5,0.8,0.8,0.8]
    see_through_error = np.array(see_through)-0.3
    plt.figure(dpi=300) 
    plt.axvspan(-1, 4, color='gray', alpha=0.5)
    plt.text(1, 14.5, 'Stiffness', ha='center', va='center')
    plt.text(7, 14.5, 'Density Sensing', ha='center', va='center')
    #plt.fill_between(x, y-error, y+error)
    for i in range(6):
        mean = np.mean(bundle_receptor_step[i,:,:],axis = 0)
        error = np.std(bundle_receptor_step[i,:,:],axis = 0)
        plt.plot(np.arange(20),mean,'o-', markersize=3, color =["blue","green","red","yellow","pink","orange"][i],alpha=see_through[i], label = f"R{i+1}")
        plt.fill_between(np.arange(20),mean-error, mean+error,color =["blue","green","red","yellow","pink","orange"][i],alpha=see_through_error[i])
    #plt.tight_layout()
    plt.xlim(-1,20)
    plt.xlabel("Time in Hours") 
    plt.ylabel("Stepsize") 
    plt.xticks(np.arange(20))
    plt.legend()
    plt.savefig(f"{folder_path_1}length_of_step_median_std.png")

def speed_scaling():
    """
    finding new speed scaling    
    """
    #heels_desity, fronts_desity,heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg=creat_start(False)
    bundle_receptor_step = np.zeros((6,10*6,20))
    for fil_files in range(10):
        with open(f"{folder_path}test_step_size_{fil_files}.npy", 'rb') as f:
            way_matrix_x = np.load(f)
            way_matrix_y = np.load(f)
            grid_x = np.load(f)
            grid_y = np.load(f)
        bundles = np.array([14, 15, 20, 21, 26, 27])*6
        for receptor in np.arange(6):
            for ind, bundle in enumerate(bundles):
                bundle_receptor_step[receptor,fil_files*6+ind,:] +=np.sqrt((way_matrix_x[bundle,1:] - way_matrix_x[bundle,:-1])**2+(way_matrix_y[bundle,1:] - way_matrix_y[bundle,:-1])**2)
            bundles+=1

    new_speed = np.zeros(6)
    for i in range(6):
        mean = np.mean(bundle_receptor_step[i,:,:],axis = 0)
        stiffness = sum(mean[:5])/len(mean[:5])
        filopodia = sum(mean[5:])/len(mean[5:])
        new_speed[i] = filopodia/stiffness

    bundle_receptor_step = np.zeros((6,10*6,20))
    for fil_files in range(10):
        with open(f"{folder_path}test_{fil_files}.npy", 'rb') as f:
            way_matrix_x = np.load(f)
            way_matrix_y = np.load(f)
            grid_x = np.load(f)
            grid_y = np.load(f)
        bundles = np.array([14, 15, 20, 21, 26, 27])*6
        for receptor in np.arange(6):
            for ind, bundle in enumerate(bundles):
                bundle_receptor_step[receptor,fil_files*6+ind,:] +=np.sqrt((way_matrix_x[bundle,1:] - way_matrix_x[bundle,:-1])**2+(way_matrix_y[bundle,1:] - way_matrix_y[bundle,:-1])**2)
            bundles+=1

    new_speed_2 = np.zeros(6)
    for i in range(6):
        mean = np.mean(bundle_receptor_step[i,:,:],axis = 0)
        stiffness = sum(mean[:5])/len(mean[:5])
        filopodia = sum(mean[5:])/len(mean[5:])
        new_speed_2[i] = filopodia/stiffness

    new_speed = (new_speed + new_speed_2)/2
    return new_speed

def length_of_step_keep_fil(folder_path_1):
    """
    comparing the steplength of different keep_fil_numbers for each receptor type for the six inner bundles for each time step
    """
    #heels_desity, fronts_desity,heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg=creat_start(False)
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axes = axes.flatten()
    cmap = ['#0000FF','#1A1AFF','#3333CC','#4D4D99','#666666','#808080','#999966','#B2B233','#CCCCCC','#00FF00']
    
    for keep_fil in np.arange(10):
        bundle_receptor_step = np.zeros((6,10*6,20))
        for fil_files in range(10):
            with open(f"{folder_path_1}test_keep_{keep_fil}_fil_{fil_files}.npy", 'rb') as f:
                way_matrix_x = np.load(f)
                way_matrix_y = np.load(f)
            bundles = np.array([14, 15, 20, 21, 26, 27])*6
            #bundle =14*6
            for receptor in np.arange(6):
                for ind, bundle in enumerate(bundles):
                    bundle_receptor_step[receptor,fil_files*6+ind,:] =np.sqrt((way_matrix_x[bundle,1:] - way_matrix_x[bundle,:-1])**2+(way_matrix_y[bundle,1:] - way_matrix_y[bundle,:-1])**2)
                bundles+=1
            #for receptor in range(6):
        for i in range(6):
            mean = np.mean(bundle_receptor_step[i,:,:],axis = 0)
            error = np.std(bundle_receptor_step[i,:,:],axis = 0)
            axes[i].plot(np.arange(20),mean,'o-', markersize=3, color =cmap[keep_fil],alpha=0.5, label = f"{keep_fil}")
            axes[i].fill_between(np.arange(20),mean-error, mean+error,color =cmap[keep_fil],alpha=0.5)
    for i in range(6):
        axes[i].set_title(f"R{i+1}")
    
    
    plt.tight_layout()
    plt.xlim(-1,20)
    plt.xlabel("Time in Hours") 
    plt.ylabel("Stepsize") 
    plt.xticks(np.arange(20))
    plt.legend()
    plt.savefig(f"{folder_path_1}new_test_length_of_step_median_std.png")
    """
    found error in keep fil file
    """

def length_of_stiff_fil_std(folder_path_1):
    """
    comparing the length of stiffness and filopodia drag for each receptor type for the six inner bundles for each time step
    """
    #heels_desity, fronts_desity,heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg=creat_start(False)
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axes = axes.flatten()
    cmap = ['#0000FF','#1A1AFF','#3333CC','#4D4D99','#666666','#808080','#999966','#B2B233','#CCCCCC','#00FF00']
    

    bundle_receptor_stiff = np.zeros((6,10*6,20))
    bundle_receptor_fil = np.zeros((6,10*6,20))
    for fil_files in range(10):
        with open(f"{folder_path_1}test_step_size_{fil_files}.npy", 'rb') as f:
            way_matrix_x = np.load(f)
            way_matrix_y = np.load(f)
            grid_x = np.load(f)
            grid_y = np.load(f)
            step_fil = np.load(f)
            step_stiff = np.load(f)
        bundles = np.array([14, 15, 20, 21, 26, 27])*6
        for receptor in np.arange(6):
            for ind, bundle in enumerate(bundles):
                bundle_receptor_stiff[receptor,fil_files*6+ind,:] =step_stiff[bundle+ind,:]
                bundle_receptor_fil[receptor,fil_files*6+ind,:] =step_fil[bundle+ind,:]
            bundles+=1
    for i in range(6):
        mean = np.mean(bundle_receptor_fil[i,:,:],axis = 0)
        mean_stiff = np.mean(bundle_receptor_stiff[i,:,:],axis = 0)
        error = np.std(bundle_receptor_fil[i,:,:],axis = 0)
        error_stiff = np.std(bundle_receptor_stiff[i,:,:],axis = 0)
        axes[i].plot(np.arange(20),mean,'o-', markersize=3, color ="red",alpha=1, label = f"filopodium")
        axes[i].fill_between(np.arange(20),mean-error, mean+error,color ="red",alpha=0.5)
        axes[i].plot(np.arange(20),mean_stiff,'o-', markersize=3, color ="blue",alpha=1, label = f"stiffness")
        axes[i].fill_between(np.arange(20),mean_stiff-error_stiff, mean_stiff+error_stiff,color ="blue",alpha=0.5)
        axes[i].set_title(f"R{i+1}")
    
    plt.tight_layout()
    plt.xlim(-1,20)
    plt.xlabel("Time in Hours") 
    plt.ylabel("Stepsize") 
    plt.legend(["stiffness","filopodia"])
    plt.savefig(f"{folder_path_1}stiffness_vs_filopodium_std.png")

def length_of_stiff_fil_scatter(folder_path_1):
    """
    comparing the length of stiffness and filopodia drag for each receptor type for the six inner bundles for each time step
    """
    #heels_desity, fronts_desity,heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg=creat_start(False)
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axes = axes.flatten()
    cmap = ['#0000FF','#1A1AFF','#3333CC','#4D4D99','#666666','#808080','#999966','#B2B233','#CCCCCC','#00FF00']
    

    bundle_receptor_stiff = np.zeros((6,10*6,20))
    bundle_receptor_fil = np.zeros((6,10*6,20))
    for fil_files in range(10):
        with open(f"{folder_path_1}test_step_size_{fil_files}.npy", 'rb') as f:
            way_matrix_x = np.load(f)
            way_matrix_y = np.load(f)
            grid_x = np.load(f)
            grid_y = np.load(f)
            step_fil = np.load(f)
            step_stiff = np.load(f)
        bundles = np.array([14, 15, 20, 21, 26, 27])*6
        for receptor in np.arange(6):
            for ind, bundle in enumerate(bundles):
                bundle_receptor_stiff[receptor,fil_files*6+ind,:] =step_stiff[bundle+ind,:]
                bundle_receptor_fil[receptor,fil_files*6+ind,:] =step_fil[bundle+ind,:]
                axes[receptor].scatter(np.arange(20),step_stiff[bundle+ind,:], color = "blue",s =1,alpha=0.5)
                axes[receptor].scatter(np.arange(20),step_fil[bundle+ind,:], color = "red",s =1,alpha=0.5)
            bundles+=1

    for i in range(6):
        mean = np.mean(bundle_receptor_fil[i,:,:],axis = 0)
        mean_stiff = np.mean(bundle_receptor_stiff[i,:,:],axis = 0)
        #axes[i].plot(np.arange(20),mean,'o-', markersize=3, color ="red",alpha=1, label = f"filopodium")
        #axes[i].plot(np.arange(20),mean_stiff,'o-', markersize=3, color ="blue",alpha=1, label = f"stiffness")
        axes[i].set_title(f"R{i+1}")
    
    plt.tight_layout()
    plt.xlim(-1,20)
    plt.xlabel("Time in Hours") 
    plt.ylabel("Stepsize") 
    plt.legend(["stiffness","filopodia"])
    plt.savefig(f"{folder_path_1}stiffness_vs_filopodium_scatter.png")

def length_of_stiff_fil_violin(folder_path_1):
    """
    comparing the length of stiffness and filopodia drag for each receptor type for the six inner bundles for each time step
    """
    #heels_desity, fronts_desity,heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg=creat_start(False)

    bundle_receptor_stiff = np.zeros((6,10*6,20))
    bundle_receptor_fil = np.zeros((6,10*6,20))
    for fil_files in range(10):
        with open(f"{folder_path_1}test_step_size_{fil_files}.npy", 'rb') as f:
            way_matrix_x = np.load(f)
            way_matrix_y = np.load(f)
            grid_x = np.load(f)
            grid_y = np.load(f)
            step_fil = np.load(f)
            step_stiff = np.load(f)
        bundles = np.array([14, 15, 20, 21, 26, 27])*6
        for receptor in np.arange(6):
            for ind, bundle in enumerate(bundles):
                bundle_receptor_stiff[receptor,fil_files*6+ind,:] =step_stiff[bundle+ind,:]
                bundle_receptor_fil[receptor,fil_files*6+ind,:] =step_fil[bundle+ind,:]
            bundles+=1
    
    fig, axes = plt.subplots(dpi = 350, nrows=2, ncols=3, figsize=(12, 8))
    axes = axes.flatten()
    for receptor in range(6):
        vio = axes[receptor].violinplot(bundle_receptor_stiff[receptor,:,:],positions = np.arange(20)+1.5, showmedians=True)
        for pc in vio["bodies"]:
            pc.set_color("blue")
        vio = axes[receptor].violinplot(bundle_receptor_fil[receptor,:,:],positions = np.arange(20)+1, showmedians=True)
        for pc in vio["bodies"]:
            pc.set_color("red")
        axes[receptor].set_title(f'Rezeptor {receptor + 1}')
    legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='Stiffness'),
    Line2D([0], [0], color='red', lw=2, label='Filopodial'),
    ]   
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    
    plt.xlabel("Time in Hours") 
    plt.ylabel("Stepsize") 
    plt.savefig(f"{folder_path_1}stiffness_vs_filopodium_violin.png")

def rec_miss_change_rate():
    """
    calculating differences when receptor loss for 6 bundles around bundle with receptor loss
    """
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['red', 'yellow', 'green'], N=256)
    bounds = [-5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    
    heels_desity, fronts_desity,heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg=creat_start(False)
    colors = ['#0000FF', '#1E90FF', '#4169E1', '#6495ED', '#67d1fc', '#79bbd1', '#67ffff', '#ffbc2d', '#FFA07A', '#FF0000']
    index_col = [0,0,1,1,1,2,2]
    index_row = [0,1,0,1,2,0,1]
    b_indexs = np.array([14,15,20,21,22,26,27])
    for i in range(7):
        folder_path = f"./modell_tanh_stiffness_full_funct_adjust_grid_size/"
        all_receptor = np.zeros((6))
        b_index = b_indexs[i]
        for sweeps in range(10):
            with open(f"{folder_path}test_{sweeps}.npy", 'rb') as f:
                way_matrix_x = np.load(f)
                way_matrix_y = np.load(f)
                grid_x = np.load(f)
                grid_y = np.load(f)
            for receptor in np.arange(1,7):
                rec_index = np.arange(receptor-1,nr_of_rec*6,6)
                first_pos = np.array(list(zip(heel_pos_x[rec_index],heel_pos_y[rec_index])))
                last_pos = np.array(list(zip(way_matrix_x[rec_index,15],way_matrix_y[rec_index,15])))
                voronoi_results = distance_to_exp(first_pos,last_pos, grid_x, grid_y, v1, v2 ,receptor,"voronoi")
                all_receptor[(receptor-1)] += voronoi_results[b_index].astype(int)
        folder_path = f"./modell_tanh_stiffness_receptor_loss/"

        rec_missing = np.zeros((6,6))
        for rec_miss in range(6):
            for sweeps in range(10):
                with open(f"{folder_path}{rec_miss}_rt_21_b_test_{sweeps}.npy", 'rb') as f:
                    way_matrix_x = np.load(f)
                    way_matrix_y = np.load(f)
                    grid_x = np.load(f)
                    grid_y = np.load(f)
                for receptor in np.arange(1,7):
                    rec_index = np.arange(receptor-1,nr_of_rec*6,6)
                    first_pos = np.array(list(zip(heel_pos_x[rec_index],heel_pos_y[rec_index])))
                    last_pos = np.array(list(zip(way_matrix_x[rec_index,15],way_matrix_y[rec_index,15])))
                    voronoi_results = distance_to_exp(first_pos,last_pos, grid_x, grid_y, v1, v2 ,receptor,"voronoi")
                    rec_missing[(receptor-1),rec_miss] += voronoi_results[b_index].astype(int)
        results = rec_missing-np.transpose(np.tile(all_receptor, (6, 1)))
        #print(all_receptor, rec_missing, (rec_missing-all_receptor[:,np.newaxis]), np.tile(all_receptor, (6, 1)))
        np.fill_diagonal(results, np.nan)
        #print(axs[index_col[i],index_row[i]])
        axs[index_col[i],index_row[i]].imshow(results,origin="upper" ,cmap=cmap)
        axs[index_col[i],index_row[i]].set_xticks(np.arange(6), ['R1', 'R2', 'R3', 'R4', 'R5','R6'])
        axs[index_col[i],index_row[i]].set_yticks(np.arange(6), ['R1', 'R2', 'R3', 'R4', 'R5','R6'])
        #plt.yaxis.tick_right()
        axs[index_col[i],index_row[i]].set_ylabel("Performance of Receptor")
        axs[index_col[i],index_row[i]].set_xlabel("Killed Receptors")
    
    plt.show()
    #plt.savefig(f"{folder_path}receptor_missing.png")  
  
def similar_neightbours():
    """
    calculating simularity of neightbouring bundles receptor subtype spezific
    """
    heels_desity, fronts_desity,heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg=creat_start(False)
    voronoi_matrix = np.zeros((nr_of_rec*6,10))
    colors = ['#0000FF', '#1E90FF', '#4169E1', '#6495ED', '#67d1fc', '#79bbd1', '#67ffff', '#ffbc2d', '#FFA07A', '#FF0000']
    for sweeps in range(10):
        with open(f"{folder_path}test_{sweeps}.npy", 'rb') as f:
            way_matrix_x = np.load(f)
            way_matrix_y = np.load(f)
            grid_x = np.load(f)
            grid_y = np.load(f)
        for receptor in np.arange(1,7):
            rec_index = np.arange(receptor-1,nr_of_rec*6,6)
            first_pos = np.array(list(zip(heel_pos_x[rec_index],heel_pos_y[rec_index])))
            last_pos = np.array(list(zip(way_matrix_x[rec_index,15],way_matrix_y[rec_index,15])))
            voronoi_results = distance_to_exp(first_pos,last_pos, grid_x, grid_y, v1, v2 ,receptor,"voronoi")
            voronoi_matrix[(receptor-1)::6,sweeps] = voronoi_results.astype(int)
    fig, axes = plt.subplots(dpi=300 ,nrows=6, ncols=1,figsize=(10,10))  
    axes = axes.flatten()
    for receptor in np.arange(6):
        axes[receptor].imshow(np.transpose(voronoi_matrix[np.arange(receptor,42*6,6)]),origin="lower", cmap='viridis')
        axes[receptor].set_title(f'Rezeptor {receptor + 1}') 
    plt.tight_layout() 

    plt.savefig(f"{folder_path}neighbouring_bundles.png")

def comp_perfomance_bar():
    """
    comparing the output of the experiments of the stiffness as the performance of the 
    """
    plt.bar(np.arange(6)-0.3,same_modell_distance("C:/Users/hei-1/OneDrive/BCHLRARBT/modell_tanh_stiffness_full_funct_adjust_grid_size/test_",10), 0.1)
    plt.bar(np.arange(6)-0.1,modell_distance("C:/Users/hei-1/OneDrive/BCHLRARBT/modell_constant_stiffness_anaylse/test_0.0_stiffness_",4, "C:/Users/hei-1/OneDrive/BCHLRARBT/modell_tanh_stiffness_full_funct_adjust_grid_size/test_",10), 0.1)
    plt.bar(np.arange(6),modell_distance("C:/Users/hei-1/OneDrive/BCHLRARBT/modell_constant_stiffness_anaylse/test_0.2_stiffness_",4, "C:/Users/hei-1/OneDrive/BCHLRARBT/modell_tanh_stiffness_full_funct_adjust_grid_size/test_",10), 0.1)
    plt.bar(np.arange(6)+0.1,modell_distance("C:/Users/hei-1/OneDrive/BCHLRARBT/modell_constant_stiffness_anaylse/test_0.4_stiffness_",4, "C:/Users/hei-1/OneDrive/BCHLRARBT/modell_tanh_stiffness_full_funct_adjust_grid_size/test_",10), 0.1)
    plt.bar(np.arange(6)+0.2,modell_distance("C:/Users/hei-1/OneDrive/BCHLRARBT/modell_constant_stiffness_anaylse/test_0.6000000000000001_stiffness_",4, "C:/Users/hei-1/OneDrive/BCHLRARBT/modell_tanh_stiffness_full_funct_adjust_grid_size/test_",10), 0.1)
    plt.bar(np.arange(6)+0.3,modell_distance("C:/Users/hei-1/OneDrive/BCHLRARBT/modell_constant_stiffness_anaylse/test_0.8_stiffness_",4, "C:/Users/hei-1/OneDrive/BCHLRARBT/modell_tanh_stiffness_full_funct_adjust_grid_size/test_",10), 0.1)
    plt.xticks(np.arange(6), ['R1', 'R2', 'R3', 'R4', 'R5','R6'])
    plt.ylim(0,80)
    plt.xlabel("Receptor Type")
    plt.ylabel("Distance to same Receptors")
    plt.legend(["Controll Modell with Original Function", "Modell Constant Stiffness 0.0", "Modell Constant Stiffness 0.2", "Modell Constant Stiffness 0.4", "Modell Constant Stiffness 0.6", "Modell Constant Stiffness 0.8"], loc='upper left', borderaxespad=1)
    #plt.show()
    plt.savefig("C:/Users/hei-1/OneDrive/BCHLRARBT/modell_constant_stiffness_anaylse/distance_of_modells.png")

def comp_performance_plot():
    """
    calculating the performance of inner bundle comparing different flashlight width or stiffness or so on
    with the plot for each subtype and all together
    """
    range_ = np.arange(10)

    plt.figure(dpi = 300, figsize=(10,6))
    heels_desity, fronts_desity,heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg=creat_start(False)
    performance = np.zeros((range_.size,6))
    performance_all = np.zeros(range_.size)
    index_perf = 0
    
    bundle_index = [14, 15, 20, 21, 26, 27]
    #bundle_index = [14, 15, 20, 21, 22, 26, 27]
    for rt in range_:
        count =0
        voronoi_added = np.zeros(6)
        voronoi_matrix = np.zeros(nr_of_rec*6)
        for sweeps in range(10):
            with open(f"{folder_path}keep_{rt}_fil_test_{sweeps}.npy", 'rb') as f:
                way_matrix_x = np.load(f)
                way_matrix_y = np.load(f)
                grid_x = np.load(f)
                grid_y = np.load(f)
            for receptor in np.arange(1,7):
                count +=len(bundle_index)
                rec_index = np.arange(receptor-1,nr_of_rec*6,6)
                first_pos = np.array(list(zip(heel_pos_x[rec_index],heel_pos_y[rec_index])))
                last_pos = np.array(list(zip(way_matrix_x[rec_index,15],way_matrix_y[rec_index,15])))
                voronoi_results = distance_to_exp(first_pos,last_pos, grid_x, grid_y, v1, v2 ,receptor,"voronoi")
                voronoi_added[receptor-1] += np.sum(voronoi_results[bundle_index].astype(int)) 
        performance[index_perf,:] = voronoi_added/(len(bundle_index)*10)
        performance_all[index_perf]  = np.sum(voronoi_added)/count
        index_perf +=1
    """
    count = 0
    voronoi_added = np.zeros(6)
    for sweeps in range(10):
        with open(f"./modell_tanh_stiffness_full_funct_adjust_grid_size/test_{sweeps}.npy", 'rb') as f:
            way_matrix_x = np.load(f)
            way_matrix_y = np.load(f)
            grid_x = np.load(f)
            grid_y = np.load(f)
        for receptor in np.arange(1,7):
            count += len(bundle_index)
            rec_index = np.arange(receptor-1,nr_of_rec*6,6)
            first_pos = np.array(list(zip(heel_pos_x[rec_index],heel_pos_y[rec_index])))
            last_pos = np.array(list(zip(way_matrix_x[rec_index,15],way_matrix_y[rec_index,15])))
            voronoi_results = distance_to_exp(first_pos,last_pos, grid_x, grid_y, v1, v2 ,receptor,"voronoi")
            voronoi_added[receptor-1] += np.sum(voronoi_results[bundle_index].astype(int))
    #print(voronoi_added)
    performance[index_perf,:] = voronoi_added/(len(bundle_index)*10)
    performance_all[index_perf]  = np.sum(voronoi_added)/count
    #print(performance)
    #plt.scatter([50,60,70,80,90,100],performance)
    """
    see_through = [0.5,0.5,0.5,0.8,0.8,0.8]
    for i in range(6):
        plt.plot(range_,performance[:,i], ['o-','*-','s-','d-','x-','v-'][i], markersize=8, color =["blue","green","red","yellow","pink","orange"][i],alpha=see_through[i], label = f"R{i+1}")
    plt.plot(range_,performance_all, 'p-', markersize=8, color ="black",alpha=1, label = "all RT")
    plt.ylabel("Performance of Inner Bundles")
    plt.ylim(0,1.1)
    #plt.xticks(np.arange(11), ['0','1', '2', '3', '4', '5','6','7','8','9','original'])
    #plt.xticks(np.arange(5), ['50','', '2', '3', '4', '5','6','7','8','9','original'])
    plt.xlabel("Number of kept filopodial per round")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.title("The performance of correct connected receptors depending on flashlight angle percentage")
    plt.savefig(f"{folder_path}performance_of_keep_fil.png")
    #plt.legend(["R1","R2","R3","R4","R5","R6"], loc='lower left', borderaxespad=1)

def voronoi_grid(folder_path, time_index):
    """
    calcualting the performance based on the placement in the grid
    """
    heels_desity, fronts_desity,heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg=creat_start(False)
    plt.figure(dpi = 300)
    voronoi_matrix = np.zeros(nr_of_rec*6)
    for sweeps in range(10):
        with open(f"{folder_path}test_{sweeps}.npy", 'rb') as f:
            way_matrix_x = np.load(f)
            way_matrix_y = np.load(f)
            grid_x = np.load(f)
            grid_y = np.load(f)
        for receptor in np.arange(1,7):
            rec_index = np.arange(receptor-1,nr_of_rec*6,6)
            first_pos = np.array(list(zip(heel_pos_x[rec_index],heel_pos_y[rec_index])))
            
            last_pos = np.array(list(zip(way_matrix_x[rec_index,time_index],way_matrix_y[rec_index,time_index])))
            
            voronoi_results = distance_to_exp(first_pos,last_pos, grid_x, grid_y, v1, v2 ,receptor,"voronoi")
            voronoi_matrix[(receptor-1)::6] += voronoi_results.astype(int)
            
    #
    #plt.xlim(400,1100)
    #plt.ylim(200,650)
    #plt.figure(dpi = 300)
    plt.scatter(heel_pos_x,heel_pos_y,c=voronoi_matrix,cmap='viridis', s=10)
    cmap = mpl.cm.viridis
    bounds = [0, 1, 2, 3, 4, 5,6,7,8,9,10]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    plt.title("Correct connected receptors out of 10 rounds")
    #plt.show()
    plt.savefig(f"{folder_path}voronoi_matrix.png")

def voronoi_grid_repeat():
    """
    calcualting the performance based on the placement in the grid for the loss of a receptor or for different stiffness
    """
    cmap = mpl.cm.viridis
    bounds = [0, 1, 2, 3, 4, 5,6,7,8,9,10]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    plt.title("Correct connected receptors out of 10 rounds")
    heels_desity, fronts_desity,heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg=creat_start(False)
    for rt in np.arange(10):
        rt = np.round(rt,1)
        voronoi_matrix = np.zeros(nr_of_rec*6)
        for sweeps in range(10):
            with open(f"{folder_path}test_keep_{rt}_fil_{sweeps}.npy", 'rb') as f:
                way_matrix_x = np.load(f)
                way_matrix_y = np.load(f)
                grid_x = np.load(f)
                grid_y = np.load(f)
            for receptor in np.arange(1,7):
                rec_index = np.arange(receptor-1,nr_of_rec*6,6)
                first_pos = np.array(list(zip(heel_pos_x[rec_index],heel_pos_y[rec_index])))
                
                last_pos = np.array(list(zip(way_matrix_x[rec_index,15],way_matrix_y[rec_index,15])))
                
                voronoi_results = distance_to_exp(first_pos,last_pos, grid_x, grid_y, v1, v2 ,receptor,"voronoi")
                voronoi_matrix[(receptor-1)::6] += voronoi_results.astype(int)
                
        #plt.figure(dpi = 300,figsize=(7,3.5))
        #plt.xlim(400,1100)
        #plt.ylim(200,650)
        #plt.figure(dpi = 300)

        plt.scatter(heel_pos_x,heel_pos_y,c=voronoi_matrix,cmap='viridis', s=10)
        
        
        plt.savefig(f"{folder_path}keep_{rt}_fil_voronoi_matrix.png")
        print(rt)

def plot_way_matrix():
    """
    making plot of way_matrix
    """
    #for sweeps in range(10):
    plt.figure(dpi = 300)
    with open(f"{folder_path}angle_2.0_test_0.npy", 'rb') as f:
                way_matrix_x = np.load(f)
                way_matrix_y = np.load(f)
                grid_x = np.load(f)
                grid_y = np.load(f)
    #print(way_matrix_x[21*6+3,:], way_matrix_y[21*6+3,:])
    for R in range(42*6):
        plt.plot(way_matrix_x[R,:16],way_matrix_y[R,:16],color=["blue","green","red","yellow","pink","orange"][np.mod(R,6)])
        if np.mod(R,6) ==5:
            plt.plot(way_matrix_x[R-5:R+1,0],way_matrix_y[R-5:R+1,0],color = "gray")
    plt.savefig(f"{folder_path}way_matrix_angle_2.0_test_0.png")

def run_main(save_fil, file_name):
    heels_desity, fronts_desity,heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg=creat_start(True)
    for sweeps in range(10):
        way_matrix_x, way_matrix_y, grid_x, grid_y, step_fil, step_stiff= main(heels_desity, fronts_desity,heel_pos_x, heel_pos_y, rows, cols, POS, starting_pos_x,starting_pos_y, radius_fronts_avg) 
        with open(f"{folder_path}{file_name}{sweeps}.npy", 'w+b') as f:
            np.save(f, way_matrix_x)
            np.save(f, way_matrix_y)
            np.save(f, grid_x)
            np.save(f, grid_y)
            if save_fil:
                np.save(f, step_fil)
                np.save(f, step_stiff)
    
if __name__ == '__main__':
    """import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()"""
    v1=np.around((np.array([[3.51,3.05],
                 [3.71,3.80],
                 [4.0,3.91],
                 [3.77,3.84]]).mean(axis=0))*25.2).astype(int)
    v2=np.around((np.array([[6.78,-0.25],
                 [7.03,0.001],
                 [7.88,0.06],
                 [7.42,0.01]]).mean(axis=0))*25.2).astype(int)
    a_ell=np.around((np.array([1.27,1.35]).mean(axis=0))*25.2).astype(int)
    b_ell=np.around((np.array([2.18,2.38]).mean(axis=0))*25.2).astype(int)
    making_movie = False
    nr_of_rec = 42 #number of bundles
    include_equator = False
    r3r4swap = False    
    folder_path = f"./modell_h2f_stiff_spsc_normal_change_R3_init_angle/"
    const_stiff = False
    angle_per = 2
    change_angle = True
    change = ( np.pi/180 * 10)
    run_main(False,"plus_10_degree_test_" )
    change = -( np.pi/180 * 10)
    run_main(False,"minus_10_degree_test_" )
    folder_path = f"./modell_h2f_stiff_spsc_constant_stiff_change_R3_init_angle/"
    const_stiff = True
    constanct_stiff = 1
    change = (np.pi/180 * 10)
    run_main(False,"plus_10_degree_test_" )
    change = -(np.pi/180 * 10)
    run_main(False,"minus_10_degree_test_" )
    """
    folder_path = f"./modell_h2f_stiff_spsc_constant_stiffness_anaylse/"
    const_stiff = True
    for constanct_stiff in np.around(np.arange(0,1.1,0.1),1):
        run_main(False, f"stiffness_{constanct_stiff}_test_")
    folder_path = f"./modell_h2f_stiff_spsc_tanh_stiffness_flashlight_width_analyse/"
    const_stiff = False
    for angle_per in np.around(np.arange(1,2.1,0.2),1):
        run_main(False, f"angle_{angle_per}_test_")
    """
    #plot_way_matrix()
    #voronoi_grid(folder_path, 15)
    #comp_performance_plot()
    #length_of_step(folder_path,10)
    #length_of_stiff_fil_violin(folder_path)
    #run_main(False,"test_")
    
    """profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats('dump_stats_4')"""