import numpy as np
import numexpr as ne
from scipy.signal import fftconvolve
from scipy import signal
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import matplotlib as mpl

def quantify_correct_way(grid_x, grid_y, receptor_x, receptor_y):
    print("grid_x",grid_x,"gird_y",grid_y)
    expected_closest,actual_closest = np.array([]), np.array([])
    grid_size= grid_x.size/2
    grid_sqrt = np.sqrt(grid_size)
    for grid_num in range(grid_x.size):
        for rec_type in range(6):
            if rec_type ==0:
                if grid_num<grid_size:
                    expected_closest = np.append(expected_closest,grid_num+grid_size-grid_sqrt-1)
                else:
                    expected_closest = np.append(expected_closest,grid_num-grid_size)
            elif rec_type ==1:
                if grid_num<grid_size:
                    expected_closest = np.append(expected_closest,grid_num+grid_size-grid_sqrt)
                else:
                    expected_closest = np.append(expected_closest,grid_num-grid_size+1)
            elif rec_type ==2:
                if grid_num<grid_size:
                    expected_closest = np.append(expected_closest,grid_num+grid_size-grid_sqrt+1)
                else:
                    expected_closest = np.append(expected_closest,grid_num-grid_size+2)
            elif rec_type ==3:
                expected_closest = np.append(expected_closest,grid_num+1)
            elif rec_type ==4:
                if grid_num<grid_size:
                    expected_closest = np.append(expected_closest,grid_num+grid_size)
                else:
                    expected_closest = np.append(expected_closest,grid_num-grid_size+grid_sqrt+1)
            elif rec_type ==5:
                if grid_num<grid_size:
                    expected_closest = np.append(expected_closest,grid_num+grid_size-1)
                else:
                    expected_closest = np.append(expected_closest,grid_num-grid_size+grid_sqrt)

            #trying to find sideline cases
            if expected_closest[-1]<0:
                expected_closest[-1] = grid_num
        
            actual_closest = np.append(actual_closest,np.argmin(np.sqrt((grid_x-receptor_x[grid_num*6+rec_type])**2+(grid_y-receptor_y[grid_num*6+rec_type])**2)))

    print("actual",actual_closest,"expected",expected_closest)
    correct_way= expected_closest == actual_closest
    return correct_way.reshape(-1,6)

def count_rec_correct_way(quantify_output):
    size,col = quantify_output.shape
    row=np.sqrt(size/2)
    rec = np.array([0,0,0,0,0,0])
    all_rec = 0
    for i in range(size):
        if i<row or size/2-row<=i<size/2+row or size-row<=i or np.mod(i,row)==0 or np.mod(i,row)==row-1:
            continue
        all_rec = all_rec + 1
        rec = quantify_output[i].astype(int)+rec
    print(rec, " out of ", all_rec)
    return rec

#print(count_rec_correct_way(np.array([[False, False, False, False, False, False],[False, False, False,  True, False, False], [False, False, False,  True,  True, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [ True,  True, False,  True,  True,  True], [ True,  True, False,  True,  True,  True], [ True,  True, False, False,  True,  True], [False, False, False, False, False, False], [ True,  True, False,  True,  True,  True], [ True,  True,  True,  True,  True,  True], [ True,  True, False, False,  True, False], [False, False, False, False, False, False], [ True, False,  True,  True,  True,  True], [ True,  True,  True,  True,  True,  True], [ True,  True, False, False,  True,  True], [ True,  True, False, False, False,  True], [ True,  True, False,  True, False,  True], [ True,  True, False,  True,  True,  True], [ True, False, False, False, False, False], [ True,  True, False, False, False,  True], [ True,  True,  True,  True,  True,  True], [ True, False, False, False, False, False], [ True, False, False, False, False,  True], [ True,  True,  True, False,  True,  True], [ True,  True, False,  True,  True, False], [ True, False, False, False, False, False], [False, False,  True, False, False, False], [False,  True,  True,  True, False, False], [False,  True, False,  True, False, False], [False, False, False, False, False, False]])))
#print(count_rec_correct_way(np.array([[True,True,True,True,True,True],[True,True,True,True,True,True],[True,True,True,True,True,True],[True,True,True,True,True,True],[True,True,True,True,True,True],[True,True,True,True,True,True],[True,True,True,True,True,True],[True,True,True,True,True,True],[True,True,True,True,True,True],[True,True,True,True,True,True],[True,True,True,True,True,True],[True,True,True,True,True,True],[True,True,True,True,True,True],[True,True,True,True,True,True],[True,True,True,True,True,True],[True,True,True,True,True,True],[True,True,True,True,True,True],[True,True,True,True,True,True]])))#print(quantify_correct_way(np.array([10,20,30,10,20,30,10,20,30,15,25,35,15,25,35,15,25,35]),np.array([10,10,10,20,20,20,30,30,30,15,15,15,25,25,25,35,35,35]),np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,23,33,32,27,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,17,14,22,23,26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])))

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

def kernal_parabel(A, input_x, center_x,w_x, input_y, center_y,w_y):
    
    return A* np.max((-(input_x-center_x)**2)/(w_x**2)+(-(input_y-center_y)**2)/(w_y**2)+1,0)

"""
def kernel_parabola(A, input_x, center_x, w_x, input_y, center_y):
    # Calculate the distance between each input coordinate and all the centers
    squared_distance_x = (input_x[:,:, np.newaxis] - center_x )**2
    squared_distance_y = (input_y[:,:,np.newaxis] - center_y)**2
    # Calculate the kernel values for each center
    kernel = A * np.maximum(-((squared_distance_x) / (w_x**2)) - ((squared_distance_y) / (w_x**2)) + 1, 0)
    # Sum the kernel values over all centers to get the final result
    result = np.sum(kernel, axis=2)

    return result
"""
#using this one, uses alot of time !!!
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
#should try this?
def potential_parabola(loc, mu_loc, width, amp):
    """
    square parabola with support defined by width,
    different extent for x-axis and y-axis
    """
    return amp*np.maximum(-np.sum((loc[:, :, None]-mu_loc[:, None, :])**2
                              /width[:, None, None]**2, axis=0) + 1, 0)

#print(potential_parabola(np.array([[[1,2,3],[1,2,3],[1,2,3]],[[1,1,1],[2,2,2],[3,3,3]]]),np.array([[[1]],[[1]]]),np.array[[[1]]],1))
"""
mesh = np.meshgrid(np.arange(100),np.arange(100))
save = kernel_parabola(1,mesh[0],np.array([40,80,60]),20,mesh[1],np.array([40,80,60]),20)
print(save.shape)
plt.imshow(save)
plt.show()

v1, v2= [3.4, 3.2],[6.8, 0]
n1, n2 = np.meshgrid(np.arange(1, 5.1, 1), np.arange(1, 5.1, 1))
Xgrid = (n1.reshape(1,25)*v1[0] + n2.reshape(1,25)*v2[0])
Ygrid = (n1.reshape(1,25)*v1[1] + n2.reshape(1,25)*v2[1])
print(Xgrid.shape,Ygrid)
horseshoe = create_starting_grid(np.array([0]),np.array([0]))
#print(horseshoe[0]+Xgrid)
plt.scatter(Xgrid +horseshoe[0],Ygrid+horseshoe[1],s=1,color = "b")
plt.scatter(Xgrid,Ygrid, s = 1, color= "r")
plt.show()


"""

def calc_closest_point_on_ellipse(a_ell, b_ell, point):
    """
    for a given point, calculate the closest point on the periphery
    of an ellipse with the two axes a_ell and b_ell
    - assume that the center of the ellipse is at (0, 0)
    """
    xr = np.sqrt(a_ell**2 * b_ell**2 / (b_ell**2 + a_ell**2*(point[:, :, 1]/point[:, :, 0])**2))
    yr = point[:, :, 1]/point[:, :, 0] * xr
    return np.sign(point[:, :, 0])*xr, np.sign(point[:, :, 1])*np.abs(yr)

def distance_to_exp(firstpos, pos_eval, Xint, Yint, v1int, v2int, mode='target_area'):
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
        plt.scatter(Xint,Yint)
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

def main():
#print(create_starting_grid(np.array([0]),np.array([0])))
#creating starting postions

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
    
    heels_desity = kernel_parabola(1, POS[0],heel_pos_x,radius_heels_avg,POS[1],heel_pos_y)
    fronts_desity = kernel_parabola(0.5, POS[0],heel_pos_x,radius_fronts_avg,POS[1],heel_pos_y)
    #min and max values for histogram given with range = [[xmin, xmax], [ymin, ymax]]
    #bins is the controll for how blurry the graph is
    #postitions = np.histogram2d(heel_pos_y,heel_pos_x, bins = 720, range = [[0, 720], [0, 720]])
    #heels_desity = signal.fftconvolve(postitions[0], kernel, mode='same')
    dat2_inter = heels_desity+fronts_desity
    if making_movie:
        plt.figure(dpi=300)
        ### xlim und ylim anpassen an neue Gridgröße
        plt.xlim(250,700)
        plt.ylim(170,600)
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
                               [82.29011450458808, 70.63559641125377, 69.51033837975177, 58.19852457593135]])
    roi_degree = np.radians(circ_width_all.mean(axis=1)) # angular width of the 'flashlight'
    roi_radius  = 25.2*np.array([4.41,3.6,5.15,3.58,4.48,4.74])
    n_fil = 10
    startangs_all = np.pi/180 * np.array([-140.6786, -64.3245, -17.25796667, 5.312072, 63.2865, 135.0751667])
    speeds = np.array([0.093, 0.053, 0.148, 0.09, 0.052, 0.077])
    s_steap = 0.75
    s_x_move = 25

    histogram_input = np.zeros((42,20))
    way_matrix_x = np.zeros((42*6,21))
    way_matrix_y = np.zeros((42*6,21))
    fil_matrix_x = np.zeros((42*6,10))
    fil_matrix_y = np.zeros((42*6,10))
    
    #time verändert von 20,40,1 zu 20
    for time in np.arange(20):
        s_time = s(time+20, s_steap, s_x_move) 
        if making_movie:
            plt.figure(dpi = 300)
            plt.imshow(dat2_inter,  interpolation='nearest',origin="lower",vmin = 3,vmax = 8)  
            plt.colorbar()     
        for R in np.arange(42*6):
            #stop the growth of the filopodia if mask is out of range of matrix
            if  way_matrix_x[R,time-1] < roi_radius[np.mod(R,6)] or way_matrix_x[R,time-1] > rows- roi_radius[np.mod(R,6)] or way_matrix_y[R,time-1] < roi_radius[np.mod(R,6)] or way_matrix_y[R,time-1] > rows- roi_radius[np.mod(R,6)]:
                way_matrix_x[R,time] =way_matrix_x[R,time-1]
                way_matrix_y[R,time] =way_matrix_y[R,time-1]
                continue
            k_fil_scale = speeds[np.mod(R,6)]
            mask = np.zeros((rows, cols), dtype= bool) #mask_emtpy
            if time == 0:
                way_matrix_x[R,time] = heel_pos_x[R]
                way_matrix_y[R,time] = heel_pos_y[R]
                front_x, front_y = way_matrix_x[R,time],way_matrix_y[R,time]
                angle = startangs_all[np.mod(R,6)]
            else:
                last_front_x,last_front_y, front_x, front_y = way_matrix_x[R,time-1],way_matrix_y[R,time-1], way_matrix_x[R,time],way_matrix_y[R,time]
                angle = find_degree(last_front_x,last_front_y, front_x, front_y)
            heel_x, heel_y = way_matrix_x[R,0],way_matrix_y[R,0]
            ind = creat_mask(angle, front_x, front_y, roi_radius[np.mod(R,6)], roi_degree[np.mod(R,6)], mask)
            if making_movie and R == 80:
                plt.imshow(ind,origin="lower",alpha=0.2)
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
                new_stiff_x, new_stiff_y = find_point_2degree(front_x, front_y,2*roi_radius[np.mod(R,6)]*np.sin(angle)*k_fil_scale/(3*angle), angle)

            else:
                new_stiff_x = ((-front_x + heel_x)/(-time)) + front_x
                new_stiff_y = ((-front_y + heel_y)/(-time)) + front_y

            way_matrix_x[R,time+1] = round( s_time * new_stiff_x + (1-s_time) * new_fil_x)
            way_matrix_y[R,time+1] = round( s_time * new_stiff_y + (1-s_time) * new_fil_y)
            if making_movie:
                plt.plot(way_matrix_x[R,:time+1],way_matrix_y[R,:time+1],color=["blue","green","red","yellow","pink","orange"][np.mod(R,6)])
                if np.mod(R,6) ==5:
                    plt.plot(way_matrix_x[R-5:R+1,0],way_matrix_y[R-5:R+1,0],color = "gray",zorder=0)
        #landscape doesnt work and plotting doesnt work, but at least it is quick!
        if making_movie:
            #plt.imshow(ind, alpha=0.2, cmap ="hot",interpolation='bilinear',origin="lower")
            plt.scatter(way_matrix_x[:,time+1],way_matrix_y[:,time+1], s = 1, color= "r")
            plt.scatter(way_matrix_x[:,0],way_matrix_y[:,0], s = 1,color = "black")
            plt.title(label = "Densitylandscape")
            plt.xlim(250,700)
            plt.ylim(170,600)
            plt.scatter(starting_pos_x, starting_pos_y,s=3,color="black")
            #plt.show()
            plt.savefig(f"{folder_path}{time+1}.png")
            plt.close()
        
        #kernel = np.outer(signal.windows.gaussian(70,100), signal.windows.gaussian(70, 100))
        #postitions = np.histogram2d(way_matrix_y[:,time+1],way_matrix_x[:,time+1], bins = 720, range = [[0, 720], [0, 720]])

        ## histogram ##
        #density_middel_point = dat2_inter[starting_pos_x, starting_pos_y]
        #plt.figure()
        #plt.hist(density_middel_point, bins=100) 
        #plt.show()
        #plt.close()

        #histogram_input[:,time] = density_middel_point
        #delete theses indexes from the input for the dis_to_ellipse function
        #density_middel_point = np.delete(density_middel_point,index_outofrange)

        fronts_desity = kernel_parabola(0.5,POS[0], way_matrix_x[:,time+1],radius_fronts_avg, POS[1],way_matrix_y[:,time+1])
        dat2_inter =heels_desity+ fronts_desity
        
    return way_matrix_x,way_matrix_y,starting_pos_x, starting_pos_y #, histogram_input

    
    """
    saveing = quantify_correct_way(starting_pos_x, starting_pos_y, way_matrix_x[:,20],way_matrix_y[:,20])
    print(saveing)
    count_rec_correct_way(saveing)"""

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
    folder_path = f"./modell_tanh_stiffness_full_funct_adjust_grid_size/"
    nr_of_rec = 42
    include_equator = False
    r3r4swap = False
    voronoi_matrix = np.zeros((14,18))
    for sweeps in range(10):
        
        way_matrix_x, way_matrix_y, grid_x, grid_y= main() 
        with open(f"{folder_path}test_{sweeps}.npy", 'w+b') as f:
            np.save(f, way_matrix_x)
            np.save(f, way_matrix_y)
            np.save(f, grid_x)
            np.save(f, grid_y)
    for sweeps in range(10):
        """with open(f"{folder_path}test_{sweeps}.npy", 'rb') as f:
        way_matrix_x = np.load(f)
        way_matrix_y = np.load(f)
        grid_x = np.load(f)
        grid_y = np.load(f)"""
        start = 0
        for receptor in np.arange(1,7):
            rec_index = np.arange(receptor-1,nr_of_rec*6,6)
            first_pos = np.array(list(zip(way_matrix_x[rec_index,0],way_matrix_y[rec_index,0])))
            last_pos = np.array(list(zip(way_matrix_x[rec_index,15],way_matrix_y[rec_index,15])))
            voronoi_results = distance_to_exp(first_pos,last_pos, grid_x, grid_y, v1, v2 ,"voronoi")
            if receptor==4:
                start =1
            voronoi_matrix[start::2,np.arange(np.mod(receptor-1,3),18,3)] += voronoi_results.astype(int).reshape(7,6)
        
    #fig, ax = plt.subplots()
    plt.figure(dpi = 300)
    plt.imshow(voronoi_matrix,origin="lower")
    
    plt.axhline(y = -0.5,linewidth=1, color='black')
    plt.axhline(y = 1.5,linewidth=1, color='black')
    plt.axhline(y = 3.5,linewidth=1, color='black')
    plt.axhline(y = 5.5,linewidth=1, color='black')
    plt.axhline(y = 7.5,linewidth=1, color='black')
    plt.axhline(y = 9.5,linewidth=1, color='black')
    plt.axhline(y = 11.5,linewidth=1, color='black')
    plt.axhline(y = 13.5,linewidth=1, color='black')
    plt.axvline(x = -0.5,linewidth=1, color='black')
    plt.axvline(x = 2.5,linewidth=1, color='black')
    plt.axvline(x = 5.5,linewidth=1, color='black')
    plt.axvline(x = 8.5,linewidth=1, color='black')
    plt.axvline(x = 11.5,linewidth=1, color='black')
    

    cmap = mpl.cm.viridis
    bounds = [0, 1, 2, 3, 4, 5,6,7,8,9,10]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    plt.title("Correct conected receptors out of 10 rounds")

    plt.savefig(f"{folder_path}voronoi_matrix.png")
    """#histogram_input = np.load(f)
        #print(histogram_input[:,14]>5.5)
        #plt.hist(histogram_input[:,14], bins=30) 
        #plt.show()
    #histogram
    #finding cut off?
    cutoff =1
    #finding index of values smaller then cutoff
    #index_outofrange =np.argwhere(density_middel_point<cutoff)
    #density_middel_point[index_outofrange] = 0
    
    #print("way_x",way_matrix_x, "way_y", way_matrix_y)
    
    print(voronoi_results)

    plt.scatter(grid_x, grid_y, c = ["green" if x  else "blue" for x in voronoi_results])
    plt.scatter(last_pos[:,0],last_pos[:,1],c = ["yellow" if x  else "orange" for x in voronoi_results])
    #plt.xlim(250,700)
    #plt.ylim(170,600)
    plt.show()
    #print("first_pos",first_pos,"last_pos", last_pos,"ell_a", ell_a, "ell_b",ell_b,"grid_x",grid_x, "grid_y",grid_y, "v1",v1, "v2",v2 )"""
    """profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats('dump_stats_4')"""