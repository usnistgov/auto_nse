from matplotlib.colors import LogNorm
from numpy.matlib import repmat

import copy as copy
import scipy.stats 
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd  
from sklearn.neighbors import KernelDensity


def say_hi():
    print("Hello ...")
    
#Notice it returns a list instead of a numpy array like in flatten function
def flatten_simple(a):
    """ Super flatten"""
    return [ia for alist in a for ia in alist]

def finding_max_min(list_of_lists):
    max = np.amax(flatten_simple(list_of_lists[0]))
    min =np.amin(flatten_simple(list_of_lists[0]))
    
    for elem in list_of_lists:
        max_local = np.amax(flatten_simple(elem))
        min_local = np.amax(flatten_simple(elem))
    
        if max_local > max:
            max = max_local
        if min_local < min:
            min = min_local
    
    return min, max
    
###############################################################################################################    
#First we define a helper function to plot Y-profiles from any apporach (on-the-fly, entropy, or control)
def plot_y_profiles(this_yprofs, this_x,FOM=None, this_title= "y profiles", this_x_label=r'$\phi$ $(\degree)$', 
                    this_y_label='y(x)', ax=None):
    """ 
    Inputs:
    this_yprofs: It is a yprofs array that contains samples of y's at each position of the measurement space.
                It is the yprofiles at a given iteration of the main loop.
                n_samples*m_places to measure. 
    this_x: measurement space        
    this_title: title of the plot
    this_x_label: label for x-axis, default is 'Measurement Space'
    this_y_label: label for y-axis, default is 'y(x)'
    FOM: an optional array that contains the FOM for the given Y-profs. It will be
          plotted as a separate  subplot with the same x-axis as the main plot.
          If it is not provided, simplily it won't be plotted
    
    Returns: None
    """
    #Entropy Approach! Do the same for On-the-fly approach 

    # Duplicate the x vector to be the same shape as the yprofs matrix
    X = np.tile(this_x, (this_yprofs.shape[0], 1))
    
    # Make a histogram with each value of x has its own bin, and 256 bins in the y direction
    y_profs_hist, x_edges, y_edges = np.histogram2d(X.flatten(), this_yprofs.flatten(), bins=[X.shape[1], 256])
    
    # Choose a color map for the histogram 
    cmap = copy.copy(plt.cm.plasma)
    
    # Choose how to color "bad" values of the histogram 
    cmap.set_bad(cmap(0))
    
    # Set up the figure and plot the main subplot (histogram)
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()
    
    ax.pcolormesh(x_edges, y_edges, y_profs_hist.T, cmap=cmap,
                  rasterized=False, norm=LogNorm(vmax=np.max(y_profs_hist)))
    ax.set_facecolor(cmap(0))
    #ax.set_xticks(np.arange(min(this_x), max(this_x)+1, 120.0))
    ax.set_xlabel(this_x_label)
    ax.set_ylabel(this_y_label)
    ax.set_title(this_title)
    
    # Add a colorbar
    cbar = fig.colorbar(ax.get_children()[0], ax=ax)
    cbar.set_label("# points")
    
    # Show the histogram plot
    #plt.show()

    fig2, ax2 = None, None
    # Plot the line plot, if array2 is provided
    if FOM is not None:
        fig2, ax2 = plt.subplots(figsize=(10, 10))
        ax2.plot(this_x, FOM, color='black')
        ax2.set_xlabel(this_x_label, color = "C0")
        ax2.set_ylabel('Figure of Merit', color='C0')
        ax2.set_title('Figure of Merits for each x')
        ax2.tick_params(axis='y', labelcolor='C0')

        # Show the line plot
        #plt.show()

    return fig, ax, fig2, ax2


###############################################################################################################
    
    
"""
Getting the Lists for the Parameters
The containers total_pts (and similars) stores 2-D arrays on them; each 2-d array represents an iteration. The 2-d array contain the samples for all the parameters n_samples*n_parameters (where the number of samples can change per iteration, but the number of columns is constant). First,based on the columns of each array (each column represents a parameter) we will create lists that represent each parameter where each sub-list inside the list for a given paremeter represents the samples at an iterations, n_iterations*n_samples

Note: 
1) Any of the 2-d arays in this_total_pts and this_total_pts2 have as columns representing each parameter. The columns represent respectvily A, I0, T,and phi0. They are in the same order than in the console output from the loop.                  

2) We Cannot create 2D arrays for each parameter because the number of samples per iteration can change. Thus, we create lists for each parameter. 

"""

def getting_list_each_par(this_total_pts):
    
    """ From a list of of 2-d arrays with the same number of columns or parameters(Otherwise this gives an UNEXPECTED OUTPUT) 
    
        Inputs:
        this_total_pts -- It is a list of of 2-d arrays with the same number of columns but the number of rows can be different.
                        Because We use in the main loop the function mark_outliers() that rules out outliers samples, the number 
                        of rows can change. Thus, we cannot create a 2-d array for each parameter. So, we chose to create a list.
                        Note: We could create a ragged 2-d array, but it is not recomended for future versions
                        
                        Assumpitons: We do not assume a specific number of parameters, but we assume the number of parameters is 
                        the same for all iterations. 
                        
                        example:
                          iter1 is a 2d array
                          [ iter1[n_samples1*n_par], iter2[n_samples2*n_par],..      ]
        
        Returns:
        list_par_separated -- a list of lists where each sublist contains elements from a given column 
                                example the first sublist contains all the first columns of the input list as sub-sub-lists 
                                
                                [par1[ [iter1], [iter2]...],   par2[ [iter1], [iter2]...], ...par_m[...]]
    """
    
    #remember to erase the first terms of the total_pts 
    n_iters = len(this_total_pts)
    numb_par = this_total_pts[0].shape[1] #We assume all the 2-d arrays have the same nuber of coulmns (parameters)
    
    list_par_separated = [ [] for _ in range(numb_par) ] # each sublist represent a parameter. 
                                                    #inside each sub-sub list represents the sample of a given parameter 
                                                    #at a given iteration 
    for i in range(n_iters):
        for j in range(numb_par):
            list_par_separated[j].append(list(this_total_pts[i][:,j]))
                  
    return list_par_separated
    #each row in one of the 2-d arrays inside list_par_separated is a sample of a given parameter at iteraion = row (omitting the 
    #initial guess) 
    
    
    
"""
Here we create an auxiliary funtion to plot each of the sublists (that represent the samples at all the iterations for a given parameter) of the output of using the function above. 

WARNING: Remember that the number of samples per iteration can change. Thus, a histogram may not be the best way to visualize converge. we have two options. 
1. Normalize the number of samples for al iterations.
2. commnet out the line of code mark_outliers() in the main loop. So, we do not rule outliers and then we will have the same number of samples for all iterations. 
"""

def plotting_hist(list_for_par, name_approach, name_par, y_min, y_max, ax=None):
    
    """ 
        an auxiliary funtion to plot each of the arrays in the parameters
        Inputs:
        list_for_par -- a list that contains sublists n_iterations*n_samples (the number of samples can change) 
                          par1[ [iter1], [iter2]...]
        name_approach -- an string representing the name of the apporach
        name_par -- an string representing the name of the parameter 
        y_min: the minimum value for the y axis in the histogram
        y_max: the maximin value for the y axis in the histogram
        
        Returns:
        None. It prints a picture 
    """
    #############
    iter = len(list_for_par)
    indexes_iter_list = [] #this list will have the correspoinding iterations numbers reapeted a number of times equal to the 
                            #length of the list represnting that iterations in the list_for_par list of lists 
    #filling the list 
    
    for i in range(iter):
        indexes_iter_list = indexes_iter_list + ([i]*len(list_for_par[i]))
    

    #Use the same number of bins for all histograms, but adjust the y-axis limits
    n_bins = 300
    y_edges = np.linspace(y_min, y_max, n_bins)
    y_range = y_max - y_min
    
    #Make a histogram with each value of x has it's own bin, and 256 bins in the y direction
    pars_hist, x_edges, y_edges = np.histogram2d(indexes_iter_list, flatten_simple(list_for_par), bins=[iter, y_edges])
    
    #Choose a color map for the histogram 
    cmap = copy.copy(plt.cm.plasma)
    #Choose how to color "bad" values of the histogram 
    cmap.set_bad(cmap(0))

    #Plot the 2D histogram:
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(10,10))
    else:
        fig = ax.get_figure()
        
    ax.pcolormesh(x_edges, y_edges, pars_hist.T, cmap=cmap,
                   rasterized=False, norm=LogNorm(vmax=np.max(pars_hist)))
    plt.colorbar(label="# points", ax=ax)
    ax.set_xticks(np.arange(0,iter))
    ax.set_title("Histogram of Parameter: "+ name_par+" "+"-" + name_approach+" approach" )
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Units of parameter')
    
    #Adjust the y-axis limits for all histograms
    #plt.ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    return fig, ax


#DEPRECATED VERSION
# def plotting_hist(list_for_par, name_approach, name_par):
    
#     """ 
#         an auxiliary funtion to plot each of the arrays in the parameters
#         Inputs:
#         list_for_par -- a list that contains sublists n_iterations*n_samples (the number of samples can change) 
#         name_approach -- an string representing the name of the apporach
#         name_par -- an string representing the name of the parameter 
        
#         Returns:
#         None. It prints a picture 
#     """
#     #############
#     iter = len(list_for_par)
#     indexes_iter_list = [] #this list will have the correspoinding iterations numbers reapeted a number of times equal to the 
#                             #length of the list represnting that iterations in the list_for_par list of lists 
#     #filling the list 
    
#     for i in range(iter):
#         indexes_iter_list = indexes_iter_list + ([i]*len(list_for_par[i]))
    
#     #Make a histogram with each value of x has it's own bin, and 256 bins in the y direction
#     pars_hist, x_edges, y_edges = np.histogram2d(indexes_iter_list, flatten_simple(list_for_par), bins=[iter, 300])
    
#     #Choose a color map for the histogram 
#     cmap = plt.cm.plasma.copy()
#     #Choose how to color "bad" values of the histogram 
#     cmap.set_bad(cmap(0))

#     #Plot the 2D histogram:
#     plt.figure(figsize=(10,10))
        
#     plt.pcolormesh(x_edges, y_edges, pars_hist.T, cmap=cmap,
#                    rasterized=False, norm=LogNorm(vmax=np.max(pars_hist)))
#     plt.colorbar(label="# points")
#     plt.xticks(np.arange(0,iter))
#     plt.title("Histogram of Parameter: "+ name_par+" "+"-" + name_approach+" approach" )
    
#     plt.xlabel('Iteration')
#     plt.ylabel('Units of parameter')
    
#     plt.show()
#################################################################################################################################
def plotting_hist_logtime(list_for_par, name_approach, name_par, y_min, y_max, list_times, ax=None):
    
    """ 
        an auxiliary function to plot each of the arrays in the parameters similar to plotting_hist, but now the x axis(time) will be
        in log scale
        Inputs:
        list_for_par -- a list that contains sublists n_iterations*n_samples (the number of samples can change) 
                        par[iter1[], iter2[], ...iter_n[] ]
        name_approach -- an string representing the name of the approach
        name_par -- an string representing the name of the parameter 
        y_min: the minimum value for the y axis in the histogram
        y_max: the maximin value for the y axis in the histogram
        list_times: a list of cumulative times for each iteration 
                        [time1,...time_n]
        
        Warning the length of list_for_par(# of sublists) and list_times must be the same
        Returns:
        None. It prints a picture 
    """
    #############
    iter = len(list_for_par)
    
    
    #indexes_iter_list = []
    #for i in range(iter):
    #    indexes_iter_list = indexes_iter_list + ([i]*len(list_for_par[i]))
    
    #Ask: I am not sure if this is the desire code given that we will deal with cumulative time
    logt = np.log10(list_times) #32
    edges = 0.5 * (logt[:-1] + logt[1:]) #31
    left_edge = logt[0] - 0.5 * (logt[1] - logt[0])
    right_edge = logt[-1] + 0.5 * (logt[-1] - logt[-2])
    x_edges = np.concatenate(([left_edge], edges, [right_edge]))
    
    
    #print("my x edges")
    #print(x_edges)
    
    indexes_iter_list = []
    
    for i in range(iter):
        indexes_iter_list = indexes_iter_list + ([logt[i]]*len(list_for_par[i]))
    
    
    # Use the same number of bins for all histograms, but adjust the y-axis limits
    n_bins = 300
    y_edges = np.linspace(y_min, y_max, n_bins)
    y_range = y_max - y_min
    
    # Flatten the list of parameter values
    flat_pars = flatten_simple(list_for_par)
    
    
    #CHANGES HERE AS WELL SO IT WORKS WHEN X IS TIME 
    # Make a histogram with each value of x has it's own bin, and 256 bins in the y direction
    pars_hist, _, _ = np.histogram2d(indexes_iter_list, flat_pars, bins=[x_edges, y_edges])
    
    # Choose a color map for the histogram 
    cmap = copy.copy(plt.cm.plasma)
    # Choose how to color "bad" values of the histogram 
    cmap.set_bad(cmap(0))

    # Plot the 2D histogram:
    if not ax:
        fig, ax = plt.subplots(figsize=(10,10))
    else:
        fig = ax.get_figure()
        
    pcm = ax.pcolormesh(x_edges, y_edges, pars_hist.T, cmap=cmap,
                   rasterized=False, norm=LogNorm(vmax=np.max(pars_hist)))
    fig.colorbar(pcm, ax=ax)
    
    # Update the x-axis label to reflect the time difference instead of the iteration number
    ax.set_xlabel('Log10(Time (s))')
    ax.set_ylabel('Parameter units')
    
    #ax.set_title("Histogram of Parameter: "+ name_par+" "+"-" + name_approach+" approach", size='large')
    ax.set_title(name_par)
    
    # Adjust the y-axis limits for all histograms
    #plt.ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    return fig, ax


#Example of Histogram Log

## specify y-edges for all three histograms
#y_min, y_max = MyPlots.finding_max_min([list_par_separated_e1[0]]) 
#plotting_hist(list_par_separated_e1[0], "Entropy 1 (Selected) MVN", "A", y_min, y_max, exp_entropy1.totaltimes())





##############################################################################################
#Function for plottting Entropies and time 
def plot_entropy_times(list_times, list_entropies, list_names, this_title='', yaxis_title='Entropy (bits)', ax=None):
    """   
        WARNING: 1)we assume the arguments have the same length, i.e, the number of methods for the 
                    times and entropies is the same 
                2) we assume each element in a list correponds with the one in the same positiion
                ecample: list_times[0] and list_entropies[0] represent the times and entropies for a given method and the are
                        in order. Each time in a sublist correponds to a ntropy in its correposing sublist
                3)Usually we put control data at the end of the lists 
        Inputs:
        list_times -- a list that contains sublist of times times for each of the methods to estimate the parameter(s). 
        We suggest to use the total time = meas_times+mov_time for each of the methods
        
        list_entropies -- a list of sublist containing the entropies for each of the methods 
        list_names -- a list that contains a name fro each of the methhods to get each of the suublists in list_entropies
        this_tile -- string title for the plot
        Returns:
        
    """
    n_methods = len(list_times)
    
    if not ax:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        fig = ax.get_figure()

    plt.rcParams['axes.facecolor'] = '#f7f7f7'
    plt.rcParams['axes.edgecolor'] = '#f0f0f0'
    plt.rcParams['axes.edgecolor'] = '0.1'
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.color'] = '#444444'
    plt.rcParams['ytick.color'] = '#444444'

    for i in range(n_methods):
        ax.semilogx(list_times[i], list_entropies[i], color='C'+str(i), label= list_names[i])
    
    if False:
        n_slope = 10 #20 
            
        for i in range(n_methods):
            if len(list_times[i]) > n_slope: #if the number of iterations is greater than n_slope, we make a linear
                pall = np.polyfit(np.log(list_times[i])[n_slope:], list_entropies[i][n_slope:], 1)
                ax.plot(list_times[i], np.polyval(pall, np.log(list_times[i])), color='C'+str(i), linestyle='--') 
        #Rememeber c0 = Azul, C1 = Verde, C2 = orange

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(yaxis_title)
    ax.set_title(this_title)
    ax.tick_params(axis='both', which='major')
    
    ax.legend(loc="best")
    
    
    # Customize plot appearance 
    #Delete this section if you do not want the grids in the background 
    ax.grid(True, which='both', alpha=0.5)
    ax.tick_params(axis='both', which='both', length=0)
    
    #fig.tight_layout()
    #########################
    
    return ax
        

##############################################################################################
"""
Note: In my opinion, the estimator_avg_at causes an error in some cases bc at some point of the iteration the values are 
concentrate around the ground truth. Thus, there are no many pints to take on the left or right of the interval. So the 
checking below alert us!

    if (left_value_index < 0) or (right_value_index > (NUMBER_SAMPLES - 1)): Alert! 
"""


# This function calculates the pdf estimator at a given ground truth
#For more details in how to calculate this estimator, read the Approximation AVG jupyter notebook


def estimator_avg_at(datay, ground_y,  SIZE_SAMPLE = 10):
    
    """ Function for calculating the estimator for the pdf given some data
        Inputs:
        datay -- array of Y random variable realizations. For this project, datay is a column of a yprofs array or the total_pts
                    Thus, datay contains the infromation for only one characteristic(y at a position or a given parameter)
                    Array_like 1*n_samples
                    
        ground_y-- the ground truth value where we want to estimate the pdf
                    scalar (only one number)
                    
        size_sample -- the size of the points we will use to compute the average of the variation at a given ground
                        y truth value. Greater or equal to 1.
                        Scalar
                        
        Return:
        pdf_approx -- the approximation for the pdf's at the given ground truth
                        scalar
        
    """
    # We will try to estimate the pdf of y at each measurement. 



    #Remember in this project, we are using 
    #WITH len(datay) = 800 or 840
    NUMBER_SAMPLES = len(datay) 

    #Sorting values for random varibale
    ordered_y = np.sort(datay)
    
    #We find the two closest points to each ground_xs to estimate the pdf
    left_values_index = np.searchsorted(ordered_y, ground_y, 'left') 
    
    right_value_index = left_values_index + SIZE_SAMPLE -1 
    
    #This is the right value of the left indexes. 
    left_value_index = left_values_index- SIZE_SAMPLE
    
    #Checking we have enough realiaztions of y on the right and left of the ground truth
    if (left_value_index < 0) or (right_value_index > (NUMBER_SAMPLES - 1)):
        print("ERROR: the ground truth to estimate is too close to one extreme. So, we do not have enought samples either"
             +"on the left or right of the ground truth")
        print("This is the left index")
        print(left_value_index)
        print("this is rigth index")
        print(right_value_index)
        
        
    
    intervals = ordered_y[left_value_index:right_value_index+1] #we add one on the right bc Python takes until index_right-1
        
    average_delta = np.mean(np.diff(intervals))
    
    #Now, we estimate the pdf for the ground_xs with the formula derived during the discussion 
    pdf_approx = 1/(average_delta*NUMBER_SAMPLES) #approximations for the pdf's in the ground_xs 

    return pdf_approx



#Notice that in (compute_likelihoods func )this case we are assuming implicitly that the columns of each list inside in 
#this_total_y_porfiles are independent. For instance, if we are working with total_pts, the this function assumes that at
#a given iteration the parameters are 
#independent 

def compute_likelihoods(this_total_y_profiles, this_truey, this_size_sample = 10):
    
    """ Function for calculating the sum of the log-likelihoods for all ground true values in this_truey based on 
        this_total_y_profiles, an array of arrays with the shape described in the return of the sampler function.
        This function works with (total_y_prof, true_ys) or (total_pts, true_arranged_par) like tuples 
        
        Inputs:
        This_total_y_profiles -- list of arrays of the shape N(number of samples)* n_features_of interest (y at a given place
                                    or parameters). 
                                    feature1 is a column representing the first feature
                                    [iteration_1[feture1, feature2,...feeature_j],...  ,iteration_i[feature1, feature2,...]]
        
        this_true_y-- Array containing the true values of the fetures of interst
                        true_feataure_x is an scalar
                        [true_feature1,...true_feature_j]
        
        size_sample -- the size of the points we will use to compute the average of the variation at a given ground
                        y truth value. Greater or equal to 1. It will be used as input for the estimator_avg_at function
                        scalar

        Return:
        all_total_likelihoods -- the approximation for the sum of likelihoods at the ground truth for each iteration 
                              [iteration_1_likelihoood,...., iteration_j_likelihood]  
        
    """
    #To store the sum of log-likelihoods at each y-profiles
    all_total_likelihoods = []
    
    iterations = len(this_total_y_profiles)
    #Iterating thought the array of y-profiles
    for i in range(iterations):
        _ , columns = this_total_y_profiles[i].shape #remember each column of total_y_profiles[i] has the y-profiles at x = column
        sum_of_loglikelihoods = 0
        
        #given a y-profile, we are iterating throught its columns 
        for col in range(columns):
            my_column = this_total_y_profiles[i][:, col]
            #print("I am in "+str(i)+ "element of total y profiles in col "+ str(col))
            
            pdf_at = estimator_avg_at(my_column, this_truey[col],  SIZE_SAMPLE = this_size_sample)
            sum_of_loglikelihoods += math.log(pdf_at)
        
        all_total_likelihoods.append(sum_of_loglikelihoods)
        
    return all_total_likelihoods

    
"""
Notes likelihood_helper_kde:

1. Using KDE to estimate the pdf. Notice that KDE is a non-parametric estimator. However, it assumes independent samples; 
this could be a problem. Remember each sample is measurememnt point in the main lopp is chosen based in the previous results. 

2. Ask about the bandwidth for this function. In the next functions, we use a kde in a single feature (a parameter or a y at a 
given measurement place); so we can use the std as the bandwidth. We use the std bc all the other methods s.a. silverman or 
scott gave us terrible estimators.
In this case, I am using "silverman". Since it is a multidimesonal pdf I cannot use just the std. Should I tried to use 
something smilar to the std?  
"""

def likelihood_helper_kde(this_total_yprofiles, this_true_ys):
    
    """ Function for calling recursively the KerndelDensityFunction througth an array of arrays with the shape 
        n_samples X n_features such as one element of the y-profiles. Then it gets the loglikelihood at each of 
        these inner arrays. 
        It can be used with (total_yporfiles, true_y) or (total pts, arranged_true_par) 
        Inputs:
        This_total_y_profiles -- array of arrays of the shape n_samples* n_dimensions (each dimension is a feature like parame) 
                                You can use this with tota_pts, total_yprofs or similars 
        
        this_true_y-- Array containing the true values of the features. 1-D Notice we assume is in order in the sense that 
                    the first element in this_true_y represents the true value for the first columns in the 2-d arrays inside 
                    this_total_yprofiles
        
        I was also thinking of including some parameters that are use in the KDE function as the arguments  
        
        Return:
        my_log_likeelihoods -- A list of the same sizze of this_total_y_profiles.
                                the approximation for the log-likelihood of finding the given true_y values at a givent pdf
                                 we estimate using each of the elements in this_total_yprofiles 
        
    """
    my_log_likelihoods = []
    number_arrays = len(this_total_yprofiles)
    
    for i in range(number_arrays):
        #we are  using a default bandwidth called "silverman" to compute automatically the baneidth. Another opt. is "scott"
        kde = KernelDensity( bandwidth = "silverman", kernel='gaussian').fit(this_total_yprofiles[i])
        log_density = kde.score_samples(this_true_ys.reshape(1,-1)) #we reshpae so it accepts the array. It has to be an array 
                                                                    #arrays
        my_log_likelihoods.append(log_density)
    
    return my_log_likelihoods













####################################################################################################
#Plotting function for likelihoods vs iterations

def plot_likelihood_iterations(rows, cols, likelihoods_list, titles_list):
    """ 
        Inputs:
        rows -- number of rows of the final plot 
        cols -- number of columns of the final plot  
        likelihoods_list -- a list that contains sublists likelihoods for each method
                        example:
                        likelihoods_list[entropy_likelihoods[iter1, iter2...itern], control_likelihoods[iter1,iter2...itern]...]
        titles_list-- a list that contains a title for each of the subplots, each of the plots of likelihoods vs iteration
        
        Note:
        You can plot one columns or 1 row plots or 2-D array of subplots 
        If it is a 2-d array of plots, the elements in the list will start filling row by row
        
        len(likelihoods_list) = len(titles_list)
        rows*cols = len(likelihhos_list)
        the inputs are parallel lists 
        
        Return:
        A plot with subplots 
    """
    n_subplots = rows*cols
    plt.rcParams["figure.autolayout"] = True
    fig, subplots = plt.subplots(rows, cols, sharex=True, sharey = True , gridspec_kw={'hspace': 0}, figsize=(15, 10))
    
    flat_subplots = subplots.flatten() #flattening the subplots in case we have a 2-d subplots (not just a row or a column)
    
    for i in range(n_subplots):
        #plt.rcParams["figure.autolayout"] = True
        flat_subplots[i].plot(likelihoods_list[i])
        plt.tight_layout()
        flat_subplots[i].set_ylabel('Log likelihoods')
        flat_subplots[i].set_xlabel('Iteration')
        flat_subplots[i].set_title(titles_list[i])
        plt.tight_layout()
        
    plt.tight_layout()

######################################################################################################################################

#Another plotting function 
#noTICE YOU MAY want to delete the first time that is for the random point So number iterations = number of times for a method
def plot_likelihood_time(rows, cols, likelihoods_list, times_list, titles_list):
    """ 
        Inputs:
        rows -- number of rows of the final plot 
        cols -- number of columns of the final plot  
        likelihodds_list -- a list that contains sublists likelihoods for each method
                        example:
                        likelihoods_list[entropy_likelihoods[iter1, iter2...itern],  control_likelihoods[iter1,iter2...itern]...]
                        
        times_list -- a list that contain sublist for each of the methods. This will be the cumulative time 
                      for an autonomous approach and the total time spend in one iteration for control data
                        
        titles_list-- a list that contains a title for each of the subplots, each of the plots of likelihoods vs iteration
        
        Note:
        You can plot one columns or 1 row plots or 2-D array of subplots 
        If it is a 2-d array of plots, the elements in the list will start filling row by row
        
        len(likelihoods_list) = len(titles_list)= len(times(list)) = number_iteration in the main loop 
        rows*cols = len(likelihhos_list)
        the inputs are parallel lists 
        
        Return:
        A plot with subplots 
    """
    
    n_subplots = rows*cols
    #plt.rcParams["figure.autolayout"] = True
    fig, subplots = plt.subplots(rows, cols, sharex=True, sharey = True , gridspec_kw={'hspace': 0}, figsize=(12, 7))
    
    flat_subplots = subplots.flatten() #flattening the subplots in case we have a 2-d subplots (not just a row or a column)
    
    for i in range(n_subplots):
        #plt.rcParams["figure.autolayout"] = True
        flat_subplots[i].plot(times_list[i] ,likelihoods_list[i])
        plt.tight_layout()
        flat_subplots[i].set_ylabel('Log likelihoods')
        flat_subplots[i].set_xlabel('Cumulative Time')
        flat_subplots[i].set_title(titles_list[i])
        plt.tight_layout()
        
    plt.tight_layout()
    
    
################################################################################################################################

#First, we define an auxiliary function to compute the kde's of each parameter at a given iteration 
#generaluze later the algorithm to work also with the total_y_profiles 

#Note we use the std of each feature instead of the other methods like silverman bc when we tried to use it for total_pts 
#it did not work as well as with the std

#2, as it is this will work only with total_pts or total_pts2 or any list like that for only parameters 
def kdes_and_loglikelihoods_for_pars(this_total_pts, this_arranged_true_pars):
    
    """ Function for getting the kdes of each parameter at a given iteration. It estimates 1-d kde using the data of 
        only one parameter. From those kde's we estimate the log_likelihoodsof of getting the true parameters
        
        Inputs:
        This_total_pts -- array of arrays of the shape n_samlples* n_dimensions similar to the containers of total_pts 
                          
        this_arranged_parameters -- an array having the the parameters. We assume they have the same order of the columns in 
                                    the elements of this_total_pts
                                    Error: we assume that the number of parameters is the same for all the iterations,
                                    elements of this_total_pts
        
        Return:
        kdes -- a 2-d array, n_iterations_main_loop * n_parameters. Thus element at (i,j) is the kde for iteration i and 
                paramter j
        log_likelihoods_true_pars -- the loglikelihoods of finding the true parameters for each paremeter
                                        2d array n_iteratiosn* n_pars
    """
    
    number_iterations = len(this_total_pts)
    
    #Error: here we assume all the arrays in the this_total_pts have the same number of paremeters(columns)
    _, numb_par = this_total_pts[1].shape
    
    
    #2-d array to store the kde 's for each parameter at a givem iteration 
    #example (i,j) is the kde in i iteration for parameter j 
    #row = Active_learning_loops columns = len(truepars)
    kdes = np.array([[None] * numb_par] * number_iterations)
                    
    for i in range(number_iterations):
        for j in range(numb_par):
            kdes[i,j] = KernelDensity( bandwidth = np.std(this_total_pts[i][:,j]), kernel='gaussian').fit(this_total_pts[i][:,j].reshape(-1,1))
        
        
        
    ##Now we compute the loglikelihoods of seeing the true pars given the pdf we estimate 
    #Container for the log-likelihoods for each kde in the array above 

    log_likelihoods_true_pars = np.array([[None] * numb_par] *  number_iterations)


    for i in range(number_iterations):
        for j in range(numb_par):
            log_likelihoods_true_pars[i,j] = kdes[i,j].score(this_arranged_true_pars[j].reshape(1,1))
        
    return kdes, log_likelihoods_true_pars
                    
    
#######################################################################################################################################

def plot_hist_1d_kde(sample, kde, method_name, par_name, iter_int):
    """ 
        Inputs:
        sample_par -- a list that contains a sample   
        kde_for_par -- a kde object. Notice this kde is 1-d. Thus, it was built using only the data of the sample 
        method_name -- the name of the method used to get the sample
        par_name -- the name of the parameter 
        iter_int -- a integer that represents the iteration in the main loop that produced the sample 
            
        Note: 
        You can get a sample par from the getting_list_each_par and the kde from kdes_and_loglikelihoods function
        the pdf is plot from the smallest to the largest value in the sample
        
        Return:
        A plot that has a histogram of the sample and a 1D KDE plot
    """
    
    # sample space
    my_space = np.linspace(min(sample), max(sample), 100)
    
    # pdf evaluated at the sample space
    pdf_space = np.exp(kde.score_samples(my_space.reshape(-1, 1) ))#my_space.reshape(-1, 1)
    
    # mean and standard deviation of the sample
    sample_mean = np.mean(sample)
    sample_std = np.std(sample)
    
    # Plotting
    fig, ax = plt.subplots()
    ax.set_title(f"{method_name} Method, 1-D KDE for {par_name}\n(iteration {iter_int}), Sample Mean: {sample_mean:.4f}, Sample Std: {sample_std:.4f}")
    ax.hist(sample, bins="auto", alpha=0.5, density=True, label="Sample Histogram", color="lightgreen")
    ax.plot(my_space, pdf_space, color='r', label="1D KDE")
    ax.legend(loc="upper left")
    ax.set_xlabel(par_name)
    ax.set_ylabel("Density")
    plt.show()

    
#####################################################################################################################################


    



    
    