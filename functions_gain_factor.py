from datastruct import Settings, Experiment, ExperimentStep, DataPoint
from entropy import calc_entropy, default_entropy_options
import numpy as np

#Helper Function for comparing with GMM. We may move this to My_Plots.py
def recalculating_entropy(exp, this_select, this_options):
    """"
    Recalculates the entropy
    
    exp -> experiment object 
    this_select -> the list of parameters selected
    this_options -> a map that has the options for how to compute the entropy. The options paramter in the calc_entropy
                    function. 
                    example: gmm_setting= {'method': 'gmm', 'n_components': None}
                    
    """
    #We include a select bc usually we want to recalculate because all parameter were selected. 
    predictor = None
    entro1 = []
    for this_pts in exp.load_pts():
        H, _, predictor = calc_entropy(this_pts, select_pars=this_select, 
                            options= this_options, predictor=predictor)
        entro1.append(H)
        
    return np.array(entro1)

######################################################################################################




def find_last_factor_of_10_subset(times):
    """"
    times -> a list of  times 
    
    return:
        index_from -> index from all the values after that are greater or equal to the 
                      last power of 10. 
    
    WARNING we assume times is increasing. This will be true bc for the autonomous cases the times is 
    cumulative and for the control data we are increasing the time that is given at each iteration
    """
    #WARNING we assume times is increasing. This will be true bc for the autonomous cases the times is 
    #cumulative and for the control data we are increasing the time that is given at each iteration
    
    max_time = times[-1]
    
    greatest_factor_of_10 = max_time/10.0 #10 ** np.floor(np.log10(max_time))
    
    #We find the index of the first time that is greater to the 
    index_from = None
    for i in range(len(times)):
        if times[i] >= greatest_factor_of_10:
            index_from = i
            break
    
    return index_from
            
            
###################################################################################################################    

def getting_smaller(Hs, times):
    """
    gets a sublist of the Hs that is decreasing with their respective times
    """
    #Warning assumes len Hs is the same as times
    iter = len(Hs)
    if iter != len(times):
        print("ERROR THE LENGHT OF TIMES IS DIFFERENT TO THE ENTROPIES. THEY MUST HAVE THE SAME LENGTH")
        return None, None 
    
    result_Hs = []
    result_times = []
    
    min_seen = float('inf')
    
    for i in range(iter):
        if Hs[i] < min_seen:
            min_seen = Hs[i]
            result_Hs.append(Hs[i])
            result_times.append(times[i])
            
    return np.array(result_Hs), np.array(result_times)


def speedgain(t, H, tctrl, Hctrl):
    """
    This function computes the gain factor. Use as a reference the entropy v.s log(time). Given a level of entropy
    in the autonomous case. How long will it take to get the same level of entropy using control data 
    
    return:
    """
    
    if len(t) != len(H):
        print("Thre is an error in spped gain you are providing entropies and times of different length")
        
    if len(tctrl) != len(Hctrl):
        print("There is an error you are providing control entropies and control times of different length")
        
        
    # interpolate autonomous values to control values
    # note: only works if Hctrl is monotonically increasing
    
    Hctrl, tctrl = getting_smaller(Hctrl, tctrl)
    
    #We are actually interested in log-times. Remember the plot Entropy v.s log(time)
    #t = np.log10(t)
    #tctrl = np.log10(tctrl)
    
        
    #ti = np.interp(H, np.flip(Hctrl), np.flip(tctrl), left=np.nan, right=np.nan) #We flip the Hctrl bc the inter function
                                                                            #needs increasing values 
    
    ti = np.interp(-H, -Hctrl, tctrl, left=np.nan, right=np.nan) #We are using -Hctrl so it is incraesing. 
                                                                 #We also have to use -H in order to get the correct estimat
    
    # calculate ratio at each interpolated point
    rat = ti / t #This will be used as the gain factor. since ti-t is in log scale a substraction of times is 
                 #equivalent to the ratio

    # throw away anything that has a nan in it
    trat = t[~np.isnan(rat)]
    rat = rat[~np.isnan(rat)]
    
    return trat, rat 

 

# average the speed gain over all points within a factor of 10 of the maximum time


#############################EXAMPLE ##################
#EXAMPLE 
#determining the entropies and times of the autonomous approach that we have to use to compute the gain factor

##from_here = find_last_factor_of_10_subset(exp_entropy1.totaltimes())

#We did not include the find_last_factor... inside speed gain bc we want to give the option to the user to choose for 
#which values he/she wants to use 

##times, gain_factor = speedgain(exp_entropy1.totaltimes()[from_here:], exp_entropy1.entropy_marg()[from_here:], 
##                        exp_control.totaltimes(), exp_control.entropy_marg())


###########End Example#################################

##########################################################################################
def printing_speed_gain_info(times, entropies, control_times, control_entropies):
    """
    times -> a list of times for a experiment 
    entropies -> a list of entropies for a experiment
    
    """
    #determining the entropies and times of the autonomous approach that we have to use to compute the gain factor. 
    #We want to use only the last times, namely, the times greater than the lst power of 10. 
    from_here = find_last_factor_of_10_subset(times)
    
    #We did not include the find_last_factor... inside speed gain bc we want to give the option to the user to choose for 
    #which values he/she wants to use 

    times, gain_factors = speedgain(times[from_here:], entropies[from_here:], 
                        control_times, control_entropies)
    
    this_mean = np.mean(gain_factors)
    this_std = np.std(gain_factors)

    return len(gain_factors), this_mean, this_std
############################################################

##################Example###################


#aux_list = [exp_entropy1, exp_entropy2, exp_on_the_fly1, exp_on_the_fly2]

#titles = ["Entropy 1", "Entropy2", "On-the-fly 1", "On-the-fly 2", "Control"]
#print("For Total Entropy")
#for this_exp, this_title in zip(aux_list, titles):
#    printing_speed_gain_info(this_exp.totaltimes(), this_exp.entropy(),
#                            exp_control.totaltimes(), exp_control.entropy())

###############end Example#################











