# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:55:07 2024

@author: Ilan
""" 
from scipy.stats import norm, lognorm, t, invgamma
import numpy as np 
from scipy.special import factorial, comb, beta, digamma
from scipy.special import gamma as gamma_func 

class err: 
    
    """ 
    Initalize a member of the error class. 
    default_option: string that initializes corresponding pre-defined error type 
    args: dictionary of args to provide if no default is provided 
    n: spare parameter to use in error distributions 
    """
    def __init__(self,default_option=None,args=None,n=None) :
        
        """ 
        Error distributions 
        phi: the pdf of outcome, xx. The action, aa, moves the pdf   
        dphi: derivative of pdf w.r.t action 
        y_grid_func: for numerical integration, tell comp where probability distribution mass is
        print_error: The name of the error function, used in print(prob) 
        expected_revenue: The principle's expected revenue as a function of a 
        error_type: says whether x is discrete or continuous 
        """ 
        
        # if args is given, use the args rather than using a default option 
        if args:
            self.phi = args['phi'] 
            self.dphi = args['dphi'] 
            self.print_error = args['print_error'] 
            self.y_grid_func = args['y_grid_func']
            self.expected_revenue = args['expected_revenue'] 
            self.error_type = args['error_type'] 
        
        if default_option=='norm': 
            if n is None: # use normal default for variance 
                n=1
            self.phi = lambda xx,aa: norm.pdf(xx,aa,n)
            self.dphi = lambda xx,aa: self.phi(xx,aa)*(xx - aa)/(n**2)
            self.y_grid_func = lambda aa,num_points: np.linspace(aa-5*n,aa+5*n,num_points) # etc 
            self.print_error = f"Normal, sigma: {n}" 
            self.expected_revenue = lambda aa: aa
            self.error_type = 'continuous'
            
        if default_option=='ln_norm':  
            if n is None: # use log normal default for variance 
                n=1
            self.phi = lambda xx, aa: lognorm.pdf(xx, s=n, scale=np.exp(aa)) 
            self.dphi = lambda xx, aa: self.phi(xx,aa)*(np.log(xx) - aa)/(n**2)
            self.y_grid_func = lambda aa,num_points: np.exp(np.linspace(aa-5*n,aa+5*n,num_points))
            self.print_error = "Log Normal" 
            self.expected_revenue = lambda aa: np.exp(aa+.5*n)
            self.error_type = 'continuous'  
            
        if default_option=='ln_norm_mult': 
            if n is None: 
                n=1
            self.phi = lambda xx, aa: lognorm.pdf(xx, s=n, scale=aa) 
            self.dphi = lambda xx, aa: self.phi(xx,aa)*(np.log(xx)-np.log(aa))/(aa*(n**2)) 
            self.y_grid_func = lambda aa,num_points: np.exp(np.linspace(np.log(aa)-5*n,np.log(aa)+5*n,num_points))
            self.print_error = "Multiplicative Log Normal" 
            self.expected_revenue = lambda aa: aa*np.exp(.5*n) 
            self.error_type = 'continuous'  
            
        if default_option=='poisson': 
            self.phi = lambda xx,aa: (aa**xx)*np.exp(-aa)/factorial(xx) 
            self.dphi = lambda xx, aa: self.phi(xx,aa)*(xx-aa)/aa 
            self.print_error = "Poisson" 
            self.y_grid_func = lambda aa,num_points: np.linspace(0,round(10*aa),round(10*aa)+1)
            self.expected_revenue = lambda aa: aa 
            self.error_type = 'discrete' 
            
        if default_option=='exponential': 
            self.phi = lambda xx,aa: (1/aa)*np.exp(-xx/aa) 
            self.dphi = lambda xx,aa: self.phi(xx,aa)*(xx-aa)/(aa**2)
            self.y_grid_func = lambda aa,num_points: np.linspace(.0001,20*aa,num_points)  
            self.print_error = "Exponential"   
            self.expected_revenue = lambda aa: aa 
            self.error_type = 'continuous'
        
        if default_option=='bernoulli': 
            self.phi = lambda xx,aa: (aa**xx)*(1-aa)**(1-xx)
            self.dphi = lambda xx,aa: self.phi(xx,aa)*(xx-aa)/(aa-aa**2) 
            self.print_error = "bernoulli" 
            self.y_grid_func = lambda aa,num_points: np.array([0,1]) 
            self.expected_revenue = lambda aa: aa 
            self.error_type = 'discrete'
            
        if default_option=='geometric': 
            self.phi = lambda xx, aa: (1-(1/aa))**(xx-1)*(1/aa)
            self.dphi = lambda xx, aa: self.phi(xx,aa)*(xx-aa)/(aa**2-aa) 
            self.print_error = "Geometric"
            self.y_grid_func = lambda aa,num_points: np.linspace(1,round(40*aa),round(40*aa))
            self.expected_revenue = lambda aa: aa 
            self.error_type = 'discrete'
            
        if default_option=='binomial':  
            if n is None: # use binomial default 
                n = 10 
            self.phi = lambda xx, aa: comb(n, xx) * (aa ** xx) * ((1 - aa) ** (n - xx))
            self.dphi = lambda xx, aa: self.phi(xx,aa)*(xx-n*aa)/(aa-aa**2)
            self.print_error = "Binomial"
            self.y_grid_func = lambda aa,num_points: np.linspace(0,n,n+1)
            self.expected_revenue = lambda aa: aa*n 
            self.error_type = 'discrete' 
            
        if default_option=='gamma':   
            
            EPSILON = 1e-12  # Smallest allowed probability to avoid numerical issues
            if n is None: # use gamma default 
                n = 2 
                
            """   
            self.phi = lambda xx, aa: np.maximum(
                (xx**(n - 1) * np.exp(-xx / aa)) / (gamma_func(n) * aa**n), EPSILON
            )  """
            
            def safe_phi(xx, aa, n):
                try:
                    result = np.where(
                        aa == 0, EPSILON,
                        np.maximum((xx**(n - 1) * np.exp(-xx / aa)) / (gamma_func(n) * aa**n), EPSILON)
                    )
                    return result
                except FloatingPointError as e:  # Catch divide-by-zero or invalid computations
                    print(f"FloatingPointError detected: {e}")
                    print(f"xx range: {xx[:5]} ... {xx[-5:]}")
                    print(f"aa: {aa}, n: {n}")
                    return np.full_like(xx, EPSILON)  # Return safe values instead of crashing
                
                return result
        
            self.phi = lambda xx, aa: safe_phi(xx, aa, n)
            
            self.dphi = lambda xx, aa: self.phi(xx, aa) * ((xx - aa*n) / aa**2) 
            self.print_error = "Gamma" 
            self.y_grid_func = lambda aa,num_points: np.linspace(.00001,10+aa+10*aa*n,num_points)  
            self.expected_revenue = lambda aa: aa*n 
            self.error_type = 'continuous'  
            
        if default_option=='inverse_gamma': 
            if n is None: # use inverse gamma default 
                n = 2 
            self.phi = lambda xx, aa: (
                    (aa**n / gamma_func(n))
                    * (xx ** (-(n + 1)))
                    * np.exp(-aa / xx)) 
            self.dphi = lambda xx, aa: self.phi(xx,aa)*(n*xx - aa)/(xx*aa) 
            self.print_error = "Inverse Gamma" 
            self.y_grid_func = lambda aa, num_points: np.linspace(
                0.00001, invgamma.ppf(0.9999, a=n, scale=aa), num_points)
            self.expected_revenue = lambda aa: aa/n-1
            self.error_type = 'continuous'  
            
        if default_option == 'beta':
            # Provide a default for 'n' if not given
            if n is None:
                n = 2
            self.phi = lambda xx, aa: (xx**(aa - 1) * (1 - xx)**(n - 1)) / beta(aa, n)      
            self.dphi = lambda xx, aa: self.phi(xx, aa) * (
                np.log(xx) - digamma(aa) + digamma(aa + n))
            self.print_error = "Beta" 
            self.y_grid_func = lambda aa, num_points: np.linspace(.0000001,1,num_points) 
            self.expected_revenue = lambda aa: aa/(aa+n)
            self.error_type = 'continuous'
            
        if default_option=='t': 
            if n is None: 
                n = {'df': 10, 'sigma': 1} 
            self.phi = lambda xx, aa: t.pdf(xx, n['df'], aa, n['sigma'])
            self.dphi = lambda xx, aa: self.phi(xx,aa)*(n['df']+1)*(xx-aa)/(n['df']*n['sigma']**2+(xx-aa)**2) 
            self.print_error = f"t, df: {n['df']}, sigma: {n['sigma']}" 
            self.y_grid_func = lambda aa,num_points: np.linspace(aa-12*n['sigma'],aa+12*n['sigma'],num_points) 
            self.expected_revenue = lambda aa: aa 
            self.error_type = 'continuous' 
            
        if default_option=='geometric_ce': 
            self.phi = lambda xx, aa: (aa)**(xx-1)*(1-aa)
            self.dphi = lambda xx, aa: (xx - 1) * (aa ** (xx - 2)) * (1 - aa) - (aa ** (xx - 1))
            self.print_error = "Geometric"
            self.y_grid_func = lambda aa,num_points: np.linspace(1,round(40/(1-aa)),round(40/(1-aa)))
            self.expected_revenue = lambda aa: 1/(1-aa)
            self.error_type = 'discrete' 
            