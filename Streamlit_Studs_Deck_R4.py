import streamlit as st
import pandas as pd
from pickle import load
import joblib
import pickle
import numpy as np
import math
import os
from config.definitions import ROOT_DIR
from PIL import Image
import matplotlib.pyplot as plt

#Load model and scaler
NGBoost=joblib.load(os.path.join(ROOT_DIR,'Studs_Deck_NGBoost.joblib'))
NGBoost_sc=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Deck_NGBoost.pkl'),'rb'))
k_red_NGBoost_T=0.90
k_red_NGBoost_R=0.94

st.write('### Probabilistic Predictions of the Shear Resistance of Welded Studs in Deck Slab Ribs Transverse to Beams through Natural Gradient Boosting')

st.sidebar.header('User input parameters')

design_practice=st.sidebar.radio('Design Practice',('Europe','United States'))

if design_practice=='Europe':
    fck=st.sidebar.selectbox('$f_\mathrm{ck}$ (MPa)',(16,20,25,30,35,40,45,50,55,60))

    welding=st.sidebar.selectbox("Welding",('Through-deck','Through-holes'))

    if welding=='Through-deck': d=19
    else: d_min, d_max, d_step=(19,22,3)

    if welding=='Through-holes': d=st.sidebar.selectbox("$d$ (mm)",(19,22))    
    
    nr=st.sidebar.selectbox('$n_\mathrm{r}$',(1,2))
    
    if nr==2: stud_position=st.sidebar.selectbox("Stud position",('Two studs in parallel','Two studs in series','Two staggered studs'))
    else: stud_position="N/A"
    
    if stud_position=="Two studs in parallel": sy_min, sy_max, sy_step, sx=(math.ceil(2.6*d/1)*1.0, math.floor(10*d/1)*1.0, 1.0, 0.0)
    elif stud_position=="Two studs in series": sy, sx_min, sx_max, sx_step=(0.0, math.ceil(2.6*d/1)*1.0, math.floor(4.7*d/1)*1.0, 1.0)
    elif stud_position=="Two staggered studs": sy_min, sy_max, sy_step, sx_min, sx_max, sx_step=(math.ceil(2.6*d/1)*1.0, math.floor(10*d/1)*1.0, 1.0, math.ceil(2.6*d/1)*1.0, math.floor(4.7*d/1)*1.0, 1.0)

    if stud_position=="Two studs in parallel": sy=st.sidebar.slider("$s_\mathrm{y}$ (mm)",min_value=sy_min, max_value=sy_max, step=sy_step, format="%.0f")
    elif stud_position=="Two studs in series": sx=st.sidebar.slider("$s_\mathrm{x}$ (mm)",min_value=sx_min, max_value=sx_max, step=sx_step, format="%.0f") 
    elif stud_position=="Two staggered studs": sy, sx=(st.sidebar.slider("$s_\mathrm{y}$ (mm)",min_value=sy_min, max_value=sy_max, step=sy_step, format="%.0f"), st.sidebar.slider("$s_\mathrm{x}$ (mm)",min_value=sx_min, max_value=sx_max, step=sx_step, format="%.0f"))  
    if nr==1: sy, sx=(0.0, 0.0)    

    deck_type=st.sidebar.selectbox("Deck type",('Trapezoidal','Re-entrant'))
    
    if deck_type=='Trapezoidal': hpn_min, hpn_max, hpn_step=(38.0,136.0, 1.0)
    else: hpn_min, hpn_max, hpn_step=(51.0,56.0, 1.0)    
    hpn=st.sidebar.slider("$h_\mathrm{pn}$ (mm)",min_value=hpn_min, max_value=hpn_max, step=hpn_step, format="%.0f")
    
    if deck_type=='Trapezoidal': bbot_min, bbot_max, bbot_step=(math.ceil(max(40.0,sx+d)/1)*1.0, 160.0, 1.0)
    else: bbot_min, bbot_max, bbot_step=(137.0, 138.0, 1.0)
    bbot=st.sidebar.slider("$b_\mathrm{bot}$ (mm)",min_value=bbot_min, max_value=bbot_max, step=bbot_step, format="%.0f")
    
    if deck_type=='Trapezoidal': btop_min, btop_max, btop_step=(math.ceil(max(bbot/0.82,63.0,0.95*hpn-bbot+d)/1)*1.0, math.floor(min(bbot/0.29,240.0)/1)*1.0, 1.0)
    else: btop_min, btop_max, btop_step=(math.ceil(max(bbot/1.25,110.0,sx+d)/1)*1.0, math.floor(min(bbot/1.21,114.0)/1)*1.0, 1.0)
    btop=st.sidebar.slider("$b_\mathrm{top}$ (mm)",min_value=btop_min, max_value=btop_max, step=btop_step, format="%.0f")
    
    if deck_type=='Trapezoidal': t_min, t_max, t_step=(math.ceil(max(0.006*hpn,0.75)/0.01)*0.01, math.floor(min(0.03*hpn,1.52)/0.01)*0.01, 0.01)
    else: t_min, t_max, t_step=(math.ceil(max(0.015*hpn,0.75)/0.01)*0.01, math.floor(min(0.024*hpn,1.20)/0.01)*0.01, 0.01)
    t=st.sidebar.slider("$t$ (mm)",min_value=t_min, max_value=t_max, step=t_step, format="%.2f")
        
    if deck_type=='Trapezoidal': hsc_min, hsc_max, hsc_step=(math.ceil(max(3.9*d,1.58*d+hpn)/1)*1.0, math.floor(min(9.2*d,4.21*d+hpn)/1)*1.0, 1.0)
    else: hsc_min, hsc_max, hsc_step=(math.ceil(max(3.9*d,1.26*d+hpn)/1)*1.0, math.floor(min(6.4*d,3.49*d+hpn)/1)*1.0, 1.0)
    hsc=st.sidebar.slider("$h_\mathrm{sc}$ (mm)",min_value=hsc_min, max_value=hsc_max, step=hsc_step, format="%.0f") 
        
    if deck_type=='Trapezoidal': et_min, et_max, et_step=(math.ceil(max(0.46*hpn,31,(btop-bbot+d)/2)/1)*1.0, math.floor(min(2.73*hpn,154,(btop+bbot-d)/2)/1)*1.0, 1.0)
    else: et_min, et_max, et_step=(math.ceil(max(0.46*hpn,23)/1)*1.0, math.floor(min(1.78*hpn,91)/1)*1.0, 1.0)
    et=st.sidebar.slider("$e_\mathrm{t}$ (mm)",min_value=et_min, max_value=et_max, step=et_step, format="%.0f")
    
    fu_min, fu_max, fu_step=(400.0, 500.0, 25.0)
    fu=st.sidebar.slider("$f_\mathrm{u}$ (MPa)",min_value=fu_min, max_value=fu_max, step=fu_step, format="%.0f")
    


    data1 = {"fck (MPa)": "{:.0f}".format(fck),
            "Deck type": deck_type,
            "hpn (mm)": "{:.0f}".format(hpn),
            "bbot (mm)": "{:.0f}".format(bbot),
            "btop (mm)": "{:.0f}".format(btop),
            "t (mm)": "{:.2f}".format(t),
            "Welding": welding              
            }
    data2 = {"d (mm)": "{:.0f}".format(d),
            "hsc (mm)": "{:.0f}".format(hsc),
            "et (mm)": "{:.0f}".format(et),
            "fu (MPa)": "{:.0f}".format(fu),
            "nr": nr,
#            "Stud position": stud_position,
            "sy (mm)": "{:.0f}".format(sy),
            "sx (mm)": "{:.0f}".format(sx)                
            }                        
            
else:
    fprc_psi=st.sidebar.selectbox("f'c (psi)",(2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500 ))
    
    welding=st.sidebar.selectbox("Welding",('Through-deck','Through-holes'))
   
    if welding=='Through-deck': d_in=0.75
    else: d_in_min, d_in_max, d_in_step=(0.75,0.875,0.125)

    if welding=='Through-holes': d_in=st.sidebar.selectbox("d (in)",(0.75,0.875))

    nr=st.sidebar.selectbox('$n_\mathrm{r}$',(1,2)) 
    
    if nr==2: stud_position=st.sidebar.selectbox("Stud position",('Two studs in parallel','Two studs in series','Two staggered studs'))
    else: stud_position="N/A"
    
    if stud_position=="Two studs in parallel": sy_in_min, sy_in_max, sy_in_step, sx_in=(math.ceil(2.6*d_in/0.125)*0.125, math.floor(10*d_in/0.125)*0.125, 0.125, 0.0)
    elif stud_position=="Two studs in series": sy_in, sx_in_min, sx_in_max, sx_in_step=(0.0, math.ceil(2.6*d_in/0.125)*0.125, math.floor(4.7*d_in/0.125)*0.125, 0.125)
    elif stud_position=="Two staggered studs": sy_in_min, sy_in_max, sy_in_step, sx_in_min, sx_in_max, sx_in_step=(math.ceil(2.6*d_in/0.125)*0.125, math.floor(10*d_in/0.125)*0.125, 0.125, math.ceil(2.6*d_in/0.125)*0.125, math.floor(4.7*d_in/0.125)*0.125, 0.125)
    
    if stud_position=="Two studs in parallel": sy_in=st.sidebar.slider("$s_\mathrm{y}$ (in.)",min_value=sy_in_min, max_value=sy_in_max, step=sy_in_step, format="%.3f")
    elif stud_position=="Two studs in series": sx_in=st.sidebar.slider("$s_\mathrm{x}$ (in.)",min_value=sx_in_min, max_value=sx_in_max, step=sx_in_step, format="%.3f") 
    elif stud_position=="Two staggered studs": sy_in, sx_in=(st.sidebar.slider("$s_\mathrm{y}$ (in.)",min_value=sy_in_min, max_value=sy_in_max, step=sy_in_step, format="%.3f"), st.sidebar.slider("$s_\mathrm{x}$ (in.)",min_value=sx_in_min, max_value=sx_in_max, step=sx_in_step, format="%.3f"))  
    if nr==1: sy_in, sx_in=(0.0, 0.0)
    
    deck_type=st.sidebar.selectbox("Deck type",('Trapezoidal','Re-entrant'))    
 
    if deck_type=='Trapezoidal': hpn_in_min, hpn_in_max, hpn_in_step=(1.50,5.25, 0.125)
    else: hpn_in_min, hpn_in_max, hpn_in_step=(2.00,2.25, 0.125)    
    hpn_in=st.sidebar.slider("$h_\mathrm{pn}$ (in.)",min_value=hpn_in_min, max_value=hpn_in_max, step=hpn_in_step, format="%.3f")

    if deck_type=='Trapezoidal': bbot_in_min, bbot_in_max, bbot_in_step=(math.ceil(max(1.50,sx_in+d_in)/0.125)*0.125, 6.25, 0.125)
    else: bbot_in_min, bbot_in_max, bbot_in_step=(5.25, 5.5, 0.125)
    bbot_in=st.sidebar.slider("$b_\mathrm{bot}$ (in.)",min_value=bbot_in_min, max_value=bbot_in_max, step=bbot_in_step, format="%.3f")

    if deck_type=='Trapezoidal': btop_in_min, btop_in_max, btop_in_step=(math.ceil(max(bbot_in/0.82,2.5,0.95*hpn_in-bbot_in+d_in)/0.125)*0.125, math.floor(min(bbot_in/0.29,9.5)/0.125)*0.125, 0.125)
    else: btop_in_min, btop_in_max, btop_in_step=(math.ceil(max(bbot_in/1.25,4.25,sx_in+d_in)/0.0625)*0.0625, math.floor(min(bbot_in/1.21,4.5)/0.0625)*0.0625, 0.0625)
    btop_in=st.sidebar.slider("$b_\mathrm{top}$ (in.)",min_value=btop_in_min, max_value=btop_in_max, step=btop_in_step, format="%.4f")
  
    if deck_type=='Trapezoidal': t_in_min, t_in_max, t_in_step=(math.ceil(max(0.006*hpn_in,0.0295)/0.0001)*0.0001, math.floor(min(0.03*hpn_in,0.0598)/0.0001)*0.0001, 0.0001)
    else: t_in_min, t_in_max, t_in_step=(math.ceil(max(0.015*hpn_in,0.0295)/0.0001)*0.0001, math.floor(min(0.024*hpn_in,0.0474)/0.0001)*0.0001, 0.0001)
    t_in=st.sidebar.slider("$t$ (in.)",min_value=t_in_min, max_value=t_in_max, step=t_in_step, format="%.4f")
 
    if deck_type=='Trapezoidal': hsc_in_min, hsc_in_max, hsc_in_step=(math.ceil(max(3.9*d_in,1.58*d_in+hpn_in)/0.125)*0.125, math.floor(min(9.2*d_in,4.21*d_in+hpn_in)/0.125)*0.125, 0.125)
    else: hsc_in_min, hsc_in_max, hsc_in_step=(math.ceil(max(3.9*d_in,1.26*d_in+hpn_in)/0.125)*0.125, math.floor(min(6.4*d_in,3.49*d_in+hpn_in)/0.125)*0.125, 0.125)
    hsc_in=st.sidebar.slider("$h_\mathrm{sc}$ (in.)",min_value=hsc_in_min, max_value=hsc_in_max, step=hsc_in_step, format="%.3f") 
 
    if deck_type=='Trapezoidal': et_in_min, et_in_max, et_in_step=(math.ceil(max(0.46*hpn_in,1.25,(btop_in-bbot_in+d_in)/2)/0.0625)*0.0625, math.floor(min(2.73*hpn_in,6.0,(btop_in+bbot_in-d_in)/2)/0.0625)*0.0625, 0.0625)
    else: et_in_min, et_in_max, et_in_step=(math.ceil(max(0.46*hpn_in,0.875)/0.125)*0.125, math.floor(min(1.78*hpn_in,3.5)/0.125)*0.125, 0.125)
    et_in=st.sidebar.slider("$e_\mathrm{t}$ (in.)",min_value=et_in_min, max_value=et_in_max, step=et_in_step, format="%.4f")

    fu_ksi_min, fu_ksi_max, fu_ksi_step=(58.0, 72.0, 3.5)
    fu_ksi=st.sidebar.slider("$f_\mathrm{u}$ (ksi)",min_value=fu_ksi_min, max_value=fu_ksi_max, step=fu_ksi_step, format="%.1f")



    data1 = {"f'c (psi)": "{:.0f}".format(fprc_psi),
            "hpn (in)": "{:.2f}".format(hpn_in),
            "bbot (in)": "{:.2f}".format(bbot_in),
            "btop (in)": "{:.2f}".format(btop_in),
            "t (in)": "{:.4f}".format(t_in),
            "Welding": welding              
            }
    data2 = {"d (in)": "{:.3f}".format(d_in),
            "hsc (in)": "{:.2f}".format(hsc_in),
            "et (in)": "{:.3f}".format(et_in),
            "fu (ksi)": "{:.1f}".format(fu_ksi),
            "nr": nr,
#            "Stud position": stud_position,
            "sy (in)": "{:.3f}".format(sy_in),
            "sx (in)": "{:.3f}".format(sx_in)                
            }            

st.write('##### Geometric parameters of welded stud connections')
image1 = Image.open(os.path.join(ROOT_DIR,'Geom_R1.png'))
st.image(image1)

st.write('##### Positions of two studs in concrete ribs')
image2 = Image.open(os.path.join(ROOT_DIR,'Two_stud_positions.png'))
st.image(image2)

st.write('##### Specified input parameters')
            
if design_practice=='Europe':
  
    s_diag=((sy**2)+(sx**2))**0.5

    st.write('Characteristic cylinder compressive strength of the concrete with a density $\geq$ 1750 kg/m$^{3}$: $f_\mathrm{ck}$=', "{:.0f}".format(fck),' MPa')
    st.write('Deck type: ', deck_type)
    st.write('Deck depth excluding longitudinal stiffener on the crest: $h_\mathrm{pn}$=', "{:.0f}".format(hpn),' mm')
    st.write('Width of the concrete rib bottom: $b_\mathrm{bot}$=', "{:.0f}".format(bbot),' mm')
    st.write('Width of the concrete rib top: $b_\mathrm{top}$=', "{:.0f}".format(btop),' mm')
    st.write('Deck thickness: $t$=', "{:.2f}".format(t),' mm')
    st.write('Stud shank diameter: $d$=', "{:.0f}".format(d),' mm')
    st.write('Stud height after welding: $h_\mathrm{sc}$=', "{:.0f}".format(hsc),' mm')
    st.write('Ultimate tensile strength of stud: $f_\mathrm{u}$=', "{:.0f}".format(fu),' MPa')
    st.write('Stud welding: ', welding)
    st.write('The number of studs per rib: $n_\mathrm{r}$=', "{:.0f}".format(nr))
    st.write('Longitudinal distance from the rib top corner to the nearest stud center: $e_\mathrm{t}$=', "{:.0f}".format(et),' mm')
    st.write('Transverse spacing between studs within rib: $s_\mathrm{y}$=', "{:.0f}".format(sy),' mm')
    st.write('Longitudinal spacing between studs within rib: $s_\mathrm{x}$=', "{:.0f}".format(sx),' mm')
    if stud_position=="Two staggered studs": st.write('Diagonal distance between staggered studs within rib: $s$=', "{:.0f}".format(s_diag),' mm')

    fcm=fck+8
    X_ML_N=np.array([[nr,t,btop,bbot,hpn,sy,sx,et,fcm,fu,d,hsc]]) 
    X_ML_D=np.array([[nr,t,btop,bbot,hpn,sy,sx,et,fck,fu,d,hsc]])    
    X_ML_N_NGBoost=NGBoost_sc.transform(X_ML_N) 
    X_ML_D_NGBoost=NGBoost_sc.transform(X_ML_D)       

    if bbot/btop<=1.0: k_red_NGBoost=k_red_NGBoost_T
    else: k_red_NGBoost=k_red_NGBoost_R

    Prd_NGBoost=0.001*k_red_NGBoost*NGBoost.predict(X_ML_D_NGBoost)/1.25
    Pn_NGBoost=0.001*NGBoost.predict(X_ML_N_NGBoost)
    Pn_NGBoost_dist=NGBoost.pred_dist(X_ML_N_NGBoost)

    Pn_NGBoost_68p3lower=0.001*Pn_NGBoost_dist.dist.interval(0.683)[0]
    Pn_NGBoost_68p3upper=0.001*Pn_NGBoost_dist.dist.interval(0.683)[1]

    Pn_NGBoost_95p4lower=0.001*Pn_NGBoost_dist.dist.interval(0.954)[0]
    Pn_NGBoost_95p4upper=0.001*Pn_NGBoost_dist.dist.interval(0.954)[1]

    Pn_NGBoost_99p7lower=0.001*Pn_NGBoost_dist.dist.interval(0.997)[0]
    Pn_NGBoost_99p7upper=0.001*Pn_NGBoost_dist.dist.interval(0.997)[1]

    st.write('##### Predicted stud shear resistance')
    
    st.write('Mean nominal stud shear resistance, $P_\mathrm{n,mean}$=', "{:.2f}".format(Pn_NGBoost[0]),' kN')
    st.write('99.7% lower bound of nominal stud shear resistance, $P_\mathrm{n,mean}-3\sigma$=', "{:.2f}".format(Pn_NGBoost_99p7lower[0]),' kN') 
    st.write('99.7% upper bound of nominal stud shear resistance, $P_\mathrm{n,mean}+3\sigma$=', "{:.2f}".format(Pn_NGBoost_99p7upper[0]),' kN')     
    st.write('Design stud shear resistance, $P_\mathrm{Rd}$=',"{:.2f}".format(Prd_NGBoost[0]),' kN')
    
    
    rc = {"font.family" : "sans-serif", 
      "mathtext.fontset" : "stix"}

    plt.rcParams.update(rc)
    plt.rcParams["font.sans-serif"] = ["Source Sans Pro"] + plt.rcParams["font.sans-serif"]
       
    
    st.write('##### Stud shear resistance distribution plot')
    
    val_dist_plot_min=Pn_NGBoost_dist.dist.interval(0.999999)[0]
    val_dist_plot_max=Pn_NGBoost_dist.dist.interval(0.9999)[1]
    distval = Pn_NGBoost_dist.pdf(np.linspace(val_dist_plot_min, val_dist_plot_max,1000).reshape(1,-1)).transpose().reshape(-1,)
    distval_68p3lower=Pn_NGBoost_dist.pdf(Pn_NGBoost_dist.dist.interval(0.683)[0])
    distval_68p3upper=Pn_NGBoost_dist.pdf(Pn_NGBoost_dist.dist.interval(0.683)[1])
    distval_95p4lower=Pn_NGBoost_dist.pdf(Pn_NGBoost_dist.dist.interval(0.954)[0])
    distval_95p4upper=Pn_NGBoost_dist.pdf(Pn_NGBoost_dist.dist.interval(0.954)[1])
    distval_99p7lower=Pn_NGBoost_dist.pdf(Pn_NGBoost_dist.dist.interval(0.997)[0])
    distval_99p7upper=Pn_NGBoost_dist.pdf(Pn_NGBoost_dist.dist.interval(0.997)[1])
    xrange = 0.001*np.linspace(val_dist_plot_min, val_dist_plot_max,1000).reshape(-1,)
    f1 = plt.figure(figsize=(6.75,5), dpi=200)
    ax1=f1.add_subplot(1, 1, 1)
    ax1.plot(xrange,distval,c='#08519c',linewidth=1.5,linestyle='solid',label='$P_\mathrm{n}$ distribution')
    ax1.vlines(Pn_NGBoost,0,np.max(distval),'#e41a1c',linewidth=1.5,linestyle='solid',label='$P_\mathrm{n,mean}$')
    ax1.fill_between(xrange, distval, 0, where=(xrange>Pn_NGBoost_68p3lower)&(xrange<Pn_NGBoost_68p3upper),color= "#4292c6",alpha= 0.75,label="$P_\mathrm{n,mean}\pm \sigma$ (68.3% bound)", lw=0.25)
    ax1.fill_between(xrange, distval, 0, where=(xrange>Pn_NGBoost_95p4lower)&(xrange<Pn_NGBoost_68p3lower), color= "#4292c6", alpha= 0.5, label="$P_\mathrm{n,mean}\pm 2\sigma$ (95.4% bound)", lw=0.25)
    ax1.fill_between(xrange, distval, 0, where=(xrange<Pn_NGBoost_95p4upper)&(xrange>Pn_NGBoost_68p3upper), color= "#4292c6", alpha= 0.5, lw=0.25)
    ax1.fill_between(xrange, distval, 0, where=(xrange>Pn_NGBoost_99p7lower)&(xrange<Pn_NGBoost_95p4lower), color= "#4292c6", alpha= 0.25, label="$P_\mathrm{n,mean}\pm 3\sigma$ (99.7% bound)", lw=0.25)
    ax1.fill_between(xrange, distval, 0, where=(xrange<Pn_NGBoost_99p7upper)&(xrange>Pn_NGBoost_95p4upper), color= "#4292c6", alpha= 0.25, lw=0.25)
    ax1.vlines(Prd_NGBoost,0,np.max(distval),'#4daf4a',linewidth=1.5,linestyle='solid',label='$P_\mathrm{Rd}$')
    ax1.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax1.set_ylabel('PDF', fontsize=12)
    ax1.set_xlabel('Stud shear resistance (kN)', fontsize=12)
    ax1.legend(loc="lower center", fontsize=10, ncol=2, bbox_to_anchor=(0.5, -0.4))
    f1.tight_layout()
    st.pyplot(f1)

    st.write('##### Stud resistance plots as functions of design variables')
    
    fck1=np.array([16,20,25,30,35,40,45,50,55,60])
    fck1=fck1.reshape(len(fck1),1)
    fcm1=fck1+8
        
    nr1=np.full((10,1),nr)
    t1=np.full((10,1),t)   
    btop1=np.full((10,1),btop)    
    bbot1=np.full((10,1),bbot)
    hpn1=np.full((10,1),hpn)    
    sy1=np.full((10,1),sy)    
    sx1=np.full((10,1),sx)    
    et1=np.full((10,1),et)
    fu1=np.full((10,1),fu)
    d1=np.full((10,1),d)
    hsc1=np.full((10,1),hsc)
    X_ML_D_1=np.concatenate((nr1,t1,btop1,bbot1,hpn1,sy1,sx1,et1,fck1,fu1,d1,hsc1), axis=1)
    X_ML_N_1=np.concatenate((nr1,t1,btop1,bbot1,hpn1,sy1,sx1,et1,fcm1,fu1,d1,hsc1), axis=1)
    X_ML_D_NGBoost_1=NGBoost_sc.transform(X_ML_D_1) 
    X_ML_N_NGBoost_1=NGBoost_sc.transform(X_ML_N_1) 
    k_red_NGBoost_1_1=[]
    for i in range(10):
        if btop1[i]/bbot1[i]>=1: new_value=k_red_NGBoost_T
        else: new_value=k_red_NGBoost_R
        k_red_NGBoost_1_1.append(new_value)
    k_red_NGBoost_1=np.array(k_red_NGBoost_1_1)
    Prd_NGBoost_1=0.001*k_red_NGBoost_1*NGBoost.predict(X_ML_D_NGBoost_1)/1.25
    Pn_NGBoost_1=0.001*NGBoost.predict(X_ML_N_NGBoost_1)
    Pn_NGBoost_dist_1=NGBoost.pred_dist(X_ML_N_NGBoost_1)
    Pn_NGBoost_1_68p3lower=0.001*Pn_NGBoost_dist_1.dist.interval(0.683)[0]
    Pn_NGBoost_1_68p3upper=0.001*Pn_NGBoost_dist_1.dist.interval(0.683)[1]
    Pn_NGBoost_1_95p4lower=0.001*Pn_NGBoost_dist_1.dist.interval(0.954)[0]
    Pn_NGBoost_1_95p4upper=0.001*Pn_NGBoost_dist_1.dist.interval(0.954)[1]
    Pn_NGBoost_1_99p7lower=0.001*Pn_NGBoost_dist_1.dist.interval(0.997)[0]
    Pn_NGBoost_1_99p7upper=0.001*Pn_NGBoost_dist_1.dist.interval(0.997)[1]
    
    
    hpn2=np.arange(hpn_min,hpn_max+1,hpn_step)
    hpn2=hpn2.reshape(len(hpn2),1)
    fck2=np.full((len(hpn2),1), fck)
    fcm2=fck2+8
    nr2=np.full((len(hpn2),1),nr)
    t2=np.full((len(hpn2),1),t)   
    btop2=np.full((len(hpn2),1),btop)    
    bbot2=np.full((len(hpn2),1),bbot)
    sy2=np.full((len(hpn2),1),sy)    
    sx2=np.full((len(hpn2),1),sx)    
    et2=np.full((len(hpn2),1),et)
    fu2=np.full((len(hpn2),1),fu)
    d2=np.full((len(hpn2),1),d)
    hsc2=np.full((len(hpn2),1),hsc)

    X_ML_D_2=np.concatenate((nr2,t2,btop2,bbot2,hpn2,sy2,sx2,et2,fck2,fu2,d2,hsc2), axis=1)
    X_ML_N_2=np.concatenate((nr2,t2,btop2,bbot2,hpn2,sy2,sx2,et2,fcm2,fu2,d2,hsc2), axis=1)
    X_ML_D_NGBoost_2=NGBoost_sc.transform(X_ML_D_2) 
    X_ML_N_NGBoost_2=NGBoost_sc.transform(X_ML_N_2) 

    k_red_NGBoost_2_1=[]
    for i in range(len(hpn2)):
        if btop2[i]/bbot2[i]>=1: new_value=k_red_NGBoost_T
        else: new_value=k_red_NGBoost_R
        k_red_NGBoost_2_1.append(new_value)
    k_red_NGBoost_2=np.array(k_red_NGBoost_2_1)

    Prd_NGBoost_2=0.001*k_red_NGBoost_2*NGBoost.predict(X_ML_D_NGBoost_2)/1.25
    Pn_NGBoost_2=0.001*NGBoost.predict(X_ML_N_NGBoost_2)
    Pn_NGBoost_dist_2=NGBoost.pred_dist(X_ML_N_NGBoost_2)
    Pn_NGBoost_2_68p3lower=0.001*Pn_NGBoost_dist_2.dist.interval(0.683)[0]
    Pn_NGBoost_2_68p3upper=0.001*Pn_NGBoost_dist_2.dist.interval(0.683)[1]
    Pn_NGBoost_2_95p4lower=0.001*Pn_NGBoost_dist_2.dist.interval(0.954)[0]
    Pn_NGBoost_2_95p4upper=0.001*Pn_NGBoost_dist_2.dist.interval(0.954)[1]
    Pn_NGBoost_2_99p7lower=0.001*Pn_NGBoost_dist_2.dist.interval(0.997)[0]
    Pn_NGBoost_2_99p7upper=0.001*Pn_NGBoost_dist_2.dist.interval(0.997)[1]
    
    
    bbot3=np.arange(bbot_min,bbot_max+1,bbot_step)
    bbot3=bbot3.reshape(len(bbot3),1)
    hpn3=np.full((len(bbot3),1), hpn)
    fck3=np.full((len(bbot3),1), fck)
    fcm3=fck3+8
    nr3=np.full((len(bbot3),1),nr)
    t3=np.full((len(bbot3),1),t)   
    btop3=np.full((len(bbot3),1),btop)    
    sy3=np.full((len(bbot3),1),sy)    
    sx3=np.full((len(bbot3),1),sx)    
    et3=np.full((len(bbot3),1),et)
    fu3=np.full((len(bbot3),1),fu)
    d3=np.full((len(bbot3),1),d)
    hsc3=np.full((len(bbot3),1),hsc)

    X_ML_D_3=np.concatenate((nr3,t3,btop3,bbot3,hpn3,sy3,sx3,et3,fck3,fu3,d3,hsc3), axis=1)
    X_ML_N_3=np.concatenate((nr3,t3,btop3,bbot3,hpn3,sy3,sx3,et3,fcm3,fu3,d3,hsc3), axis=1)
    X_ML_D_NGBoost_3=NGBoost_sc.transform(X_ML_D_3) 
    X_ML_N_NGBoost_3=NGBoost_sc.transform(X_ML_N_3) 

    k_red_NGBoost_3_1=[]
    for i in range(len(bbot3)):
        if btop3[i]/bbot3[i]>=1: new_value=k_red_NGBoost_T
        else: new_value=k_red_NGBoost_R
        k_red_NGBoost_3_1.append(new_value)
    k_red_NGBoost_3=np.array(k_red_NGBoost_3_1)

    Prd_NGBoost_3=0.001*k_red_NGBoost_3*NGBoost.predict(X_ML_D_NGBoost_3)/1.25
    Pn_NGBoost_3=0.001*NGBoost.predict(X_ML_N_NGBoost_3)
    Pn_NGBoost_dist_3=NGBoost.pred_dist(X_ML_N_NGBoost_3)
    Pn_NGBoost_3_68p3lower=0.001*Pn_NGBoost_dist_3.dist.interval(0.683)[0]
    Pn_NGBoost_3_68p3upper=0.001*Pn_NGBoost_dist_3.dist.interval(0.683)[1]
    Pn_NGBoost_3_95p4lower=0.001*Pn_NGBoost_dist_3.dist.interval(0.954)[0]
    Pn_NGBoost_3_95p4upper=0.001*Pn_NGBoost_dist_3.dist.interval(0.954)[1]
    Pn_NGBoost_3_99p7lower=0.001*Pn_NGBoost_dist_3.dist.interval(0.997)[0]
    Pn_NGBoost_3_99p7upper=0.001*Pn_NGBoost_dist_3.dist.interval(0.997)[1]
    
    
    btop4=np.arange(btop_min,btop_max+1,btop_step)
    btop4=btop4.reshape(len(btop4),1)
    hpn4=np.full((len(btop4),1), hpn)
    fck4=np.full((len(btop4),1), fck)
    fcm4=fck4+8
    nr4=np.full((len(btop4),1),nr)
    t4=np.full((len(btop4),1),t)   
    bbot4=np.full((len(btop4),1),bbot)    
    sy4=np.full((len(btop4),1),sy)    
    sx4=np.full((len(btop4),1),sx)    
    et4=np.full((len(btop4),1),et)
    fu4=np.full((len(btop4),1),fu)
    d4=np.full((len(btop4),1),d)
    hsc4=np.full((len(btop4),1),hsc)

    X_ML_D_4=np.concatenate((nr4,t4,btop4,bbot4,hpn4,sy4,sx4,et4,fck4,fu4,d4,hsc4), axis=1)
    X_ML_N_4=np.concatenate((nr4,t4,btop4,bbot4,hpn4,sy4,sx4,et4,fcm4,fu4,d4,hsc4), axis=1)
    X_ML_D_NGBoost_4=NGBoost_sc.transform(X_ML_D_4) 
    X_ML_N_NGBoost_4=NGBoost_sc.transform(X_ML_N_4) 

    k_red_NGBoost_4_1=[]
    for i in range(len(btop4)):
        if btop4[i]/bbot4[i]>=1: new_value=k_red_NGBoost_T
        else: new_value=k_red_NGBoost_R
        k_red_NGBoost_4_1.append(new_value)
    k_red_NGBoost_4=np.array(k_red_NGBoost_4_1)

    Prd_NGBoost_4=0.001*k_red_NGBoost_4*NGBoost.predict(X_ML_D_NGBoost_4)/1.25
    Pn_NGBoost_4=0.001*NGBoost.predict(X_ML_N_NGBoost_4)
    Pn_NGBoost_dist_4=NGBoost.pred_dist(X_ML_N_NGBoost_4)
    Pn_NGBoost_4_68p3lower=0.001*Pn_NGBoost_dist_4.dist.interval(0.683)[0]
    Pn_NGBoost_4_68p3upper=0.001*Pn_NGBoost_dist_4.dist.interval(0.683)[1]
    Pn_NGBoost_4_95p4lower=0.001*Pn_NGBoost_dist_4.dist.interval(0.954)[0]
    Pn_NGBoost_4_95p4upper=0.001*Pn_NGBoost_dist_4.dist.interval(0.954)[1]
    Pn_NGBoost_4_99p7lower=0.001*Pn_NGBoost_dist_4.dist.interval(0.997)[0]
    Pn_NGBoost_4_99p7upper=0.001*Pn_NGBoost_dist_4.dist.interval(0.997)[1]
    
    
    t5=np.arange(t_min,t_max+0.01,t_step)
    t5=t5.reshape(len(t5),1)
    hpn5=np.full((len(t5),1), hpn)
    fck5=np.full((len(t5),1), fck)
    fcm5=fck5+8
    nr5=np.full((len(t5),1),nr)
    btop5=np.full((len(t5),1),btop)   
    bbot5=np.full((len(t5),1),bbot)    
    sy5=np.full((len(t5),1),sy)    
    sx5=np.full((len(t5),1),sx)    
    et5=np.full((len(t5),1),et)
    fu5=np.full((len(t5),1),fu)
    d5=np.full((len(t5),1),d)
    hsc5=np.full((len(t5),1),hsc)

    X_ML_D_5=np.concatenate((nr5,t5,btop5,bbot5,hpn5,sy5,sx5,et5,fck5,fu5,d5,hsc5), axis=1)
    X_ML_N_5=np.concatenate((nr5,t5,btop5,bbot5,hpn5,sy5,sx5,et5,fcm5,fu5,d5,hsc5), axis=1)
    X_ML_D_NGBoost_5=NGBoost_sc.transform(X_ML_D_5) 
    X_ML_N_NGBoost_5=NGBoost_sc.transform(X_ML_N_5) 

    k_red_NGBoost_5_1=[]
    for i in range(len(t5)):
        if btop5[i]/bbot5[i]>=1: new_value=k_red_NGBoost_T
        else: new_value=k_red_NGBoost_R
        k_red_NGBoost_5_1.append(new_value)
    k_red_NGBoost_5=np.array(k_red_NGBoost_5_1)

    Prd_NGBoost_5=0.001*k_red_NGBoost_5*NGBoost.predict(X_ML_D_NGBoost_5)/1.25
    Pn_NGBoost_5=0.001*NGBoost.predict(X_ML_N_NGBoost_5)
    Pn_NGBoost_dist_5=NGBoost.pred_dist(X_ML_N_NGBoost_5)
    Pn_NGBoost_5_68p3lower=0.001*Pn_NGBoost_dist_5.dist.interval(0.683)[0]
    Pn_NGBoost_5_68p3upper=0.001*Pn_NGBoost_dist_5.dist.interval(0.683)[1]
    Pn_NGBoost_5_95p4lower=0.001*Pn_NGBoost_dist_5.dist.interval(0.954)[0]
    Pn_NGBoost_5_95p4upper=0.001*Pn_NGBoost_dist_5.dist.interval(0.954)[1]
    Pn_NGBoost_5_99p7lower=0.001*Pn_NGBoost_dist_5.dist.interval(0.997)[0]
    Pn_NGBoost_5_99p7upper=0.001*Pn_NGBoost_dist_5.dist.interval(0.997)[1]   
 
    if welding=='Through-holes':   
        d6=np.arange(d_min,d_max+1,d_step)
        d6=d6.reshape(len(d6),1)
        hpn6=np.full((len(d6),1), hpn)
        fck6=np.full((len(d6),1), fck)
        fcm6=fck6+8
        nr6=np.full((len(d6),1),nr)
        btop6=np.full((len(d6),1),btop)   
        bbot6=np.full((len(d6),1),bbot)    
        sy6=np.full((len(d6),1),sy)    
        sx6=np.full((len(d6),1),sx)    
        et6=np.full((len(d6),1),et)
        fu6=np.full((len(d6),1),fu)
        t6=np.full((len(d6),1),t)
        hsc6=np.full((len(d6),1),hsc)

        X_ML_D_6=np.concatenate((nr6,t6,btop6,bbot6,hpn6,sy6,sx6,et6,fck6,fu6,d6,hsc6), axis=1)
        X_ML_N_6=np.concatenate((nr6,t6,btop6,bbot6,hpn6,sy6,sx6,et6,fcm6,fu6,d6,hsc6), axis=1)
        X_ML_D_NGBoost_6=NGBoost_sc.transform(X_ML_D_6) 
        X_ML_N_NGBoost_6=NGBoost_sc.transform(X_ML_N_6) 

        k_red_NGBoost_6_1=[]
        for i in range(len(d6)):
            if btop6[i]/bbot6[i]>=1: new_value=k_red_NGBoost_T
            else: new_value=k_red_NGBoost_R
            k_red_NGBoost_6_1.append(new_value)
        k_red_NGBoost_6=np.array(k_red_NGBoost_6_1)

        Prd_NGBoost_6=0.001*k_red_NGBoost_6*NGBoost.predict(X_ML_D_NGBoost_6)/1.25
        Pn_NGBoost_6=0.001*NGBoost.predict(X_ML_N_NGBoost_6)
        Pn_NGBoost_dist_6=NGBoost.pred_dist(X_ML_N_NGBoost_6)
        Pn_NGBoost_6_68p3lower=0.001*Pn_NGBoost_dist_6.dist.interval(0.683)[0]
        Pn_NGBoost_6_68p3upper=0.001*Pn_NGBoost_dist_6.dist.interval(0.683)[1]
        Pn_NGBoost_6_95p4lower=0.001*Pn_NGBoost_dist_6.dist.interval(0.954)[0]
        Pn_NGBoost_6_95p4upper=0.001*Pn_NGBoost_dist_6.dist.interval(0.954)[1]
        Pn_NGBoost_6_99p7lower=0.001*Pn_NGBoost_dist_6.dist.interval(0.997)[0]
        Pn_NGBoost_6_99p7upper=0.001*Pn_NGBoost_dist_6.dist.interval(0.997)[1]
    
    
    hsc7=np.arange(hsc_min,hsc_max+1,hsc_step)
    hsc7=hsc7.reshape(len(hsc7),1)
    hpn7=np.full((len(hsc7),1), hpn)
    fck7=np.full((len(hsc7),1), fck)
    fcm7=fck7+8
    nr7=np.full((len(hsc7),1),nr)
    btop7=np.full((len(hsc7),1),btop)   
    bbot7=np.full((len(hsc7),1),bbot)    
    sy7=np.full((len(hsc7),1),sy)    
    sx7=np.full((len(hsc7),1),sx)    
    et7=np.full((len(hsc7),1),et)
    fu7=np.full((len(hsc7),1),fu)
    t7=np.full((len(hsc7),1),t)
    d7=np.full((len(hsc7),1),d)

    X_ML_D_7=np.concatenate((nr7,t7,btop7,bbot7,hpn7,sy7,sx7,et7,fck7,fu7,d7,hsc7), axis=1)
    X_ML_N_7=np.concatenate((nr7,t7,btop7,bbot7,hpn7,sy7,sx7,et7,fcm7,fu7,d7,hsc7), axis=1)
    X_ML_D_NGBoost_7=NGBoost_sc.transform(X_ML_D_7) 
    X_ML_N_NGBoost_7=NGBoost_sc.transform(X_ML_N_7) 

    k_red_NGBoost_7_1=[]
    for i in range(len(hsc7)):
        if btop7[i]/bbot7[i]>=1: new_value=k_red_NGBoost_T
        else: new_value=k_red_NGBoost_R
        k_red_NGBoost_7_1.append(new_value)
    k_red_NGBoost_7=np.array(k_red_NGBoost_7_1)

    Prd_NGBoost_7=0.001*k_red_NGBoost_7*NGBoost.predict(X_ML_D_NGBoost_7)/1.25
    Pn_NGBoost_7=0.001*NGBoost.predict(X_ML_N_NGBoost_7)
    Pn_NGBoost_dist_7=NGBoost.pred_dist(X_ML_N_NGBoost_7)
    Pn_NGBoost_7_68p3lower=0.001*Pn_NGBoost_dist_7.dist.interval(0.683)[0]
    Pn_NGBoost_7_68p3upper=0.001*Pn_NGBoost_dist_7.dist.interval(0.683)[1]
    Pn_NGBoost_7_95p4lower=0.001*Pn_NGBoost_dist_7.dist.interval(0.954)[0]
    Pn_NGBoost_7_95p4upper=0.001*Pn_NGBoost_dist_7.dist.interval(0.954)[1]
    Pn_NGBoost_7_99p7lower=0.001*Pn_NGBoost_dist_7.dist.interval(0.997)[0]
    Pn_NGBoost_7_99p7upper=0.001*Pn_NGBoost_dist_7.dist.interval(0.997)[1]
    
    
    et8=np.arange(et_min,et_max+1,et_step)
    et8=et8.reshape(len(et8),1)
    hpn8=np.full((len(et8),1), hpn)
    fck8=np.full((len(et8),1), fck)
    fcm8=fck8+8
    nr8=np.full((len(et8),1),nr)
    btop8=np.full((len(et8),1),btop)   
    bbot8=np.full((len(et8),1),bbot)    
    sy8=np.full((len(et8),1),sy)    
    sx8=np.full((len(et8),1),sx)    
    hsc8=np.full((len(et8),1),hsc)
    fu8=np.full((len(et8),1),fu)
    t8=np.full((len(et8),1),t)
    d8=np.full((len(et8),1),d)

    X_ML_D_8=np.concatenate((nr8,t8,btop8,bbot8,hpn8,sy8,sx8,et8,fck8,fu8,d8,hsc8), axis=1)
    X_ML_N_8=np.concatenate((nr8,t8,btop8,bbot8,hpn8,sy8,sx8,et8,fcm8,fu8,d8,hsc8), axis=1)
    X_ML_D_NGBoost_8=NGBoost_sc.transform(X_ML_D_8) 
    X_ML_N_NGBoost_8=NGBoost_sc.transform(X_ML_N_8) 

    k_red_NGBoost_8_1=[]
    for i in range(len(et8)):
        if btop8[i]/bbot8[i]>=1: new_value=k_red_NGBoost_T
        else: new_value=k_red_NGBoost_R
        k_red_NGBoost_8_1.append(new_value)
    k_red_NGBoost_8=np.array(k_red_NGBoost_8_1)

    Prd_NGBoost_8=0.001*k_red_NGBoost_8*NGBoost.predict(X_ML_D_NGBoost_8)/1.25
    Pn_NGBoost_8=0.001*NGBoost.predict(X_ML_N_NGBoost_8)
    Pn_NGBoost_dist_8=NGBoost.pred_dist(X_ML_N_NGBoost_8)
    Pn_NGBoost_8_68p3lower=0.001*Pn_NGBoost_dist_8.dist.interval(0.683)[0]
    Pn_NGBoost_8_68p3upper=0.001*Pn_NGBoost_dist_8.dist.interval(0.683)[1]
    Pn_NGBoost_8_95p4lower=0.001*Pn_NGBoost_dist_8.dist.interval(0.954)[0]
    Pn_NGBoost_8_95p4upper=0.001*Pn_NGBoost_dist_8.dist.interval(0.954)[1]
    Pn_NGBoost_8_99p7lower=0.001*Pn_NGBoost_dist_8.dist.interval(0.997)[0]
    Pn_NGBoost_8_99p7upper=0.001*Pn_NGBoost_dist_8.dist.interval(0.997)[1]
    
    
    fu9=np.arange(fu_min,fu_max+1,fu_step)
    fu9=fu9.reshape(len(fu9),1)
    hpn9=np.full((len(fu9),1), hpn)
    fck9=np.full((len(fu9),1), fck)
    fcm9=fck9+8
    nr9=np.full((len(fu9),1),nr)
    btop9=np.full((len(fu9),1),btop)   
    bbot9=np.full((len(fu9),1),bbot)    
    sy9=np.full((len(fu9),1),sy)    
    sx9=np.full((len(fu9),1),sx)    
    hsc9=np.full((len(fu9),1),hsc)
    et9=np.full((len(fu9),1),et)
    t9=np.full((len(fu9),1),t)
    d9=np.full((len(fu9),1),d)

    X_ML_D_9=np.concatenate((nr9,t9,btop9,bbot9,hpn9,sy9,sx9,et9,fck9,fu9,d9,hsc9), axis=1)
    X_ML_N_9=np.concatenate((nr9,t9,btop9,bbot9,hpn9,sy9,sx9,et9,fcm9,fu9,d9,hsc9), axis=1)
    X_ML_D_NGBoost_9=NGBoost_sc.transform(X_ML_D_9) 
    X_ML_N_NGBoost_9=NGBoost_sc.transform(X_ML_N_9) 

    k_red_NGBoost_9_1=[]
    for i in range(len(fu9)):
        if btop9[i]/bbot9[i]>=1: new_value=k_red_NGBoost_T
        else: new_value=k_red_NGBoost_R
        k_red_NGBoost_9_1.append(new_value)
    k_red_NGBoost_9=np.array(k_red_NGBoost_9_1)

    Prd_NGBoost_9=0.001*k_red_NGBoost_9*NGBoost.predict(X_ML_D_NGBoost_9)/1.25
    Pn_NGBoost_9=0.001*NGBoost.predict(X_ML_N_NGBoost_9)
    Pn_NGBoost_dist_9=NGBoost.pred_dist(X_ML_N_NGBoost_9)
    Pn_NGBoost_9_68p3lower=0.001*Pn_NGBoost_dist_9.dist.interval(0.683)[0]
    Pn_NGBoost_9_68p3upper=0.001*Pn_NGBoost_dist_9.dist.interval(0.683)[1]
    Pn_NGBoost_9_95p4lower=0.001*Pn_NGBoost_dist_9.dist.interval(0.954)[0]
    Pn_NGBoost_9_95p4upper=0.001*Pn_NGBoost_dist_9.dist.interval(0.954)[1]
    Pn_NGBoost_9_99p7lower=0.001*Pn_NGBoost_dist_9.dist.interval(0.997)[0]
    Pn_NGBoost_9_99p7upper=0.001*Pn_NGBoost_dist_9.dist.interval(0.997)[1]
    
    
    if sy>0:
        sy10=np.arange(sy_min,sy_max+1,sy_step)
        sy10=sy10.reshape(len(sy10),1)
        hpn10=np.full((len(sy10),1), hpn)
        fck10=np.full((len(sy10),1), fck)
        fcm10=fck10+8
        nr10=np.full((len(sy10),1),nr)
        btop10=np.full((len(sy10),1),btop)   
        bbot10=np.full((len(sy10),1),bbot)    
        fu10=np.full((len(sy10),1),fu)    
        sx10=np.full((len(sy10),1),sx)    
        hsc10=np.full((len(sy10),1),hsc)
        et10=np.full((len(sy10),1),et)
        t10=np.full((len(sy10),1),t)
        d10=np.full((len(sy10),1),d)

        X_ML_D_10=np.concatenate((nr10,t10,btop10,bbot10,hpn10,sy10,sx10,et10,fck10,fu10,d10,hsc10), axis=1)
        X_ML_N_10=np.concatenate((nr10,t10,btop10,bbot10,hpn10,sy10,sx10,et10,fcm10,fu10,d10,hsc10), axis=1)
        X_ML_D_NGBoost_10=NGBoost_sc.transform(X_ML_D_10) 
        X_ML_N_NGBoost_10=NGBoost_sc.transform(X_ML_N_10) 

        k_red_NGBoost_10_1=[]
        for i in range(len(sy10)):
            if btop10[i]/bbot10[i]>=1: new_value=k_red_NGBoost_T
            else: new_value=k_red_NGBoost_R
            k_red_NGBoost_10_1.append(new_value)
        k_red_NGBoost_10=np.array(k_red_NGBoost_10_1)

        Prd_NGBoost_10=0.001*k_red_NGBoost_10*NGBoost.predict(X_ML_D_NGBoost_10)/1.25
        Pn_NGBoost_10=0.001*NGBoost.predict(X_ML_N_NGBoost_10)
        Pn_NGBoost_dist_10=NGBoost.pred_dist(X_ML_N_NGBoost_10)
        Pn_NGBoost_10_68p3lower=0.001*Pn_NGBoost_dist_10.dist.interval(0.683)[0]
        Pn_NGBoost_10_68p3upper=0.001*Pn_NGBoost_dist_10.dist.interval(0.683)[1]
        Pn_NGBoost_10_95p4lower=0.001*Pn_NGBoost_dist_10.dist.interval(0.954)[0]
        Pn_NGBoost_10_95p4upper=0.001*Pn_NGBoost_dist_10.dist.interval(0.954)[1]
        Pn_NGBoost_10_99p7lower=0.001*Pn_NGBoost_dist_10.dist.interval(0.997)[0]
        Pn_NGBoost_10_99p7upper=0.001*Pn_NGBoost_dist_10.dist.interval(0.997)[1]
        
        
    if sx>0:
        sx11=np.arange(sx_min,sx_max+1,sx_step)
        sx11=sx11.reshape(len(sx11),1)
        hpn11=np.full((len(sx11),1), hpn)
        fck11=np.full((len(sx11),1), fck)
        fcm11=fck11+8
        nr11=np.full((len(sx11),1),nr)
        btop11=np.full((len(sx11),1),btop)   
        bbot11=np.full((len(sx11),1),bbot)    
        fu11=np.full((len(sx11),1),fu)    
        sy11=np.full((len(sx11),1),sy)    
        hsc11=np.full((len(sx11),1),hsc)
        et11=np.full((len(sx11),1),et)
        t11=np.full((len(sx11),1),t)
        d11=np.full((len(sx11),1),d)

        X_ML_D_11=np.concatenate((nr11,t11,btop11,bbot11,hpn11,sy11,sx11,et11,fck11,fu11,d11,hsc11), axis=1)
        X_ML_N_11=np.concatenate((nr11,t11,btop11,bbot11,hpn11,sy11,sx11,et11,fcm11,fu11,d11,hsc11), axis=1)
        X_ML_D_NGBoost_11=NGBoost_sc.transform(X_ML_D_11) 
        X_ML_N_NGBoost_11=NGBoost_sc.transform(X_ML_N_11) 

        k_red_NGBoost_11_1=[]
        for i in range(len(sx11)):
            if btop11[i]/bbot11[i]>=1: new_value=k_red_NGBoost_T
            else: new_value=k_red_NGBoost_R
            k_red_NGBoost_11_1.append(new_value)
        k_red_NGBoost_11=np.array(k_red_NGBoost_11_1)

        Prd_NGBoost_11=0.001*k_red_NGBoost_11*NGBoost.predict(X_ML_D_NGBoost_11)/1.25
        Pn_NGBoost_11=0.001*NGBoost.predict(X_ML_N_NGBoost_11)
        Pn_NGBoost_dist_11=NGBoost.pred_dist(X_ML_N_NGBoost_11)
        Pn_NGBoost_11_68p3lower=0.001*Pn_NGBoost_dist_11.dist.interval(0.683)[0]
        Pn_NGBoost_11_68p3upper=0.001*Pn_NGBoost_dist_11.dist.interval(0.683)[1]
        Pn_NGBoost_11_95p4lower=0.001*Pn_NGBoost_dist_11.dist.interval(0.954)[0]
        Pn_NGBoost_11_95p4upper=0.001*Pn_NGBoost_dist_11.dist.interval(0.954)[1]
        Pn_NGBoost_11_99p7lower=0.001*Pn_NGBoost_dist_11.dist.interval(0.997)[0]
        Pn_NGBoost_11_99p7upper=0.001*Pn_NGBoost_dist_11.dist.interval(0.997)[1]
        
        
    f1 = plt.figure(figsize=(6.75,4*3), dpi=200)

    ax1 = f1.add_subplot(6,2,1)
    ax1.plot(fck1, Prd_NGBoost_1, color='#4daf4a',linewidth=1.5, label='$P_\mathrm{Rd}$',linestyle='solid')
    ax1.plot(fck1, Pn_NGBoost_1, color='#08519c',linewidth=1.5, label='$P_\mathrm{n,mean}$',linestyle='solid')

    ax1.fill_between(fck1.reshape(len(fck1),),Pn_NGBoost_1_68p3lower,Pn_NGBoost_1_68p3upper, color= "#4292c6",alpha= 0.75,label="$P_\mathrm{n,mean}\pm \sigma$ (68.3% bound)", lw=0.25)
    ax1.fill_between(fck1.reshape(len(fck1),),Pn_NGBoost_1_68p3lower,Pn_NGBoost_1_95p4lower, color= "#4292c6",alpha= 0.5,label="$P_\mathrm{n,mean}\pm 2\sigma$ (95.4% bound)", lw=0.25)
    ax1.fill_between(fck1.reshape(len(fck1),),Pn_NGBoost_1_68p3upper,Pn_NGBoost_1_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax1.fill_between(fck1.reshape(len(fck1),),Pn_NGBoost_1_95p4lower,Pn_NGBoost_1_99p7lower, color= "#4292c6",alpha= 0.25,label="$P_\mathrm{n,mean}\pm 3\sigma$ (99.7% bound)", lw=0.25)
    ax1.fill_between(fck1.reshape(len(fck1),),Pn_NGBoost_1_95p4upper,Pn_NGBoost_1_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

    fck_loc=np.where(fck1==fck)[0].item()
    ax1.scatter(fck,Prd_NGBoost_1[fck_loc],marker='o',facecolors='#4daf4a')
    ax1.scatter(fck,Pn_NGBoost_1[fck_loc],marker='o',facecolors='#08519c')

    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax1.set_ylabel('$P$ (kN)', fontsize=10)
    ax1.set_xlabel('$f_\mathrm{ck}$ (MPa)', fontsize=10)


    ax2 = f1.add_subplot(6,2,2)
    ax2.plot(hpn2, Prd_NGBoost_2, color='#4daf4a',linewidth=1.5, linestyle='solid')
    ax2.plot(hpn2, Pn_NGBoost_2, color='#08519c',linewidth=1.5, linestyle='solid')

    ax2.fill_between(hpn2.reshape(len(hpn2),),Pn_NGBoost_2_68p3lower,Pn_NGBoost_2_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
    ax2.fill_between(hpn2.reshape(len(hpn2),),Pn_NGBoost_2_68p3lower,Pn_NGBoost_2_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax2.fill_between(hpn2.reshape(len(hpn2),),Pn_NGBoost_2_68p3upper,Pn_NGBoost_2_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax2.fill_between(hpn2.reshape(len(hpn2),),Pn_NGBoost_2_95p4lower,Pn_NGBoost_2_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
    ax2.fill_between(hpn2.reshape(len(hpn2),),Pn_NGBoost_2_95p4upper,Pn_NGBoost_2_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

    hpn_loc=np.where(hpn2==hpn)[0].item()
    ax2.scatter(hpn,Prd_NGBoost_2[hpn_loc],marker='o',facecolors='#4daf4a')
    ax2.scatter(hpn,Pn_NGBoost_2[hpn_loc],marker='o',facecolors='#08519c')

    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax2.set_ylabel('$P$ (kN)', fontsize=10)
    ax2.set_xlabel('$h_\mathrm{pn}$ (mm)', fontsize=10)


    ax3 = f1.add_subplot(6,2,3)
    ax3.plot(bbot3, Prd_NGBoost_3, color='#4daf4a',linewidth=1.5, linestyle='solid')
    ax3.plot(bbot3, Pn_NGBoost_3, color='#08519c',linewidth=1.5, linestyle='solid')

    ax3.fill_between(bbot3.reshape(len(bbot3),),Pn_NGBoost_3_68p3lower,Pn_NGBoost_3_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
    ax3.fill_between(bbot3.reshape(len(bbot3),),Pn_NGBoost_3_68p3lower,Pn_NGBoost_3_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax3.fill_between(bbot3.reshape(len(bbot3),),Pn_NGBoost_3_68p3upper,Pn_NGBoost_3_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax3.fill_between(bbot3.reshape(len(bbot3),),Pn_NGBoost_3_95p4lower,Pn_NGBoost_3_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
    ax3.fill_between(bbot3.reshape(len(bbot3),),Pn_NGBoost_3_95p4upper,Pn_NGBoost_3_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

    bbot_loc=np.where(bbot3==bbot)[0].item()
    ax3.scatter(bbot,Prd_NGBoost_3[bbot_loc],marker='o',facecolors='#4daf4a')
    ax3.scatter(bbot,Pn_NGBoost_3[bbot_loc],marker='o',facecolors='#08519c')

    ax3.set_ylim(bottom=0)
    ax3.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax3.set_ylabel('$P$ (kN)', fontsize=10)
    ax3.set_xlabel('$b_\mathrm{bot}$ (mm)', fontsize=10)



    ax4 = f1.add_subplot(6,2,4)
    ax4.plot(btop4, Prd_NGBoost_4, color='#4daf4a',linewidth=1.5, linestyle='solid')
    ax4.plot(btop4, Pn_NGBoost_4, color='#08519c',linewidth=1.5, linestyle='solid')

    ax4.fill_between(btop4.reshape(len(btop4),),Pn_NGBoost_4_68p3lower,Pn_NGBoost_4_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
    ax4.fill_between(btop4.reshape(len(btop4),),Pn_NGBoost_4_68p3lower,Pn_NGBoost_4_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax4.fill_between(btop4.reshape(len(btop4),),Pn_NGBoost_4_68p3upper,Pn_NGBoost_4_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax4.fill_between(btop4.reshape(len(btop4),),Pn_NGBoost_4_95p4lower,Pn_NGBoost_4_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
    ax4.fill_between(btop4.reshape(len(btop4),),Pn_NGBoost_4_95p4upper,Pn_NGBoost_4_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

    btop_loc=np.where(btop4==btop)[0].item()
    ax4.scatter(btop,Prd_NGBoost_4[btop_loc],marker='o',facecolors='#4daf4a')
    ax4.scatter(btop,Pn_NGBoost_4[btop_loc],marker='o',facecolors='#08519c')

    ax4.set_ylim(bottom=0)
    ax4.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax4.set_ylabel('$P$ (kN)', fontsize=10)
    ax4.set_xlabel('$b_\mathrm{top}$ (mm)', fontsize=10)



    ax5 = f1.add_subplot(6,2,5)
    ax5.plot(t5, Prd_NGBoost_5, color='#4daf4a',linewidth=1.5, linestyle='solid')
    ax5.plot(t5, Pn_NGBoost_5, color='#08519c',linewidth=1.5, linestyle='solid')

    ax5.fill_between(t5.reshape(len(t5),),Pn_NGBoost_5_68p3lower,Pn_NGBoost_5_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
    ax5.fill_between(t5.reshape(len(t5),),Pn_NGBoost_5_68p3lower,Pn_NGBoost_5_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax5.fill_between(t5.reshape(len(t5),),Pn_NGBoost_5_68p3upper,Pn_NGBoost_5_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax5.fill_between(t5.reshape(len(t5),),Pn_NGBoost_5_95p4lower,Pn_NGBoost_5_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
    ax5.fill_between(t5.reshape(len(t5),),Pn_NGBoost_5_95p4upper,Pn_NGBoost_5_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

    t_loc=np.where(abs(t5-t)<0.001)[0].item()
    ax5.scatter(t,Prd_NGBoost_5[t_loc],marker='o',facecolors='#4daf4a')
    ax5.scatter(t,Pn_NGBoost_5[t_loc],marker='o',facecolors='#08519c')

    ax5.set_ylim(bottom=0)
    ax5.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax5.set_ylabel('$P$ (kN)', fontsize=10)
    ax5.set_xlabel('$t$ (mm)', fontsize=10)


    if welding=='Through-holes': 
        ax6 = f1.add_subplot(6,2,6)
        ax6.plot(d6, Prd_NGBoost_6, color='#4daf4a',linewidth=1.5, linestyle='solid')
        ax6.plot(d6, Pn_NGBoost_6, color='#08519c',linewidth=1.5, linestyle='solid')

        ax6.fill_between(d6.reshape(len(d6),),Pn_NGBoost_6_68p3lower,Pn_NGBoost_6_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
        ax6.fill_between(d6.reshape(len(d6),),Pn_NGBoost_6_68p3lower,Pn_NGBoost_6_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
        ax6.fill_between(d6.reshape(len(d6),),Pn_NGBoost_6_68p3upper,Pn_NGBoost_6_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
        ax6.fill_between(d6.reshape(len(d6),),Pn_NGBoost_6_95p4lower,Pn_NGBoost_6_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
        ax6.fill_between(d6.reshape(len(d6),),Pn_NGBoost_6_95p4upper,Pn_NGBoost_6_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

        d_loc=np.where(d6==d)[0].item()
        ax6.scatter(d,Prd_NGBoost_6[d_loc],marker='o',facecolors='#4daf4a')
        ax6.scatter(d,Pn_NGBoost_6[d_loc],marker='o',facecolors='#08519c')

        ax6.set_ylim(bottom=0)
        ax6.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
        ax6.set_ylabel('$P$ (kN)', fontsize=10)
        ax6.set_xlabel('$d$ (mm)', fontsize=10)



    ax7 = f1.add_subplot(6,2,7 if welding=='Through-holes' else 6)
    ax7.plot(hsc7, Prd_NGBoost_7, color='#4daf4a',linewidth=1.5, linestyle='solid')
    ax7.plot(hsc7, Pn_NGBoost_7, color='#08519c',linewidth=1.5, linestyle='solid')

    ax7.fill_between(hsc7.reshape(len(hsc7),),Pn_NGBoost_7_68p3lower,Pn_NGBoost_7_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
    ax7.fill_between(hsc7.reshape(len(hsc7),),Pn_NGBoost_7_68p3lower,Pn_NGBoost_7_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax7.fill_between(hsc7.reshape(len(hsc7),),Pn_NGBoost_7_68p3upper,Pn_NGBoost_7_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax7.fill_between(hsc7.reshape(len(hsc7),),Pn_NGBoost_7_95p4lower,Pn_NGBoost_7_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
    ax7.fill_between(hsc7.reshape(len(hsc7),),Pn_NGBoost_7_95p4upper,Pn_NGBoost_7_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

    hsc_loc=np.where(hsc7==hsc)[0].item()
    ax7.scatter(hsc,Prd_NGBoost_7[hsc_loc],marker='o',facecolors='#4daf4a')
    ax7.scatter(hsc,Pn_NGBoost_7[hsc_loc],marker='o',facecolors='#08519c')

    ax7.set_ylim(bottom=0)
    ax7.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax7.set_ylabel('$P$ (kN)', fontsize=10)
    ax7.set_xlabel('$h_\mathrm{sc}$ (mm)', fontsize=10)



    ax8 = f1.add_subplot(6,2,8 if welding=='Through-holes' else 7)
    ax8.plot(et8, Prd_NGBoost_8, color='#4daf4a',linewidth=1.5, linestyle='solid')
    ax8.plot(et8, Pn_NGBoost_8, color='#08519c',linewidth=1.5, linestyle='solid')

    ax8.fill_between(et8.reshape(len(et8),),Pn_NGBoost_8_68p3lower,Pn_NGBoost_8_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
    ax8.fill_between(et8.reshape(len(et8),),Pn_NGBoost_8_68p3lower,Pn_NGBoost_8_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax8.fill_between(et8.reshape(len(et8),),Pn_NGBoost_8_68p3upper,Pn_NGBoost_8_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax8.fill_between(et8.reshape(len(et8),),Pn_NGBoost_8_95p4lower,Pn_NGBoost_8_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
    ax8.fill_between(et8.reshape(len(et8),),Pn_NGBoost_8_95p4upper,Pn_NGBoost_8_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

    et_loc=np.where(et8==et)[0].item()
    ax8.scatter(et,Prd_NGBoost_8[et_loc],marker='o',facecolors='#4daf4a')
    ax8.scatter(et,Pn_NGBoost_8[et_loc],marker='o',facecolors='#08519c')

    ax8.set_ylim(bottom=0)
    ax8.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax8.set_ylabel('$P$ (kN)', fontsize=10)
    ax8.set_xlabel('$e_\mathrm{t}$ (mm)', fontsize=10)



    ax9 = f1.add_subplot(6,2,9 if welding=='Through-holes' else 8)
    ax9.plot(fu9, Prd_NGBoost_9, color='#4daf4a',linewidth=1.5, linestyle='solid')
    ax9.plot(fu9, Pn_NGBoost_9, color='#08519c',linewidth=1.5, linestyle='solid')

    ax9.fill_between(fu9.reshape(len(fu9),),Pn_NGBoost_9_68p3lower,Pn_NGBoost_9_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
    ax9.fill_between(fu9.reshape(len(fu9),),Pn_NGBoost_9_68p3lower,Pn_NGBoost_9_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax9.fill_between(fu9.reshape(len(fu9),),Pn_NGBoost_9_68p3upper,Pn_NGBoost_9_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax9.fill_between(fu9.reshape(len(fu9),),Pn_NGBoost_9_95p4lower,Pn_NGBoost_9_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
    ax9.fill_between(fu9.reshape(len(fu9),),Pn_NGBoost_9_95p4upper,Pn_NGBoost_9_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

    fu_loc=np.where(fu9==fu)[0].item()
    ax9.scatter(fu,Prd_NGBoost_9[fu_loc],marker='o',facecolors='#4daf4a')
    ax9.scatter(fu,Pn_NGBoost_9[fu_loc],marker='o',facecolors='#08519c')

    ax9.set_ylim(bottom=0)
    ax9.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax9.set_ylabel('$P$ (kN)', fontsize=10)
    ax9.set_xlabel('$f_\mathrm{u}$ (MPa)', fontsize=10)

    if sy>0:
        ax10 = f1.add_subplot(6,2,10 if welding=='Through-holes' else 9)
        ax10.plot(sy10, Prd_NGBoost_10, color='#4daf4a',linewidth=1.5, linestyle='solid')
        ax10.plot(sy10, Pn_NGBoost_10, color='#08519c',linewidth=1.5, linestyle='solid')

        ax10.fill_between(sy10.reshape(len(sy10),),Pn_NGBoost_10_68p3lower,Pn_NGBoost_10_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
        ax10.fill_between(sy10.reshape(len(sy10),),Pn_NGBoost_10_68p3lower,Pn_NGBoost_10_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
        ax10.fill_between(sy10.reshape(len(sy10),),Pn_NGBoost_10_68p3upper,Pn_NGBoost_10_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
        ax10.fill_between(sy10.reshape(len(sy10),),Pn_NGBoost_10_95p4lower,Pn_NGBoost_10_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
        ax10.fill_between(sy10.reshape(len(sy10),),Pn_NGBoost_10_95p4upper,Pn_NGBoost_10_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

        sy_loc=np.where(sy10==sy)[0].item()
        ax10.scatter(sy,Prd_NGBoost_10[sy_loc],marker='o',facecolors='#4daf4a')
        ax10.scatter(sy,Pn_NGBoost_10[sy_loc],marker='o',facecolors='#08519c')

        ax10.set_ylim(bottom=0)
        ax10.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
        ax10.set_ylabel('$P$ (kN)', fontsize=10)
        ax10.set_xlabel('$s_\mathrm{y}$ (mm)', fontsize=10)
        
        
    if sx>0:    
        ax11 = f1.add_subplot(6,2,11 if welding=='Through-holes' and nr==1 else 10 if stud_position=='Two studs in series' and welding=='Through-holes' else 9 if stud_position=='Two studs in series' and welding=='Through-deck' else 11 if stud_position=='Two staggered studs' and welding=='Through-holes' else 10)
        ax11.plot(sx11, Prd_NGBoost_11, color='#4daf4a',linewidth=1.5, linestyle='solid')
        ax11.plot(sx11, Pn_NGBoost_11, color='#08519c',linewidth=1.5, linestyle='solid')

        ax11.fill_between(sx11.reshape(len(sx11),),Pn_NGBoost_11_68p3lower,Pn_NGBoost_11_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
        ax11.fill_between(sx11.reshape(len(sx11),),Pn_NGBoost_11_68p3lower,Pn_NGBoost_11_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
        ax11.fill_between(sx11.reshape(len(sx11),),Pn_NGBoost_11_68p3upper,Pn_NGBoost_11_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
        ax11.fill_between(sx11.reshape(len(sx11),),Pn_NGBoost_11_95p4lower,Pn_NGBoost_11_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
        ax11.fill_between(sx11.reshape(len(sx11),),Pn_NGBoost_11_95p4upper,Pn_NGBoost_11_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

        sx_loc=np.where(sx11==sx)[0].item()
        ax11.scatter(sx,Prd_NGBoost_11[sx_loc],marker='o',facecolors='#4daf4a')
        ax11.scatter(sx,Pn_NGBoost_11[sx_loc],marker='o',facecolors='#08519c')

        ax11.set_ylim(bottom=0)
        ax11.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
        ax11.set_ylabel('$P$ (kN)', fontsize=10)
        ax11.set_xlabel('$s_\mathrm{x}$ (mm)', fontsize=10)

    if (welding=='Through-deck') and (nr==1): legend_yloc=0.26
    elif stud_position=='Two staggered studs' and welding=='Through-holes': legend_yloc=-0.05
    else:legend_yloc=0.095
    
    f1.legend(ncol=3, fontsize=10, bbox_to_anchor=(0.52, legend_yloc), loc='lower center')
    f1.tight_layout()
    st.pyplot(f1)
 

else:
 
    s_diag_in=((sy_in**2)+(sx_in**2))**0.5

    st.write("Specified compressive strength of the concrete with a density $\geq$ 110 pcf: $f'_\mathrm{c}$=", "{:.0f}".format(fprc_psi),' psi')
    st.write('Deck type: ', deck_type)
    st.write('Deck depth excluding longitudinal stiffener on the crest: $h_\mathrm{pn}$=', "{:.3f}".format(hpn_in),' in.')
    st.write('Width of the concrete rib bottom: $b_\mathrm{bot}$=', "{:.3f}".format(bbot_in),' in.')
    st.write('Width of the concrete rib top: $b_\mathrm{top}$=', "{:.3f}".format(btop_in),' in.')
    st.write('Deck thickness: $t$=', "{:.4f}".format(t_in),' in.')
    st.write('Stud shank diameter: $d$=', "{:.3f}".format(d_in),' in.')
    st.write('Stud height after welding: $h_\mathrm{sc}$=', "{:.2f}".format(hsc_in),' in.')
    st.write('Ultimate tensile strength of stud: $f_\mathrm{u}$=', "{:.0f}".format(fu_ksi),' ksi')
    st.write('Stud welding: ', welding)
    st.write('The number of studs per rib: $n_\mathrm{r}$=', "{:.0f}".format(nr))
    st.write('Longitudinal distance from the rib top corner to the nearest stud center: $e_\mathrm{t}$=', "{:.3f}".format(et_in),' in.')
    st.write('Transverse spacing between studs within rib: $s_\mathrm{y}$=', "{:.3f}".format(sy_in),' in.')
    st.write('Longitudinal spacing between studs within rib: $s_\mathrm{x}$=', "{:.3f}".format(sx_in),' in.')
    if stud_position=="Two staggered studs": st.write('Diagonal distance between staggered studs within rib: $s$=', "{:.3f}".format(s_diag_in),' in.')
    
    fprc=fprc_psi/145.038
    hpn=hpn_in*25.4
    bbot=bbot_in*25.4
    btop=btop_in*25.4
    t=t_in*25.4
    d=d_in*25.4
    hsc=hsc_in*25.4
    fu=fu_ksi/0.145038
    et=et_in*25.4
    sy=sy_in*25.4
    sx=sx_in*25.4
    s_diag=s_diag_in*25.4
    
    X_ML_N=np.array([[nr,t,btop,bbot,hpn,sy,sx,et,fprc,fu,d,hsc]]) 
    X_ML_N_NGBoost=NGBoost_sc.transform(X_ML_N) 

    if bbot/btop<=1.0: k_red_NGBoost=k_red_NGBoost_T
    else: k_red_NGBoost=k_red_NGBoost_R

    Qn_NGBoost=NGBoost.predict(X_ML_N_NGBoost)
    Qn_NGBoost_kips=Qn_NGBoost*0.2248/1000
    
    Qn_NGBoost_dist=NGBoost.pred_dist(X_ML_N_NGBoost)

    Qn_NGBoost_68p3lower=(0.2248/1000)*Qn_NGBoost_dist.dist.interval(0.683)[0]
    Qn_NGBoost_68p3upper=(0.2248/1000)*Qn_NGBoost_dist.dist.interval(0.683)[1]

    Qn_NGBoost_95p4lower=(0.2248/1000)*Qn_NGBoost_dist.dist.interval(0.954)[0]
    Qn_NGBoost_95p4upper=(0.2248/1000)*Qn_NGBoost_dist.dist.interval(0.954)[1]

    Qn_NGBoost_99p7lower=(0.2248/1000)*Qn_NGBoost_dist.dist.interval(0.997)[0]
    Qn_NGBoost_99p7upper=(0.2248/1000)*Qn_NGBoost_dist.dist.interval(0.997)[1]

    st.write('##### Predicted stud shear strength')
    
    st.write('Mean nominal stud shear strength, $Q_\mathrm{n,mean}$=', "{:.3f}".format(Qn_NGBoost_kips[0]),' kips')
    st.write('99.7% lower bound of nominal stud shear strength, $Q_\mathrm{n,mean}-3\sigma$=', "{:.3f}".format(Qn_NGBoost_99p7lower[0]),' kips') 
    st.write('99.7% upper bound of nominal stud shear strength, $Q_\mathrm{n,mean}+3\sigma$=', "{:.3f}".format(Qn_NGBoost_99p7upper[0]),' kips') 


    st.write('##### Stud shear strength distribution plot')
    
    val_dist_plot_min=Qn_NGBoost_dist.dist.interval(0.999999)[0]
    val_dist_plot_max=Qn_NGBoost_dist.dist.interval(0.9999)[1]
    distval = Qn_NGBoost_dist.pdf(np.linspace(val_dist_plot_min, val_dist_plot_max,1000).reshape(1,-1)).transpose().reshape(-1,)
    distval_68p3lower=Qn_NGBoost_dist.pdf(Qn_NGBoost_dist.dist.interval(0.683)[0])
    distval_68p3upper=Qn_NGBoost_dist.pdf(Qn_NGBoost_dist.dist.interval(0.683)[1])
    distval_95p4lower=Qn_NGBoost_dist.pdf(Qn_NGBoost_dist.dist.interval(0.954)[0])
    distval_95p4upper=Qn_NGBoost_dist.pdf(Qn_NGBoost_dist.dist.interval(0.954)[1])
    distval_99p7lower=Qn_NGBoost_dist.pdf(Qn_NGBoost_dist.dist.interval(0.997)[0])
    distval_99p7upper=Qn_NGBoost_dist.pdf(Qn_NGBoost_dist.dist.interval(0.997)[1])
    xrange = (0.2248/1000)*np.linspace(val_dist_plot_min, val_dist_plot_max,1000).reshape(-1,)
    f1 = plt.figure(figsize=(6.75,5), dpi=200)
    ax1=f1.add_subplot(1, 1, 1)
    ax1.plot(xrange,distval,c='#08519c',linewidth=1.5,linestyle='solid',label='$Q_\mathrm{n}$ distribution')
    ax1.vlines(Qn_NGBoost_kips,0,np.max(distval),'#e41a1c',linewidth=1.5,linestyle='solid',label='$Q_\mathrm{n,mean}$')
    ax1.fill_between(xrange, distval, 0, where=(xrange>Qn_NGBoost_68p3lower)&(xrange<Qn_NGBoost_68p3upper),color= "#4292c6",alpha= 0.75,label="$Q_\mathrm{n,mean}\pm \sigma$ (68.3% bound)", lw=0.25)
    ax1.fill_between(xrange, distval, 0, where=(xrange>Qn_NGBoost_95p4lower)&(xrange<Qn_NGBoost_68p3lower), color= "#4292c6", alpha= 0.5, label="$Q_\mathrm{n,mean}\pm 2\sigma$ (95.4% bound)", lw=0.25)
    ax1.fill_between(xrange, distval, 0, where=(xrange<Qn_NGBoost_95p4upper)&(xrange>Qn_NGBoost_68p3upper), color= "#4292c6", alpha= 0.5, lw=0.25)
    ax1.fill_between(xrange, distval, 0, where=(xrange>Qn_NGBoost_99p7lower)&(xrange<Qn_NGBoost_95p4lower), color= "#4292c6", alpha= 0.25, label="$Q_\mathrm{n,mean}\pm 3\sigma$ (99.7% bound)", lw=0.25)
    ax1.fill_between(xrange, distval, 0, where=(xrange<Qn_NGBoost_99p7upper)&(xrange>Qn_NGBoost_95p4upper), color= "#4292c6", alpha= 0.25, lw=0.25)
    ax1.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax1.set_ylabel('PDF', fontsize=12)
    ax1.set_xlabel('Stud shear strength (kips)', fontsize=12)
    ax1.legend(loc="lower center", fontsize=10, ncol=2, bbox_to_anchor=(0.5, -0.4))
    f1.tight_layout()
    st.pyplot(f1)    
    
    
    
    
    
    st.write('##### Stud strength plots as functions of design variables')
    
    fprc_psi1=np.array([2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500])
    fprc_psi1=fprc_psi1.reshape(len(fprc_psi1),1)
    fprc1=fprc_psi1/145.038
        
    nr1=np.full((13,1),nr)
    t1=np.full((13,1),t)   
    btop1=np.full((13,1),btop)    
    bbot1=np.full((13,1),bbot)
    hpn1=np.full((13,1),hpn)    
    sy1=np.full((13,1),sy)    
    sx1=np.full((13,1),sx)    
    et1=np.full((13,1),et)
    fu1=np.full((13,1),fu)
    d1=np.full((13,1),d)
    hsc1=np.full((13,1),hsc)
    X_ML_N_1=np.concatenate((nr1,t1,btop1,bbot1,hpn1,sy1,sx1,et1,fprc1,fu1,d1,hsc1), axis=1)
    X_ML_N_NGBoost_1=NGBoost_sc.transform(X_ML_N_1) 
    k_red_NGBoost_1_1=[]
    for i in range(10):
        if btop1[i]/bbot1[i]>=1: new_value=k_red_NGBoost_T
        else: new_value=k_red_NGBoost_R
        k_red_NGBoost_1_1.append(new_value)
    k_red_NGBoost_1=np.array(k_red_NGBoost_1_1)
    Qn_NGBoost_1=NGBoost.predict(X_ML_N_NGBoost_1)
    Qn_NGBoost_1_kips=Qn_NGBoost_1*0.2248/1000
    Qn_NGBoost_dist_1=NGBoost.pred_dist(X_ML_N_NGBoost_1)
    Qn_NGBoost_1_68p3lower=(0.2248/1000)*Qn_NGBoost_dist_1.dist.interval(0.683)[0]
    Qn_NGBoost_1_68p3upper=(0.2248/1000)*Qn_NGBoost_dist_1.dist.interval(0.683)[1]
    Qn_NGBoost_1_95p4lower=(0.2248/1000)*Qn_NGBoost_dist_1.dist.interval(0.954)[0]
    Qn_NGBoost_1_95p4upper=(0.2248/1000)*Qn_NGBoost_dist_1.dist.interval(0.954)[1]
    Qn_NGBoost_1_99p7lower=(0.2248/1000)*Qn_NGBoost_dist_1.dist.interval(0.997)[0]
    Qn_NGBoost_1_99p7upper=(0.2248/1000)*Qn_NGBoost_dist_1.dist.interval(0.997)[1]
    
    
    hpn_in2=np.arange(hpn_in_min,hpn_in_max+0.125,hpn_in_step)
    hpn_in2=hpn_in2.reshape(len(hpn_in2),1)
    hpn2=hpn_in2*25.4
    fprc2=np.full((len(hpn2),1), fprc)
    nr2=np.full((len(hpn2),1),nr)
    t2=np.full((len(hpn2),1),t)   
    btop2=np.full((len(hpn2),1),btop)    
    bbot2=np.full((len(hpn2),1),bbot)
    sy2=np.full((len(hpn2),1),sy)    
    sx2=np.full((len(hpn2),1),sx)    
    et2=np.full((len(hpn2),1),et)
    fu2=np.full((len(hpn2),1),fu)
    d2=np.full((len(hpn2),1),d)
    hsc2=np.full((len(hpn2),1),hsc)

    X_ML_N_2=np.concatenate((nr2,t2,btop2,bbot2,hpn2,sy2,sx2,et2,fprc2,fu2,d2,hsc2), axis=1)
    X_ML_N_NGBoost_2=NGBoost_sc.transform(X_ML_N_2) 

    k_red_NGBoost_2_1=[]
    for i in range(len(hpn2)):
        if btop2[i]/bbot2[i]>=1: new_value=k_red_NGBoost_T
        else: new_value=k_red_NGBoost_R
        k_red_NGBoost_2_1.append(new_value)
    k_red_NGBoost_2=np.array(k_red_NGBoost_2_1)

    Qn_NGBoost_2=NGBoost.predict(X_ML_N_NGBoost_2)
    Qn_NGBoost_2_kips=Qn_NGBoost_2*0.2248/1000
    Qn_NGBoost_dist_2=NGBoost.pred_dist(X_ML_N_NGBoost_2)
    Qn_NGBoost_2_68p3lower=(0.2248/1000)*Qn_NGBoost_dist_2.dist.interval(0.683)[0]
    Qn_NGBoost_2_68p3upper=(0.2248/1000)*Qn_NGBoost_dist_2.dist.interval(0.683)[1]
    Qn_NGBoost_2_95p4lower=(0.2248/1000)*Qn_NGBoost_dist_2.dist.interval(0.954)[0]
    Qn_NGBoost_2_95p4upper=(0.2248/1000)*Qn_NGBoost_dist_2.dist.interval(0.954)[1]
    Qn_NGBoost_2_99p7lower=(0.2248/1000)*Qn_NGBoost_dist_2.dist.interval(0.997)[0]
    Qn_NGBoost_2_99p7upper=(0.2248/1000)*Qn_NGBoost_dist_2.dist.interval(0.997)[1]
    
    
    bbot_in3=np.arange(bbot_in_min,bbot_in_max+0.125,bbot_in_step)
    bbot_in3=bbot_in3.reshape(len(bbot_in3),1)
    bbot3=bbot_in3*25.4
    hpn3=np.full((len(bbot3),1), hpn)
    fprc3=np.full((len(bbot3),1), fprc)
    nr3=np.full((len(bbot3),1),nr)
    t3=np.full((len(bbot3),1),t)   
    btop3=np.full((len(bbot3),1),btop)    
    sy3=np.full((len(bbot3),1),sy)    
    sx3=np.full((len(bbot3),1),sx)    
    et3=np.full((len(bbot3),1),et)
    fu3=np.full((len(bbot3),1),fu)
    d3=np.full((len(bbot3),1),d)
    hsc3=np.full((len(bbot3),1),hsc)

    X_ML_N_3=np.concatenate((nr3,t3,btop3,bbot3,hpn3,sy3,sx3,et3,fprc3,fu3,d3,hsc3), axis=1)
    X_ML_N_NGBoost_3=NGBoost_sc.transform(X_ML_N_3) 

    k_red_NGBoost_3_1=[]
    for i in range(len(bbot3)):
        if btop3[i]/bbot3[i]>=1: new_value=k_red_NGBoost_T
        else: new_value=k_red_NGBoost_R
        k_red_NGBoost_3_1.append(new_value)
    k_red_NGBoost_3=np.array(k_red_NGBoost_3_1)

    Qn_NGBoost_3=NGBoost.predict(X_ML_N_NGBoost_3)
    Qn_NGBoost_3_kips=Qn_NGBoost_3*0.2248/1000
    Qn_NGBoost_dist_3=NGBoost.pred_dist(X_ML_N_NGBoost_3)
    Qn_NGBoost_3_68p3lower=(0.2248/1000)*Qn_NGBoost_dist_3.dist.interval(0.683)[0]
    Qn_NGBoost_3_68p3upper=(0.2248/1000)*Qn_NGBoost_dist_3.dist.interval(0.683)[1]
    Qn_NGBoost_3_95p4lower=(0.2248/1000)*Qn_NGBoost_dist_3.dist.interval(0.954)[0]
    Qn_NGBoost_3_95p4upper=(0.2248/1000)*Qn_NGBoost_dist_3.dist.interval(0.954)[1]
    Qn_NGBoost_3_99p7lower=(0.2248/1000)*Qn_NGBoost_dist_3.dist.interval(0.997)[0]
    Qn_NGBoost_3_99p7upper=(0.2248/1000)*Qn_NGBoost_dist_3.dist.interval(0.997)[1]
    
    
    btop_in4=np.arange(btop_in_min,btop_in_max+0.125,btop_in_step)
    btop_in4=btop_in4.reshape(len(btop_in4),1)
    btop4=btop_in4*25.4
    hpn4=np.full((len(btop4),1), hpn)
    fprc4=np.full((len(btop4),1), fprc)
    nr4=np.full((len(btop4),1),nr)
    t4=np.full((len(btop4),1),t)   
    bbot4=np.full((len(btop4),1),bbot)    
    sy4=np.full((len(btop4),1),sy)    
    sx4=np.full((len(btop4),1),sx)    
    et4=np.full((len(btop4),1),et)
    fu4=np.full((len(btop4),1),fu)
    d4=np.full((len(btop4),1),d)
    hsc4=np.full((len(btop4),1),hsc)

    X_ML_N_4=np.concatenate((nr4,t4,btop4,bbot4,hpn4,sy4,sx4,et4,fprc4,fu4,d4,hsc4), axis=1)
    X_ML_N_NGBoost_4=NGBoost_sc.transform(X_ML_N_4) 

    k_red_NGBoost_4_1=[]
    for i in range(len(btop4)):
        if btop4[i]/bbot4[i]>=1: new_value=k_red_NGBoost_T
        else: new_value=k_red_NGBoost_R
        k_red_NGBoost_4_1.append(new_value)
    k_red_NGBoost_4=np.array(k_red_NGBoost_4_1)

    Qn_NGBoost_4=NGBoost.predict(X_ML_N_NGBoost_4)
    Qn_NGBoost_4_kips=Qn_NGBoost_4*0.2248/1000
    Qn_NGBoost_dist_4=NGBoost.pred_dist(X_ML_N_NGBoost_4)
    Qn_NGBoost_4_68p3lower=(0.2248/1000)*Qn_NGBoost_dist_4.dist.interval(0.683)[0]
    Qn_NGBoost_4_68p3upper=(0.2248/1000)*Qn_NGBoost_dist_4.dist.interval(0.683)[1]
    Qn_NGBoost_4_95p4lower=(0.2248/1000)*Qn_NGBoost_dist_4.dist.interval(0.954)[0]
    Qn_NGBoost_4_95p4upper=(0.2248/1000)*Qn_NGBoost_dist_4.dist.interval(0.954)[1]
    Qn_NGBoost_4_99p7lower=(0.2248/1000)*Qn_NGBoost_dist_4.dist.interval(0.997)[0]
    Qn_NGBoost_4_99p7upper=(0.2248/1000)*Qn_NGBoost_dist_4.dist.interval(0.997)[1]
    
    
    t_in5=np.arange(t_in_min,t_in_max+0.0001,t_in_step)
    t_in5=t_in5.reshape(len(t_in5),1)
    t5=t_in5*25.4
    hpn5=np.full((len(t5),1), hpn)
    fprc5=np.full((len(t5),1), fprc)
    nr5=np.full((len(t5),1),nr)
    btop5=np.full((len(t5),1),btop)   
    bbot5=np.full((len(t5),1),bbot)    
    sy5=np.full((len(t5),1),sy)    
    sx5=np.full((len(t5),1),sx)    
    et5=np.full((len(t5),1),et)
    fu5=np.full((len(t5),1),fu)
    d5=np.full((len(t5),1),d)
    hsc5=np.full((len(t5),1),hsc)

    X_ML_N_5=np.concatenate((nr5,t5,btop5,bbot5,hpn5,sy5,sx5,et5,fprc5,fu5,d5,hsc5), axis=1)
    X_ML_N_NGBoost_5=NGBoost_sc.transform(X_ML_N_5) 

    k_red_NGBoost_5_1=[]
    for i in range(len(t5)):
        if btop5[i]/bbot5[i]>=1: new_value=k_red_NGBoost_T
        else: new_value=k_red_NGBoost_R
        k_red_NGBoost_5_1.append(new_value)
    k_red_NGBoost_5=np.array(k_red_NGBoost_5_1)

    Qn_NGBoost_5=NGBoost.predict(X_ML_N_NGBoost_5)
    Qn_NGBoost_5_kips=Qn_NGBoost_5*0.2248/1000
    Qn_NGBoost_dist_5=NGBoost.pred_dist(X_ML_N_NGBoost_5)
    Qn_NGBoost_5_68p3lower=(0.2248/1000)*Qn_NGBoost_dist_5.dist.interval(0.683)[0]
    Qn_NGBoost_5_68p3upper=(0.2248/1000)*Qn_NGBoost_dist_5.dist.interval(0.683)[1]
    Qn_NGBoost_5_95p4lower=(0.2248/1000)*Qn_NGBoost_dist_5.dist.interval(0.954)[0]
    Qn_NGBoost_5_95p4upper=(0.2248/1000)*Qn_NGBoost_dist_5.dist.interval(0.954)[1]
    Qn_NGBoost_5_99p7lower=(0.2248/1000)*Qn_NGBoost_dist_5.dist.interval(0.997)[0]
    Qn_NGBoost_5_99p7upper=(0.2248/1000)*Qn_NGBoost_dist_5.dist.interval(0.997)[1]  
 
    if welding=='Through-holes':   
        d_in6=np.arange(d_in_min,d_in_max+0.125,d_in_step)
        d_in6=d_in6.reshape(len(d_in6),1)
        d6=d_in6*25.4
        hpn6=np.full((len(d6),1), hpn)
        fprc6=np.full((len(d6),1), fprc)
        nr6=np.full((len(d6),1),nr)
        btop6=np.full((len(d6),1),btop)   
        bbot6=np.full((len(d6),1),bbot)    
        sy6=np.full((len(d6),1),sy)    
        sx6=np.full((len(d6),1),sx)    
        et6=np.full((len(d6),1),et)
        fu6=np.full((len(d6),1),fu)
        t6=np.full((len(d6),1),t)
        hsc6=np.full((len(d6),1),hsc)

        X_ML_N_6=np.concatenate((nr6,t6,btop6,bbot6,hpn6,sy6,sx6,et6,fprc6,fu6,d6,hsc6), axis=1)
        X_ML_N_NGBoost_6=NGBoost_sc.transform(X_ML_N_6)  

        k_red_NGBoost_6_1=[]
        for i in range(len(d6)):
            if btop6[i]/bbot6[i]>=1: new_value=k_red_NGBoost_T
            else: new_value=k_red_NGBoost_R
            k_red_NGBoost_6_1.append(new_value)
        k_red_NGBoost_6=np.array(k_red_NGBoost_6_1)

        Qn_NGBoost_6=NGBoost.predict(X_ML_N_NGBoost_6)
        Qn_NGBoost_6_kips=Qn_NGBoost_6*0.2248/1000
        Qn_NGBoost_dist_6=NGBoost.pred_dist(X_ML_N_NGBoost_6)
        Qn_NGBoost_6_68p3lower=(0.2248/1000)*Qn_NGBoost_dist_6.dist.interval(0.683)[0]
        Qn_NGBoost_6_68p3upper=(0.2248/1000)*Qn_NGBoost_dist_6.dist.interval(0.683)[1]
        Qn_NGBoost_6_95p4lower=(0.2248/1000)*Qn_NGBoost_dist_6.dist.interval(0.954)[0]
        Qn_NGBoost_6_95p4upper=(0.2248/1000)*Qn_NGBoost_dist_6.dist.interval(0.954)[1]
        Qn_NGBoost_6_99p7lower=(0.2248/1000)*Qn_NGBoost_dist_6.dist.interval(0.997)[0]
        Qn_NGBoost_6_99p7upper=(0.2248/1000)*Qn_NGBoost_dist_6.dist.interval(0.997)[1] 
    
    
    hsc_in7=np.arange(hsc_in_min,hsc_in_max+0.125,hsc_in_step)
    hsc_in7=hsc_in7.reshape(len(hsc_in7),1)
    hsc7=hsc_in7*25.4
    hpn7=np.full((len(hsc7),1), hpn)
    fprc7=np.full((len(hsc7),1), fprc)
    nr7=np.full((len(hsc7),1),nr)
    btop7=np.full((len(hsc7),1),btop)   
    bbot7=np.full((len(hsc7),1),bbot)    
    sy7=np.full((len(hsc7),1),sy)    
    sx7=np.full((len(hsc7),1),sx)    
    et7=np.full((len(hsc7),1),et)
    fu7=np.full((len(hsc7),1),fu)
    t7=np.full((len(hsc7),1),t)
    d7=np.full((len(hsc7),1),d)

    X_ML_N_7=np.concatenate((nr7,t7,btop7,bbot7,hpn7,sy7,sx7,et7,fprc7,fu7,d7,hsc7), axis=1)
    X_ML_N_NGBoost_7=NGBoost_sc.transform(X_ML_N_7) 

    k_red_NGBoost_7_1=[]
    for i in range(len(hsc7)):
        if btop7[i]/bbot7[i]>=1: new_value=k_red_NGBoost_T
        else: new_value=k_red_NGBoost_R
        k_red_NGBoost_7_1.append(new_value)
    k_red_NGBoost_7=np.array(k_red_NGBoost_7_1)

    Qn_NGBoost_7=NGBoost.predict(X_ML_N_NGBoost_7)
    Qn_NGBoost_7_kips=Qn_NGBoost_7*0.2248/1000
    Qn_NGBoost_dist_7=NGBoost.pred_dist(X_ML_N_NGBoost_7)
    Qn_NGBoost_7_68p3lower=(0.2248/1000)*Qn_NGBoost_dist_7.dist.interval(0.683)[0]
    Qn_NGBoost_7_68p3upper=(0.2248/1000)*Qn_NGBoost_dist_7.dist.interval(0.683)[1]
    Qn_NGBoost_7_95p4lower=(0.2248/1000)*Qn_NGBoost_dist_7.dist.interval(0.954)[0]
    Qn_NGBoost_7_95p4upper=(0.2248/1000)*Qn_NGBoost_dist_7.dist.interval(0.954)[1]
    Qn_NGBoost_7_99p7lower=(0.2248/1000)*Qn_NGBoost_dist_7.dist.interval(0.997)[0]
    Qn_NGBoost_7_99p7upper=(0.2248/1000)*Qn_NGBoost_dist_7.dist.interval(0.997)[1]  
    
    
    et_in8=np.arange(et_in_min,et_in_max+0.125,et_in_step)
    et_in8=et_in8.reshape(len(et_in8),1)
    et8=et_in8*25.4
    hpn8=np.full((len(et8),1), hpn)
    fprc8=np.full((len(et8),1), fprc)
    nr8=np.full((len(et8),1),nr)
    btop8=np.full((len(et8),1),btop)   
    bbot8=np.full((len(et8),1),bbot)    
    sy8=np.full((len(et8),1),sy)    
    sx8=np.full((len(et8),1),sx)    
    hsc8=np.full((len(et8),1),hsc)
    fu8=np.full((len(et8),1),fu)
    t8=np.full((len(et8),1),t)
    d8=np.full((len(et8),1),d)

    X_ML_N_8=np.concatenate((nr8,t8,btop8,bbot8,hpn8,sy8,sx8,et8,fprc8,fu8,d8,hsc8), axis=1)
    X_ML_N_NGBoost_8=NGBoost_sc.transform(X_ML_N_8) 

    k_red_NGBoost_8_1=[]
    for i in range(len(et8)):
        if btop8[i]/bbot8[i]>=1: new_value=k_red_NGBoost_T
        else: new_value=k_red_NGBoost_R
        k_red_NGBoost_8_1.append(new_value)
    k_red_NGBoost_8=np.array(k_red_NGBoost_8_1)

    Qn_NGBoost_8=NGBoost.predict(X_ML_N_NGBoost_8)
    Qn_NGBoost_8_kips=Qn_NGBoost_8*0.2248/1000
    Qn_NGBoost_dist_8=NGBoost.pred_dist(X_ML_N_NGBoost_8)
    Qn_NGBoost_8_68p3lower=(0.2248/1000)*Qn_NGBoost_dist_8.dist.interval(0.683)[0]
    Qn_NGBoost_8_68p3upper=(0.2248/1000)*Qn_NGBoost_dist_8.dist.interval(0.683)[1]
    Qn_NGBoost_8_95p4lower=(0.2248/1000)*Qn_NGBoost_dist_8.dist.interval(0.954)[0]
    Qn_NGBoost_8_95p4upper=(0.2248/1000)*Qn_NGBoost_dist_8.dist.interval(0.954)[1]
    Qn_NGBoost_8_99p7lower=(0.2248/1000)*Qn_NGBoost_dist_8.dist.interval(0.997)[0]
    Qn_NGBoost_8_99p7upper=(0.2248/1000)*Qn_NGBoost_dist_8.dist.interval(0.997)[1]  
    
    
    fu_ksi9=np.arange(fu_ksi_min,fu_ksi_max+1,fu_ksi_step)
    fu_ksi9=fu_ksi9.reshape(len(fu_ksi9),1)
    fu9=fu_ksi9/0.145038
    hpn9=np.full((len(fu9),1), hpn)
    fprc9=np.full((len(fu9),1), fprc)
    nr9=np.full((len(fu9),1),nr)
    btop9=np.full((len(fu9),1),btop)   
    bbot9=np.full((len(fu9),1),bbot)    
    sy9=np.full((len(fu9),1),sy)    
    sx9=np.full((len(fu9),1),sx)    
    hsc9=np.full((len(fu9),1),hsc)
    et9=np.full((len(fu9),1),et)
    t9=np.full((len(fu9),1),t)
    d9=np.full((len(fu9),1),d)

    X_ML_N_9=np.concatenate((nr9,t9,btop9,bbot9,hpn9,sy9,sx9,et9,fprc9,fu9,d9,hsc9), axis=1)
    X_ML_N_NGBoost_9=NGBoost_sc.transform(X_ML_N_9) 

    k_red_NGBoost_9_1=[]
    for i in range(len(fu9)):
        if btop9[i]/bbot9[i]>=1: new_value=k_red_NGBoost_T
        else: new_value=k_red_NGBoost_R
        k_red_NGBoost_9_1.append(new_value)
    k_red_NGBoost_9=np.array(k_red_NGBoost_9_1)

    Qn_NGBoost_9=NGBoost.predict(X_ML_N_NGBoost_9)
    Qn_NGBoost_9_kips=Qn_NGBoost_9*0.2248/1000
    Qn_NGBoost_dist_9=NGBoost.pred_dist(X_ML_N_NGBoost_9)
    Qn_NGBoost_9_68p3lower=(0.2248/1000)*Qn_NGBoost_dist_9.dist.interval(0.683)[0]
    Qn_NGBoost_9_68p3upper=(0.2248/1000)*Qn_NGBoost_dist_9.dist.interval(0.683)[1]
    Qn_NGBoost_9_95p4lower=(0.2248/1000)*Qn_NGBoost_dist_9.dist.interval(0.954)[0]
    Qn_NGBoost_9_95p4upper=(0.2248/1000)*Qn_NGBoost_dist_9.dist.interval(0.954)[1]
    Qn_NGBoost_9_99p7lower=(0.2248/1000)*Qn_NGBoost_dist_9.dist.interval(0.997)[0]
    Qn_NGBoost_9_99p7upper=(0.2248/1000)*Qn_NGBoost_dist_9.dist.interval(0.997)[1]
    
    
    if sy_in>0:
        sy_in10=np.arange(sy_in_min,sy_in_max+0.125,sy_in_step)
        sy_in10=sy_in10.reshape(len(sy_in10),1)
        sy10=sy_in10*25.4
        hpn10=np.full((len(sy10),1), hpn)
        fprc10=np.full((len(sy10),1), fprc)
        nr10=np.full((len(sy10),1),nr)
        btop10=np.full((len(sy10),1),btop)   
        bbot10=np.full((len(sy10),1),bbot)    
        fu10=np.full((len(sy10),1),fu)    
        sx10=np.full((len(sy10),1),sx)    
        hsc10=np.full((len(sy10),1),hsc)
        et10=np.full((len(sy10),1),et)
        t10=np.full((len(sy10),1),t)
        d10=np.full((len(sy10),1),d)

        X_ML_N_10=np.concatenate((nr10,t10,btop10,bbot10,hpn10,sy10,sx10,et10,fprc10,fu10,d10,hsc10), axis=1)
        X_ML_N_NGBoost_10=NGBoost_sc.transform(X_ML_N_10) 

        k_red_NGBoost_10_1=[]
        for i in range(len(sy10)):
            if btop10[i]/bbot10[i]>=1: new_value=k_red_NGBoost_T
            else: new_value=k_red_NGBoost_R
            k_red_NGBoost_10_1.append(new_value)
        k_red_NGBoost_10=np.array(k_red_NGBoost_10_1)

        Qn_NGBoost_10=NGBoost.predict(X_ML_N_NGBoost_10)
        Qn_NGBoost_10_kips=Qn_NGBoost_10*0.2248/1000
        Qn_NGBoost_dist_10=NGBoost.pred_dist(X_ML_N_NGBoost_10)
        Qn_NGBoost_10_68p3lower=(0.2248/1000)*Qn_NGBoost_dist_10.dist.interval(0.683)[0]
        Qn_NGBoost_10_68p3upper=(0.2248/1000)*Qn_NGBoost_dist_10.dist.interval(0.683)[1]
        Qn_NGBoost_10_95p4lower=(0.2248/1000)*Qn_NGBoost_dist_10.dist.interval(0.954)[0]
        Qn_NGBoost_10_95p4upper=(0.2248/1000)*Qn_NGBoost_dist_10.dist.interval(0.954)[1]
        Qn_NGBoost_10_99p7lower=(0.2248/1000)*Qn_NGBoost_dist_10.dist.interval(0.997)[0]
        Qn_NGBoost_10_99p7upper=(0.2248/1000)*Qn_NGBoost_dist_10.dist.interval(0.997)[1]
        
        
    if sx_in>0:
        sx_in11=np.arange(sx_in_min,sx_in_max+0.125,sx_in_step)
        sx_in11=sx_in11.reshape(len(sx_in11),1)
        sx11=sx_in11*25.4
        hpn11=np.full((len(sx11),1), hpn)
        fprc11=np.full((len(sx11),1), fprc)
        nr11=np.full((len(sx11),1),nr)
        btop11=np.full((len(sx11),1),btop)   
        bbot11=np.full((len(sx11),1),bbot)    
        fu11=np.full((len(sx11),1),fu)    
        sy11=np.full((len(sx11),1),sy)    
        hsc11=np.full((len(sx11),1),hsc)
        et11=np.full((len(sx11),1),et)
        t11=np.full((len(sx11),1),t)
        d11=np.full((len(sx11),1),d)

        X_ML_N_11=np.concatenate((nr11,t11,btop11,bbot11,hpn11,sy11,sx11,et11,fprc11,fu11,d11,hsc11), axis=1)
        X_ML_N_NGBoost_11=NGBoost_sc.transform(X_ML_N_11) 

        k_red_NGBoost_11_1=[]
        for i in range(len(sx11)):
            if btop11[i]/bbot11[i]>=1: new_value=k_red_NGBoost_T
            else: new_value=k_red_NGBoost_R
            k_red_NGBoost_11_1.append(new_value)
        k_red_NGBoost_11=np.array(k_red_NGBoost_11_1)

        Qn_NGBoost_11=NGBoost.predict(X_ML_N_NGBoost_11)
        Qn_NGBoost_11_kips=Qn_NGBoost_11*0.2248/1000
        Qn_NGBoost_dist_11=NGBoost.pred_dist(X_ML_N_NGBoost_11)
        Qn_NGBoost_11_68p3lower=(0.2248/1000)*Qn_NGBoost_dist_11.dist.interval(0.683)[0]
        Qn_NGBoost_11_68p3upper=(0.2248/1000)*Qn_NGBoost_dist_11.dist.interval(0.683)[1]
        Qn_NGBoost_11_95p4lower=(0.2248/1000)*Qn_NGBoost_dist_11.dist.interval(0.954)[0]
        Qn_NGBoost_11_95p4upper=(0.2248/1000)*Qn_NGBoost_dist_11.dist.interval(0.954)[1]
        Qn_NGBoost_11_99p7lower=(0.2248/1000)*Qn_NGBoost_dist_11.dist.interval(0.997)[0]
        Qn_NGBoost_11_99p7upper=(0.2248/1000)*Qn_NGBoost_dist_11.dist.interval(0.997)[1]
        
        
    f1 = plt.figure(figsize=(6.75,4*3), dpi=200)

    ax1 = f1.add_subplot(6,2,1)
    ax1.plot(fprc_psi1, Qn_NGBoost_1_kips, color='#08519c',linewidth=1.5, label='$Q_\mathrm{n,mean}$',linestyle='solid')

    ax1.fill_between(fprc_psi1.reshape(len(fprc_psi1),),Qn_NGBoost_1_68p3lower,Qn_NGBoost_1_68p3upper, color= "#4292c6",alpha= 0.75,label="$Q_\mathrm{n,mean}\pm \sigma$ (68.3% bound)", lw=0.25)
    ax1.fill_between(fprc_psi1.reshape(len(fprc_psi1),),Qn_NGBoost_1_68p3lower,Qn_NGBoost_1_95p4lower, color= "#4292c6",alpha= 0.5,label="$Q_\mathrm{n,mean}\pm 2\sigma$ (95.4% bound)", lw=0.25)
    ax1.fill_between(fprc_psi1.reshape(len(fprc_psi1),),Qn_NGBoost_1_68p3upper,Qn_NGBoost_1_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax1.fill_between(fprc_psi1.reshape(len(fprc_psi1),),Qn_NGBoost_1_95p4lower,Qn_NGBoost_1_99p7lower, color= "#4292c6",alpha= 0.25,label="$Q_\mathrm{n,mean}\pm 3\sigma$ (99.7% bound)", lw=0.25)
    ax1.fill_between(fprc_psi1.reshape(len(fprc_psi1),),Qn_NGBoost_1_95p4upper,Qn_NGBoost_1_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

    fprc_loc=np.where(fprc_psi1==fprc_psi)[0].item()
    ax1.scatter(fprc_psi,Qn_NGBoost_1_kips[fprc_loc],marker='o',facecolors='#08519c')

    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax1.set_ylabel('$Q$ (kips)', fontsize=10)
    ax1.set_xlabel("$f'_\mathrm{c}$ (psi)", fontsize=10)


    ax2 = f1.add_subplot(6,2,2)
    ax2.plot(hpn_in2, Qn_NGBoost_2_kips, color='#08519c',linewidth=1.5, linestyle='solid')

    ax2.fill_between(hpn_in2.reshape(len(hpn_in2),),Qn_NGBoost_2_68p3lower,Qn_NGBoost_2_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
    ax2.fill_between(hpn_in2.reshape(len(hpn_in2),),Qn_NGBoost_2_68p3lower,Qn_NGBoost_2_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax2.fill_between(hpn_in2.reshape(len(hpn_in2),),Qn_NGBoost_2_68p3upper,Qn_NGBoost_2_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax2.fill_between(hpn_in2.reshape(len(hpn_in2),),Qn_NGBoost_2_95p4lower,Qn_NGBoost_2_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
    ax2.fill_between(hpn_in2.reshape(len(hpn_in2),),Qn_NGBoost_2_95p4upper,Qn_NGBoost_2_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

    hpn_in_loc=np.where(hpn_in2==hpn_in)[0].item()
    ax2.scatter(hpn_in,Qn_NGBoost_2_kips[hpn_in_loc],marker='o',facecolors='#08519c')

    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax2.set_ylabel('$Q$ (kips)', fontsize=10)
    ax2.set_xlabel('$h_\mathrm{pn}$ (in.)', fontsize=10)


    ax3 = f1.add_subplot(6,2,3)
    ax3.plot(bbot_in3, Qn_NGBoost_3_kips, color='#08519c',linewidth=1.5, linestyle='solid')

    ax3.fill_between(bbot_in3.reshape(len(bbot_in3),),Qn_NGBoost_3_68p3lower,Qn_NGBoost_3_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
    ax3.fill_between(bbot_in3.reshape(len(bbot_in3),),Qn_NGBoost_3_68p3lower,Qn_NGBoost_3_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax3.fill_between(bbot_in3.reshape(len(bbot_in3),),Qn_NGBoost_3_68p3upper,Qn_NGBoost_3_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax3.fill_between(bbot_in3.reshape(len(bbot_in3),),Qn_NGBoost_3_95p4lower,Qn_NGBoost_3_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
    ax3.fill_between(bbot_in3.reshape(len(bbot_in3),),Qn_NGBoost_3_95p4upper,Qn_NGBoost_3_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

    bbot_in_loc=np.where(bbot_in3==bbot_in)[0].item()
    ax3.scatter(bbot_in,Qn_NGBoost_3_kips[bbot_in_loc],marker='o',facecolors='#08519c')

    ax3.set_ylim(bottom=0)
    ax3.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax3.set_ylabel('$Q$ (kips)', fontsize=10)
    ax3.set_xlabel('$b_\mathrm{bot}$ (in.)', fontsize=10)



    ax4 = f1.add_subplot(6,2,4)
    ax4.plot(btop_in4, Qn_NGBoost_4_kips, color='#08519c',linewidth=1.5, linestyle='solid')

    ax4.fill_between(btop_in4.reshape(len(btop_in4),),Qn_NGBoost_4_68p3lower,Qn_NGBoost_4_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
    ax4.fill_between(btop_in4.reshape(len(btop_in4),),Qn_NGBoost_4_68p3lower,Qn_NGBoost_4_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax4.fill_between(btop_in4.reshape(len(btop_in4),),Qn_NGBoost_4_68p3upper,Qn_NGBoost_4_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax4.fill_between(btop_in4.reshape(len(btop_in4),),Qn_NGBoost_4_95p4lower,Qn_NGBoost_4_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
    ax4.fill_between(btop_in4.reshape(len(btop_in4),),Qn_NGBoost_4_95p4upper,Qn_NGBoost_4_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

    btop_in_loc=np.where(btop_in4==btop_in)[0].item()
    ax4.scatter(btop_in,Qn_NGBoost_4_kips[btop_in_loc],marker='o',facecolors='#08519c')

    ax4.set_ylim(bottom=0)
    ax4.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax4.set_ylabel('$Q$ (kips)', fontsize=10)
    ax4.set_xlabel('$b_\mathrm{top}$ (in.)', fontsize=10)



    ax5 = f1.add_subplot(6,2,5)
    ax5.plot(t_in5, Qn_NGBoost_5_kips, color='#08519c',linewidth=1.5, linestyle='solid')

    ax5.fill_between(t_in5.reshape(len(t_in5),),Qn_NGBoost_5_68p3lower,Qn_NGBoost_5_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
    ax5.fill_between(t_in5.reshape(len(t_in5),),Qn_NGBoost_5_68p3lower,Qn_NGBoost_5_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax5.fill_between(t_in5.reshape(len(t_in5),),Qn_NGBoost_5_68p3upper,Qn_NGBoost_5_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax5.fill_between(t_in5.reshape(len(t_in5),),Qn_NGBoost_5_95p4lower,Qn_NGBoost_5_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
    ax5.fill_between(t_in5.reshape(len(t_in5),),Qn_NGBoost_5_95p4upper,Qn_NGBoost_5_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

    t_in_loc=np.where(abs(t_in5-t_in)<0.00001)[0].item()
    ax5.scatter(t_in,Qn_NGBoost_5_kips[t_in_loc],marker='o',facecolors='#08519c')

    ax5.set_ylim(bottom=0)
    ax5.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax5.set_ylabel('$Q$ (kips)', fontsize=10)
    ax5.set_xlabel('$t$ (in.)', fontsize=10)


    if welding=='Through-holes': 
        ax6 = f1.add_subplot(6,2,6)
        ax6.plot(d_in6, Qn_NGBoost_6_kips, color='#08519c',linewidth=1.5, linestyle='solid')

        ax6.fill_between(d_in6.reshape(len(d_in6),),Qn_NGBoost_6_68p3lower,Qn_NGBoost_6_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
        ax6.fill_between(d_in6.reshape(len(d_in6),),Qn_NGBoost_6_68p3lower,Qn_NGBoost_6_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
        ax6.fill_between(d_in6.reshape(len(d_in6),),Qn_NGBoost_6_68p3upper,Qn_NGBoost_6_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
        ax6.fill_between(d_in6.reshape(len(d_in6),),Qn_NGBoost_6_95p4lower,Qn_NGBoost_6_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
        ax6.fill_between(d_in6.reshape(len(d_in6),),Qn_NGBoost_6_95p4upper,Qn_NGBoost_6_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

        d_in_loc=np.where(d_in6==d_in)[0].item()
        ax6.scatter(d_in,Qn_NGBoost_6_kips[d_in_loc],marker='o',facecolors='#08519c')

        ax6.set_ylim(bottom=0)
        ax6.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
        ax6.set_ylabel('$Q$ (kips)', fontsize=10)
        ax6.set_xlabel('$d$ (in.)', fontsize=10)



    ax7 = f1.add_subplot(6,2,7 if welding=='Through-holes' else 6)
    ax7.plot(hsc_in7, Qn_NGBoost_7_kips, color='#08519c',linewidth=1.5, linestyle='solid')

    ax7.fill_between(hsc_in7.reshape(len(hsc_in7),),Qn_NGBoost_7_68p3lower,Qn_NGBoost_7_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
    ax7.fill_between(hsc_in7.reshape(len(hsc_in7),),Qn_NGBoost_7_68p3lower,Qn_NGBoost_7_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax7.fill_between(hsc_in7.reshape(len(hsc_in7),),Qn_NGBoost_7_68p3upper,Qn_NGBoost_7_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax7.fill_between(hsc_in7.reshape(len(hsc_in7),),Qn_NGBoost_7_95p4lower,Qn_NGBoost_7_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
    ax7.fill_between(hsc_in7.reshape(len(hsc_in7),),Qn_NGBoost_7_95p4upper,Qn_NGBoost_7_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

    hsc_in_loc=np.where(hsc_in7==hsc_in)[0].item()
    ax7.scatter(hsc_in,Qn_NGBoost_7_kips[hsc_in_loc],marker='o',facecolors='#08519c')

    ax7.set_ylim(bottom=0)
    ax7.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax7.set_ylabel('$Q$ (kpis)', fontsize=10)
    ax7.set_xlabel('$h_\mathrm{sc}$ (in.)', fontsize=10)



    ax8 = f1.add_subplot(6,2,8 if welding=='Through-holes' else 7)
    ax8.plot(et_in8, Qn_NGBoost_8_kips, color='#08519c',linewidth=1.5, linestyle='solid')

    ax8.fill_between(et_in8.reshape(len(et_in8),),Qn_NGBoost_8_68p3lower,Qn_NGBoost_8_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
    ax8.fill_between(et_in8.reshape(len(et_in8),),Qn_NGBoost_8_68p3lower,Qn_NGBoost_8_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax8.fill_between(et_in8.reshape(len(et_in8),),Qn_NGBoost_8_68p3upper,Qn_NGBoost_8_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax8.fill_between(et_in8.reshape(len(et_in8),),Qn_NGBoost_8_95p4lower,Qn_NGBoost_8_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
    ax8.fill_between(et_in8.reshape(len(et_in8),),Qn_NGBoost_8_95p4upper,Qn_NGBoost_8_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

    et_in_loc=np.where(et_in8==et_in)[0].item()
    ax8.scatter(et_in,Qn_NGBoost_8_kips[et_in_loc],marker='o',facecolors='#08519c')

    ax8.set_ylim(bottom=0)
    ax8.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax8.set_ylabel('$Q$ (kips)', fontsize=10)
    ax8.set_xlabel('$e_\mathrm{t}$ (in.)', fontsize=10)



    ax9 = f1.add_subplot(6,2,9 if welding=='Through-holes' else 8)
    ax9.plot(fu_ksi9, Qn_NGBoost_9_kips, color='#08519c',linewidth=1.5, linestyle='solid')

    ax9.fill_between(fu_ksi9.reshape(len(fu_ksi9),),Qn_NGBoost_9_68p3lower,Qn_NGBoost_9_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
    ax9.fill_between(fu_ksi9.reshape(len(fu_ksi9),),Qn_NGBoost_9_68p3lower,Qn_NGBoost_9_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax9.fill_between(fu_ksi9.reshape(len(fu_ksi9),),Qn_NGBoost_9_68p3upper,Qn_NGBoost_9_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
    ax9.fill_between(fu_ksi9.reshape(len(fu_ksi9),),Qn_NGBoost_9_95p4lower,Qn_NGBoost_9_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
    ax9.fill_between(fu_ksi9.reshape(len(fu_ksi9),),Qn_NGBoost_9_95p4upper,Qn_NGBoost_9_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

    fu_ksi_loc=np.where(fu_ksi9==fu_ksi)[0].item()
    ax9.scatter(fu_ksi,Qn_NGBoost_9_kips[fu_ksi_loc],marker='o',facecolors='#08519c')

    ax9.set_ylim(bottom=0)
    ax9.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
    ax9.set_ylabel('$Q$ (kips)', fontsize=10)
    ax9.set_xlabel('$f_\mathrm{u}$ (ksi)', fontsize=10)

    if sy>0:
        ax10 = f1.add_subplot(6,2,10 if welding=='Through-holes' else 9)
        ax10.plot(sy_in10, Qn_NGBoost_10_kips, color='#08519c',linewidth=1.5, linestyle='solid')

        ax10.fill_between(sy_in10.reshape(len(sy_in10),),Qn_NGBoost_10_68p3lower,Qn_NGBoost_10_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
        ax10.fill_between(sy_in10.reshape(len(sy_in10),),Qn_NGBoost_10_68p3lower,Qn_NGBoost_10_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
        ax10.fill_between(sy_in10.reshape(len(sy_in10),),Qn_NGBoost_10_68p3upper,Qn_NGBoost_10_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
        ax10.fill_between(sy_in10.reshape(len(sy_in10),),Qn_NGBoost_10_95p4lower,Qn_NGBoost_10_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
        ax10.fill_between(sy_in10.reshape(len(sy_in10),),Qn_NGBoost_10_95p4upper,Qn_NGBoost_10_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

        sy_in_loc=np.where(sy_in10==sy_in)[0].item()
        ax10.scatter(sy_in,Qn_NGBoost_10_kips[sy_in_loc],marker='o',facecolors='#08519c')

        ax10.set_ylim(bottom=0)
        ax10.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
        ax10.set_ylabel('$Q$ (kips)', fontsize=10)
        ax10.set_xlabel('$s_\mathrm{y}$ (in.)', fontsize=10)
        
        
    if sx>0:    
        ax11 = f1.add_subplot(6,2,11 if welding=='Through-holes' and nr==1 else 10 if stud_position=='Two studs in series' and welding=='Through-holes' else 9 if stud_position=='Two studs in series' and welding=='Through-deck' else 11 if stud_position=='Two staggered studs' and welding=='Through-holes' else 10)
        ax11.plot(sx_in11, Qn_NGBoost_11_kips, color='#08519c',linewidth=1.5, linestyle='solid')

        ax11.fill_between(sx_in11.reshape(len(sx_in11),),Qn_NGBoost_11_68p3lower,Qn_NGBoost_11_68p3upper, color= "#4292c6",alpha= 0.75, lw=0.25)
        ax11.fill_between(sx_in11.reshape(len(sx_in11),),Qn_NGBoost_11_68p3lower,Qn_NGBoost_11_95p4lower, color= "#4292c6",alpha= 0.5, lw=0.25)
        ax11.fill_between(sx_in11.reshape(len(sx_in11),),Qn_NGBoost_11_68p3upper,Qn_NGBoost_11_95p4upper, color= "#4292c6",alpha= 0.5, lw=0.25)
        ax11.fill_between(sx_in11.reshape(len(sx_in11),),Qn_NGBoost_11_95p4lower,Qn_NGBoost_11_99p7lower, color= "#4292c6",alpha= 0.25, lw=0.25)
        ax11.fill_between(sx_in11.reshape(len(sx_in11),),Qn_NGBoost_11_95p4upper,Qn_NGBoost_11_99p7upper, color= "#4292c6",alpha= 0.25, lw=0.25)

        sx_in_loc=np.where(sx_in11==sx_in)[0].item()
        ax11.scatter(sx_in,Qn_NGBoost_11_kips[sx_in_loc],marker='o',facecolors='#08519c')

        ax11.set_ylim(bottom=0)
        ax11.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
        ax11.set_ylabel('$Q$ (kips)', fontsize=10)
        ax11.set_xlabel('$s_\mathrm{x}$ (in.)', fontsize=10)

    if (welding=='Through-deck') and (nr==1): legend_yloc=0.26
    elif stud_position=='Two staggered studs' and welding=='Through-holes': legend_yloc=-0.05
    else:legend_yloc=0.095
    
    f1.legend(ncol=2, fontsize=10, bbox_to_anchor=(0.52, legend_yloc), loc='lower center')
    f1.tight_layout()
    st.pyplot(f1)

st.write('##### Reference')
st.write('Degtyarev, V.V., Hicks, S.J., Machine learning-based probabilistic predictions of shear resistance of welded studs in deck slab ribs transverse to beams, Steel and Composite Structures. In Press.')    

st.write('##### Source code')
st.markdown('[GitHub](https://github.com/vitdegtyarev/Studs_in_Deck_Slabs)', unsafe_allow_html=True)

st.write('##### Test database')
st.markdown('[Mendeley Data](https://data.mendeley.com/datasets/nfmhnzbfy9/2)', unsafe_allow_html=True)