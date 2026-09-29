import configparser

def main(): 

    # read in parameters
    cfg = configparser.ConfigParser()
    cfg.read('cfg.ini')

    fraction_negative_curve = float(cfg['_']['fraction_negative_curve']) 
    reduced_head_diameter = float(cfg['_']['reduced_head_diameter']) 
    seed = int(cfg['_']['seed'])
    # pressure = float(cfg['_']['pressure'])
    temperature = float(cfg['_']['temperature']) 

    # if fraction_bilayer ==0.0:
    #     if km==0.:
    #         temperature=1.2    
    #     else:
    #         temperature=1.3+(km/10.)
    # elif fraction_bilayer ==1.0:
    #     temperature=1.1 
    # else:                           # Any time there is a micture between bilayers and bolapipids, to match miguel figures lets only use this for when kbola=2 and T=1.3
    #     temperature= 1.3            # From miguel paper membrane is fluid for this case from bilayer fractions>=0.1

    sigma = 1
    omega = 1.5
    epsilon=1./temperature
    
    sigma_11=sigma*0.95 * 2**(1/6)
    cutoff_11=sigma_11
    
    sigma_12=sigma*0.95 * 2**(1/6) 
    cutoff_12=sigma_12

    sigma_22=sigma * 2**(1/6)
    cutoff_22=sigma_22+omega
    
    sigma_33=sigma*reduced_head_diameter * 2**(1/6)
    cutoff_33=sigma_33
    
    sigma_23=sigma*reduced_head_diameter * 2**(1/6)
    cutoff_23=sigma_23
    
    sigma_13=(sigma_11+sigma_33)/2      # simple mixing rule
    cutoff_13=sigma_13                  



    outfile='in.local'
    f=open(outfile,'w') #open for writing, appending to the end of file if it exists
    f.write("processors * * 1" + '\n')
    f.write('\n')
    f.write("units lj" + '\n')
    f.write("atom_style angle" + '\n')
    f.write("dimension 3" + '\n')
    f.write("boundary p p p" + '\n')  
    f.write("timestep 0.1" + '\n')  
    f.write('\n')
    f.write("read_data vesicle.data" + '\n')  
    f.write("group all type 1 2 3" + '\n')  
    f.write("group membrane type 1 2 3" + '\n')
    f.write("group heads type 1 3" + "\n")    
    f.write("bond_style hybrid fene" + '\n') 
    f.write("special_bonds lj 0.0 1.0 1.0" + '\n') 
    f.write("bond_coeff 1 fene 30.0 1.5 1.0 1.0" + '\n') 
    f.write('\n')
    f.write("angle_style hybrid harmonic" + '\n')  
    f.write("angle_coeff 1 harmonic 5.0 180.0" + '\n')    
    f.write('\n')
    f.write("comm_modify cutoff 2.0" + '\n')  
    f.write("pair_style soft 1.0" + '\n')
    f.write('\n')  
    f.write("pair_coeff * * 1 0.5" + '\n') 
    f.write("fix 1 all nve/limit 0.1" + '\n')  
    f.write("run 5000" + '\n')     
    f.write('\n')
    f.write("pair_coeff * * 3 0.7" + '\n') 
    f.write("fix 1 all nve/limit 0.1" + '\n')  
    f.write("run 5000" + '\n')     
    f.write('\n')
    f.write("pair_coeff * * 5 0.9" + '\n')  
    f.write("fix 1 all nve/limit 0.1" + '\n')  
    f.write("run 5000" + '\n')  
    f.write('\n')
    f.write("pair_coeff * * 7 1.0" + '\n')  
    f.write("fix 1 all nve/limit 0.01" + '\n')  
    f.write("run 5000" + '\n') 
    f.write('\n')
    f.write("pair_coeff * * 9 1.0" + '\n')  
    f.write("fix 1 all nve/limit 0.01" + '\n')  
    f.write("run 5000" + '\n') 
    f.write('\n')
    f.write("unfix 1" + '\n')  
    f.write("neigh_modify every 1 delay 1" + '\n')  
    f.write("timestep 0.01" + '\n')
    f.write("comm_modify cutoff 3.022462048309373" + '\n')  
    f.write("pair_coeff * * 0 0" + '\n')    
    f.write('\n')
    f.write("pair_style hybrid cosine/squared" + " " + str(cutoff_22)+ '\n')         # 1 head 2 tail 3 water
    f.write("pair_coeff 1 1 cosine/squared" + " " + str(epsilon) + " " + str(sigma_11) + " " + str(cutoff_11) + " " + "wca\n")
    f.write("pair_coeff 1 2 cosine/squared" + " " + str(epsilon) + " " + str(sigma_12) + " " + str(cutoff_12) + " " + "wca\n")
    f.write("pair_coeff 1 3 cosine/squared" + " " + str(epsilon) + " " + str(sigma_13) + " " + str(cutoff_13) + " " + "wca\n")
    f.write("pair_coeff 2 2 cosine/squared" + " " + str(epsilon) + " " + str(sigma_22) + " " + str(cutoff_22) + " " + "wca\n")
    f.write("pair_coeff 2 3 cosine/squared" + " " + str(epsilon) + " " + str(sigma_23) + " " + str(cutoff_23) + " " + "wca\n")
    f.write("pair_coeff 3 3 cosine/squared" + " " + str(epsilon) + " " + str(sigma_33) + " " + str(cutoff_33) + " " + "wca\n\n")

    # f.write("variable lx equal lx"+ '\n')
    # f.write("variable ly equal ly"+ '\n')
    # f.write("variable lz equal lz"+ '\n')
    # f.write("variable vol equal vol"+ '\n')
    # f.write("#Computes"+ '\n')
    # f.write("compute temp_all all temp" + '\n')
    # f.write("compute temp_mem membrane temp" + '\n')
    # f.write("compute temp_water water temp" + '\n')

   
    # f.write("variable thermodump     equal   1000					#Thermo info dump period"+ '\n')
    # f.write("variable nevery_thermo  equal   $(v_thermodump/10) 				#Use input values to av. thermo quantities every this many time steps"+ '\n')
    # f.write("variable nrepeat_thermo equal   $(v_thermodump/v_nevery_thermo) #Number of times to use input values for calculating averages"+ '\n')

    # f.write("compute total_press all pressure temp_all"+ '\n')


    # f.write("# Compute total1000 virial pressure (virial part only)" + '\n')
    # f.write("compute press_virial all pressure NULL virial" + '\n')
    # f.write("# Compute per-atom stress tensor (total stress)" + '\n')
    # f.write("compute stress_water water stress/atom temp_water" + '\n')
    # f.write("compute stress_membrane membrane stress/atom temp_mem" + '\n')
    # f.write("# Compute per-atom stress tensor (virial component only)" + '\n')
    # f.write("compute stress_water_virial water stress/atom NULL virial" + '\n')
    # f.write("compute stress_membrane_virial membrane stress/atom NULL virial" + '\n')
    # f.write("# Compute total pressure with kinetic energy component" + '\n')
    # f.write("compute total_press_ke all pressure temp_all ke" + '\n')
    # f.write("compute stress_water_ke water stress/atom temp_water ke" + '\n')
    # f.write("compute stress_membrane_ke membrane stress/atom temp_mem ke" + '\n')
    # f.write("# Sum per-atom stress to get total stress for membrane and water" + '\n')
    # f.write("compute sum_stress_water water reduce sum c_stress_water[1] c_stress_water[2] c_stress_water[3]" + '\n')
    # f.write("compute sum_stress_membrane membrane reduce sum c_stress_membrane[1] c_stress_membrane[2] c_stress_membrane[3]" + '\n')
    # f.write("# Sum per-atom virial stress for membrane and water" + '\n')
    # f.write("compute sum_stress_water_virial water reduce sum c_stress_water_virial[1] c_stress_water_virial[2] c_stress_water_virial[3]" + '\n')
    # f.write("compute sum_stress_membrane_virial membrane reduce sum c_stress_membrane_virial[1] c_stress_membrane_virial[2] c_stress_membrane_virial[3]" + '\n')
    # f.write("# Sum per-atom ke stress for membrane and water" + '\n')
    # f.write("compute sum_stress_water_ke water reduce sum c_stress_water_ke[1] c_stress_water_ke[2] c_stress_water_ke[3]" + '\n')
    # f.write("compute sum_stress_membrane_ke membrane reduce sum c_stress_membrane_ke[1] c_stress_membrane_ke[2] c_stress_membrane_ke[3]" + '\n')

    # f.write("variable total_virial_xx equal c_press_virial[1]" + '\n')
    # f.write("variable total_virial_yy equal c_press_virial[2]" + '\n')
    # f.write("variable total_virial_zz equal c_press_virial[3]" + '\n')
    # f.write("variable total_kinetic_xx equal c_total_press_ke[1]" + '\n')
    # f.write("variable total_kinetic_yy equal c_total_press_ke[2]" + '\n')
    # f.write("variable total_kinetic_zz equal c_total_press_ke[3]" + '\n')




    # f.write("# Time-averaged values" + '\n')
    # f.write("fix avg_total all ave/time ${nevery_thermo} ${nrepeat_thermo} ${thermodump} c_total_press[1] c_total_press[2] c_total_press[3]" + '\n')
    # f.write("fix avg_virial all ave/time ${nevery_thermo} ${nrepeat_thermo} ${thermodump} v_total_virial_xx v_total_virial_yy v_total_virial_zz" + '\n')
    # f.write("fix avg_kinetic all ave/time ${nevery_thermo} ${nrepeat_thermo} ${thermodump} v_total_kinetic_xx v_total_kinetic_yy v_total_kinetic_zz" + '\n')
    # f.write("fix avg_membrane all ave/time ${nevery_thermo} ${nrepeat_thermo} ${thermodump} c_sum_stress_membrane[1] c_sum_stress_membrane[2] c_sum_stress_membrane[3]" + '\n')
    # f.write("fix avg_water all ave/time ${nevery_thermo} ${nrepeat_thermo} ${thermodump} c_sum_stress_water[1] c_sum_stress_water[2] c_sum_stress_water[3]" + '\n')
    # f.write("fix avg_virial_membrane all ave/time ${nevery_thermo} ${nrepeat_thermo} ${thermodump} c_sum_stress_membrane_virial[1] c_sum_stress_membrane_virial[2] c_sum_stress_membrane_virial[3]" + '\n')
    # f.write("fix avg_virial_water all ave/time ${nevery_thermo} ${nrepeat_thermo} ${thermodump} c_sum_stress_water_virial[1] c_sum_stress_water_virial[2] c_sum_stress_water_virial[3]" + '\n')
    # f.write("fix avg_kinetic_membrane all ave/time ${nevery_thermo} ${nrepeat_thermo} ${thermodump} c_sum_stress_membrane_ke[1] c_sum_stress_membrane_ke[2] c_sum_stress_membrane_ke[3]" + '\n')
    # f.write("fix avg_kinetic_water all ave/time ${nevery_thermo} ${nrepeat_thermo} ${thermodump} c_sum_stress_water_ke[1] c_sum_stress_water_ke[2] c_sum_stress_water_ke[3]" + '\n')
    # # Print output section
    # f.write("# Output pressures and decompositions" + "\n")
    # f.write("# 1:step 2:total_xx 3:total_yy 4:total_zz 5:virial_xx 6:virial_yy 7:virial_zz 8:kinetic_xx 9:kinetic_yy 10:kinetic_zz " +
    #     "11:membrane_total_xx 12:membrane_total_yy 13:membrane_total_zz " +
    #     "14:water_total_xx 15:water_total_yy 16:water_total_zz " +
    #     "17:membrane_virial_xx 18:membrane_virial_yy 19:membrane_virial_zz " +
    #     "20:water_virial_xx 21:water_virial_yy 22:water_virial_zz " +
    #     "23:membrane_kinetic_xx 24:membrane_kinetic_yy 25:membrane_kinetic_zz " +
    #     "26:water_kinetic_xx 27:water_kinetic_yy 28:water_kinetic_zz " +
    #     "29:Lx 30:Ly 31:Lz" + "\n")


    # f.write("fix print_pressures all print ${thermodump} &" + "\n")
    # f.write("  \"$(step) $(f_avg_total[1]) $(f_avg_total[2]) $(f_avg_total[3]) " +
    #     "$(f_avg_virial[1]) $(f_avg_virial[2]) $(f_avg_virial[3]) " +
    #     "$(f_avg_kinetic[1]) $(f_avg_kinetic[2]) $(f_avg_kinetic[3]) " +
    #     "$(f_avg_membrane[1]/-v_vol) $(f_avg_membrane[2]/-v_vol) $(f_avg_membrane[3]/-v_vol) " +
    #     "$(f_avg_water[1]/-v_vol) $(f_avg_water[2]/-v_vol) $(f_avg_water[3]/-v_vol) " +
    #     "$(f_avg_virial_membrane[1]/-v_vol) $(f_avg_virial_membrane[2]/-v_vol) $(f_avg_virial_membrane[3]/-v_vol) " +
    #     "$(f_avg_virial_water[1]/-v_vol) $(f_avg_virial_water[2]/-v_vol) $(f_avg_virial_water[3]/-v_vol) " +
    #     "$(f_avg_kinetic_membrane[1]/-v_vol) $(f_avg_kinetic_membrane[2]/-v_vol) $(f_avg_kinetic_membrane[3]/-v_vol) " +
    #     "$(f_avg_kinetic_water[1]/-v_vol) $(f_avg_kinetic_water[2]/-v_vol) $(f_avg_kinetic_water[3]/-v_vol) " +
    #     "${lx} ${ly} ${lz}\" &" + "\n")
    # f.write("  file pressures_output_all.txt screen no" + "\n")


    f.write("minimize 1.0e-4 1.0e-6 100 1000 "+'\n')
    f.write("fix langevin all langevin 1.0 1.0 1.0" + ' ' + str(seed) +  ' zero yes'+ '\n')
    #f.write("fix fNPH membrane nph x" +' '+str(pressure)+' '+str(pressure) +' '+"10. y"+' '+str(pressure)+' '+str(pressure)+' '+ "10. couple xy dilate membrane" + '\n')  
    f.write("fix nve all nve\n")
    f.write(f"dump dump all netcdf 50000 output_rhd{reduced_head_diameter:.2f}_x{fraction_negative_curve:.2f}_seed{seed}.nc xu yu zu\n\n")
    
    f.write("restart 5000 out.a.restart out.b.restart" + '\n')
    f.write("fix f_avetime all ave/time 1 100 100 c_thermo_temp[*] c_thermo_press[*] mode scalar ave one"+ '\n') 
    f.write("thermo 2000"+ '\n')
    f.write("thermo_style custom step dt time cpu fmax fnorm temp ke pe ebond eangle epair pxx pyy pzz xlo xhi ylo yhi zlo zhi c_thermo_temp[*] c_thermo_press[*] f_f_avetime[*]"+ '\n') 
    f.write("run 10000" + '\n')
           

    f.close()

main()    