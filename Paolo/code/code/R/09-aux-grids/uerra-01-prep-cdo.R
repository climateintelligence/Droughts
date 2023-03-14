# prep monthly UERRA for extraction

library(fs)




# totprec -----------------------------------------------------------------



all_files <- dir_ls("/mnt/CEPH_PROJECTS/CLIRSNOW/uerra/total_precipitation/")
all_files

for(i_fn in all_files){
  
  in_file <- i_fn
  interm_file <- path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/monthly/UERRA_totprec/", 
                      path_file(i_fn))
  out_file <- path_ext_set(interm_file, "nc")
    
  # monthly
  cdo_call <- paste0("cdo monsum",
                     # " -sellonlatbox,-410000,840000,-880000,480000", # does not work with proj
                     " ", in_file,
                     " ", interm_file)
  
  system(cdo_call)
  
  
  # convert to ncdf
  cdo_call <- paste0("cdo -f nc copy",
                     " ", interm_file,
                     " ", out_file)
  
  system(cdo_call)
  
  file_delete(interm_file)
  
}



# temp -----------------------------------------------------------------



all_files <- dir_ls("/mnt/CEPH_PROJECTS/CLIRSNOW/uerra/2m_temperature/")
all_files

for(i_fn in all_files){
  
  in_file <- i_fn
  interm_file <- path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/monthly/UERRA_t2m/", 
                      path_file(i_fn))
  out_file <- path_ext_set(interm_file, "nc")
  
  # monthly
  cdo_call <- paste0("cdo monmean",
                     # " -sellonlatbox,-410000,840000,-880000,480000", # does not work with proj
                     " ", in_file,
                     " ", interm_file)
  
  system(cdo_call)
  
  
  # convert to ncdf
  cdo_call <- paste0("cdo -f nc copy",
                     " ", interm_file,
                     " ", out_file)
  
  system(cdo_call)
  
  file_delete(interm_file)
  
}



# elev -----------------------------------------------------------------


in_file <- "/mnt/CEPH_PROJECTS/CLIRSNOW/uerra/mescan-orography.grib"
out_file <- "/mnt/CEPH_PROJECTS/CLIRSNOW/uerra/mescan-orography.nc"
# convert to ncdf

cdo_call <- paste0("cdo -f nc copy",
                   " ", in_file,
                   " ", out_file)

system(cdo_call)



  

