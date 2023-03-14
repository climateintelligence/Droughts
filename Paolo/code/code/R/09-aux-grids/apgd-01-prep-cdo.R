# prep monthly APGD for extraction

library(fs)


all_files <- dir_ls("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/APGD/",
                    recurse = T,
                    regexp = "00.nc$")
all_files

for(i_fn in all_files){
  
  in_file <- i_fn
  out_file <- path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/monthly/APGD/", 
                   path_file(i_fn))
  
  cdo_call <- paste0("cdo monsum",
                     " ", in_file,
                     " ", out_file)
  
  system(cdo_call)
  
  
}