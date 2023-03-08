# prep monthly E-OBS mean temp for extraction

library(fs)
library(data.table)

# meta for box
dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_long_HN_HS.rds")
summary(dat_meta)

in_file <- "/mnt/CEPH_BASEDATA/METEO/EUROPE/E_OBS/v20.0e/tg_ens_mean_0.1deg_reg_v20.0e.nc"
out_file <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/monthly/E-OBS/tg_ens_mean_0.1deg_reg_v20.0e_monmean_alps.nc"

cdo_call <- paste0("cdo sellonlatbox,3,18,42,50", 
                   " -monmean",
                   " ", in_file,
                   " ", out_file)

system(cdo_call)

