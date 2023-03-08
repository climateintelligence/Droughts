# extract monthly stn series from APGD

library(data.table)
library(magrittr)
library(fs)
library(foreach)
library(sf)
library(raster)
library(lubridate)

dat_meta <- 
  unique(rbind(
    readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020/meta_long_HN_HS.rds"),
    readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/meta_long_HS.rds")
  ))  

rr0 <- raster("/mnt/CEPH_PROJECTS/CLIRSNOW/chelsa/prec/CHELSA_prec_1979_01_V1.2.1.tif")
sf_meta_lonlat <- st_as_sf(dat_meta, coords = c("Longitude", "Latitude"), crs = 4326)
sp_meta_lonlat <- as(sf_meta_lonlat, "Spatial")
icells <- cellFromXY(rr0, sp_meta_lonlat)

all_files_prec <- dir_ls("/mnt/CEPH_PROJECTS/CLIRSNOW/chelsa/prec/")
all_files_tmean <- dir_ls("/mnt/CEPH_PROJECTS/CLIRSNOW/chelsa/tmean/")

dat_stn <- foreach(
  i_fn_prec = all_files_prec,
  i_fn_tmean = all_files_tmean,
  .final = rbindlist
) %do% {

  rr_prec <- raster(i_fn_prec)    
  rr_tmean <- raster(i_fn_tmean)  
  
  i_fn_prec %>% 
    path_file %>% 
    stringr::str_split("_") %>% 
    .[[1]] -> i_fn_prec_split

  i_year <- as.numeric(i_fn_prec_split[3])
  i_month <- as.numeric(i_fn_prec_split[4])
  
  data.table(Name = sf_meta_lonlat$Name,
             prec = rr_prec[icells],
             tmean = rr_tmean[icells]/10 - 273.15,
             year = i_year,
             month = i_month)
  
}


saveRDS(dat_stn,
        file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/chelsa-01-series.rds")
