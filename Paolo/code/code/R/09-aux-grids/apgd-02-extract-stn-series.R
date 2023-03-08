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
    readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_all/meta_long_HS.rds")
  ))  

rr0 <- raster("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/monthly/APGD/RapdD_al05.etrs.laea_19710100.nc")
sf_meta_lonlat <- st_as_sf(dat_meta, coords = c("Longitude", "Latitude"), crs = 4326)
sf_meta_apgd <- st_transform(sf_meta_lonlat, st_crs(rr0))
sp_meta_apgd <- as(sf_meta_apgd, "Spatial")
icells <- cellFromXY(rr0, sp_meta_apgd)

all_files <- dir_ls("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/monthly/APGD/")

dat_stn <- foreach(
  i_fn = all_files,
  .final = rbindlist
) %do% {

  rr <- raster(i_fn)    
  i_fn %>% 
    path_file %>% 
    stringr::str_extract("[0-9]{6}") %>% 
    ymd(truncated = 1) -> i_date
  
  data.table(Name = sf_meta_apgd$Name,
             prec = rr[icells],
             year = year(i_date),
             month = month(i_date))
  
}


saveRDS(dat_stn,
        file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/apgd-01-series.rds")
