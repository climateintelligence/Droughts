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

rr <- raster("/mnt/CEPH_BASEDATA/METEO/EUROPE/E_OBS/v20.0e/elev_ens_0.1deg_reg_v20.0e.nc")
sf_meta_lonlat <- st_as_sf(dat_meta, coords = c("Longitude", "Latitude"), crs = 4326)
sp_meta_lonlat <- as(sf_meta_lonlat, "Spatial")

extract(rr, sp_meta_lonlat) %>% 
  cbind(dat_meta[, .(Name)], elev_eobs = .) -> dat_stn


saveRDS(dat_stn,
        file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/eobs-02-elev.rds")
