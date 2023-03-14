# extract monthly stn series from UERRA

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

# rr0 <- raster("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/monthly/UERRA_totprec/mescan_tp_1961.grib")
rr0 <- raster("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/monthly/UERRA_totprec/mescan_tp_1961.nc")
sf_meta_lonlat <- st_as_sf(dat_meta, coords = c("Longitude", "Latitude"), crs = 4326)
sf_meta_uerra <- st_transform(sf_meta_lonlat, crs(rr0))
sp_meta_uerra <- as(sf_meta_uerra, "Spatial")
icells <- cellFromXY(rr0, sp_meta_uerra)

all_files_prec <- dir_ls("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/monthly/UERRA_totprec/")
all_files_tmean <- dir_ls("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/monthly/UERRA_t2m/")

dat_stn <- foreach(
  i_fn_prec = all_files_prec,
  i_fn_tmean = all_files_tmean,
  .final = rbindlist
) %do% {

  rr_prec <- brick(i_fn_prec)    
  rr_tmean <- brick(i_fn_tmean)  
  
  rr_prec[icells] %>% 
    as.data.table %>% 
    cbind(dat_meta[, .(Name)]) %>% 
    melt(id.vars = "Name", value.name = "prec") -> dat_prec
  dat_prec[, date_time := ymd_hms(substr(variable, 2, 100))]
  dat_prec[, year := year(date_time)]
  dat_prec[, month := month(date_time)]

  rr_tmean[icells] %>% 
    as.data.table %>% 
    cbind(dat_meta[, .(Name)]) %>% 
    melt(id.vars = "Name", value.name = "tmeanK") -> dat_tmean
  dat_tmean[, date_time := ymd_hms(substr(variable, 2, 100))]
  dat_tmean[, year := year(date_time)]
  dat_tmean[, month := month(date_time)]
  dat_tmean[, tmean := tmeanK - 273.15]
  
  merge(dat_prec[, .(Name, year, month, prec)],
        dat_tmean[, .(Name, year, month, tmean)])
  
  
   
}


saveRDS(dat_stn,
        file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/uerra-01-series.rds")
