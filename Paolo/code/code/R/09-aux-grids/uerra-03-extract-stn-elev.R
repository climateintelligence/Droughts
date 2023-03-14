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


rr0 <- raster("/mnt/CEPH_PROJECTS/CLIRSNOW/uerra/mescan-orography.nc")
sf_meta_lonlat <- st_as_sf(dat_meta, coords = c("Longitude", "Latitude"), crs = 4326)
sf_meta_uerra <- st_transform(sf_meta_lonlat, crs(rr0))
sp_meta_uerra <- as(sf_meta_uerra, "Spatial")


dat_stn <- data.table(Name = dat_meta$Name,
                      elev_uerra = extract(rr0, sp_meta_uerra))


saveRDS(dat_stn,
        file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/uerra-02-elev.rds")
