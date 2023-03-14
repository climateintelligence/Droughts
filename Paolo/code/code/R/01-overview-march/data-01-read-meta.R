# read in all original data

library(fs)
library(data.table)
library(magrittr)
library(readxl)
library(lubridate)


# meta --------------------------------------------------------------------

l_meta <- list(
  fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/AT_HZB/list_sites.csv"),
  fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/BOLZANO/list_manual_sites.csv"),
  fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/CH_METEOSWISS/list_manual_sites.csv"),
  fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/CH_SLF/list_manual_sites.csv"),
  fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/FR_METEOFRANCE/list_manual_sites.csv"),
  fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/GERMANY/list_sites.csv"),
  fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/PIEMONTE/list_manual_sites.csv"),
  fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/TRENTINO/list_manual_sites.csv"),
  fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/VENETO/list_manual_sites_filledNAmanual.csv")
)
names(l_meta) <- c("AT_HZB", "IT_BZ", "CH_METEOSWISS", "CH_SLF", "FR_METEOFRANCE", "DE_DWD", 
                   "IT_PIEMONTE", "IT_TN", "IT_VENETO")
lapply(l_meta, names)

tbl <- read_excel("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/FVG/list_AINEVA_stations.xlsx")
tbl %>% 
  as.data.table() %>% 
  .[, .(Name, Longitude = Long, Latitude = Lat, Elevation = Alt)] -> dat_fvg

tbl <- read_excel("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/LOMBARDIA/list_of_manual_stations.xlsx")
tbl %>% 
  as.data.table() %>% 
  .[, .(Name, Longitude = Long, Latitude = Lat, Elevation = Alt)] -> dat_lombardia

dat_slovenia <- rbind(
  fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/SLOVENIA/List_stations_HN.csv", encoding = "Latin-1"),
  fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/SLOVENIA/List_stations_HS.csv", encoding = "Latin-1")
) %>% unique

source("R/functions/safe_names.R")
dat_slovenia[, stn_name_safe := stringi::stri_trans_totitle(safe_names(tolower(NAME)))]
dat_slovenia[stn_name_safe %in% c("Bilje", "Novo_mesto", "Celje_medlog", "Ratece","Murska_sobota_rakican"),
             stn_name_safe := paste0(stn_name_safe, "_", ID)]

# dat_slovenia %>% .[, .N, stn_name_safe] %>% .[N > 1, stn_name_safe]
# dat_slovenia %>% .[, .N, stn_name_safe] %>% .[N > 1] %>% 
#   merge(dat_slovenia, by = "stn_name_safe")

dat_slovenia_out <- dat_slovenia[, .(Name = stn_name_safe, Longitude = LONG, Latitude = LAT, Elevation = ELE)]


dat_vda <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/VDA/list_manual_sites.csv",
                 encoding = "Latin-1")
dat_vda <- dat_vda[!is.na(Longitude), .(Name, Longitude, Latitude, Elevation)]
dat_vda[, Name := safe_names(Name)]

l_meta2 <- list(dat_fvg, dat_lombardia, dat_slovenia_out, dat_vda)
names(l_meta2) <- c("IT_FVG", "IT_LOMBARDIA", "SI_ARSO", "IT_VDA")

dat_meta <- rbindlist(c(l_meta, l_meta2), use.names = T, fill = T, idcol = "provider")

saveRDS(dat_meta, file = "data/meta-01-original.rds")
saveRDS(dat_meta, file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/meta-01-original.rds")



