# export time series for Michele Master

library(data.table)
library(magrittr)
library(fs)

dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_long_HN_HS.rds")
dat_meta_out <- dat_meta[Provider %in% c("IT_TN", "IT_BZ")]

dat_hshn <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/data_long_HN_HS.rds")
dat_hshn_out <- dat_hshn[Name %in% dat_meta_out$Name]

fwrite(dat_meta_out, "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/09_EXPORT/data-michele-master/TNBZ-meta.csv")
fwrite(dat_hshn_out, "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/09_EXPORT/data-michele-master/TNBZ-data-HNHS.csv")



# add meteotrentino rest

# read in data from meteotrentino


library(htmltab)
library(data.table)
library(fs)
library(foreach)

# meta

dat_meta <- fread("~/projects-r/alps-past-snow/data-raw/meteotrentino/elenco_stazioni_filled.csv", encoding = "UTF-8")
dat_meta <- dat_meta[!is.na(year_end)]

source("R/functions/safe_names.R")
dat_meta[, stn_name_safe := stringi::stri_trans_totitle(safe_names(stn_name))]

dat_meta[, c("lon", "lat") := data.table(sf::sf_project("+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0",
                                                    "+proj=longlat +datum=WGS84",
                                                    dat_meta[, .(coord1, coord2)]))]
fwrite(dat_meta, 
       file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/09_EXPORT/data-michele-master/meteotrentino-raw-meta.csv")


# data 

mitmatmisc::init_parallel_ubuntu()

all_files <- dir_ls("~/projects-r/alps-past-snow/data-raw/meteotrentino/xls_html/", recurse = T, glob = "*.xls")

l_tn <- list()
for(i in seq_along(all_files)){
  
  i_file <- all_files[i]
  tbl <- htmltab(i_file, which = 1)
  dat <- data.table(tbl)
  l_tn[[i]] <- dat
  
  cat(i, " of ", length(all_files), ": ", i_file, "\n")
  
}

dat_tn <- rbindlist(l_tn, use.names = T, fill = T)
fwrite(dat_tn, 
       file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/09_EXPORT/data-michele-master/meteotrentino-raw-series.csv")

