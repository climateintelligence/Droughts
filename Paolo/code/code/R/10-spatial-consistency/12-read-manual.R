# read in spatial consistency overview


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(directlabels)
library(readxl)
library(writexl)
library(fs)

dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/meta_wide_HS.rds")

dat_sc <- read_xlsx("manual-qc/v02/spatcons_overview_table_filled.xlsx")
setDT(dat_sc)


stns_to_remove <- dat_sc[suspicious == 1 | partly == 1, Name]
stns_ok <- dat_meta[! Name %in% stns_to_remove, Name]

# save only stns ----------------------------------------------------------

saveRDS(stns_ok, "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/rds/stns_ok.rds")
saveRDS(stns_to_remove, "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/rds/stns_to_remove.rds")


saveRDS(stns_ok, "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/spatcons_stns_ok.rds")
saveRDS(stns_to_remove, "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/spatcons_stns_to_remove.rds")

