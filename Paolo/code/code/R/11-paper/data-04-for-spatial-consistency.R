# prep data for spatial consistency


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(scico)

library(forcats)
library(foreach)
library(flextable)
library(officer)

library(fs)



# read and load -----------------------------------------------------------

load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/data4corr-01-merged.rda")
load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/regions-01-sinkr-eof.rda")

dat_t1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/trends-01-mw-hydro-year.rds")
dat_t2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/trends-02-mw-calendar-year.rds")
dat_t3 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/trends-03-full_1971-2019-calyear.rds")


common_stns <- sort(unique(c(
  dat_hs_apgd_eobs$Name, dat_hs_chelsa$Name, dat_hs_uerra$Name, dat_hs_uerra_full$Name,
  dat_t1$Name, dat_t2$Name, dat_t3$Name,
  colnames(mat_eof)
)))

# subset ------------------------------------------------------------------

dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/meta_long_HS.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/data_long_HS.rds")

dat_meta2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_all/meta_long_HS.rds")
dat_hs2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_all/data_long_HS.rds")

dat_meta3 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020/meta_long_HN_HS.rds")
dat_hs3 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020/data_long_HN_HS.rds")


dat_meta[Name %in% common_stns]

common_stn_missing <- common_stns[! common_stns %in% dat_meta$Name]

dat_hs2[Name %in% common_stn_missing]
dat_hs3[Name %in% common_stn_missing]



dat_out <- rbindlist(use.names = T, fill = T, list(
  dat_hs[Name %in% common_stns],
  dat_hs3[Name %in% common_stn_missing]
))

dat_out_meta <- dat_meta3[Name %in% common_stns]

# write -------------------------------------------------------------------



out_data_long <- dat_out[!is.na(HS), 
                               .(Name, year, month, HS)]
out_meta_long <- dat_out_meta[Name %in% unique(out_data_long$Name)]

out_data_long[, .(year = min(year) : max(year)), .(Name)] %>% 
  .[, .(month = 1:12), .(Name, year)] %>% 
  merge(out_data_long, all.x = T) -> out_data_long

data_wide_hs <- dcast(out_data_long, year + month ~ Name, value.var = "HS")
lgl_cols_to_keep <- sapply(data_wide_hs, function(x) !all(is.na(x)))
chr_cols_to_keep <- names(lgl_cols_to_keep)[lgl_cols_to_keep]
out_data_wide_hs <- data_wide_hs[, chr_cols_to_keep, with = F]
out_meta_wide_hs <- out_meta_long[Name %in% chr_cols_to_keep]


path_out <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/"
if(!dir_exists(path_out)) dir_create(path_out)
saveRDS(out_data_long, file = path(path_out, "data_long_HS.rds"))
saveRDS(out_data_wide_hs, file = path(path_out, "data_wide_HS.rds"))
saveRDS(out_meta_long, file = path(path_out, "meta_long_HS.rds"))
saveRDS(out_meta_wide_hs, file = path(path_out, "meta_wide_HS.rds"))


