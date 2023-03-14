# table of # of stns


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



# data --------------------------------------------------------------------

# initially HN
dat_meta_hn <- unique(rbind(
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_wide_HN.rds"),
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/meta_wide_HN.rds")
))

dat_meta_hs <- unique(rbind(
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_wide_HS.rds"),
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/meta_wide_HS.rds")
))

# sub for paper
dat_meta_hs_paper <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/meta_wide_HS.rds") 

# subset to spatcons
stns_ok <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/spatcons_stns_ok.rds")
dat_meta_hs_paper <- dat_meta_hs_paper[Name %in% stns_ok]

# HS full period 1971-2019?




# prep for table ----------------------------------------------------------

l_all <- list(dat_meta_hn, dat_meta_hs, dat_meta_hs_paper)
names(l_all) <- c("HN", "HS", "HS used in analysis")
dat_all <- rbindlist(l_all, idcol = "to_cast")

dat_all_n <- dat_all[, .N, .(to_cast, Provider)]
dat_all_n_margins <- dat_all_n[, .(N = sum(N), Provider = "Total sum"), to_cast]

rbind(dat_all_n, dat_all_n_margins, use.names = T) %>% 
  dcast(Provider ~ to_cast, value.var = "N") -> dat_table


dat_table %>% 
  flextable %>% 
  set_header_labels("Provider" = "Data source") %>% 
  colformat_int() %>% 
  hline(i = 16, border = fp_border()) %>% 
  autofit -> ft1

read_docx() %>% 
  body_add_flextable(ft1) %>% 
  print(target = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/table/Table 1.docx")




