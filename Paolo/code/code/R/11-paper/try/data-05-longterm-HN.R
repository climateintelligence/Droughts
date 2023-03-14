# prep figure on longterm HS/HN changes

library(data.table)
library(magrittr)
library(ggplot2)
library(forcats)
library(slider)
#
dat <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/06_SEASONAL/Oct-Sep/data_long_HN_HS.rds")
meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/06_SEASONAL/Oct-Sep/meta_long_HN_HS.rds")



# subset data -------------------------------------------------------------




# ** HS -------------------------------------------------------------------

dat[!is.na(HS), .N, Name] %>% summary

dat_hs_overview <- dat[!is.na(HS), 
                       .(nn = .N, 
                         year_start = min(hydro_year), 
                         year_end = max(hydro_year),
                         max_years = max(hydro_year) - min(hydro_year) + 1), 
                       Name]
ggplot(dat_hs_overview, aes(year_start, year_end, fill = nn))+
  geom_tile()+
  scale_fill_viridis_c()

dat_hs_overview %>% 
  merge(meta) %>% 
  ggplot(aes(year_start, year_end))+
  geom_tile()+
  facet_wrap(~Provider)+
  theme_bw()

# dat_hs_overview[year_start < 1920]
# stn_sub_hs <- dat_hs_overview[year_start < 1920, Name]


dat_hs_overview[year_start < 1930 & year_end > 2000 & nn >= (max_years - 5)]
stn_sub_hs <- dat_hs_overview[year_start < 1930 & year_end > 2000 & nn >= (max_years - 5), Name]


dat_hs <- dat[Name %in% stn_sub_hs]
dat_meta_hs <- meta[Name %in% stn_sub_hs]

# ** HN -------------------------------------------------------------------


dat[!is.na(HN), .N, Name] %>% summary

dat_hn_overview <- dat[!is.na(HN), 
                       .(nn = .N, 
                         year_start = min(hydro_year), 
                         year_end = max(hydro_year),
                         max_years = max(hydro_year) - min(hydro_year) + 1), 
                       Name]
dat_hn_overview <- merge(dat_hn_overview, meta[, .(Name, Provider)])
ggplot(dat_hn_overview, aes(year_start, year_end, fill = nn))+
  geom_tile()+
  scale_fill_viridis_c()

# dat_hn_overview[year_start < 1900 & nn > 80 & year_end > 2000]
# stn_sub_hn <- dat_hn_overview[year_start < 1900 & nn > 80 & year_end > 2000, Name]

# 
# dat_hn_overview[year_start < 1900 & year_end > 2000 & nn >= (max_years - 5)]
# stn_sub_hn <- dat_hn_overview[year_start < 1900 & year_end > 2000 & nn >= (max_years - 5), Name]



dat_hn_overview[year_start < 1900 & year_end > 2000 & nn >= (max_years - 10)]
stn_sub_hn <- dat_hn_overview[year_start < 1900 & year_end > 2000 & nn >= (max_years - 10), Name]



dat_hn <- dat[Name %in% stn_sub_hn]
dat_meta_hn <- meta[Name %in% stn_sub_hn]



# save --------------------------------------------------------------------


save(dat_hs, dat_meta_hs, dat_hn, dat_meta_hn,
     file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/data-longterm-HN-HS.rda")

