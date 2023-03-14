# check meteoswiss availability


dat_meta_hs <- unique(rbind(
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_wide_HS.rds"),
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/meta_wide_HS.rds")
))


dat_daily_hnhs_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/data_long_HN_HS.rds")
dat_daily_hnhs_2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/data_long_HN_HS.rds")
dat_daily_hnhs <- rbind(dat_daily_hnhs_1, dat_daily_hnhs_2)

stn_mswiss <- dat_meta_hs[Provider == "CH_METEOSWISS", Name]
dat_daily_mswiss <- dat_daily_hnhs[Name %in% stn_mswiss]

dat_hs_overview <- dat_daily_mswiss[!is.na(HS),
                                    .(n_avail = .N),
                                    .(Name, year(Date))]


dat_hs_overview[year >= 1930] %>%
  ggplot(aes(year, fct_reorder(as.factor(Name), year, .fun = min), fill = n_avail))+
  geom_tile()+
  # geom_hline(yintercept = 1:5*20, alpha = 0.5)+
  scale_fill_viridis_c(direction = -1)+
  theme_bw()+
  scale_x_continuous(expand = c(0,0))

ggsave(file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/extra-test/meteoswiss-stn-avail.png",
       width = 12, height = 8)




# subset to not missing in period 1981 - 1997 ---------------------------

dat_zz <- dat_hs_overview[, .N, keyby = year]

stns_keep1 <- dat_hs_overview[, min(year), Name] %>% .[V1 > 1997, Name]

stns_start_before_1981 <- dat_hs_overview[, min(year), Name] %>% .[V1 < 1981, Name]
dat_hs_overview[Name %in% stns_start_before_1981 & year %in% c(1981:1997)] %>% 
  .[, unique(Name)] -> stns_keep2

# stns_stop_1981 <- dat_hs_overview[, max(year), Name] %>% .[V1 < 1981, Name]
# not needed, included in other

# at least 70% of data Dec

stns_keep_mswiss <- c(stns_keep1, stns_keep2)


dat_hs_overview[year >= 1930 & Name %in% stns_keep_mswiss] %>%
  ggplot(aes(year, fct_reorder(as.factor(Name), year, .fun = min), fill = n_avail))+
  geom_tile()+
  # geom_hline(yintercept = 1:5*20, alpha = 0.5)+
  scale_fill_viridis_c(direction = -1)+
  theme_bw()+
  scale_x_continuous(expand = c(0,0))

ggsave(file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/extra-test/meteoswiss-stn-avail2.png",
       width = 12, height = 8)


# source table-01-n-stn
stns_paper <- dat_meta_hs_paper[Provider == "CH_METEOSWISS", Name]

stns_paper[stns_paper %in% stns_keep_mswiss]

dat_hs_overview[, inpaper := Name %in% stns_paper]

dat_hs_overview[year >= 1930] %>%
  ggplot(aes(year, fct_reorder(as.factor(Name), year, .fun = min), fill = n_avail, alpha = inpaper))+
  geom_tile()+
  # geom_hline(yintercept = 1:5*20, alpha = 0.5)+
  scale_fill_viridis_c(direction = -1)+
  theme_bw()+
  scale_x_continuous(expand = c(0,0))


ggsave(file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/extra-test/meteoswiss-stn-avail3.png",
       width = 12, height = 8)



# check gapfill -----------------------------------------------------------

dat_hs_gf <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds")


dat_hs_gf_overview <- dat_hs_gf[Name %in% stn_mswiss & !is.na(HS),
                                .(n_avail = .N,
                                  n_gapfill = sum(HS_fillcode == 222)),
                                .(Name, year(Date))]

dat_hs_gf_overview[, inpaper := Name %in% stns_paper]


dat_hs_gf_overview[year >= 1930] %>%
  ggplot(aes(year, fct_reorder(as.factor(Name), year, .fun = min), fill = n_avail, alpha = inpaper))+
  geom_tile()+
  # geom_hline(yintercept = 1:5*20, alpha = 0.5)+
  scale_fill_viridis_c(direction = -1)+
  theme_bw()+
  scale_x_continuous(expand = c(0,0))


ggsave(file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/extra-test/meteoswiss-stn-avail4.png",
       width = 12, height = 8)


dat_hs_gf_overview[year >= 1930] %>%
  ggplot(aes(year, fct_reorder(as.factor(Name), year, .fun = min), fill = n_avail))+
  geom_tile()+
  # geom_hline(yintercept = 1:5*20, alpha = 0.5)+
  scale_fill_viridis_c(direction = -1)+
  theme_bw()+
  scale_x_continuous(expand = c(0,0))+
  ggtitle("MeteoSwiss data after gapfilling")


ggsave(file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/extra-test/meteoswiss-stn-avail5.png",
       width = 12, height = 8)

