# figure overview 
# - map
# - altitude distribution
# - time with # (maybe divide by country)



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(scico)
library(directlabels)
library(patchwork)

library(forcats)
library(foreach)
library(flextable)
library(officer)

library(fs)



# data --------------------------------------------------------------------

# hs paper
# dat_meta_paper <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/meta_wide_HS.rds") 
# stns_hs_used <- dat_meta_paper$Name
# dat_meta_hs_used <- copy(dat_meta_paper)

# adapt from spat cons data
load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/sensitivity-gapfill/data4corr-01-merged.rda")
load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/sensitivity-gapfill/regions-01-sinkr-eof.rda")

dat_t1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/sensitivity-gapfill/trends-01-mw-hydro-year.rds")
dat_t2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/sensitivity-gapfill/trends-02-mw-calendar-year.rds")
dat_t3 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/sensitivity-gapfill/trends-03-full_1971-2019-calyear.rds")


common_stns <- sort(unique(c(
  dat_hs_apgd_eobs$Name, dat_hs_chelsa$Name, dat_hs_uerra$Name, dat_hs_uerra_full$Name,
  dat_t1$Name, dat_t2$Name, dat_t3$Name,
  colnames(mat_eof)
)))
dat_meta3 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020/meta_long_HN_HS.rds")
dat_meta_hs_used <- dat_meta3[Name %in% common_stns]
stns_hs_used <- common_stns

# subset to spatcons
stns_ok <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/spatcons_stns_ok.rds")
stns_hs_used <- stns_hs_used[stns_hs_used %in% stns_ok]
dat_meta_hs_used <- dat_meta_hs_used[Name %in% stns_ok]

dat_meta_hs <- unique(rbind(
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_wide_HS.rds"),
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/meta_wide_HS.rds")
))




# time series
dat_daily_hnhs_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/data_long_HN_HS.rds")
dat_daily_hnhs_2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/data_long_HN_HS.rds")
dat_daily_hnhs <- rbind(dat_daily_hnhs_1, dat_daily_hnhs_2)

# time series summary
dat_ts_hn <- dat_daily_hnhs[!is.na(HN), .(n_stn = length(unique(Name))), .(Provider, year(Date))]
dat_ts_hs <- dat_daily_hnhs[!is.na(HS), .(n_stn = length(unique(Name))), .(Provider, year(Date))]



# plot elev hist ----------------------------------------------------------

cols_2grey <- setNames(scales::grey_pal()(2),
                       c("used", "available"))


gg_elev <-
  dat_meta_hs %>% 
  ggplot(aes(Elevation))+
  geom_histogram(breaks = seq(0, 3300, by = 50), aes(fill = "available"))+
  geom_histogram(breaks = seq(0, 3300, by = 50),
                 data = dat_meta_hs_used,
                 aes(fill = "used"))+
  # geom_point(data = dat_meta_hn_used, aes(y = -5, colour = "used"), shape = 1)+
  scale_fill_manual("HS", values = cols_2grey)+
  # scale_colour_manual("HN", values = cols_2grey)+
  theme_bw(16)+
  theme(legend.position = c(0.8, 0.6),
        legend.direction = "vertical",
        legend.title = element_blank(),
        legend.background = element_rect(colour = "black"))+
  xlab("Elevation [m]")+
  ylab("Number of HS stations")


# plot ts HN and HS -------------------------------------------------------------------

cols_country <- c(scales::brewer_pal(type = "qual")(6), "black")
cols_country <- c(scales::hue_pal()(6), "black")
lty_country <- c(rep("solid", 6), "dashed")
xlim_common <- range(c(dat_ts_hn$year, dat_ts_hs$year))

# 
# dat_ts_hn[, country := substr(Provider, 1, 2)]
# dat_plot_ts_hn <- rbind(use.names = T,
#                           dat_ts_hn[, .(n_stn = sum(n_stn)), .(year, country)],
#                           dat_ts_hn[, .(n_stn = sum(n_stn), country = "Total"), .(year)])
#   
#   
# gg_ts_hn <- dat_plot_ts_hn %>% 
#   ggplot(aes(year, n_stn, colour = country, linetype = country))+
#   geom_line()+
#   scale_color_manual("", values = cols_country)+
#   scale_linetype_manual("", values = lty_country)+
#   xlim(xlim_common)+
#   theme_bw(16)+
#   xlab(NULL)+
#   ylab("Number of HN stations")
# 
# # -> maybe only one ts summary over all countries
# # -> maybe not
# 
# # gg_ts_hn_dl <- direct.label(gg_ts_hn, "extreme.grid")
# # gg_ts_hn_dl <- direct.label(gg_ts_hn, "first.bumpup")
# gg_ts_hn_dl <- direct.label(gg_ts_hn, "top.bumpup")



dat_ts_hs[, country := substr(Provider, 1, 2)]
dat_plot_ts_hs <- rbind(use.names = T,
                        dat_ts_hs[, .(n_stn = sum(n_stn)), .(year, country)],
                        dat_ts_hs[, .(n_stn = sum(n_stn), country = "Total"), .(year)])


gg_ts_hs <- dat_plot_ts_hs %>% 
  ggplot(aes(year, n_stn, colour = country, linetype = country))+
  geom_line()+
  scale_color_manual("", values = cols_country)+
  scale_linetype_manual("", values = lty_country)+
  # xlim(xlim_common)+
  theme_bw(16)+
  xlab(NULL)+
  ylab("Number of HS stations")

# gg_ts_hs_dl <- direct.label(gg_ts_hs, "extreme.grid")
# gg_ts_hs_dl <- direct.label(gg_ts_hs, "first.bumpup")
gg_ts_hs_dl <- direct.label(gg_ts_hs, "top.bumpup")


# combine and save --------------------------------------------------------

gg_out <- gg_elev + gg_ts_hs + 
  plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")")

ggsave(gg_out,
       file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/sensitivity-gapfill/Figure 2.png",
       width = 12, height = 4)
