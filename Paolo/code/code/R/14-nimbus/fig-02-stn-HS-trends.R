
# calc monthly and seasonal means based on EOF results --------------------



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(scico)
library(patchwork)
library(forcats)
library(foreach)

# dat_hs_month <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/rds/data-04-monthly-1981-2010.rds")
# dat_hs_season <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/rds/data-05-seasonal-1981-2010.rds")
dat_meta_cluster <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/rds/meta-with-cluster-01.rds")

load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/rds/trends-01-1971-2019-ols-gls.rda")


# map: monthly -----------------------------------------------------------------


dat_plot_month <- dat_month_gls[term == "year0"] %>% 
  merge(dat_meta_cluster, by = "Name")

mitmatmisc::add_month_fct(dat_plot_month, 10)



gg_month <-
dat_plot_month %>% 
  ggplot(aes(Longitude, Latitude, colour = estimate*10))+
  borders()+
  geom_point(size = 0.5)+
  facet_wrap(~month_fct, ncol = 4)+
  scale_color_scico("Trends 1971-2019 [cm / decade]",
                    palette = "vik", rescaler = scales::rescale_mid, direction = -1)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"), breaks = c(5,10,15))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"), breaks = c(44, 46, 48))+
  coord_quickmap(xlim = range(dat_meta_cluster$Longitude), ylim = range(dat_meta_cluster$Latitude))+
  theme_bw()+
  theme(panel.grid.minor = element_blank(),
        legend.position = "bottom")

ggsave(gg_month,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/fig/map_hs-trend_month.png",
       width = 12,
       height = 5)
  

# map: seasonal -----------------------------------------------------------------


dat_plot_season <- dat_seasonal_gls[term == "year0" & !startsWith(variable, "SCD")] %>% 
  merge(dat_meta_cluster, by = "Name")


dat_plot_season[, variable_fct := fct_relevel(variable, "maxHS_NDJFMAM", after = Inf)]

gg_season <-
  dat_plot_season %>% 
  ggplot(aes(Longitude, Latitude, colour = estimate*10))+
  borders()+
  geom_point(size = 0.5)+
  facet_wrap(~ variable_fct)+
  scale_color_scico("Trends 1971-2019 [cm / decade]",
                    palette = "vik", rescaler = scales::rescale_mid, direction = -1)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"), breaks = c(5,10,15))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"), breaks = c(44, 46, 48))+
  coord_quickmap(xlim = range(dat_meta_cluster$Longitude), ylim = range(dat_meta_cluster$Latitude))+
  theme_bw()+
  theme(panel.grid.minor = element_blank(),
        legend.position = "bottom")

ggsave(gg_season,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/fig/map_hs-trend_season.png",
       width = 6.5,
       height = 5)




# hist: month --------------------------------------------------------------



dat_hist_month <- dat_month_gls[term == "year0"] %>% 
  merge(dat_meta_cluster, by = "Name")
mitmatmisc::add_month_fct(dat_hist_month, 10)
# dat_hist_month[, elev_fct := cut(Elevation, seq(0,3000,by=500), dig.lab = 6)]
dat_hist_month[, elev_fct := cut(Elevation, c(seq(0,2000,by=500), 3000), dig.lab = 6)]

gg_hist_month <- dat_hist_month %>% 
  ggplot(aes(estimate*10, fill = cluster_fct))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_histogram(binwidth = 2, boundary = 0)+
  scale_fill_brewer("Region", palette = "Set1")+
  facet_grid(fct_rev(elev_fct) ~ month_fct, scales = "free_y")+
  theme_bw()+
  theme(panel.grid.minor = element_blank(),
        legend.position = "bottom")+
  xlab("Trend 1971-2019 [cm / decade]")+
  ylab("# of stations")


ggsave(gg_hist_month,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/fig/hist_hs-trend_month.png",
       width = 12,
       height = 6)



# hist: season HS --------------------------------------------------------------



dat_hist_season <- dat_seasonal_gls[term == "year0" & !startsWith(variable, "SCD")] %>% 
  merge(dat_meta_cluster, by = "Name")

dat_hist_season[, variable_fct := fct_relevel(variable, "maxHS_NDJFMAM", after = Inf)]
# dat_hist_month[, elev_fct := cut(Elevation, seq(0,3000,by=500), dig.lab = 6)]
dat_hist_season[, elev_fct := cut(Elevation, c(seq(0,2000,by=500), 3000), dig.lab = 6)]

gg_hist_season <-
dat_hist_season %>% 
  ggplot(aes(estimate*10, fill = cluster_fct))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_histogram(binwidth = 2, boundary = 0)+
  scale_fill_brewer("Region", palette = "Set1")+
  facet_grid(fct_rev(elev_fct) ~ variable_fct, scales = "free_y")+
  theme_bw()+
  theme(panel.grid.minor = element_blank(),
        legend.position = "bottom")+
  xlab("Trend 1971-2019 [cm / decade]")+
  ylab("# of stations")


ggsave(gg_hist_season,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/fig/hist_hs-trend_seasonHS.png",
       width = 12,
       height = 6)


# hist: season SCD --------------------------------------------------------------



dat_hist_season2 <- dat_seasonal_gls[term == "year0" & startsWith(variable, "SCD")] %>% 
  merge(dat_meta_cluster, by = "Name")

dat_hist_season2[, variable_fct := fct_relevel(variable, "SCD_NDJF")]
# dat_hist_month[, elev_fct := cut(Elevation, seq(0,3000,by=500), dig.lab = 6)]
dat_hist_season2[, elev_fct := cut(Elevation, c(seq(0,2000,by=500), 3000), dig.lab = 6)]

gg_hist_season2 <-
  dat_hist_season2 %>% 
  ggplot(aes(estimate*10, fill = cluster_fct))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_histogram(binwidth = 1, boundary = 0)+
  scale_fill_brewer("Region", palette = "Set1")+
  facet_grid(fct_rev(elev_fct) ~ variable_fct, scales = "free_y")+
  theme_bw()+
  theme(panel.grid.minor = element_blank(),
        legend.position = "bottom")+
  xlab("Trend 1971-2019 [days / decade]")+
  ylab("# of stations")


ggsave(gg_hist_season2,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/fig/hist_hs-trend_seasonSCD.png",
       width = 12,
       height = 6)


