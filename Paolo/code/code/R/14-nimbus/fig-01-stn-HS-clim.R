
# calc monthly and seasonal means based on EOF results --------------------



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(scico)
library(patchwork)
library(forcats)
library(foreach)

dat_hs_month <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/rds/data-04-monthly-1981-2010.rds")
dat_hs_season <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/rds/data-05-seasonal-1981-2010.rds")
dat_meta_cluster <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/rds/meta-with-cluster-01.rds")

# calc clim
dat_hs_month_clim <- dat_hs_month[, .(HS = mean(HS), nn_HS = .N), .(Name, month)]
dat_hs_season_clim <- dat_hs_season[, .(value = mean(value), nn_value = .N), .(Name, variable)]


# monthly -----------------------------------------------------------------


dat_plot_month <- dat_hs_month_clim %>% 
  merge(dat_meta_cluster, by = "Name")

mitmatmisc::add_month_fct(dat_plot_month, 10)



gg_month <- dat_plot_month %>% 
  ggplot(aes(Longitude, Latitude, colour = HS))+
  borders()+
  geom_point(size = 0.5)+
  facet_wrap(~month_fct, ncol = 4)+
  scale_color_viridis_c("Average monthly snow depth (1981-2010) [cm]", direction = -1)+
  # scale_color_binned(type = "viridis", n.breaks = 8, direction = -1)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"), breaks = c(5,10,15))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"), breaks = c(44, 46, 48))+
  coord_quickmap(xlim = range(dat_meta_cluster$Longitude), ylim = range(dat_meta_cluster$Latitude))+
  theme_bw()+
  theme(panel.grid.minor = element_blank(),
        legend.position = "bottom")

ggsave(gg_month,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/fig/map_hs-clim_month.png",
       width = 12,
       height = 5)
  

# seasonal -----------------------------------------------------------------


dat_plot_season <- dat_hs_season_clim[!startsWith(variable, "SCD")] %>% 
  merge(dat_meta_cluster, by = "Name")


dat_plot_season[, variable_fct := fct_relevel(variable, "maxHS_NDJFMAM", after = Inf)]

gg_season <-
  dat_plot_season %>% 
  ggplot(aes(Longitude, Latitude, colour = value))+
  borders()+
  geom_point(size = 0.5)+
  facet_wrap(~ variable_fct)+
  scale_color_viridis_c("Average seasonal snow depth index (1981-2010) [cm]", direction = -1)+
  # scale_color_binned(type = "viridis", n.breaks = 8, direction = -1)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"), breaks = c(5,10,15))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"), breaks = c(44, 46, 48))+
  coord_quickmap(xlim = range(dat_meta_cluster$Longitude), ylim = range(dat_meta_cluster$Latitude))+
  theme_bw()+
  theme(panel.grid.minor = element_blank(),
        legend.position = "bottom")

ggsave(gg_season,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/fig/map_hs-clim_season.png",
       width = 6.5,
       height = 5)

