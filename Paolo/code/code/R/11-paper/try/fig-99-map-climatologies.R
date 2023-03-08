# plot monthly? climatologies?
# or winter?

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
# library(foreach)
library(scico)
library(lemon)


# prep data ---------------------------------------------------------------

dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/data_long_HS.rds")
dat_meta<- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/meta_long_HS.rds")

dat_hs <- mitmatmisc::add_hydro_year(dat_hs)
mitmatmisc::add_month_fct(dat_hs, 10)



# by month ----------------------------------------------------------------



dat_hs_clim <- dat_hs[hydro_year %in% 1981:2010 & month %in% c(11:12,1:5),
                      .(HS = mean(HS)), 
                      .(Name, month_fct)]

dat_plot <- merge(dat_hs_clim, dat_meta, by = "Name")

dat_plot %>% 
  ggplot(aes(Longitude, Latitude, colour = HS))+
  geom_point(size = 0.5)+
  # scale_color_scico(palette = "davos", direction = -1)+
  facet_wrap(~month_fct)+
  # borders()+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "°E"))+
  scale_y_continuous(labels = function(x) paste0(x, "°N"))+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_bw()



# only seasonal -----------------------------------------------------------



season_months <- c(12, 1:2)

dat_hs_clim_seasonal <- dat_hs[!is.na(HS) & hydro_year %in% 1981:2010 & month %in% season_months,
                      .(HS = mean(HS), nn = .N), 
                      .(Name)]

dat_hs_clim_seasonal <- dat_hs_clim_seasonal[nn == length(season_months)*30]

dat_plot_seasonal <- merge(dat_hs_clim_seasonal, dat_meta, by = "Name")

dat_plot_seasonal %>% 
  ggplot(aes(Longitude, Latitude, colour = HS))+
  # borders(fill = "grey90")+
  borders()+
  geom_point(size = 1)+
  scale_color_scico(palette = "imola", direction = -1)+
  # scale_color_scico(palette = "davos", direction = -1)+
  # scale_color_binned(direction = -1)+
  # scale_color_viridis_c(direction = -1)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "°E"))+
  scale_y_continuous(labels = function(x) paste0(x, "°N"))+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_bw()



# not very exciting

