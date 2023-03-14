# compare HS trends to T & P clim
# maybe also try comparing to trends?





library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(scico)

library(forcats)
library(foreach)
library(flextable)
library(officer)




# data --------------------------------------------------------------------

dat_meta_clust <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/meta-with-cluster-01.rds")
load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/data4corr-01-merged.rda")


# need to adjust by lapse rate!
dat_elev_eobs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/eobs-02-elev.rds")

dat_hs_apgd_eobs2 <- merge(dat_hs_apgd_eobs, dat_elev_eobs, by = "Name") %>% 
  merge(dat_meta_clust)
dat_hs_apgd_eobs2[, tmean_lapse := tmean + (elev_eobs - Elevation) * 6.4 / 1000]


dat_elev_uerra <-  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/uerra-02-elev.rds")

dat_hs_uerra_full %>% 
  merge(dat_meta_clust) %>% 
  merge(dat_elev_uerra) -> dat_hs_uerra_full2

dat_hs_uerra_full2[, tmean_lapse := tmean + (elev_uerra - Elevation) * 6.4 / 1000]




# djf hs t p --------------------------------------------------------------

mitmatmisc::add_hydro_year(dat_hs_uerra_full2)
mitmatmisc::add_season_fct(dat_hs_uerra_full2)

dat_hs_uerra_full2[hydro_year %in% c(1981:2010),
                   .(HS = mean(HS),
                     tmean_lapse = mean(tmean_lapse),
                     prec = sum(prec),
                     nn = .N),
                   .(Name, hydro_year, season)] %>% 
  .[nn == 3] %>% 
  .[, 
    .(HS = mean(HS),
      tmean_lapse = mean(tmean_lapse),
      prec = mean(prec),
      nn = .N),
    .(Name, season)] %>% 
  .[nn == 30] -> dat_clim



dat_hs_uerra_full2[hydro_year %in% c(1981:2010) & season == "DJF",
                   .(HS = mean(HS),
                     tmean_lapse = mean(tmean_lapse),
                     prec = sum(prec),
                     nn = .N),
                   .(Name, hydro_year, season)] %>% 
  .[nn == 3] %>% 
  .[, 
    .(corr = c(cor(HS, tmean_lapse), cor(HS, prec)),
      climval = c("tmean", "prec"),
      nn = .N),
    .(Name, season)] %>% 
  .[nn == 30] -> dat_cor_djf



dat_hs_uerra_full2[hydro_year %in% c(1981:2010),
                   .(HS = mean(HS),
                     tmean_lapse = mean(tmean_lapse),
                     prec = sum(prec),
                     nn = .N),
                   .(Name, hydro_year)] %>% 
  .[nn == 7] %>% 
  .[, 
    .(corr = c(cor(HS, tmean_lapse), cor(HS, prec)),
      climval = c("tmean", "prec"),
      nn = .N),
    .(Name)] %>% 
  .[nn == 30] -> dat_cor_ndjfmam

# plot clim --------------------------------------------------------------------

dat_clim %>% 
  merge(dat_meta_clust, by = "Name") -> dat_plot

dat_plot[!is.na(HS)] %>% 
  ggplot(aes(Longitude, Latitude, color = HS))+
  borders()+
  geom_point()+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_quickmap(xlim = range(dat_meta_clust$Longitude), ylim = range(dat_meta_clust$Latitude))+
  scale_color_scico(palette = "imola", direction = -1)+
  theme_bw()+
  facet_wrap(~season)

dat_plot %>% 
  ggplot(aes(Elevation, HS, color = cluster_fct))+
  geom_point()+
  geom_smooth(se = F)+
  scale_color_brewer(palette = "Set1")+
  theme_bw()+
  facet_wrap(~season)

dat_plot %>% 
  ggplot(aes(Elevation, tmean_lapse, color = cluster_fct))+
  geom_point()+
  # geom_smooth(se = F, method = lm)+
  scale_color_brewer(palette = "Set1")+
  theme_bw()+
  facet_wrap(~season, scales = "free_y")


dat_plot %>% 
  ggplot(aes(Elevation, prec, color = cluster_fct))+
  geom_point()+
  # geom_smooth(se = F, method = lm)+
  scale_color_brewer(palette = "Set1")+
  theme_bw()+
  facet_wrap(~season, scales = "free_y")


dat_plot %>% 
  ggplot(aes(Elevation, prec, color = cluster_fct))+
  geom_point()+
  # geom_smooth(se = F, method = lm)+
  scale_color_brewer(palette = "Set1")+
  theme_bw()+
  facet_grid(season ~ cluster_fct, scales = "free", space = "free")


dat_plot[season == "DJF"] %>% 
  ggplot(aes(prec, HS, color = cluster_fct))+
  geom_point()+
  # geom_smooth(se = F, method = lm)+
  scale_color_brewer(palette = "Set1")+
  theme_bw()+
  facet_grid(season ~ cluster_fct, scales = "free", space = "free")


dat_plot[season == "DJF"] %>% 
  ggplot(aes(tmean_lapse, HS, color = cluster_fct))+
  geom_point()+
  # geom_smooth(se = F, method = lm)+
  scale_color_brewer(palette = "Set1")+
  theme_bw()+
  facet_grid(season ~ cluster_fct, scales = "free", space = "free")




# plot cor  ----------------------------------------------------------------

dat_cor_djf %>% 
  merge(dat_meta_clust, by = "Name") %>% 
  ggplot(aes(corr, Elevation, color = cluster_fct))+
  geom_point()+
  scale_color_brewer(palette = "Set1")+
  facet_grid(season ~ climval)+
  theme_bw()


dat_cor_ndjfmam %>% 
  merge(dat_meta_clust, by = "Name") %>% 
  ggplot(aes(corr, Elevation, color = cluster_fct))+
  geom_point()+
  scale_color_brewer(palette = "Set1")+
  facet_grid(. ~ climval)+
  theme_bw()

