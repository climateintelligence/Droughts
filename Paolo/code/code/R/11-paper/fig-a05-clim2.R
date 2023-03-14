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




# djf + mam hs t p --------------------------------------------------------------

mitmatmisc::add_hydro_year(dat_hs_apgd_eobs2)
mitmatmisc::add_season_fct(dat_hs_apgd_eobs2)

dat_hs_apgd_eobs2[hydro_year %in% c(1981:2010),
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
  .[nn == 28] -> dat_clim



dat_hs_apgd_eobs2[hydro_year %in% c(1981:2010) & season == "DJF",
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
  .[nn == 28] -> dat_cor_djf



dat_hs_apgd_eobs2[hydro_year %in% c(1981:2010),
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
  .[nn == 28] -> dat_cor_ndjfmam

# plot clim --------------------------------------------------------------------

dat_plot <- merge(dat_clim[!is.na(HS)], dat_meta_clust, by = "Name")


gg1_hs <- dat_plot %>% 
  ggplot(aes(HS, Elevation, color = cluster_fct))+
  geom_point(size = 0.5)+
  scale_color_brewer("Region", palette = "Set1")+
  theme_bw()+
  theme(panel.grid.minor = element_blank())+
  facet_grid(. ~ season, scales = "free")+
  ylab("Elevation [m]")+
  xlab("Mean snow depth [cm]")



gg2_tmean <- dat_plot %>% 
  ggplot(aes(tmean_lapse, Elevation, color = cluster_fct))+
  geom_point(size = 0.5)+
  scale_color_brewer("Region", palette = "Set1")+
  theme_bw()+
  theme(panel.grid.minor = element_blank())+
  facet_grid(. ~ season, scales = "free")+
  ylab("Elevation [m]")+
  xlab("Mean temperature [deg C]")


gg3_prec <- dat_plot %>% 
  ggplot(aes(prec, Elevation, color = cluster_fct))+
  geom_point(size = 0.5)+
  scale_color_brewer("Region", palette = "Set1")+
  theme_bw()+
  theme(panel.grid.minor = element_blank())+
  facet_grid(. ~ season, scales = "free")+
  ylab("Elevation [m]")+
  xlab("Mean precipitation [mm]")

gg_out <- gg1_hs + gg2_tmean + gg3_prec + 
  plot_layout(ncol = 1, guides = "collect")+
  plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")")


ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/Figure B6.png",
       width = 10, height = 8)



# summary clim table ------------------------------------------------------

dat_table <- dat_clim[!is.na(HS)] %>% merge(dat_meta_clust, by = "Name")


dat_table[, elev_fct := cut(Elevation, seq(0, 3000, by = 250), dig.lab = 5)]

dat_table_out <- dat_table[, 
                           .(HS = mean(HS),
                             tmean_lapse = mean(tmean_lapse),
                             prec = mean(prec),
                             nn = .N),
                           .(season, elev_fct, cluster_fct)]

setorder(dat_table_out, season, elev_fct, cluster_fct)
# setcolorder(dat_table_out, c("season", "elev_fct", "cluster_fct"))

dat_table_out[season == "DJF"] %>% 
  flextable() %>% 
  set_header_labels(elev_fct = "Elevation",
                    cluster_fct = "Region",
                    HS = "HS [cm]",
                    tmean_lapse = "Temperature [deg C]",
                    prec = "Precipitation [mm]",
                    nn = "Number") %>% 
  set_formatter_type(fmt_double = "%.01f") %>% 
  set_formatter(prec = function(x) sprintf("%.0f", x)) %>% 
  merge_v(j = c("season", "elev_fct")) %>% 
  valign(valign = "top") %>% 
  autofit() -> ft
  
read_docx() %>% 
  body_add_flextable(ft) %>% 
  print(target = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/table/Table C4.docx")



dat_table_out[season == "MAM"] %>% 
  flextable() %>% 
  set_header_labels(elev_fct = "Elevation",
                    cluster_fct = "Region",
                    HS = "HS [cm]",
                    tmean_lapse = "Temperature [deg C]",
                    prec = "Precipitation [mm]",
                    nn = "Number") %>% 
  set_formatter_type(fmt_double = "%.01f") %>% 
  set_formatter(prec = function(x) sprintf("%.0f", x)) %>% 
  merge_v(j = c("season", "elev_fct")) %>% 
  valign(valign = "top") %>% 
  autofit() -> ft

read_docx() %>% 
  body_add_flextable(ft) %>% 
  print(target = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/table/Table C5.docx")




# plot cor  ----------------------------------------------------------------
dat_cor_ndjfmam[, season := "NDJFMAM"]
dat_plot_cor <- rbindlist(list(dat_cor_djf, dat_cor_ndjfmam), use.names = T)

dat_plot_cor[, climval_fct := fct_recode(factor(climval),
                                              "Precipitation" = "prec",
                                              "Mean temperature" = "tmean")]

dat_plot_cor %>% 
  merge(dat_meta_clust, by = "Name") %>% 
  ggplot(aes(corr, Elevation, color = cluster_fct))+
  geom_point(size = 0.3)+
  scale_color_brewer("Region", palette = "Set1")+
  guides(color = guide_legend(override.aes = list(size = 3)))+
  facet_grid(season ~ climval_fct)+
  theme_bw()+  
  xlab("Correlation coefficient of HS with [see column header]")+
  ylab("Elevation [m]")


ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/Figure B7.png",
       width = 8, height = 4)





