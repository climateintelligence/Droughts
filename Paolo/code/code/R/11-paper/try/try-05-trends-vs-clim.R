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


dat_hs_apgd_eobs %>% summary


# need to adjust by lapse rate!
dat_elev_eobs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/eobs-02-elev.rds")

dat_hs_apgd_eobs2 <- merge(dat_hs_apgd_eobs, dat_elev_eobs, by = "Name") %>% 
  merge(dat_meta_clust)
dat_hs_apgd_eobs2[, tmean_lapse := tmean + (elev_eobs - Elevation) * 6.4 / 1000]


# trends vs clim ----------------------------------------------------------

dat_hs_apgd_eobs[, year0 := year - min(year)]

dat_hs_lm <- dat_hs_apgd_eobs[!is.na(HS),
                              broom::tidy(lm(HS ~ year0)),
                              .(Name, month)]
dat_hs_lm


dat_tp_clim <- dat_hs_apgd_eobs2[!is.na(HS),
                                .(tmean_clim = mean(tmean),
                                  tmean_lapse_clim = mean(tmean_lapse),
                                  prec_clim = mean(prec)),
                                .(Name, month)]



# ** plot -----------------------------------------------------------------


dat_hs_lm[!is.na(statistic)] %>% 
  dcast(Name + month ~ term, value.var = "estimate") %>% 
  .[! (abs(`(Intercept)`) < 1 & abs(year0) < 0.05)] %>% 
  .[, .(Name, month)] %>% 
  merge(dat_hs_lm) -> dat_hs_lm_sub


dat_hs_lm_sub[term == "year0"] %>% 
  merge(dat_tp_clim) %>% 
  merge(dat_meta_clust) -> dat_plot_clim

mitmatmisc::add_month_fct(dat_plot_clim, 10)

dat_plot_clim %>% 
  ggplot(aes(tmean_lapse_clim, estimate, color = cluster_fct))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_point(size = 0.5)+
  scale_color_brewer(palette = "Set1", guide = F)+
  facet_grid(cluster_fct ~ month_fct, scales = "free")+
  theme_bw()+
  xlab("Average mean temperature (lapse rate adjusted) [deg C]")+
  ylab("Linear trend in mean monthly HS [cm per year]")
  
ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/extra-test/scatter_tmean-hs_grid-month-region.png",
       width = 14, height = 8)


dat_plot_clim %>% 
  ggplot(aes(tmean_lapse_clim, estimate, color = cluster_fct))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_point(size = 0.5)+
  scale_color_brewer("", palette = "Set1")+
  facet_wrap( ~ month_fct, scales = "free")+
  theme_bw()+
  xlab("Average mean temperature (lapse rate adjusted) [deg C]")+
  ylab("Linear trend in mean monthly HS [cm per year]")

ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/extra-test/scatter_tmean-hs_wrap-month.png",
       width = 12, height = 8)

dat_plot_clim %>% 
  ggplot(aes(prec_clim, estimate, color = cluster_fct))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_point(size = 0.5)+
  scale_color_brewer(palette = "Set1", guide = F)+
  facet_grid(cluster_fct ~ month_fct, scales = "free")+
  theme_bw()+
  xlab("Average precipitation [mm]")+
  ylab("Linear trend in mean monthly HS [cm per year]")
  
ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/extra-test/scatter_prec-hs_grid-month-region.png",
       width = 14, height = 8)


dat_plot_clim %>% 
  ggplot(aes(prec_clim, estimate, color = cluster_fct))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_point(size = 0.5)+
  scale_color_brewer("", palette = "Set1")+
  facet_wrap( ~ month_fct, scales = "free")+
  theme_bw()+
  xlab("Average precipitation [mm]")+
  ylab("Linear trend in mean monthly HS [cm per year]")

ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/extra-test/scatter_prec-hs_wrap-month.png",
       width = 12, height = 8)


# 
# 
# dat_plot_clim %>% 
#   ggplot(aes(tmean_lapse_clim, estimate, color = prec_clim))+
#   geom_vline(xintercept = 0, linetype = "dashed")+
#   geom_hline(yintercept = 0, linetype = "dashed")+
#   geom_point(size = 0.5)+
#   # scale_color_brewer(palette = "Set1", guide = F)+
#   # scale_color_binned(type = "viridis")+
#   scale_color_viridis_c()+
#   facet_grid(cluster_fct ~ month_fct, scales = "free")+
#   theme_bw()+
#   xlab("Average mean temperature (lapse rate adjusted) [deg C]")+
#   ylab("Linear trend in mean monthly HS [cm per year]")
# 

col_lim <- c(-1, 1) * max(abs(dat_plot_clim$estimate))
dat_plot_clim %>% 
  ggplot(aes(tmean_lapse_clim, prec_clim, color = estimate))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_point(size = 0.5)+
  # scale_color_scico("Linear trend in mean monthly HS [cm per year]",
  #                   palette = "roma", limits = col_lim)+
  # scale_color_binned(type = "viridis", n.breaks = 6)+
  scale_color_viridis_c()+
  facet_grid(cluster_fct ~ month_fct, scales = "free")+
  theme_bw()+ 
  theme(legend.position = "bottom")+
  xlab("Average mean temperature (lapse rate adjusted) [deg C]")+
  ylab("Average precipitation [mm]")

ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/extra-test/scatter2_tmean-prec-hs.png",
       width = 14, height = 8)




# avg clim by elev --------------------------------------------------------

dat_plot_clim %>% 
  ggplot(aes(Elevation, tmean_lapse_clim))+
  geom_point(size = 0.5)+
  facet_grid(cluster_fct ~ month_fct, scales = "free")+
  theme_bw()+ 
  theme(legend.position = "bottom")+
  xlab("Elevation [m]")+
  ylab("Average mean temperature (lapse rate adjusted) [deg C]")

ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/extra-test/scatter_elev-tmean_grid-month-region.png",
       width = 14, height = 8)



dat_plot_clim %>% 
  ggplot(aes(Elevation, tmean_lapse_clim, colour = cluster_fct))+
  geom_point(size = 0.5)+
  scale_color_brewer("", palette = "Set1")+
  facet_wrap( ~ month_fct, scales = "free")+
  theme_bw()+ 
  xlab("Elevation [m]")+
  ylab("Average mean temperature (lapse rate adjusted) [deg C]")

ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/extra-test/scatter_elev-tmean_wrap-month.png",
       width = 12, height = 8)




dat_plot_clim %>% 
  ggplot(aes(Elevation, prec_clim))+
  geom_point(size = 0.5)+
  facet_grid(cluster_fct ~ month_fct)+
  theme_bw()+ 
  theme(legend.position = "bottom")+
  xlab("Elevation [m]")+
  ylab("Average precipitation [mm]")

ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/extra-test/scatter_elev-prec_grid-month-region.png",
       width = 14, height = 8)




dat_plot_clim %>% 
  ggplot(aes(Elevation, prec_clim, colour = cluster_fct))+
  geom_point(size = 0.5)+
  scale_color_brewer("", palette = "Set1")+
  facet_wrap( ~ month_fct)+
  theme_bw()+ 
  xlab("Elevation [m]")+
  ylab("Average precipitation [mm]")

ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/extra-test/scatter_elev-prec_wrap-month.png",
       width = 12, height = 8)







# compare trends hs clim ----------------------------------------------------------


dat_lm_full <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/trends-03-full_1971-2019-calyear.rds")
dat_elev_uerra <-  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/uerra-02-elev.rds")

dat_hs_uerra_full[Name %in% dat_lm_full$Name & year >= 1971, 
                  .N,
                  .(Name, month)] %>% summary

dat_hs_uerra_full[Name %in% dat_lm_full$Name & year >= 1971] %>% 
  merge(dat_meta_clust) %>% 
  merge(dat_elev_uerra) -> dat_hs_uerra_full_sub

dat_hs_uerra_full_sub[, tmean_lapse := tmean + (elev_uerra - Elevation) * 6.4 / 1000]

dat_hs_uerra_full_sub[, year0 := year - min(year)]

dat_lm_tmean <- dat_hs_uerra_full_sub[,
                                      broom::tidy(lm(tmean_lapse ~ year0)),
                                      .(Name, month)]


dat_plot_trend <- merge(
  dat_lm_full[!is.na(statistic)] %>% 
    dcast(Name + month ~ term, value.var = "estimate") %>% 
    .[, .(Name, month, int_hs = `(Intercept)`, slope_hs = year0)],
  dat_lm_tmean[!is.na(statistic)] %>% 
    dcast(Name + month ~ term, value.var = "estimate") %>% 
    .[, .(Name, month, int_tmean = `(Intercept)`, slope_tmean = year0)]
)


dat_plot_trend %>% 
  ggplot(aes(slope_tmean, slope_hs, colour = int_tmean))+
  geom_point()+
  scale_color_gradient2()+
  facet_wrap(~month)
