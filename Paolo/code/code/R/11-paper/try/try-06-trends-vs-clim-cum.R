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

dat_lm_mw <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/trends-01-mw-hydro-year.rds")


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





# cumulative --------------------------------------------------------------

dat_lm_mw[mw_year_start == 1981]

months <- month.abb[c(11:12,1:5)]
hydro_years <- 1981:2010


mitmatmisc::add_hydro_year(dat_hs_apgd_eobs2)
dat_hs_apgd_eobs2[, .N, hydro_year]

mitmatmisc::add_hydro_year(dat_hs_uerra_full2)
dat_hs_uerra_full2[, .N, keyby = hydro_year]



dat_tp_clim <- foreach(
  i_month = seq_along(months),
  .final = rbindlist
) %do% {
  
  dat_hs_uerra_full2[hydro_year %in% hydro_years & month_fct %in% months[1:i_month],
                     .(tmean_clim = mean(tmean),
                       tmean_lapse_clim = mean(tmean_lapse),
                       prec_clim = sum(prec),
                       nn = .N),
                     .(Name, hydro_year)] %>% 
    .[nn == i_month,
      .(tmean_clim = mean(tmean_clim),
        tmean_lapse_clim = mean(tmean_lapse_clim),
        prec_clim = mean(prec_clim),
        nn = .N),
      .(Name)] %>% 
    .[nn == length(hydro_years)] -> dat_out

  dat_out[, cum_month_fct := months[i_month]]  
  dat_out
}


dat_tp_clim[, month := match(cum_month_fct, month.abb)]
mitmatmisc::add_month_fct(dat_tp_clim, 10)





dat_tp_clim %>% 
  merge(dat_meta_clust, by = "Name")  %>% 
  ggplot(aes(Elevation, tmean_lapse_clim, colour = cluster_fct))+
  geom_point(size = 0.5)+
  scale_color_brewer("", palette = "Set1")+
  facet_wrap( ~ month_fct, scales = "free")+
  theme_bw()+ 
  xlab("Elevation [m]")+
  ylab("Average mean temperature (lapse rate adjusted) [deg C]")





dat_tp_clim %>% 
  merge(dat_meta_clust, by = "Name")  %>% 
  ggplot(aes(Elevation, prec_clim, colour = cluster_fct))+
  geom_point(size = 0.5)+
  scale_color_brewer("", palette = "Set1")+
  facet_wrap( ~ month_fct, scales = "free")+
  theme_bw()+ 
  xlab("Elevation [m]")+
  ylab("Average precipitation [mm]")




dat_lm_mw[mw_year_start == min(hydro_years) & !is.na(statistic)] %>% 
  dcast(Name + month ~ term, value.var = "estimate") %>% 
  .[! (abs(`(Intercept)`) < 1 & abs(year0) < 0.05)] %>% 
  .[, .(Name, month)] %>% 
  merge(dat_lm_mw[mw_year_start == min(hydro_years)]) -> dat_hs_lm_sub

dat_tp_clim %>% 
  merge(dat_meta_clust, by = "Name") %>% 
  merge(dat_hs_lm_sub[term == "year0"], by = c("Name", "month")) -> dat_plot_lm 



dat_plot_lm %>% 
  ggplot(aes(tmean_lapse_clim, estimate, color = cluster_fct))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_point(size = 0.5)+
  scale_color_brewer(palette = "Set1", guide = F)+
  facet_grid(cluster_fct ~ month_fct, scales = "free")+
  theme_bw()+
  xlab("Average mean temperature (lapse rate adjusted) [deg C]")+
  ylab("Linear trend in mean monthly HS [cm per year]")



dat_plot_lm %>% 
  ggplot(aes(prec_clim, estimate, color = cluster_fct))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_point(size = 0.5)+
  scale_color_brewer(palette = "Set1", guide = F)+
  facet_grid(cluster_fct ~ month_fct, scales = "free")+
  theme_bw()+
  xlab("Average precipitation [mm]")+
  ylab("Linear trend in mean monthly HS [cm per year]")




# djf only ----------------------------------------------------------------





