# graphical abstract? (full long period)


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(scico)

library(forcats)
library(foreach)
library(flextable)
library(officer)




# try trends based on season ----------------------------------------------

sub_hydro_years <- 1971:2018
# sub_hydro_years <- 1981:2017 # low 1990 snow







dat_meta_clust <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/meta-with-cluster-01.rds")
dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/meta_long_HS.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/data_long_HS.rds")

dat_hs <- mitmatmisc::add_hydro_year(dat_hs)
mitmatmisc::add_season_fct(dat_hs)

# subset to spatcons
stns_ok <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/spatcons_stns_ok.rds")
dat_meta <- dat_meta[Name %in% stns_ok]
dat_hs <- dat_hs[Name %in% stns_ok]


dat_hs_season <- dat_hs[, .(HS = mean(HS), nn = .N), .(Name, season, hydro_year)]
dat_hs_season <- dat_hs_season[nn == 3]


dat_hs_season2 <- dat_hs_season[hydro_year %in% sub_hydro_years  & !is.na(HS)]
dat_hs_season2[, nn_year := .N, .(Name, season)]
dat_hs_season_full <- dat_hs_season2[nn_year == max(nn_year)]

dat_hs_season_full[, length(unique(Name)), season]


dat_hs_season_full[, hydro_year0 := hydro_year - min(hydro_year)]
dat_hs_season_full_lm <- dat_hs_season_full[season %in% c("DJF", "MAM"),
                                            broom::tidy(lm(HS ~ hydro_year0)), 
                                            .(Name, season)]





# paper plot seasonal ------------------------------------------------------

dat_lm_full <- copy(dat_hs_season_full_lm)


dat_lm_full[!is.na(statistic)] %>% 
  dcast(Name + season ~ term, value.var = "estimate") %>% 
  .[! (abs(`(Intercept)`) < 1 & abs(hydro_year0) < 0.05)] %>% 
  .[, .(Name, season)] %>% 
  merge(dat_lm_full) -> dat_lm_sub





dat_plot_full <- dat_lm_sub[term == "hydro_year0"] %>% 
  merge(dat_meta_clust, by = "Name")
dat_plot_full[, est_low := estimate - 1.96 * std.error]
dat_plot_full[, est_high := estimate + 1.96 * std.error]
# mitmatmisc::add_month_fct(dat_plot_full, 10)

dat_plot_full


# manual limits y (elev)
dat_ylim <- dat_plot_full[, .(max_elev = max(Elevation)), .(cluster_fct)] 
setorder(dat_ylim, cluster_fct)
dat_ylim[, max_elev := c(1250, 1250, 3000, 3000, 1250)]


dat_plot_full %>% 
  ggplot(aes(estimate, Elevation, xmin = est_low, xmax = est_high, colour = cluster_fct))+
  geom_vline(xintercept = 0)+
  geom_pointrange(size = 0.1, fatten = 0.5)+
  # geom_jitter(width = 0, height = 0.3)+
  # facet_wrap(~month_fct, nrow = 2)+
  facet_grid(cluster_fct ~ season, scales = "free", space = "free")+ # free_x or free
  theme_bw()+
  theme(panel.grid.minor = element_blank())+
  scale_color_brewer("", palette = "Set1", guide = F)+
  scale_x_continuous(breaks = seq(-4, 4, by = 2))+
  scale_y_continuous(breaks = seq(0, 3000, by = 500), limits = c(0, NA))+
  geom_blank(inherit.aes = F, data = dat_ylim, aes(x = 0, y = max_elev))+
  xlab("Linear trend in mean monthly HS [cm per year]")+
  ylab("Elevation [m]")



# avg plot ----------------------------------------------------------------

dat_lm_full <- copy(dat_hs_season_full_lm)

dat_lm_full[!is.na(statistic)] %>% 
  dcast(Name + season ~ term, value.var = "estimate") %>% 
  .[! (abs(`(Intercept)`) < 1 & abs(hydro_year0) < 0.05)] %>% 
  .[, .(Name, season)] %>% 
  merge(dat_lm_full) -> dat_lm_sub

dat_lm_sub2 <- merge(dat_lm_sub, dat_meta_clust)
# dat_lm_sub2[, elev_fct := cut(Elevation, breaks = 0:3*1000, dig.lab = 5)]
dat_lm_sub2[, elev_fct := cut(Elevation, breaks = c(0, 500, 1000, 1500, 2000, 2500, 3000), dig.lab = 5)]


dat_lm_sub_avg <- dat_lm_sub2[, 
                              .(mu = mean(estimate)), 
                              .(season, term, cluster_fct, elev_fct)]

dat_lm_sub_avg %>% 
  dcast(... ~ term, value.var = "mu") -> dat_plot

setnames(dat_plot, "(Intercept)", "y1971")

dat_plot[, y2018 := y1971 + 47*hydro_year0]

dat_plot2 <- melt(dat_plot, 
                  id.vars = c("season", "cluster_fct", "elev_fct"),
                  measure.vars = c("y1971", "y2018"))

dat_plot2[, value_sc := value / max(value), .(elev_fct, season)]

dat_plot2 %>% 
  ggplot(aes(variable, elev_fct))+
  geom_point(shape = 42, aes(size = value_sc))+
  scale_size_area(max_size = 20)+
  facet_grid(season ~ cluster_fct)+
  theme_bw()


dat_plot2 %>% 
  ggplot(aes(1, elev_fct))+
  geom_point(shape = 42, aes(size = value_sc, colour = (variable)))+
  scale_size_area(max_size = 20)+
  scale_color_grey()+
  facet_grid(season ~ cluster_fct)+
  theme_bw()
