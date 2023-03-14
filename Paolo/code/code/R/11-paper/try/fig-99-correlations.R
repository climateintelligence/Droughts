# try out correlations plots



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(scico)

library(forcats)
library(foreach)
library(flextable)
library(officer)



# read data ---------------------------------------------------------------



dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/meta_long_HS.rds")
dat_meta_clust <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/meta-with-cluster-01.rds")

dat_meta_clust[, cluster_fct := fct_relevel(cluster_fct, "NW", "NE", "C" , "SW", "SE")]

dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/data_long_HS.rds")
dat_hs <- mitmatmisc::add_month_fct(dat_hs, 10)


dat_apgd <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/apgd-01-series.rds")
dat_eobs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/eobs-01-series.rds")

dat_chelsa <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/chelsa-01-series.rds")
dat_uerra <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/uerra-01-series.rds")



# obs data: apgd & eobs

dat_hs[year %in% 1971:2008 & month %in% c(11,12,1:5)] %>% 
  merge(dat_apgd) %>% 
  merge(dat_eobs) -> dat_hs_apgd_eobs

dat_hs_apgd_eobs[, 
                 n_year := sum(!is.na(HS) & !is.na(prec) & !is.na(tmean)),
                 .(Name, month)]

dat_hs_apgd_eobs <- dat_hs_apgd_eobs[n_year >= 30]



# chelsa

dat_hs_chelsa <- merge(dat_hs[month %in% c(11,12,1:5)], 
                       dat_chelsa[prec < 60000])

dat_hs_chelsa[, 
              n_year := sum(!is.na(HS) & !is.na(prec) & !is.na(tmean)),
              .(Name, month)]

dat_hs_chelsa <- dat_hs_chelsa[n_year >= 30]


# uerra (obs period)

dat_hs[year %in% 1971:2008 & month %in% c(11,12,1:5)] %>% 
  merge(dat_uerra) -> dat_hs_uerra

dat_hs_uerra[, 
             n_year := sum(!is.na(HS) & !is.na(prec) & !is.na(tmean)),
             .(Name, month)]

dat_hs_uerra <- dat_hs_uerra[n_year >= 30]



# uerra (full period)

dat_hs[month %in% c(11,12,1:5)] %>% 
  merge(dat_uerra) -> dat_hs_uerra_full

dat_hs_uerra_full[, 
             n_year := sum(!is.na(HS) & !is.na(prec) & !is.na(tmean)),
             .(Name, month)]

dat_hs_uerra_full <- dat_hs_uerra_full[n_year >= 30]


# obs corr ---------------------------------------------------------------



dat_corr_obs <- dat_hs_apgd_eobs[, 
                                 .(corr = c(cor(HS, tmean, use = "p"),
                                            cor(HS, prec, use = "p")),
                                   climval = c("tmean", "prec")),
                                 .(Name, month_fct)]


dat_plot_corr_obs <- dat_corr_obs[!is.na(corr)] %>% merge(dat_meta_clust, by = "Name")


dat_plot_corr_obs %>% 
  ggplot(aes(corr, Elevation))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_point(size = 0.5)+
  facet_grid(climval ~ month_fct)+
  theme_bw()

dat_plot_corr_obs[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 500), dig.lab = 5)]
dat_table_corr_obs <- dat_plot_corr_obs[, 
                                        .(mean_corr = mean(corr)), 
                                        .(climval, month_fct, elev_fct, cluster_fct)]

dat_table_corr_obs2 <- dcast(dat_table_corr_obs,
                             month_fct + cluster_fct ~ elev_fct + climval,
                             value.var = "mean_corr")

dat_header <- data.table(col_keys = names(dat_table_corr_obs2))
dat_header[, c("row1", "row2") := tstrsplit(col_keys, "_")]
dat_header[1:2, ":="(row1 = c("",""), row2 = c("Month", "Region"))]

dat_table_corr_obs2 %>% 
  flextable() %>%
  set_header_df(dat_header) %>% 
  colformat_num(j = names(dat_table_corr_obs2)[-c(1:2)]) %>% 
  theme_booktabs() %>% 
  merge_h(part = "header") %>% 
  align(align = "center", part = "all") %>% 
  merge_v(j = "month_fct") %>% 
  valign(j = "month_fct", valign = "top") %>% 
  fix_border_issues() # %>%
  # autofit()


# chelsa ---------------------------------------------------------------



dat_corr_chelsa <- dat_hs_chelsa[, 
                                 .(corr = c(cor(HS, tmean, use = "p"),
                                            cor(HS, prec, use = "p")),
                                   climval = c("tmean", "prec")),
                                 .(Name, month_fct)]


dat_plot_corr_chelsa <- dat_corr_chelsa[!is.na(corr)] %>% merge(dat_meta_clust, by = "Name")


dat_plot_corr_chelsa %>% 
  ggplot(aes(corr, Elevation))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_point(size = 0.5)+
  facet_grid(climval ~ month_fct)+
  theme_bw()


# very similar to obs, except low alt (<1500m) precip



# uerra -------------------------------------------------------------------


dat_corr_uerra <- dat_hs_uerra[, 
                                 .(corr = c(cor(HS, tmean, use = "p"),
                                            cor(HS, prec, use = "p")),
                                   climval = c("tmean", "prec")),
                                 .(Name, month_fct)]


dat_plot_corr_uerra <- dat_corr_uerra[!is.na(corr)] %>% merge(dat_meta_clust, by = "Name")


dat_plot_corr_uerra %>% 
  ggplot(aes(corr, Elevation))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_point(size = 0.5)+
  facet_grid(climval ~ month_fct)+
  theme_bw()


# uerra full --------------------------------------------------------------


dat_corr_uerra_full <- dat_hs_uerra_full[, 
                               .(corr = c(cor(HS, tmean, use = "p"),
                                          cor(HS, prec, use = "p")),
                                 climval = c("tmean", "prec"),
                                 n_year = c(sum(!is.na(HS) & !is.na(tmean)),
                                            sum(!is.na(HS) & !is.na(prec)))),
                               .(Name, month_fct)]


dat_plot_corr_uerra_full <- dat_corr_uerra_full[!is.na(corr)] %>% merge(dat_meta_clust, by = "Name")


dat_plot_corr_uerra_full %>% 
  ggplot(aes(corr, Elevation))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_point(size = 0.5)+
  facet_grid(climval ~ month_fct)+
  theme_bw()




# moving window uerra -----------------------------------------------------

# only complete series (as for trends)

dat_hs2 <- dat_hs[year >= 1971 & year <= 2019 & month %in% c(11,12,1:5) & !is.na(HS)]
with(dat_hs2, table(year, month))

dat_hs2[, nn_year := .N, .(Name, month)]
dat_hs_full <- dat_hs2[nn_year == max(nn_year)]

dat_meta[Name %in% dat_hs_full$Name, .N, Provider]


dat_hs_full_uerra <- merge(dat_hs_full, dat_uerra)

# settings
window_length <- 30
l_windows <- lapply(1971:1990, function(x) x + seq(0, window_length - 1))
month_sub <- c(11:12, 1:5)

# moving window correlations
dat_mw_corr <- foreach(
  i_window = seq_along(l_windows),
  .final = rbindlist
) %dopar% {
  
  # HS ------------------ #
  # subset data
  i_dat_hs <- dat_hs_full_uerra[year %in% l_windows[[i_window]] & 
                                  month %in% month_sub & 
                                  !is.na(HS)]
  i_dat_hs[, nn_years := .N, .(Name, month)]
  i_dat_hs <- i_dat_hs[nn_years == window_length]
  
  
  # corr
  dat_hs_corr <- i_dat_hs[,
                          .(corr = c(cor(HS, tmean, use = "p"),
                                     cor(HS, prec, use = "p")),
                            climval = c("tmean", "prec")),
                          .(Name, month_fct)]
  
  
  
  # return
  dat_hs_corr[, ":="(snow = "HS", 
                   mw_year_start = min(l_windows[[i_window]]),
                   mw_year_end = max(l_windows[[i_window]]))]
  
  dat_hs_corr
}


# plot
dat_mw_corr2 <- merge(dat_mw_corr, dat_meta_clust, by = "Name")
dat_mw_corr2[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 250), dig.lab = 5)]
dat_mw_corr2[, n_year := sum(!is.na(corr)), .(month_fct, climval, mw_year_start, elev_fct)]


dat_plot_mw_corr <- dat_mw_corr2[!is.na(corr),
                                 .(mean_corr = mean(corr),
                                   nn = .N),
                                 .(month_fct, climval, mw_year_start, elev_fct)]

dat_plot_mw_corr[nn < 10, 
                 .(month_fct, climval, mw_year_start, elev_fct)] %>% 
  unique %>% 
  merge(dat_mw_corr2[!is.na(corr)], 
        by = c("month_fct", "climval", "mw_year_start", "elev_fct")) -> dat_plot_mw_point
# dat_mw_corr2[!is.na(corr) & n_year < 10]

dat_plot_mw_corr[, c("ymin", "ymax") := tstrsplit(elev_fct, "[^0-9.-]", 
                                                  keep = 2:3, type.convert = T)]


# dat_plot_mw_corr[nn < 10, .(month_fct, )]
lim_corr <- max(abs(c(dat_plot_mw_corr$mean_corr, dat_plot_mw_point$corr)))

dat_plot_mw_point %>% 
  ggplot(aes(mw_year_start, Elevation, colour = corr))+
  geom_point()+
  geom_rect(inherit.aes = F, data = dat_plot_mw_corr[nn >= 10],
            aes(xmin = mw_year_start - 0.5, xmax = mw_year_start + 0.5,
                ymin = ymin, ymax = ymax, fill = mean_corr))+
  facet_grid(climval ~ month_fct)+
  theme_bw()+
  scale_fill_scico(palette = "vik", limits = c(-1,1) * lim_corr)+
  scale_colour_scico(palette = "vik", limits = c(-1,1) * lim_corr)




# ** other plot -----------------------------------------------------------


# plot
dat_mw_ts <- merge(dat_mw_corr, dat_meta_clust, by = "Name")
dat_mw_ts[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 500), dig.lab = 5)]
dat_mw_ts[, n_year := sum(!is.na(corr)), .(month_fct, climval, mw_year_start, elev_fct)]

dat_plot_mw_ts <- dat_mw_ts[!is.na(corr),
                                 .(mean_corr = mean(corr),
                                   nn = .N),
                                 .(month_fct, climval, mw_year_start, elev_fct)]

dat_plot_mw_ts %>% 
  ggplot(aes(mw_year_start, mean_corr, 
             colour = nn,
             shape = climval, linetype = climval))+
  geom_point()+
  # geom_line()+
  facet_grid(fct_rev(elev_fct) ~ month_fct)+
  scale_colour_viridis_c()+
  theme_bw()
  

# -> not really useful time series
