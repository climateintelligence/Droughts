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



# dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/meta_long_HS.rds")
dat_meta_clust <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/meta-with-cluster-01.rds")

load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/data4corr-01-merged.rda")


# plot obs corr ---------------------------------------------------------------



dat_corr_obs <- dat_hs_apgd_eobs[, 
                                 .(corr = c(cor(HS, tmean, use = "p"),
                                            cor(HS, prec, use = "p")),
                                   climval = c("tmean", "prec")),
                                 .(Name, month_fct)]


dat_plot_corr_obs <- dat_corr_obs[!is.na(corr)] %>% merge(dat_meta_clust, by = "Name")
dat_plot_corr_obs[, climval_fct := fct_recode(factor(climval),
                                              "Precipitation" = "prec",
                                              "Mean temperature" = "tmean")]

dat_plot_corr_obs %>% 
  ggplot(aes(corr, Elevation, color = cluster_fct))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_point(size = 0.2)+
  scale_color_brewer("Region", palette = "Set1")+
  facet_grid(climval_fct ~ month_fct)+
  theme_bw()+
  xlab("Correlation coefficient of HS with [see row label]")+
  ylab("Elevation [m]")

ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/Figure 7.png",
       width = 12, height = 4)




# table obs corr (needed?) ------------------------------------------------

# maybe appendix, or for writing

dat_plot_corr_obs[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 500), dig.lab = 5)]
dat_table_corr_obs <- dat_plot_corr_obs[, 
                                        .(mean_corr = mean(corr)), 
                                        .(climval, month_fct, elev_fct, cluster_fct)]


dat_table_corr_obs2 <- dcast(dat_table_corr_obs,
                             month_fct + cluster_fct ~ climval + elev_fct,
                             value.var = "mean_corr")

dat_header <- data.table(col_keys = names(dat_table_corr_obs2))
dat_header[, c("row1", "row2") := tstrsplit(col_keys, "_")]
# dat_header[1:2, ":="(row1 = c("",""), row2 = c("Month", "Region"))]
dat_header[1:2, ":="(row1 = c("Month", "Region"), row2 = c("",""))]

dat_table_corr_obs2 %>% 
  flextable() %>%
  set_header_df(dat_header) %>% 
  colformat_num(j = names(dat_table_corr_obs2)[-c(1:2)]) %>% 
  theme_booktabs() %>% 
  # merge_h(part = "header") %>%
  # align(align = "center", part = "all") %>%
  merge_v(j = "month_fct") %>%
  valign(j = "month_fct", valign = "top") %>%
  fix_border_issues() %>%
  autofit()



dat_table_corr_obs[,
                   .(mean_corr = mean(mean_corr)),
                   .(month_fct, climval)] %>% 
  dcast(climval ~ month_fct)


dat_table_corr_obs[,
                   .(mean_corr = mean(mean_corr)),
                   .(month_fct, climval, cluster_fct)] %>% 
  dcast(climval + cluster_fct ~ month_fct)



dat_table_corr_obs[,
                   .(mean_corr = mean(mean_corr)),
                   .(month_fct, climval, elev_fct)] %>% 
  dcast(climval + elev_fct ~ month_fct) 


# uerra -------------------------------------------------------------------


dat_corr_uerra <- dat_hs_uerra[, 
                                 .(corr = c(cor(HS, tmean, use = "p"),
                                            cor(HS, prec, use = "p")),
                                   climval = c("tmean", "prec")),
                                 .(Name, month_fct)]


dat_plot_corr_uerra <- dat_corr_uerra[!is.na(corr)] %>% merge(dat_meta_clust, by = "Name")

dat_plot_corr_uerra[, climval_fct := fct_recode(factor(climval),
                                              "Precipitation" = "prec",
                                              "Mean temperature" = "tmean")]

dat_plot_corr_uerra %>% 
  ggplot(aes(corr, Elevation, color = cluster_fct))+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_point(size = 0.2)+
  scale_color_brewer("Region", palette = "Set1")+
  facet_grid(climval_fct ~ month_fct)+
  theme_bw()+
  xlab("Correlation coefficient of HS with [see row label]")+
  ylab("Elevation [m]")

ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/Figure A8.png",
       width = 12, height = 4)


# uerra full --------------------------------------------------------------

# no need!, since identical to above, even when thresholding n_years 30, 40, 50
# 
# dat_corr_uerra_full <- dat_hs_uerra_full[, 
#                                .(corr = c(cor(HS, tmean, use = "p"),
#                                           cor(HS, prec, use = "p")),
#                                  climval = c("tmean", "prec"),
#                                  n_year = c(sum(!is.na(HS) & !is.na(tmean)),
#                                             sum(!is.na(HS) & !is.na(prec)))),
#                                .(Name, month_fct)]
# 
# 
# dat_plot_corr_uerra_full <- dat_corr_uerra_full[!is.na(corr)] %>% merge(dat_meta_clust, by = "Name")
# 
# dat_plot_corr_uerra_full[, climval_fct := fct_recode(factor(climval),
#                                                 "Precipitation" = "prec",
#                                                 "Mean temperature" = "tmean")]
# 
# dat_plot_corr_uerra_full[n_year > 30] %>% 
#   ggplot(aes(corr, Elevation, color = cluster_fct))+
#   geom_vline(xintercept = 0, linetype = "dashed")+
#   geom_point(size = 0.2)+
#   scale_color_brewer("Region", palette = "Set1")+
#   facet_grid(climval_fct ~ month_fct)+
#   theme_bw()+
#   xlab("Correlation coefficient of HS with [see row label]")+
#   ylab("Elevation [m]")
# 
# ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/appendix-06-correlations-uerra-full-period.png",
#        width = 12, height = 4)


