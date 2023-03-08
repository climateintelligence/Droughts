# summary table for correlations plots



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(scico)

library(forcats)
library(foreach)
library(flextable)
library(officer)



# read and prep data ---------------------------------------------------------------



# dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/meta_long_HS.rds")
dat_meta_clust <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/meta-with-cluster-01.rds")

load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/data4corr-01-merged.rda")



dat_corr_obs <- dat_hs_apgd_eobs[, 
                                 .(corr = c(cor(HS, tmean, use = "p"),
                                            cor(HS, prec, use = "p")),
                                   climval = c("tmean", "prec")),
                                 .(Name, month_fct)]


dat_plot_corr_obs <- dat_corr_obs[!is.na(corr)] %>% merge(dat_meta_clust, by = "Name")
dat_plot_corr_obs[, climval_fct := fct_recode(factor(climval),
                                              "Precipitation" = "prec",
                                              "Mean temperature" = "tmean")]


# table obs corr ------------------------------------------------



dat_plot_corr_obs[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 500), dig.lab = 5)]
dat_table_corr_obs <- dat_plot_corr_obs[, 
                                        .(mean_corr = mean(corr)), 
                                        .(climval_fct, month_fct, elev_fct)]


dat_table_corr_obs2 <- dcast(dat_table_corr_obs,
                             elev_fct ~ climval_fct + month_fct,
                             value.var = "mean_corr")

dat_header <- data.table(col_keys = names(dat_table_corr_obs2))
dat_header[, c("row1", "row2") := tstrsplit(col_keys, "_")]
# dat_header[1:2, ":="(row1 = c("",""), row2 = c("Month", "Region"))]
dat_header[1, ":="(row2 = c("Elevation"), row1 = c(""))]

ft <- dat_table_corr_obs2 %>% 
  flextable() %>%
  set_header_df(dat_header) %>% 
  colformat_num(j = names(dat_table_corr_obs2)[-c(1)]) %>% 
  theme_booktabs() %>%
  merge_h(part = "header") %>%
  align(align = "left", part = "all") %>%
  # merge_v(j = "month_fct") %>%
  # valign(j = "month_fct", valign = "top") %>%
  fix_border_issues() # %>% autofit() 

read_docx() %>% 
  body_add_flextable(ft) %>% 
  print(target = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/table/correlations-summary.docx")
