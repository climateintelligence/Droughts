# compare Crocus gapfill to Alice's functions



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(patchwork)
library(scico)
library(forcats)
library(fs)
library(officer)
library(flextable)



# prep data ---------------------------------------------------------------

dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/crocus-01-meta.rds")

dat_crocus <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/crocus-02-data.rds")
dat_gapfill <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds")
dat_gapfill

stn_fr <- unique(dat_crocus$stn_name)
dat_gapfill2 <- dat_gapfill[Name %in% stn_fr]

dat_fr <- merge(dat_gapfill2[, .(stn_name = Name, date = Date, hs_fill = HS, hs_fillcode = HS_fillcode)],
                dat_crocus,
                by = c("stn_name", "date"),
                all = F)

dat_fr
mitmatmisc::add_hydro_year(dat_fr)




# some metrics ------------------------------------------------------------

dat_eval <- dat_fr[hs_fillcode == 222 & 
                     month(date) %in% c(11,12,1:5) & 
                     year(date) %between% c(1981, 2010)]
dat_eval[, value_diff := hs_fill - hs_crocus_assim]
dat_summ <- dat_eval[,
                     .(mu_diff = mean(value_diff),
                       mu_crocus = mean(hs_crocus),
                       mu_crocus_nozero = mean(hs_crocus[hs_crocus != 0]),
                       sd = sd(value_diff),
                       mae = mean(abs(value_diff)),
                       mae_nozero = mean(abs(value_diff[hs_crocus != 0])),
                       rmse = sqrt(mean(value_diff*value_diff)),
                       n_rep = .N),
                     .(stn_name, month(date))]

dat_summ %>% summary

# remove bias from mae
dat_summ <- dat_eval[,
                     .(mu_diff = mean(value_diff),
                       mu_crocus = mean(hs_crocus),
                       mu_crocus_nozero = mean(hs_crocus[hs_crocus != 0]),
                       sd = sd(value_diff),
                       mae = mean(abs(value_diff - mean(value_diff))),
                       mae_nozero = mean(abs(value_diff[hs_crocus != 0] - mean(value_diff[hs_crocus != 0]))),
                       rmse = sqrt(mean(value_diff*value_diff)),
                       n_rep = .N),
                     .(stn_name, month(date))]


dat_summ <- dat_eval[,
                     .(mu_diff = mean(value_diff),
                       sd = sd(value_diff),
                       mae = mean(abs(value_diff)),
                       mae_bias_removed = mean(abs(value_diff - mean(value_diff))),
                       n_rep = .N),
                     .(stn_name, month(date))]


dat_summ %>% 
  merge(dat_meta, by = "stn_name") -> dat_summ
dat_summ[, month_f := factor(month.abb[month], levels = month.abb[c(10:12, 1:9)])]
dat_summ[, elev_f := cut(elev, 
                         breaks = c(0, 1000, 2000, 3000),
                         dig.lab = 4,
                         include.lowest = F)]


# ** tbl ------------------------------------------------------------------



dat_tbl <- dat_summ[n_rep > 50,
                   .(bias = mean(mu_diff),
                     mae = mean(mae),
                     mae_bias_removed = mean(mae_bias_removed)),
                   .(elev_f, month_f)]

setkey(dat_tbl, elev_f, month_f)

dat_tbl %>% 
  flextable(col_keys = c("elev_f", "month_f", "bias", "mae", "mae_bias_removed")) %>% 
  set_header_labels(elev_f = "elevation") %>% 
  set_formatter_type(fmt_double = "%.01f") %>% 
  merge_v(j = "elev_f") %>% 
  align(j = "elev_f", align = "right") %>% 
  valign(j = "elev_f", valign = "top") %>% 
  fix_border_issues() %>% 
  autofit()



# ** plot -----------------------------------------------------------------


dat_plot <- dat_summ[n_rep > 50]
# dat_plot[, elev_f2 := paste0(elev_f2, "m")]

dat_plot %>% 
  melt(measure.vars = c("mae", "mae_bias_removed")) %>% 
  ggplot(aes(month_f, value, fill = variable))+
  geom_boxplot(outlier.size = 0.5)+
  facet_wrap(~elev_f, nrow = 1, scales = "free_y")+
  scale_fill_brewer(palette = "Set1")+
  theme_bw()+
  xlab(NULL)+
  ylab("MAE / MAE_nozero [cm]")



