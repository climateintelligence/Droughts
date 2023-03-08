# visualize gapfill cross validation results
# - 



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



# read data ---------------------------------------------------------------


# meta
dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_long_HN_HS.rds")

# 1-5-month
dat_cv1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/cv-01-1day.rds")
dat_cv5 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/cv-02-5day.rds")
dat_cvm <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/cv-03-month.rds")

dat_cv1[, value_diff := value_fill - value_true]
dat_cv5[, value_diff := value_fill - value_true]
dat_cvm[, value_diff := value_fill - value_true]






# prep data ---------------------------------------------------------------

dat_cv1_summ <- dat_cv1[!is.na(i_rep) & !is.na(value_fill),
                        .(mu_diff = mean(value_diff),
                          mu_true = mean(value_true),
                          mu_true_nozero = mean(value_true[value_true != 0]),
                          sd = sd(value_diff),
                          mae = mean(abs(value_diff)),
                          mae_nozero = mean(abs(value_diff[value_true != 0])),
                          rmse = sqrt(mean(value_diff*value_diff)),
                          n_rep = .N),
                        .(Name, month)]

dat_cv1_summ %>% 
  merge(dat_meta, by = "Name") -> dat_cv1_summ
dat_cv1_summ[, country := substr(Provider, 1, 2)]
dat_cv1_summ[, month_f := factor(month.abb[month], levels = month.abb[c(10:12, 1:9)])]
dat_cv1_summ[, elev_f := cut(Elevation, 
                             breaks = c(0, 500, 1000, 1500, 2000, 3000),
                             dig.lab = 4,
                             include.lowest = F)]
dat_cv1_summ %>% summary
with(dat_cv1_summ, table(month_f, country))



dat_cv5_summ <- dat_cv5[!is.na(i_rep) & !is.na(value_fill),
                        .(mu_diff = mean(value_diff),
                          mu_true = mean(value_true),
                          mu_true_nozero = mean(value_true[value_true != 0]),
                          sd = sd(value_diff),
                          mae = mean(abs(value_diff)),
                          mae_nozero = mean(abs(value_diff[value_true != 0])),
                          rmse = sqrt(mean(value_diff*value_diff)),
                          n_rep = .N),
                        .(Name, month)]

dat_cv5_summ %>% 
  merge(dat_meta, by = "Name") -> dat_cv5_summ
dat_cv5_summ[, country := substr(Provider, 1, 2)]
dat_cv5_summ[, month_f := factor(month.abb[month], levels = month.abb[c(10:12, 1:9)])]
dat_cv5_summ[, elev_f := cut(Elevation, 
                             breaks = c(0, 500, 1000, 1500, 2000, 3000),
                             dig.lab = 4,
                             include.lowest = F)]
dat_cv5_summ %>% summary
with(dat_cv5_summ, table(month_f, country))




dat_cvm_summ <- dat_cvm[!is.na(i_year) & !is.na(value_fill),
                        .(mu_diff = mean(value_diff),
                          mu_true = mean(value_true),
                          mu_true_nozero = mean(value_true[value_true != 0]),
                          sd = sd(value_diff),
                          mae = mean(abs(value_diff)),
                          mae_nozero = mean(abs(value_diff[value_true != 0])),
                          rmse = sqrt(mean(value_diff*value_diff)),
                          n_rep = .N),
                        .(Name, month)]

dat_cvm_summ %>% 
  merge(dat_meta, by = "Name") -> dat_cvm_summ
dat_cvm_summ[, country := substr(Provider, 1, 2)]
dat_cvm_summ[, month_f := factor(month.abb[month], levels = month.abb[c(10:12, 1:9)])]
dat_cvm_summ[, elev_f := cut(Elevation, 
                             breaks = c(0, 500, 1000, 1500, 2000, 3000),
                             dig.lab = 4,
                             include.lowest = F)]
dat_cvm_summ %>% summary
with(dat_cvm_summ, table(month_f, country))


# combine all
dat_all <- rbindlist(list(cv1 = dat_cv1_summ,
                          cv5 = dat_cv5_summ,
                          cvm = dat_cvm_summ),
                     idcol = "length_na", use.names = T, fill = T)

dat_all[, elev_f2 := cut(Elevation, 
                         breaks = c(0, 1000, 2000, 3000),
                         dig.lab = 4,
                         include.lowest = F)]



# overview table ----------------------------------------------------------

dat_tbl <- dat_all[n_rep > 50,
        .(bias = mean(mu_diff),
          mae = mean(mae),
          mae_q25 = quantile(mae, 0.25),
          mae_q75 = quantile(mae, 0.75),
          mae_nozero = mean(mae_nozero, na.rm = T),
          mae_nozero_q25 = quantile(mae_nozero, 0.25, na.rm = T),
          mae_nozero_q75 = quantile(mae_nozero, 0.75, na.rm = T),
          rel_mae_nozero = mean(mae_nozero/mu_true_nozero, na.rm = T)),
        .(length_na, elev_f2)]

dat_tbl[, length_removed := fct_recode(length_na,
                                       "1 day" = "cv1",
                                       "5 days" = "cv5",
                                       "1 month" = "cvm")]
setorder(dat_tbl, elev_f2)

dat_tbl %>% 
  flextable(col_keys = c("elev_f2", "length_removed", "bias", "mae", "mae_nozero", "rel_mae_nozero")) %>% 
  set_header_labels(elev_f2 = "Elevation", length_removed = "CV period",
                    bias = "Bias", mae = "MAE", mae_nozero = "MAE no zero", 
                    rel_mae_nozero = "Rel. MAE no zero") %>% 
  set_formatter_type(fmt_double = "%.01f") %>% 
  set_formatter(rel_mae_nozero = scales::percent_format(accuracy = 0.1)) %>% 
  merge_v(j = "elev_f2") %>% 
  align(j = "elev_f2", align = "right") %>% 
  valign(j = "elev_f2", valign = "top") %>% 
  fix_border_issues() %>% 
  autofit() -> ft

read_docx() %>% 
  body_add_flextable(ft) %>% 
  print(target = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/table/Table A1.docx")


# plot --------------------------------------------------------------------


dat_plot <- dat_all[n_rep > 50 & length_na == "cvm"]
dat_plot[, elev_f2 := paste0(elev_f2, "m")]


gg1 <- dat_plot %>% 
  ggplot(aes(month_f, mu_diff))+
  geom_boxplot(outlier.size = 0.5)+
  # facet_wrap(~elev_f2, nrow = 1, scales = "free_y")+
  facet_wrap(~elev_f2, nrow = 1)+
  theme_bw()+
  xlab(NULL)+
  ylab("Bias [cm]")


gg2a <- dat_plot %>% 
  ggplot(aes(month_f, mae))+
  geom_boxplot(outlier.size = 0.5)+
  facet_wrap(~elev_f2, nrow = 1, scales = "free_y")+
  theme_bw()+
  xlab(NULL)+
  ylab("MAE [cm]")


gg2b <- dat_plot %>% 
  ggplot(aes(month_f, mae_nozero))+
  geom_boxplot(outlier.size = 0.5)+
  facet_wrap(~elev_f2, nrow = 1, scales = "free_y")+
  theme_bw()+
  xlab(NULL)+
  ylab("MAE no zero [cm]")


gg3 <- dat_plot %>% 
  ggplot(aes(month_f, mae_nozero/mu_true_nozero))+
  geom_boxplot(outlier.size = 0.5)+
  facet_wrap(~elev_f2, nrow = 1)+
  scale_y_continuous(labels = scales::percent, limits = c(NA, 2))+
  # scale_y_continuous(labels = scales::percent)+
  # facet_wrap(~elev_f2, nrow = 1, scales = "free_y")+
  theme_bw()+
  xlab(NULL)+
  ylab("relative MAE nozero")


gg_all <- wrap_plots(gg1, gg2a, gg2b, gg3, ncol = 1)+
  plot_annotation(tag_levels = "a", tag_suffix = ")", tag_prefix = "(")

ggsave(gg_all,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/Figure A2.png", 
       width = 8, height = 10)


# some numbers ------------------------------------------------------------


dat_all[n_rep > 50 & length_na == "cvm"] %>% summary

