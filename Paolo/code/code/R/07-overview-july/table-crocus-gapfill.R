# tables for overview meeting
# - crocus comparison
# - benefits of gapfilling





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

# dat_fr_sub <- dat_fr[hydro_year %between% c(1981, 2010) & month(date) %in% c(11,12,1:5)]


dat_eval <- dat_fr[hs_fillcode == 222 & is.na(hs_obs) &
                     month(date) %in% c(11,12,1:5) & 
                     hydro_year %between% c(1981, 2010)]
# dat_eval[, value_diff := hs_fill - hs_crocus_assim]
dat_eval[, value_diff := hs_crocus_assim - hs_fill]
dat_eval %>% summary

# dat_eval[!is.na(hs_obs)]
dat_eval[abs(value_diff) > 400]

dat_crocus <- dat_eval[,
                     .(mu_diff_crocus = mean(value_diff),
                       #sd = sd(value_diff),
                       mad_crocus = mean(abs(value_diff)),
                       mad_crocus_bias_removed = mean(abs(value_diff - mean(value_diff))),
                       n_rep_crocus = .N),
                     .(stn_name, month(date))]


# prep cv
dat_cvm <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/cv-03-month.rds")
dat_cvm2 <- dat_cvm[Name %in% stn_fr]
dat_cvm2[, value_diff := value_fill - value_true]

dat_cvm_summ <- dat_cvm2[!is.na(i_year) & !is.na(value_fill),
                         .(mu_diff_cv = mean(value_diff),
                           # mu_true = mean(value_true),
                           # mu_true_nozero = mean(value_true[value_true != 0]),
                           # sd = sd(value_diff),
                           mae_cv = mean(abs(value_diff)),
                           # mae_nozero = mean(abs(value_diff[value_true != 0])),
                           # rmse = sqrt(mean(value_diff*value_diff)),
                           n_rep_cv = .N),
                         .(stn_name = Name, month)]




# merge cv and crocus
dat_all <- merge(dat_crocus, dat_cvm_summ, all = T)

dat_all[is.na(n_rep_cv), table(month)]
dat_all[is.na(n_rep_crocus)]
# -> only take common stations
dat_all2 <- merge(dat_crocus, dat_cvm_summ, all = F)

summary(dat_all2)
with(dat_all2, qplot(n_rep_crocus, n_rep_cv))

dat_all2 %>% 
  merge(dat_meta, by = "stn_name") -> dat_all2
dat_all2[, month_f := factor(month.abb[month], levels = month.abb[c(10:12, 1:9)])]
dat_all2[, elev_f := cut(elev, 
                         breaks = c(500, 1000, 1500, 2000, 3000),
                         dig.lab = 4,
                         include.lowest = F)]


# table crocus ------------------------------------------------------------



dat_tbl <- dat_all2[n_rep_cv > 50 & n_rep_crocus > 50,
                    .(bias_crocus = mean(mu_diff_crocus),
                      mad_crocus = mean(mad_crocus),
                      mad_crocus_bias_removed = mean(mad_crocus_bias_removed),
                      bias_cv = mean(mu_diff_cv),
                      mae_cv = mean(mae_cv),
                      n_stn = .N),
                    .(elev_f, month_f)]

setkey(dat_tbl, elev_f, month_f)

dat_tbl %>% 
  # flextable() %>% 
  flextable(col_keys = c("elev_f", "month_f", 
                         "bias_crocus", "bias_cv",
                         "mad_crocus", "mad_crocus_bias_removed", "mae_cv")) %>%
  # set_header_labels(elev_f = "elevation") %>% 
  set_formatter_type(fmt_double = "%.01f") %>% 
  merge_v(j = "elev_f") %>% 
  align(j = "elev_f", align = "right") %>% 
  valign(j = "elev_f", valign = "top") %>% 
  fix_border_issues() %>% 
  autofit() -> ft

read_docx() %>% 
  body_add_flextable(ft) %>% 
  print(target = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/03_july_meeting/table_crocus.docx")



# extra plot crocus -------------------------------------------------------


dat_all2[n_rep_cv > 50 & n_rep_crocus > 50] %>% 
  melt(measure.vars = c("mu_diff_crocus",
                        "mad_crocus",
                        "mad_crocus_bias_removed",
                        "mu_diff_cv",
                        "mae_cv")) %>% 
  ggplot(aes(month_f, value))+
  geom_boxplot()+
  facet_grid(variable ~ elev_f, scales = "free_y")


# -> not needed



# number of gaps filled ---------------------------------------------------

# daily
dat_nfill <- dat_gapfill[, .N, .(HS_fillcode)]
dat_nfill_month <- dat_gapfill[, .N, .(HS_fillcode, month(Date))]

dat_nfill_month %>% 
  dcast(HS_fillcode ~ month)

dat_nfill[, perc := N / sum(N)]
dat_nfill[, N_mill := N / 1e6]

dat_nfill %>% 
  flextable(col_keys = c("HS_fillcode", "N_mill")) %>% 
  set_formatter_type() %>% 
  autofit -> ft_daily

# stn years
setnames(dat_gapfill, "Date", "date")
mitmatmisc::add_hydro_year(dat_gapfill)
dat_gapfill[, month := month(date)]

dat1 <- dat_gapfill[HS_fillcode == 1 & month %in% c(11,12,1:5)]
dat1_summ <- dat1[, 
                  .(nn = .N,
                    nn_max = days_in_month(date[1])),
                  .(Name, month, hydro_year)]


dat2 <- dat_gapfill[!is.na(HS_fillcode) & month %in% c(11,12,1:5)]
dat2_summ <- dat2[, 
                  .(nn = .N,
                    nn_max = days_in_month(date[1])),
                  .(Name, month, hydro_year)]


dat_tbl_month <- merge(
  dat1_summ[nn > 0.9 * nn_max, .(n_before = .N), month],
  dat2_summ[nn > 0.9 * nn_max, .(n_after = .N), month],
  by = "month"
)

dat_tbl_month[, month_f := factor(month.abb[month], levels = month.abb[c(10:12,1:9)])]
setkey(dat_tbl_month, month_f)

dat_tbl_month[, perc_inc := n_after/n_before - 1]

# dat_tbl_month %>% 
#   flextable(col_keys = c("month_f", "n_before", "n_after", "perc_inc")) %>% 
#   set_formatter(perc_inc = scales::percent_format(1)) %>% 
#   autofit() -> ft_monthly

dat_tbl_month %>% 
  flextable(col_keys = c("month_f", "n_before", "n_after")) %>% 
  autofit() -> ft_monthly

read_docx() %>% 
  body_add_flextable(ft_daily) %>% 
  body_add_par(" ") %>% 
  body_add_flextable(ft_monthly) %>% 
  print(target = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/03_july_meeting/table_gaps_filled.docx")
