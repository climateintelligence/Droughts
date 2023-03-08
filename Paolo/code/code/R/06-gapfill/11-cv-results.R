# overview of cv gapfill

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(fs)



# read in data ------------------------------------------------------------

# meta
dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_long_HN_HS.rds")

# need aux ref?
path_aux <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/aux-ref-parameter/cv-1day/"
# aux_folders <- dir_ls(path_aux)
# dat_aux1 <- foreach(
#   i_folder = aux_folders,
#   .final = rbindlist
# ) %do% {
#   
#   month_folders <- dir_ls(i_folder)
#   
#   foreach(
#     i_month_folder = month_folders,
#     .final = rbindlist
#   ) %do% {
#     
#     files_read <- dir_ls(i_month_folder, recurse = T, type = "file")
#     lapply(files_read, fread) %>% rbindlist
#     
#   }
#   
# }



# 1-5-month
dat_cv1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/cv-01-1day.rds")
dat_cv5 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/cv-02-5day.rds")
dat_cvm <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/cv-03-month.rds")

dat_cv1[, value_diff := value_fill - value_true]
dat_cv5[, value_diff := value_fill - value_true]
dat_cvm[, value_diff := value_fill - value_true]

# corr instead of v_dist
dat_cv1_corr <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/backup/backup_weight_corr_cv-01-1day.rds")
dat_cv1_corr[, value_diff := value_fill - value_true]

# first look --------------------------------------------------------------


dat_cv1[, .N, .(Name)] %>% .$N %>% qplot
dat_cv5[, .N, .(Name)] %>% .$N %>% qplot
dat_cvm[, .N, .(Name)] %>% .$N %>% qplot

dat_cv1[, .N, .(Name, month)] %>% 
  ggplot(aes(N))+
  geom_histogram()+
  facet_wrap(~month)

dat_cv1[is.na(i_rep)]
dat_cv5[is.na(i_rep)]
dat_cvm[is.na(i_year), .N, Name]


# dat_cv1[!is.na(i_rep) & !is.na(value_fill) & month == 12,
#         .(mu = mean(value_diff),
#           sd = sd(value_diff),
#           mae = mean(abs(value_diff)),
#           rmse = sqrt(mean(value_diff*value_diff))),
#         .(Name)]


dat_cv1_summ <- dat_cv1[!is.na(i_rep) & !is.na(value_fill),
                        .(mu_diff = mean(value_diff),
                          mu_true = mean(value_true),
                          mu_true_nozero = mean(value_true[value_true != 0]),
                          sd = sd(value_diff),
                          mae = mean(abs(value_diff)),
                          mae_nozero = mean(abs(value_diff[value_true != 0])),
                          # mape = mean(abs(value_diff/value_true)),
                          mape_nozero = mean(abs(value_diff[value_true != 0]/value_true[value_true != 0])),
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

dat_cv1_summ %>% 
  ggplot(aes(n_rep))+
  geom_histogram()+
  facet_wrap(~month_f)

dat_cv1_summ %>% 
  ggplot(aes(month_f, mae))+
  geom_boxplot()+
  ylim(NA, 20)

dat_cv1_summ %>% 
  ggplot(aes(mae))+
  geom_histogram()+
  facet_wrap(~month_f)

dat_cv1_summ %>%
  .[n_rep > 50] %>% 
  # ggplot(aes(mae, Provider))+
  ggplot(aes(mae, country))+
  geom_boxplot()+
  facet_wrap(~month_f, scales = "free_x")


with(dat_cv1_summ[n_rep < 50], table(Provider, month))
dat_cv1_summ[mae > 80]


dat_cv1_summ %>%
  .[n_rep > 50] %>%
  ggplot(aes(country, mae))+
  geom_boxplot()+
  facet_grid(elev_f~month_f, scales = "free_y")


dat_cv1_summ %>%
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mae))+
  geom_boxplot()+
  facet_wrap(~elev_f, scales = "free_y")

dat_cv1_summ %>%
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mae_nozero))+
  geom_boxplot()+
  facet_wrap(~elev_f, scales = "free_y")


dat_cv1_summ[, 
             .(mu_true = round(mean(mu_true, na.rm = T), 1),
               mu_true_nozero = round(mean(mu_true_nozero, na.rm = T), 1)),
             keyby = .(elev_f, month_f)]


# check relative errors ---------------------------------------------------

dat_cv1_summ %>%
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mae_nozero/mu_true_nozero))+
  geom_boxplot()+
  facet_wrap(~elev_f)+
  scale_y_continuous(labels = scales::percent, limits = c(NA, 2))

dat_cv1_summ %>%
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mape_nozero))+
  geom_boxplot()+
  facet_wrap(~elev_f)+
  scale_y_continuous(labels = scales::percent, limits = c(NA, 2))


dat_cv1_summ %>%
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mae/mu_true))+
  geom_boxplot()+
  facet_wrap(~elev_f)+
  scale_y_continuous(labels = scales::percent, limits = c(NA, 2))


# compare 1 vs 5 day vs month ------------------------------------------------------

# prep data


dat_cv1_summ <- dat_cv1[!is.na(i_rep) & !is.na(value_fill),
                        .(mu = mean(value_diff),
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
                        .(mu = mean(value_diff),
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
                        .(mu = mean(value_diff),
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



# single plots

dat_cv1_summ %>%
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mae))+
  geom_boxplot()+
  facet_wrap(~elev_f, scales = "free_y")

dat_cv5_summ %>%
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mae))+
  geom_boxplot()+
  facet_wrap(~elev_f, scales = "free_y")

dat_cvm_summ %>%
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mae))+
  geom_boxplot()+
  facet_wrap(~elev_f, scales = "free_y")

dat_cv1_summ %>%
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mae_nozero))+
  geom_boxplot()+
  facet_wrap(~elev_f, scales = "free_y")


dat_cvm_summ %>% 
  ggplot(aes(n_rep))+
  geom_histogram()+
  facet_wrap(~month_f)


# combine all three

dat_all <- rbindlist(list(cv1 = dat_cv1_summ,
                          cv5 = dat_cv5_summ,
                          cvm = dat_cvm_summ),
                     idcol = "length_na", use.names = T, fill = T)

dat_all %>% 
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mae, fill = length_na))+
  geom_boxplot()+
  facet_wrap(~elev_f, scales = "free_y")


dat_all %>% 
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mae_nozero, fill = length_na))+
  geom_boxplot()+
  facet_wrap(~elev_f, scales = "free_y")

dat_all %>% 
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mu, fill = length_na))+
  geom_boxplot()+
  facet_wrap(~elev_f, scales = "free_y")


dat_all %>% 
  .[n_rep > 50 & length_na == "cvm"] %>% 
  melt(measure.vars = c("mae", "mae_nozero"), id.vars = c("elev_f", "month_f")) %>% 
  ggplot(aes(month_f, value, fill = variable))+
  geom_boxplot()+
  facet_wrap(~elev_f, scales = "free_y")


# ->
# errors acceptable
# cvm > cv5 = cv1
# mae_nozero vs mae: errors larger the later in the snow season, 
#                    especially melting at high alt not reliable


# check HN ----------------------------------------------------------------

dat_hn_cv1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/cv-04-1day-HN.rds")
dat_hn_cv1
dat_hn_cv1[, value_diff := value_fill - value_true]

dat_hn_cv1[, .N, .(Name)] %>% .$N %>% qplot

dat_hn_cv1[, .N, .(Name, month)] %>% 
  ggplot(aes(N))+
  geom_histogram()+
  facet_wrap(~month)

# forgot to check all missing HN stations

dat_hn_cv1[is.na(i_rep)]

with(dat_hn_cv1, table(tr = value_true == 0, fi = value_fill == 0))


dat_hn_cv1_summ <- dat_hn_cv1[!is.na(i_rep) & !is.na(month) & !is.na(value_fill),
                              .(mu_diff = mean(value_diff),
                                mu_true = mean(value_true),
                                mu_true_nozero = mean(value_true[value_true != 0]),
                                sd = sd(value_diff),
                                mae = mean(abs(value_diff)),
                                mae_nozero = mean(abs(value_diff[value_true != 0])),
                                mae_nozero_both = mean(abs(value_diff[value_true != 0 & value_fill != 0])),
                                rmse = sqrt(mean(value_diff*value_diff)),
                                n_rep = .N),
                              .(Name, month)]

dat_hn_cv1_summ %>% 
  merge(dat_meta, by = "Name") -> dat_hn_cv1_summ
dat_hn_cv1_summ[, country := substr(Provider, 1, 2)]
dat_hn_cv1_summ[, month_f := factor(month.abb[month], levels = month.abb[c(10:12, 1:9)])]
dat_hn_cv1_summ[, elev_f := cut(Elevation, 
                             breaks = c(0, 500, 1000, 1500, 2000, 3000),
                             dig.lab = 4,
                             include.lowest = F)]
dat_hn_cv1_summ %>% summary
with(dat_hn_cv1_summ, table(month_f, country))

dat_hn_cv1_summ %>% 
  ggplot(aes(n_rep))+
  geom_histogram()+
  facet_wrap(~month_f)

dat_hn_cv1_summ %>%
  .[n_rep > 50] %>% 
  # ggplot(aes(mae, Provider))+
  ggplot(aes(mae, country))+
  geom_boxplot()+
  facet_wrap(~month_f, scales = "free_x")


with(dat_hn_cv1_summ[n_rep < 50], table(Provider, month_f))

# dat_hn_cv1_summ %>%
#   .[n_rep > 50] %>%
#   ggplot(aes(country, mae))+
#   geom_boxplot()+
#   facet_grid(elev_f~month_f, scales = "free_y")


dat_hn_cv1_summ %>%
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mae))+
  geom_boxplot()+
  facet_wrap(~elev_f, scales = "free_y")

with(dat_hn_cv1_summ, table(elev_f, month_f))
with(dat_hn_cv1_summ[!is.na(mae_nozero)], table(elev_f, month_f))
with(dat_hn_cv1_summ[!is.na(mae_nozero_both)], table(elev_f, month_f))

dat_hn_cv1_summ %>%
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mae_nozero))+
  geom_boxplot()+
  facet_wrap(~elev_f)

dat_hn_cv1_summ %>%
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mae_nozero_both))+
  geom_boxplot()+
  facet_wrap(~elev_f, scales = "free_y")

dat_hn_cv1_summ[, 
                .(mu_true = mean(mu_true, na.rm = T),
                  mu_true_nozero = mean(mu_true_nozero, na.rm = T)),
                keyby = .(elev_f, month_f)]

# seems we can fill HN, too?


dat_hn_cv1_summ %>%
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mae_nozero/mu_true_nozero))+
  geom_boxplot()+
  facet_wrap(~elev_f)+
  scale_y_continuous(labels = scales::percent, limits = c(NA, 2))

# but relateive errors are higher!

# check monthly again, for dependence of year -------------------------



dat_cvm_summ2 <- dat_cvm[!is.na(i_year) & !is.na(value_fill),
                        .(mu = mean(value_diff),
                          sd = sd(value_diff),
                          mae = mean(abs(value_diff)),
                          mae_nozero = mean(abs(value_diff[value_true != 0])),
                          rmse = sqrt(mean(value_diff*value_diff)),
                          n_rep = .N),
                        .(Name, month, year = i_year)]

dat_cvm_summ2 %>% 
  merge(dat_meta, by = "Name") -> dat_cvm_summ2
dat_cvm_summ2[, country := substr(Provider, 1, 2)]
dat_cvm_summ2[, month_f := factor(month.abb[month], levels = month.abb[c(10:12, 1:9)])]
dat_cvm_summ2[, elev_f := cut(Elevation, 
                             breaks = c(0, 500, 1000, 1500, 2000, 3000),
                             dig.lab = 4,
                             include.lowest = F)]


dat_cvm_summ2[n_rep > 15] %>% 
  ggplot(aes(factor(year), mae_nozero))+
  geom_boxplot()+
  facet_wrap(~month_f)+
  ylim(NA, 20)

dat_cvm_summ2[n_rep > 15, 
              .(mu_mae_nozero = mean(mae_nozero, na.rm = T)),
              .(year, month_f)] %>% 
  ggplot(aes(year, mu_mae_nozero))+
  geom_point()+
  geom_smooth()+
  facet_wrap(~month_f, scales = "free_y")


# -> no strong depenendance on year


# check ref parameter -----------------------------------------------------



# some ideas for final plot -----------------------------------------------



dat_cv1_summ[n_rep > 50] %>% 
  ggplot(aes(month_f, mae))+
  geom_boxplot()+
  facet_wrap(~elev_f, nrow = 1, scales = "free_y")

dat_cv1_summ[n_rep > 50] %>% 
  ggplot(aes(month_f, mae_nozero))+
  geom_boxplot()+
  facet_wrap(~elev_f, nrow = 1, scales = "free_y")

dat_cv1_summ[n_rep > 50] %>% 
  ggplot(aes(month_f, mae_nozero/mu_true_nozero))+
  geom_boxplot()+
  facet_wrap(~elev_f, nrow = 1)+
  scale_y_continuous(labels = scales::percent, limits = c(NA, 1.6))


# -> rbind plots together



# compare dist_v vs corr --------------------------------------------------


# prep data

dat_cv1_summ <- dat_cv1[!is.na(i_rep) & !is.na(value_fill),
                        .(mu = mean(value_diff),
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


dat_cv1_corr_summ <- dat_cv1_corr[!is.na(i_rep) & !is.na(value_fill),
                                  .(mu = mean(value_diff),
                                    sd = sd(value_diff),
                                    mae = mean(abs(value_diff)),
                                    mae_nozero = mean(abs(value_diff[value_true != 0])),
                                    rmse = sqrt(mean(value_diff*value_diff)),
                                    n_rep = .N),
                                  .(Name, month)]

dat_cv1_corr_summ %>% 
  merge(dat_meta, by = "Name") -> dat_cv1_corr_summ
dat_cv1_corr_summ[, country := substr(Provider, 1, 2)]
dat_cv1_corr_summ[, month_f := factor(month.abb[month], levels = month.abb[c(10:12, 1:9)])]
dat_cv1_corr_summ[, elev_f := cut(Elevation, 
                                  breaks = c(0, 500, 1000, 1500, 2000, 3000),
                                  dig.lab = 4,
                                  include.lowest = F)]
dat_cv1_corr_summ %>% summary
with(dat_cv1_corr_summ, table(month_f, country))



# combine all three

dat_all <- rbindlist(list(dist_v = dat_cv1_summ,
                          corr = dat_cv1_corr_summ),
                     idcol = "weight_by", use.names = T, fill = T)

dat_all %>% 
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mae, fill = weight_by))+
  geom_boxplot()+
  facet_wrap(~elev_f, scales = "free_y")


dat_all %>% 
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mae_nozero, fill = weight_by))+
  geom_boxplot()+
  facet_wrap(~elev_f, scales = "free_y")

dat_all %>% 
  .[n_rep > 50] %>%
  ggplot(aes(month_f, mu, fill = weight_by))+
  geom_boxplot()+
  facet_wrap(~elev_f, scales = "free_y")



# -> dist_v slightly worse in a few cases
#    but differences are minimal...






# EOF ---------------------------------------------------------------------



