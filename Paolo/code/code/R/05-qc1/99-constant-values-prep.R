# some figures for province meeting

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(directlabels)
library(readxl)
library(writexl)
library(foreach)
library(runner)


# prep data ---------------------------------------------------------------



dat_meta_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_long_HN_HS.rds")
dat_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/data_long_HN_HS.rds")

dat_meta_2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/meta_long_HN_HS.rds")
dat_2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/data_long_HN_HS.rds")



# dat_1[, month := month(Date)]
# dat_2[, month := month(Date)]
# 
# dat_1[, season_year := year(Date)]
# dat_2[, season_year := year(Date)]
# 
# dat_1[month <= 8, season_year := season_year - 1L]
# dat_2[month <= 8, season_year := season_year - 1L]

dat_all <- rbind(dat_2, dat_1)
setkey(dat_all, Provider, Name, Date)

dat_meta <- unique(rbind(dat_meta_1, dat_meta_2))

dat_all[, HN := as.integer(round(HN))]
dat_all[, HS := as.integer(round(HS))]

dat_all[, HN_lag := shift(HN, type = "lag"), .(Provider, Name)]
dat_all[, HS_lag := shift(HS, type = "lag"), .(Provider, Name)]
dat_all[, HS_diff := HS - HS_lag]





# 5 day window diffs ------------------------------------------------------

dat_all[, 
        HS_diff_sum5 := sum_run(HS_diff, 5, na_pad = T),
        by = .(Name)]
dat_all[, 
        HS_diff_sum10 := sum_run(HS_diff, 10, na_pad = T),
        by = .(Name)]


dat_check <- dat_all[HS_diff_sum5 == 0 & HS != 0]
dat_check <- dat_all[HS_diff_sum10 == 0 & HS != 0]

dat_check[, .N, .(Provider, Name)] %>% .[, N] %>% qplot()
dat_check[, .N, .(Provider, Name)] %>% .[, N] %>% table
dat_check[, .N, .(Provider)]


dat_check_sum <- dat_all[!is.na(HS) & !is.na(HS_diff_sum5), 
                         .(nn = .N,
                           nn_non_zero = sum(HS != 0),
                           nn_cons = sum(HS_diff_sum5 == 0 & HS != 0)),
                         .(Name)]
with(dat_check_sum, plot(nn, nn_cons))
with(dat_check_sum, plot(nn_non_zero, nn_cons))

dat_check_sum[, summary(nn_cons/nn_non_zero)]
dat_check_sum[, qplot(nn_cons/nn_non_zero)]


# zero NA -------------------------------------------------




dat_summ <- dat_all[!is.na(HS) & month %in% c(12, 1:2) &
                      Name != "La_Thuile_Ospizio_Piccolo_San_Bernardo",
                    .(n_avail = .N,
                      mean = mean(HS),
                      n_zero = sum(HS == 0),
                      mean_non_zero = mean(HS[HS != 0]),
                      n_low = sum(HS < 10),
                      mean_non_low = mean(HS[HS > 10])),
                    .(Name)]

with(dat_summ[n_zero != 0], plot(n_zero/n_avail, mean_non_zero))
with(dat_summ[n_zero != 0 & n_zero/n_avail > 0.2], plot(n_zero/n_avail, mean_non_zero))
with(dat_summ[n_zero != 0], plot(n_zero/n_avail, mean/mean_non_zero))
with(dat_summ[n_low != 0], plot(n_low/n_avail, mean_non_low))
dat_summ[n_zero == 0]
with(dat_summ[n_zero != 0], plot(mean, mean_non_zero))
with(dat_summ[n_low != 0], plot(mean, mean_non_low))



dat_summ2 <- merge(dat_summ, dat_meta[, .(Name, Elevation)])


dat_summ2 %>% 
  ggplot(aes(Elevation, mean, colour = n_zero/n_avail))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)

# check the outlier
# dat_summ2[mean > 500]
# dat_zz <- dat_all[Name == "La_Thuile_Ospizio_Piccolo_San_Bernardo"]
# dat_zz[!is.na(HS)]

dat_summ2[Elevation > 1000] %>% 
  ggplot(aes(Elevation, mean_non_zero, 
             colour = n_zero/n_avail, alpha = n_zero/n_avail > 0.2))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)

dat_summ2[Elevation > 1000 & n_zero/n_avail > 0.2]





dat_summ_month <- dat_all[!is.na(HS) & month %in% c(12, 1:2) &
                            Name != "La_Thuile_Ospizio_Piccolo_San_Bernardo",
                          .(n_avail = .N,
                            mean = mean(HS),
                            n_zero = sum(HS == 0),
                            mean_non_zero = mean(HS[HS != 0]),
                            n_low = sum(HS < 10),
                            mean_non_low = mean(HS[HS > 10])),
                          .(Name, month)]



dat_summ_month2 <- merge(dat_summ_month, dat_meta[, .(Name, Elevation)])

dat_summ_month2 %>% 
  ggplot(aes(Elevation, mean, colour = n_zero/n_avail))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)+
  facet_wrap(~month, scales = "free_y")



# try a moving window detection -------------------------------------------


dat_values <- dat_all[!is.na(HS) & month %in% c(12, 1:2) &
                      Name != "La_Thuile_Ospizio_Piccolo_San_Bernardo",
                    .(n_avail = .N,
                      mean_hs = mean(HS),
                      n_zero = sum(HS == 0),
                      mean_non_zero = mean(HS[HS != 0])),
                    .(Name)]
dat_values[, frac_zero := n_zero / n_avail]


dat_loop <- merge(dat_values[, .(Name)], 
                  dat_meta[, .(Name, Elevation)])
dat_loop[, delta_alt := 100]
dat_loop[Elevation < 300, delta_alt := 200]
dat_loop[Elevation > 1000, delta_alt := 200]
dat_loop[Elevation > 2000, delta_alt := 300]
dat_loop[Elevation > 2500, delta_alt := 500]

dat_loop$Elevation %>% qplot(binwidth = 50)

dat_loop_out <- foreach(
  i = 1:nrow(dat_loop),
  .final = rbindlist
) %do% {
  
  i_stn <- dat_loop[i, Name]
  elev_range <- dat_loop[i, Elevation + delta_alt * c(-1, 1)]
  stns_range <- dat_loop[Elevation %between% elev_range, Name]
  
  xx_frac_zero <-  dat_values[Name %chin% stns_range, frac_zero]
  xx_mean_hs <-  dat_values[Name %chin% stns_range, mean_hs]
  
  data.table(
    dat_loop[i],
    frac_zero_q05 = quantile(xx_frac_zero, 0.05),
    frac_zero_q95 = quantile(xx_frac_zero, 0.95),
    mean_hs_q05 = quantile(xx_mean_hs, 0.05),
    mean_hs_q95 = quantile(xx_mean_hs, 0.95),
    
    frac_zero_mean = mean(xx_frac_zero),
    frac_zero_sd = sd(xx_frac_zero),
    mean_hs_mean = mean(xx_mean_hs),
    mean_hs_sd = sd(xx_mean_hs)
  )
  
  
}


# both bounds --------------------------------------------------------



dat_plot <- merge(dat_values, dat_loop_out, by = c("Name"))
dat_plot[, in_zero_qr := frac_zero %between% list(frac_zero_q05, frac_zero_q95)]
dat_plot[, in_hs_qr := mean_hs %between% list(mean_hs_q05, mean_hs_q95)]
dat_plot[, in_zero_sd := (frac_zero - frac_zero_mean) %between% 
           list(-1.96 * frac_zero_sd, +1.96 * frac_zero_sd)]
dat_plot[, in_hs_sd := (mean_hs - mean_hs_mean) %between% 
           list(-1.96 * mean_hs_sd, +1.96 * mean_hs_sd)]

dat_plot %>% with(table(in_zero_qr, in_zero_sd))
dat_plot %>% with(table(in_hs_qr, in_hs_sd))


dat_plot %>% 
  ggplot(aes(Elevation, mean_hs, colour = frac_zero))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)+
  facet_grid(in_zero_qr ~ in_zero_sd, labeller = label_both)

dat_plot %>% 
  ggplot(aes(Elevation, mean_hs, colour = frac_zero))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)+
  facet_grid(in_hs_qr ~ in_hs_sd, labeller = label_both)



# one bound --------------------------------------------------------

dat_plot[, in_zero_qr := frac_zero %between% list(-Inf, frac_zero_q95)]
dat_plot[, in_hs_qr := mean_hs %between% list(mean_hs_q05, Inf)]
dat_plot[, in_zero_sd := (frac_zero - frac_zero_mean) %between% 
           list(-1.96 * Inf, +1.96 * frac_zero_sd)]
dat_plot[, in_hs_sd := (mean_hs - mean_hs_mean) %between% 
           list(-1.96 * mean_hs_sd, +1.96 * Inf)]

dat_plot %>% with(table(in_zero_qr, in_zero_sd))
dat_plot %>% with(table(in_hs_qr, in_hs_sd))


dat_plot %>% 
  ggplot(aes(Elevation, mean_hs, colour = frac_zero))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)+
  facet_grid(in_zero_qr ~ in_zero_sd, labeller = label_both)

dat_plot %>% 
  ggplot(aes(Elevation, mean_hs, colour = frac_zero))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)+
  facet_grid(in_hs_qr ~ in_hs_sd, labeller = label_both)


dat_plot %>% with(table(in_hs_qr, in_zero_qr))

dat_plot %>% 
  ggplot(aes(Elevation, mean_hs, colour = frac_zero))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)+
  facet_grid(in_hs_qr ~ in_zero_qr, labeller = label_both)






# -> most sense:
#    - quantile range 
#    - one bound
#    - hs or zero outside (not AND)


stns_to_check <- dat_plot[in_zero_qr == FALSE | in_hs_qr == FALSE, Name]




# for manual check: plots and tables ----------------------------------------------------

for(i_stn in stns_to_check){
  
  dat_i_stn <- dat_all[Name == i_stn]
  dat_i_stn[, year := year(Date)]
  
  dat_i_stn[, idn := year - min(year)]
  dat_i_stn[, idn_grp := floor(idn / 2)]
  
  
  pdf(paste0("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/fig/zero-NA/", i_stn, ".pdf"),
      width = 14, height = 7)
  
  for(i_idn_grp in sort(unique(dat_i_stn$idn_grp))){
    
    gg <- 
      dat_i_stn[idn_grp == i_idn_grp] %>% 
      ggplot(aes(Date))+
      geom_point(aes(y = HS, colour = "HS"))+
      geom_point(aes(y = HN, colour = "HN"), size = 0.5)+
      facet_wrap(~year, scales = "free", nrow = 2)+
      scale_x_date(date_labels = "%b")+
      theme_bw()+
      ylab("HS / HN  [cm]")
    
    
    print(gg)
    
    
  }
  
  dev.off()
  
  
  # table
  dat_table <- dat_i_stn[, .(year, Date, HN, HS, zeroNA = NA_integer_)]
  
  setkey(dat_table, Date)
  
  write_xlsx(
    dat_table,
    paste0("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/table/zero-NA/", i_stn, ".xlsx")
  )
  
}

# overview table

data.table(Name = stns_to_check,
           OK = NA,
           zero_NA = NA) %>% 
  write_xlsx("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/table/zero-NA-overview_empty.xlsx")



# EOF ---------------------------------------------------------------------


